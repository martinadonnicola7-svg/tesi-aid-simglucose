"""
run_campaign.py - Esegue tutte le simulazioni della tesi e salva tabelle e grafici in results/.

Uso (dalla cartella principale del repository):
    python3 run_campaign.py

Le simulazioni girano in parallelo su tutti i core disponibili. I risultati intermedi sono salvati in
results/_cache.pkl (escluso da git) e riutilizzati nelle esecuzioni successive; la cache si invalida da sola
se cambiano i controllori (dizionario CFG), il codice di aid_sim.py o la versione di simglucose.
"""
import hashlib
import importlib.metadata as md
import os
import pickle
import warnings
from multiprocessing import Pool

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

import aid_sim as A

warnings.filterwarnings("ignore")
os.makedirs("results", exist_ok=True)

CFG = A.CFG
PAZIENTI = ["adult#001", "adult#002", "adult#003"]
CONFRONTO = ["PID senza AW", "PID", "PID cons.", "PID+IOB+PLGS"]
COL = {"PID senza AW": "tab:gray", "PID": "tab:red", "PID cons.": "tab:orange", "PID+IOB+PLGS": "tab:blue"}

# --- cache (si invalida se cambiano CFG, aid_sim.py o simglucose) ------------------------------
CACHE = "results/_cache.pkl"
SIGNATURE = (repr(sorted((k, sorted(v.items())) for k, v in CFG.items()))
             + hashlib.sha1(open(A.__file__, "rb").read()).hexdigest()
             + md.version("simglucose"))
cache = {}
if os.path.exists(CACHE):
    with open(CACHE, "rb") as f:
        stored = pickle.load(f)
    if stored.get("signature") == SIGNATURE:
        cache = stored["data"]


def _sim(key):
    pat, cho, name, seed, meal_h, assorb = key
    df, _ = A.simulate(pat, cho, CFG[name], seed=seed, meal_clock_h=meal_h, assorb_scale=assorb)
    return key, (df, A.metrics(df, meal_h=meal_h))


def key(pat, cho, name, seed=1, meal_h=A.MEAL_CLOCK_H, assorb=1.0):
    return (pat, float(cho), name, seed, float(meal_h), float(assorb))


def run_all(keys):
    """Esegue in parallelo le simulazioni non ancora in cache e aggiorna la cache."""
    uniche = list(dict.fromkeys(keys))
    todo = [k for k in uniche if k not in cache]
    if todo:
        print(f"Simulazioni da eseguire: {len(todo)} (già in cache: {len(uniche) - len(todo)})")
        with Pool() as pool:
            for k, v in pool.imap_unordered(_sim, todo):
                cache[k] = v
        with open(CACHE, "wb") as f:
            pickle.dump({"signature": SIGNATURE, "data": cache}, f)


def row(k):
    pat, cho, name, seed, meal_h, assorb = k
    m = cache[k][1]
    return {
        "Paziente": pat, "CHO [g]": int(cho), "Controllore": name,
        "TIR [%]": m["TIR"], "TIR 6h post [%]": m["TIR_post6h"], "TAR [%]": m["TAR"], "TAR>250 [%]": m["TAR>250"],
        "TBR [%]": m["TBR"], "TBR<54 [%]": m["TBR<54"], "TBR tardivo [%]": m["TBR_tardivo"],
        "Picco [mg/dL]": m["picco"], "Nadir [mg/dL]": m["nadir"], "Insulina [U]": m["insulina_tot"],
    }


if __name__ == "__main__":
    # --- elenco delle simulazioni ----------------------------------------------------------
    main = [key("adult#001", c, n) for c in [0, 30, 60, 90] for n in CFG]
    main += [key(p, c, n) for p in PAZIENTI[1:] for c in [0, 60, 90] for n in CONFRONTO]
    semi = [key(p, 60, n, seed=s) for p in ["adult#001", "adult#003"] for n in ["PID", "PID cons.", "PID+IOB+PLGS"]
            for s in range(1, 6)]
    orari = [key("adult#001", 60, n, meal_h=h) for h in [10, 12, 14, 16] for n in ["PID", "PID+IOB+PLGS"]]
    assorb = [key("adult#001", 60, n, assorb=s) for s in [0.5, 1.0, 2.0] for n in ["PID", "PID+IOB+PLGS"]]
    run_all(main + semi + orari + assorb)

    # --- 1. campagna principale --------------------------------------------------------------
    R = pd.DataFrame([row(k) for k in main]).round(1)
    R.to_csv("results/metriche.csv", index=False)
    print(R.to_string(index=False))

    # --- 2. robustezza al rumore del sensore: 5 semi ----------------------------------------
    srows = []
    for p in ["adult#001", "adult#003"]:
        for n in ["PID", "PID cons.", "PID+IOB+PLGS"]:
            T = pd.DataFrame([cache[key(p, 60, n, seed=s)][1] for s in range(1, 6)])
            srows.append({"Paziente": p, "Controllore": n,
                          **{c: f"{T[c].mean():.1f} ± {T[c].std():.1f}"
                             for c in ["TIR", "TAR", "TBR", "TBR<54", "TBR_tardivo", "picco", "nadir"]}})
    S = pd.DataFrame(srows)
    S.to_csv("results/robustezza_semi.csv", index=False)
    print("\n", S.to_string(index=False))

    # --- 3. sensibilità all'orario del pasto e alla velocità di assorbimento ----------------
    prow = []
    for k in orari:
        prow.append({"Prova": "orario pasto", "Valore": f"{int(k[4]):02d}:00", **row(k)})
    for k in assorb:
        prow.append({"Prova": "velocità assorbimento x", "Valore": k[5], **row(k)})
    P = pd.DataFrame(prow).round(1)
    P.to_csv("results/sensibilita_pasto.csv", index=False)
    print("\n", P.to_string(index=False))

    # --- 4. grafici ----------------------------------------------------------------------------
    def band(ax, meal_h=A.MEAL_CLOCK_H, banda=True):
        """Linea del pasto e griglia; banda 70-180 mg/dL solo sui pannelli di glicemia."""
        if banda:
            ax.axhspan(70, 180, color="green", alpha=0.12)
        ax.axvline(meal_h, color="k", ls=":", lw=1.2)
        ax.grid(ls=":", alpha=0.5)

    def finestra(df):
        return df[df.t_h >= A.WARMUP_H]

    # 4.1 tre pannelli - adult#001, 60 g
    pz = A.make_patient("adult#001")
    iob_cap = 1.5 * A.basal_rate(pz) * 60.0
    fig, ax = plt.subplots(3, 1, figsize=(10, 9), sharex=True)
    for n in ["PID", "PID+IOB+PLGS"]:
        df = finestra(cache[key("adult#001", 60, n)][0])
        ax[0].plot(df.t_h, df.BG, color=COL[n], label=n)
        ax[1].plot(df.t_h, df.Insulin_Uh, color=COL[n], drawstyle="steps-post")
        ax[2].plot(df.t_h, df.IOB, color=COL[n], label=f"{n}: IOB stimato")
        ax[2].plot(df.t_h, df.IOB_vero, color=COL[n], ls="--", lw=1, label=f"{n}: IOB netto del modello")
    ax[1].axhline(A.basal_rate(pz) * 60, color="gray", ls=":", label="basale")
    ax[2].axhline(iob_cap, color="gray", ls="-.", label="Tetto IOB (1.5 x basale oraria)")
    band(ax[0]); band(ax[1], banda=False); band(ax[2], banda=False)
    ax[1].legend(loc="upper right")
    ax[0].set_ylabel("Glicemia plasmatica [mg/dL]"); ax[0].legend()
    ax[0].set_title("adult#001, pasto non annunciato da 60 g alle 12:00 (linea punteggiata)")
    ax[1].set_ylabel("Infusione [U/h]")
    ax[2].set_ylabel("IOB netto [U]"); ax[2].set_xlabel("Ora del giorno [h]"); ax[2].legend(fontsize=8)
    ax[2].set_xlim(A.WARMUP_H, A.ANALYSIS_END_H)
    plt.tight_layout(); plt.savefig("results/fig_3pannelli_adult001_60g.png", dpi=200); plt.close()

    # 4.2 sensibilità al carico (adult#001)
    fig, ax = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
    for i, c in enumerate([30, 60, 90]):
        for n in CONFRONTO:
            d = finestra(cache[key("adult#001", c, n)][0])
            ax[i].plot(d.t_h, d.BG, color=COL[n], label=n)
        band(ax[i]); ax[i].set_ylabel(f"BG [mg/dL] ({c} g)")
    ax[0].legend(fontsize=8); ax[0].set_title("Sensibilità al carico di carboidrati (adult#001)")
    ax[2].set_xlabel("Ora del giorno [h]"); ax[2].set_xlim(A.WARMUP_H, A.ANALYSIS_END_H)
    plt.tight_layout(); plt.savefig("results/fig_sensibilita_carico.png", dpi=200); plt.close()

    # 4.3 confronto tra pazienti (60 g)
    fig, ax = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
    for i, p in enumerate(PAZIENTI):
        for n in CONFRONTO:
            d = finestra(cache[key(p, 60, n)][0])
            ax[i].plot(d.t_h, d.BG, color=COL[n], label=n)
        band(ax[i]); ax[i].set_ylabel(f"BG [mg/dL] ({p})")
    ax[0].legend(fontsize=8); ax[0].set_title("Confronto tra pazienti virtuali, pasto non annunciato da 60 g")
    ax[2].set_xlabel("Ora del giorno [h]"); ax[2].set_xlim(A.WARMUP_H, A.ANALYSIS_END_H)
    plt.tight_layout(); plt.savefig("results/fig_multipaziente_60g.png", dpi=200); plt.close()

    # 4.4 orario del pasto e velocità di assorbimento (adult#001, 60 g)
    fig, ax = plt.subplots(2, 2, figsize=(13, 7), sharex=True, sharey=True)
    for j, n in enumerate(["PID", "PID+IOB+PLGS"]):
        for h in [10, 12, 14, 16]:
            d = finestra(cache[key("adult#001", 60, n, meal_h=h)][0])
            ax[0, j].plot(d.t_h, d.BG, label=f"pasto alle {h}:00")
        for s in [0.5, 1.0, 2.0]:
            d = finestra(cache[key("adult#001", 60, n, assorb=s)][0])
            ax[1, j].plot(d.t_h, d.BG, label=f"assorbimento x {s:g}")
        ax[0, j].set_title(n)
        for a in ax[:, j]:
            a.axhspan(70, 180, color="green", alpha=0.12); a.grid(ls=":", alpha=0.5); a.legend(fontsize=8)
    ax[0, 0].set_ylabel("BG [mg/dL] - orario"); ax[1, 0].set_ylabel("BG [mg/dL] - assorbimento")
    for a in ax[1]:
        a.set_xlabel("Ora del giorno [h]"); a.set_xlim(A.WARMUP_H, A.ANALYSIS_END_H)
    plt.tight_layout(); plt.savefig("results/fig_sensibilita_pasto.png", dpi=200); plt.close()

    print("\nFatto: tabelle e grafici in results/")
