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
    # Per ogni scenario si riportano glicemia, insulina somministrata e IOB (tre pannelli in colonna).

    def finestra(df):
        """Solo la finestra di analisi: l'assestamento non entra nei grafici."""
        return df[df.t_h >= A.WARMUP_H]

    def decora(ax, meal_h=A.MEAL_CLOCK_H, banda=False):
        if banda:
            ax.axhspan(70, 180, color="green", alpha=0.12)
        for h in (meal_h if isinstance(meal_h, (list, tuple)) else [meal_h]):
            if h is not None:
                ax.axvline(h, color="k", ls=":", lw=1.2)
        ax.grid(ls=":", alpha=0.5)
        ax.set_xlim(A.WARMUP_H, A.ANALYSIS_END_H)

    def colonna(axc, voci, ub, meal_h=A.MEAL_CLOCK_H, legenda=False, tetto=True):
        """Riempie una colonna di tre assi con glicemia, infusione e IOB.

        voci: lista di (etichetta, DataFrame, colore). Se colore è None si usa il ciclo di default.
        """
        for etichetta, df, colore in voci:
            d = finestra(df)
            kw = {} if colore is None else {"color": colore}
            axc[0].plot(d.t_h, d.BG, label=etichetta, **kw)
            axc[1].plot(d.t_h, d.Insulin_Uh, drawstyle="steps-post", **kw)
            axc[2].plot(d.t_h, d.IOB, **kw)
        axc[1].axhline(ub * 60, color="gray", ls=":", lw=1)
        if tetto:
            axc[2].axhline(1.5 * ub * 60, color="gray", ls="-.", lw=1)
        decora(axc[0], meal_h, banda=True)
        decora(axc[1], meal_h)
        decora(axc[2], meal_h)
        if legenda:
            axc[0].legend(fontsize=7)

    def etichette(ax):
        ax[0, 0].set_ylabel("BG plasmatica [mg/dL]")
        ax[1, 0].set_ylabel("Infusione [U/h]")
        ax[2, 0].set_ylabel("IOB stimato [U]")
        for a in ax[2]:
            a.set_xlabel("Ora del giorno [h]")

    ub1 = A.basal_rate(A.make_patient("adult#001"))

    # 4.1 scenario di riferimento: adult#001, 60 g, PID e controllore proposto
    fig, ax = plt.subplots(3, 1, figsize=(10, 9), sharex=True)
    for n in ["PID", "PID+IOB+PLGS"]:
        d = finestra(cache[key("adult#001", 60, n)][0])
        ax[0].plot(d.t_h, d.BG, color=COL[n], label=n)
        ax[1].plot(d.t_h, d.Insulin_Uh, color=COL[n], drawstyle="steps-post")
        ax[2].plot(d.t_h, d.IOB, color=COL[n], label=f"{n}: IOB stimato")
        ax[2].plot(d.t_h, d.IOB_vero, color=COL[n], ls="--", lw=1, label=f"{n}: IOB del modello")
    ax[1].axhline(ub1 * 60, color="gray", ls=":", label="basale")
    ax[2].axhline(1.5 * ub1 * 60, color="gray", ls="-.", label="tetto IOB")
    decora(ax[0], banda=True); decora(ax[1]); decora(ax[2])
    ax[0].set_ylabel("Glicemia plasmatica [mg/dL]"); ax[0].legend()
    ax[0].set_title("adult#001, pasto non annunciato da 60 g alle 12:00 (linea punteggiata)")
    ax[1].set_ylabel("Infusione [U/h]"); ax[1].legend(loc="upper right", fontsize=8)
    ax[2].set_ylabel("IOB netto [U]"); ax[2].set_xlabel("Ora del giorno [h]"); ax[2].legend(fontsize=8, ncol=2)
    plt.tight_layout(); plt.savefig("results/fig_3pannelli_adult001_60g.png", dpi=200); plt.close()

    # 4.2 sensibilità alla quantità di carboidrati (adult#001)
    fig, ax = plt.subplots(3, 3, figsize=(15, 9), sharex=True)
    for j, c in enumerate([30, 60, 90]):
        voci = [(n, cache[key("adult#001", c, n)][0], COL[n]) for n in CONFRONTO]
        colonna(ax[:, j], voci, ub1, legenda=(j == 0))
        ax[0, j].set_title(f"Pasto da {c} g")
    etichette(ax)
    fig.suptitle("Sensibilità alla quantità di carboidrati (adult#001)")
    plt.tight_layout(); plt.savefig("results/fig_sensibilita_carico.png", dpi=200); plt.close()

    # 4.3 confronto tra pazienti (60 g)
    fig, ax = plt.subplots(3, 3, figsize=(15, 9), sharex=True)
    for j, p in enumerate(PAZIENTI):
        ubp = A.basal_rate(A.make_patient(p))
        voci = [(n, cache[key(p, 60, n)][0], COL[n]) for n in CONFRONTO]
        colonna(ax[:, j], voci, ubp, legenda=(j == 0))
        ax[0, j].set_title(p)
    etichette(ax)
    fig.suptitle("Confronto tra pazienti virtuali, pasto non annunciato da 60 g")
    plt.tight_layout(); plt.savefig("results/fig_multipaziente_60g.png", dpi=200); plt.close()

    # 4.4 sensibilità all'istante del pasto (adult#001, 60 g)
    fig, ax = plt.subplots(3, 2, figsize=(12, 9), sharex=True)
    for j, n in enumerate(["PID", "PID+IOB+PLGS"]):
        voci = [(f"pasto alle {h}:00", cache[key("adult#001", 60, n, meal_h=h)][0], None)
                for h in [10, 12, 14, 16]]
        colonna(ax[:, j], voci, ub1, meal_h=[10, 12, 14, 16], legenda=(j == 0))
        ax[0, j].set_title(n)
    etichette(ax)
    fig.suptitle("Sensibilità all'istante del pasto (adult#001, 60 g)")
    plt.tight_layout(); plt.savefig("results/fig_orario_pasto.png", dpi=200); plt.close()

    # 4.5 sensibilità alla velocità di assorbimento (adult#001, 60 g)
    fig, ax = plt.subplots(3, 2, figsize=(12, 9), sharex=True)
    for j, n in enumerate(["PID", "PID+IOB+PLGS"]):
        voci = [(f"assorbimento $\\times$ {s:g}", cache[key("adult#001", 60, n, assorb=s)][0], None)
                for s in [0.5, 1.0, 2.0]]
        colonna(ax[:, j], voci, ub1, legenda=(j == 0))
        ax[0, j].set_title(n)
    etichette(ax)
    fig.suptitle("Sensibilità alla velocità di assorbimento del pasto (adult#001, 60 g)")
    plt.tight_layout(); plt.savefig("results/fig_assorbimento.png", dpi=200); plt.close()

    print("\nFatto: tabelle e grafici in results/")
