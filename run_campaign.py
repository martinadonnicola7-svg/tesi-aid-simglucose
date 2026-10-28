"""
run_campaign.py - Esegue tutte le simulazioni della tesi e salva tabelle e grafici in results/.

Uso (dalla cartella principale del repository):
    python3 run_campaign.py

Prima di scrivere i risultati la cartella results/ viene svuotata di tabelle (.csv) e grafici (.png), così che contenga
solo i file prodotti dalla versione corrente di questo script.

Le simulazioni girano in parallelo su tutti i core disponibili. I risultati intermedi sono salvati in
results/_cache.pkl (escluso da git) e riutilizzati nelle esecuzioni successive; la cache si invalida da sola
se cambiano i controllori (dizionario CFG), il codice di aid_sim.py o la versione di simglucose.
"""
import glob
import hashlib
import importlib.metadata as md
import inspect
import os
import pickle
import warnings
from multiprocessing import Pool

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import aid_sim as A

warnings.filterwarnings("ignore")
os.makedirs("results", exist_ok=True)

CFG = A.CFG
PAZIENTI = ["adult#001", "adult#002", "adult#003"]
POPOLAZIONE = [f"adult#{i:03d}" for i in range(1, 11)]       # tutti gli adulti di simglucose
SEMI = list(range(1, 11))                                     # 10 semi del rumore del sensore
CONFRONTO = ["PID senza AW", "PID", "PID cons.", "PID+IOB+PLGS"]
MODULI = ["PID", "PID cons.", "PID+IOB", "PID+PLGS", "PID+IOB+PLGS"]
COL = {"PID senza AW": "tab:gray", "PID": "tab:red", "PID cons.": "tab:orange",
       "PID+PLGS": "tab:green", "PID+IOB": "tab:purple", "PID+IOB+PLGS": "tab:blue"}

# Sensibilità ai parametri del tetto IOB e del PLGS: un parametro alla volta sul controllore proposto,
# gli altri fermi al valore nominale di AIDController.
SENS = {
    "iob_alpha": [0.5, 0.75, 1.0, 1.5],   # tetto IOB = iob_alpha * basale orario
    "tau_s":     [30.0, 50.0, 70.0],      # costante di tempo dello stimatore IOB [min]
    "t_pred":    [10.0, 20.0, 30.0],      # orizzonte di predizione del PLGS [min]
    "plgs_pred": [80.0, 90.0, 100.0],     # soglia sulla glicemia proiettata [mg/dL]
}
SENS_PAZ = ["adult#001", "adult#003"]
SENS_CHO = [60, 90]
NOMINALE = {k: v.default for k, v in inspect.signature(A.AIDController.__init__).parameters.items()}

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
    pat, cho, name, seed, meal_h, assorb, extra = key
    df, _ = A.simulate(pat, cho, {**CFG[name], **dict(extra)}, seed=seed, meal_clock_h=meal_h, assorb_scale=assorb)
    return key, (df, A.metrics(df, meal_h=meal_h))


# Verifica nominale a digiuno (3 giorni, senza rumore del sensore): taratura attuale contro quella precedente
DIGIUNO = {"PID": A.CFG["PID"], "PID (Ki precedente 5e-6)": {**A.CFG["PID"], "ki": 5e-6},
           "PID+IOB+PLGS": A.CFG["PID+IOB+PLGS"]}


def _digiuno(args):
    pat, name = args
    m = A.verifica_digiuno(pat, DIGIUNO[name])
    df, _ = A.simulate(pat, 0, DIGIUNO[name], end_h=48.0, rumore=False)
    return args, (m, df[["t_h", "BG"]])


def key(pat, cho, name, seed=1, meal_h=A.MEAL_CLOCK_H, assorb=1.0, **extra):
    """Chiave di una simulazione. extra: parametri di AIDController che sostituiscono quelli di CFG[name]."""
    return (pat, float(cho), name, seed, float(meal_h), float(assorb), tuple(sorted(extra.items())))


def run_all(keys):
    """Esegue in parallelo le simulazioni non ancora in cache e aggiorna la cache."""
    uniche = list(dict.fromkeys(keys))
    todo = [k for k in uniche if k not in cache]
    if todo:
        print(f"Simulazioni da eseguire: {len(todo)} (già in cache: {len(uniche) - len(todo)})")
        with Pool() as pool:
            for i, (k, v) in enumerate(pool.imap_unordered(_sim, todo), 1):
                cache[k] = v
                if i % 50 == 0:
                    print(f"  {i}/{len(todo)}")
        with open(CACHE, "wb") as f:
            pickle.dump({"signature": SIGNATURE, "data": cache}, f)


# Colonne delle tabelle: nome della metrica in aid_sim.metrics -> intestazione con unità di misura
COLONNE = {
    "TIR": "TIR [%]", "TIR_post6h": "TIR 6h post [%]", "TAR": "TAR [%]", "TAR>250": "TAR>250 [%]",
    "TBR": "TBR [%]", "TBR<54": "TBR<54 [%]", "TBR_tardivo": "TBR tardivo [%]",
    "picco": "Picco [mg/dL]", "nadir": "Nadir [mg/dL]", "insulina_tot": "Insulina [U]",
    "IAE": "IAE [mg/dL*h]", "IAE_post6h": "IAE 6h post [mg/dL*h]", "IAE_range": "IAE fuori range [mg/dL*h]",
    "t_rientro": "Rientro in range [h]",
    "n_commut": "Commutazioni vincolo [-]", "n_sosp": "Sospensioni [-]",
    "TV_u": "Variazione totale infusione [U/h]", "std_u": "Dev. std infusione [U/h]",
}


def row(k):
    pat, cho, name, seed, meal_h, assorb, extra = k
    m = cache[k][1]
    return {"Paziente": pat, "CHO [g]": int(cho), "Controllore": name, **{COLONNE[c]: m[c] for c in COLONNE}}


def media_std(T, gruppi, colonne):
    """Tabella 'media ± deviazione standard' delle colonne, raggruppando per le colonne in gruppi.
    Il tempo di rientro è mediato solo sui casi che rientrano; quelli che non rientrano sono contati a parte."""
    out = []
    for g, d in T.groupby(gruppi, sort=False):
        g = g if isinstance(g, tuple) else (g,)
        out.append({**dict(zip(gruppi, g)),
                    **{c: f"{d[c].mean():.1f} ± {d[c].std():.1f}" for c in colonne},
                    "Non rientra [n]": int(d[COLONNE["t_rientro"]].isna().sum())})
    return pd.DataFrame(out)


def pulisci_results():
    """Elimina tabelle e grafici precedenti: in results/ restano solo i file prodotti da questa esecuzione."""
    for f in glob.glob("results/*.png") + glob.glob("results/*.csv"):
        os.remove(f)


if __name__ == "__main__":
    # --- elenco delle simulazioni ----------------------------------------------------------
    main = [key("adult#001", c, n) for c in [0, 30, 60, 90] for n in CFG]
    main += [key(p, c, n) for p in PAZIENTI[1:] for c in [0, 60, 90] for n in CONFRONTO]
    pop = [key(p, c, n) for p in POPOLAZIONE for c in [0, 60, 90] for n in MODULI]
    semi = [key(p, 60, n, seed=s) for p in PAZIENTI for n in ["PID", "PID cons.", "PID+IOB+PLGS"] for s in SEMI]
    orari = [key("adult#001", 60, n, meal_h=h) for h in [10, 12, 14, 16] for n in ["PID", "PID+IOB+PLGS"]]
    assorb = [key("adult#001", 60, n, assorb=s) for s in [0.5, 1.0, 2.0] for n in ["PID", "PID+IOB+PLGS"]]
    sens = [key(p, c, "PID+IOB+PLGS", **{par: v}) for par, vals in SENS.items() for v in vals
            for p in SENS_PAZ for c in SENS_CHO]
    sens += [key(p, c, "PID") for p in SENS_PAZ for c in SENS_CHO]      # riferimento senza vincoli
    run_all(main + pop + semi + orari + assorb + sens)

    with Pool() as pool:
        dig = dict(pool.map(_digiuno, [(p, n) for p in POPOLAZIONE for n in DIGIUNO]))

    pulisci_results()

    # --- 0. verifica nominale a digiuno -----------------------------------------------------
    V = pd.DataFrame([{"Paziente": p, "Controllore": n,
                       "Ampiezza ultimo giorno [mg/dL]": dig[(p, n)][0]["ampiezza_ultimo_giorno"],
                       "Media ultimo giorno [mg/dL]": dig[(p, n)][0]["media_ultimo_giorno"],
                       "Min dalle 08 [mg/dL]": dig[(p, n)][0]["min_dalle_08"],
                       "Max dalle 08 [mg/dL]": dig[(p, n)][0]["max_dalle_08"],
                       "TBR dalle 08 [%]": dig[(p, n)][0]["TBR_dalle_08"],
                       "Superata": dig[(p, n)][0]["superata"]}
                      for n in DIGIUNO for p in POPOLAZIONE]).round(1)
    V.to_csv("results/verifica_digiuno.csv", index=False)
    print(V.to_string(index=False))

    # --- 1. campagna principale --------------------------------------------------------------
    R = pd.DataFrame([row(k) for k in main]).round(2)
    R.to_csv("results/metriche.csv", index=False)
    print(R.to_string(index=False))

    # --- 2. popolazione: 10 adulti, contributo dei singoli moduli -------------------------------
    Pop = pd.DataFrame([row(k) for k in pop]).round(2)
    Pop.to_csv("results/popolazione.csv", index=False)
    sint_col = [COLONNE[c] for c in ["TIR", "TAR", "TBR", "TBR_tardivo", "picco", "IAE_range", "t_rientro",
                                     "n_commut", "TV_u"]]
    PopS = media_std(Pop, ["CHO [g]", "Controllore"], sint_col)
    PopS.to_csv("results/popolazione_sintesi.csv", index=False)
    print("\n", PopS.to_string(index=False))

    # --- 3. robustezza al rumore del sensore: 10 semi ---------------------------------------
    Sd = pd.DataFrame([{**row(k), "Seme": k[3]} for k in semi]).round(2)
    S = media_std(Sd, ["Paziente", "Controllore"],
                  [COLONNE[c] for c in ["TIR", "TAR", "TBR", "TBR<54", "TBR_tardivo", "picco", "nadir",
                                        "IAE_range", "t_rientro", "n_commut", "TV_u"]])
    S.to_csv("results/robustezza_semi.csv", index=False)
    print("\n", S.to_string(index=False))

    # --- 4. sensibilità all'orario del pasto e alla velocità di assorbimento ----------------
    prow = []
    for k in orari:
        prow.append({"Prova": "orario pasto", "Valore": f"{int(k[4]):02d}:00", **row(k)})
    for k in assorb:
        prow.append({"Prova": "velocità assorbimento x", "Valore": k[5], **row(k)})
    P = pd.DataFrame(prow).round(2)
    P.to_csv("results/sensibilita_pasto.csv", index=False)
    print("\n", P.to_string(index=False))

    # --- 5. sensibilità ai parametri del tetto IOB e del PLGS -------------------------------
    srow = []
    for k in sens:
        par, val = k[6][0] if k[6] else ("nessuno (PID)", np.nan)
        srow.append({"Parametro": par, "Valore": val, **row(k)})
    Sens = pd.DataFrame(srow).round(2)
    Sens.to_csv("results/sensibilita_iob_plgs.csv", index=False)
    print("\n", Sens.to_string(index=False))

    # --- 6. grafici ----------------------------------------------------------------------------
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
            axc[2].axhline(A.IOB_ALPHA * ub * 60, color="gray", ls="-.", lw=1)
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

    # 6.0 verifica nominale a digiuno: 3 giorni senza pasti, senza rumore del sensore
    fig, ax = plt.subplots(1, 2, figsize=(15, 4.5), sharey=True)
    for j, n in enumerate(["PID (Ki precedente 5e-6)", "PID"]):
        for p in POPOLAZIONE:
            d = dig[(p, n)][1]
            ax[j].plot(d.t_h, d.BG, lw=1, label=p)
        ax[j].axhspan(70, 180, color="green", alpha=0.12); ax[j].axhline(120, color="k", ls="--", lw=0.8)
        ax[j].axvspan(-A.PRE_H, A.WARMUP_H, color="gray", alpha=0.1)
        ax[j].set_title(n); ax[j].set_xlabel("Ore dalla mezzanotte del giorno di analisi [h]"); ax[j].grid(ls=":", alpha=0.5)
    ax[0].set_ylabel("Glicemia plasmatica [mg/dL]"); ax[1].legend(fontsize=7, ncol=2)
    fig.suptitle("Verifica nominale a digiuno, 10 adulti, senza rumore del sensore (grigio: assestamento)")
    plt.tight_layout(); plt.savefig("results/fig_verifica_digiuno.png", dpi=200); plt.close()

    # 6.1 scenario di riferimento: adult#001, 60 g, PID e controllore proposto
    fig, ax = plt.subplots(3, 1, figsize=(10, 9), sharex=True)
    for n in ["PID", "PID+IOB+PLGS"]:
        d = finestra(cache[key("adult#001", 60, n)][0])
        ax[0].plot(d.t_h, d.BG, color=COL[n], label=n)
        ax[1].plot(d.t_h, d.Insulin_Uh, color=COL[n], drawstyle="steps-post")
        ax[2].plot(d.t_h, d.IOB, color=COL[n], label=f"{n}: IOB stimato")
        ax[2].plot(d.t_h, d.IOB_vero, color=COL[n], ls="--", lw=1, label=f"{n}: IOB del modello")
    ax[1].axhline(ub1 * 60, color="gray", ls=":", label="basale")
    ax[2].axhline(A.IOB_ALPHA * ub1 * 60, color="gray", ls="-.", label="tetto IOB")
    decora(ax[0], banda=True); decora(ax[1]); decora(ax[2])
    ax[0].set_ylabel("Glicemia plasmatica [mg/dL]"); ax[0].legend()
    ax[0].set_title("adult#001, pasto non annunciato da 60 g alle 12:00 (linea punteggiata)")
    ax[1].set_ylabel("Infusione [U/h]"); ax[1].legend(loc="upper right", fontsize=8)
    ax[2].set_ylabel("IOB netto [U]"); ax[2].set_xlabel("Ora del giorno [h]"); ax[2].legend(fontsize=8, ncol=2)
    plt.tight_layout(); plt.savefig("results/fig_3pannelli_adult001_60g.png", dpi=200); plt.close()

    # 6.2 sensibilità alla quantità di carboidrati (adult#001)
    fig, ax = plt.subplots(3, 3, figsize=(15, 9), sharex=True)
    for j, c in enumerate([30, 60, 90]):
        voci = [(n, cache[key("adult#001", c, n)][0], COL[n]) for n in CONFRONTO]
        colonna(ax[:, j], voci, ub1, legenda=(j == 0))
        ax[0, j].set_title(f"Pasto da {c} g")
    etichette(ax)
    fig.suptitle("Sensibilità alla quantità di carboidrati (adult#001)")
    plt.tight_layout(); plt.savefig("results/fig_sensibilita_carico.png", dpi=200); plt.close()

    # 6.3 confronto tra pazienti (60 g)
    fig, ax = plt.subplots(3, 3, figsize=(15, 9), sharex=True)
    for j, p in enumerate(PAZIENTI):
        ubp = A.basal_rate(A.make_patient(p))
        voci = [(n, cache[key(p, 60, n)][0], COL[n]) for n in CONFRONTO]
        colonna(ax[:, j], voci, ubp, legenda=(j == 0))
        ax[0, j].set_title(p)
    etichette(ax)
    fig.suptitle("Confronto tra pazienti virtuali, pasto non annunciato da 60 g")
    plt.tight_layout(); plt.savefig("results/fig_multipaziente_60g.png", dpi=200); plt.close()

    # 6.4 sensibilità all'istante del pasto (adult#001, 60 g)
    fig, ax = plt.subplots(3, 2, figsize=(12, 9), sharex=True)
    for j, n in enumerate(["PID", "PID+IOB+PLGS"]):
        voci = [(f"pasto alle {h}:00", cache[key("adult#001", 60, n, meal_h=h)][0], None)
                for h in [10, 12, 14, 16]]
        colonna(ax[:, j], voci, ub1, meal_h=[10, 12, 14, 16], legenda=(j == 0))
        ax[0, j].set_title(n)
    etichette(ax)
    fig.suptitle("Sensibilità all'istante del pasto (adult#001, 60 g)")
    plt.tight_layout(); plt.savefig("results/fig_orario_pasto.png", dpi=200); plt.close()

    # 6.5 sensibilità alla velocità di assorbimento (adult#001, 60 g)
    fig, ax = plt.subplots(3, 2, figsize=(12, 9), sharex=True)
    for j, n in enumerate(["PID", "PID+IOB+PLGS"]):
        voci = [(f"assorbimento $\\times$ {s:g}", cache[key("adult#001", 60, n, assorb=s)][0], None)
                for s in [0.5, 1.0, 2.0]]
        colonna(ax[:, j], voci, ub1, legenda=(j == 0))
        ax[0, j].set_title(n)
    etichette(ax)
    fig.suptitle("Sensibilità alla velocità di assorbimento del pasto (adult#001, 60 g)")
    plt.tight_layout(); plt.savefig("results/fig_assorbimento.png", dpi=200); plt.close()

    # 6.6 commutazioni dell'infusione: vincolo attivo istante per istante (adult#001 e adult#003, 60 g)
    VCOL = {"iob": "tab:purple", "plgs": "tab:green", "sat_alta": "tab:red", "sat_bassa": "tab:gray"}
    fig, ax = plt.subplots(2, 2, figsize=(14, 6), sharex=True)
    for j, p in enumerate(["adult#001", "adult#003"]):
        ubp = A.basal_rate(A.make_patient(p))
        for i, n in enumerate(["PID", "PID+IOB+PLGS"]):
            d = finestra(cache[key(p, 60, n)][0])
            a = ax[i, j]
            a.plot(d.t_h, d.Insulin_Uh, color=COL[n], drawstyle="steps-post", lw=1)
            for v, c in VCOL.items():
                on = (d.Vincolo == v).values
                if on.any():
                    a.fill_between(d.t_h, 0, 1, where=on, step="post", color=c, alpha=0.25,
                                   transform=a.get_xaxis_transform(), label=f"vincolo: {v}")
            a.axhline(ubp * 60, color="gray", ls=":", lw=1)
            m = cache[key(p, 60, n)][1]
            a.set_title(f"{p} - {n}: {m['n_commut']} commutazioni, {m['n_sosp']} sospensioni, "
                        f"TV {m['TV_u']:.0f} U/h", fontsize=9)
            decora(a)
            if a.get_legend_handles_labels()[0]:
                a.legend(fontsize=7, loc="upper right")
        ax[1, j].set_xlabel("Ora del giorno [h]")
    ax[0, 0].set_ylabel("Infusione [U/h]"); ax[1, 0].set_ylabel("Infusione [U/h]")
    fig.suptitle("Infusione e vincolo attivo, pasto non annunciato da 60 g")
    plt.tight_layout(); plt.savefig("results/fig_commutazioni_60g.png", dpi=200); plt.close()

    # 6.7 sensibilità ai parametri del tetto IOB e del PLGS
    MET = [("TIR", "TIR [%]"), ("TBR", "TBR [%]"), ("picco", "Picco [mg/dL]"), ("n_commut", "Commutazioni [-]")]
    fig, ax = plt.subplots(len(MET), len(SENS), figsize=(16, 11), sharex="col")
    for j, (par, vals) in enumerate(SENS.items()):
        for p in SENS_PAZ:
            for c, ls in zip(SENS_CHO, ["-", "--"]):
                for i, (m, lab) in enumerate(MET):
                    y = [cache[key(p, c, "PID+IOB+PLGS", **{par: v})][1][m] for v in vals]
                    line, = ax[i, j].plot(vals, y, marker="o", ls=ls, label=f"{p}, {c} g")
                    ax[i, j].axhline(cache[key(p, c, "PID")][1][m], color=line.get_color(), ls=ls, lw=0.8,
                                     alpha=0.4)
        for i, (m, lab) in enumerate(MET):
            ax[i, j].axvline(NOMINALE[par], color="k", ls=":", lw=1)
            ax[i, j].grid(ls=":", alpha=0.5)
            if j == 0:
                ax[i, j].set_ylabel(lab)
        ax[-1, j].set_xlabel(par)
    ax[0, 0].legend(fontsize=7)
    fig.suptitle("Sensibilità del controllore PID+IOB+PLGS ai parametri del tetto IOB e del PLGS\n"
                 "(linea verticale: valore nominale; linee orizzontali sottili: PID senza vincoli)")
    plt.tight_layout(); plt.savefig("results/fig_sensibilita_iob_plgs.png", dpi=200); plt.close()

    # 6.8 popolazione: distribuzione delle metriche sui 10 adulti per ciascun controllore
    MET = ["TIR [%]", "TBR [%]", "Picco [mg/dL]", "IAE fuori range [mg/dL*h]", "Commutazioni vincolo [-]"]
    fig, ax = plt.subplots(2, len(MET), figsize=(18, 7))
    for i, c in enumerate([60, 90]):
        d = Pop[Pop["CHO [g]"] == c]
        for j, colm in enumerate(MET):
            dati = [d[d.Controllore == n][colm].values for n in MODULI]
            bp = ax[i, j].boxplot(dati, patch_artist=True, widths=0.6)
            for patch, n in zip(bp["boxes"], MODULI):
                patch.set_facecolor(COL[n]); patch.set_alpha(0.5)
            for k, y in enumerate(dati, 1):
                ax[i, j].scatter(np.full(len(y), k), y, s=10, color="k", zorder=3)
            ax[i, j].set_xticks(range(1, len(MODULI) + 1))
            ax[i, j].set_xticklabels(MODULI, rotation=30, ha="right", fontsize=8)
            ax[i, j].set_title(f"{colm}, {c} g", fontsize=9)
            ax[i, j].grid(ls=":", alpha=0.5)
    fig.suptitle("Popolazione di 10 adulti virtuali (adult#001-#010), pasto non annunciato")
    plt.tight_layout(); plt.savefig("results/fig_popolazione.png", dpi=200); plt.close()

    # 6.9 compromesso sicurezza / iperglicemia: differenze del proposto rispetto al PID, paziente per paziente
    fig, ax = plt.subplots(1, 2, figsize=(13, 5.5))
    for c, colore in zip([60, 90], ["tab:blue", "tab:red"]):
        d = Pop[Pop["CHO [g]"] == c].set_index(["Paziente", "Controllore"])
        for p in POPOLAZIONE:
            a, b = d.loc[(p, "PID")], d.loc[(p, "PID+IOB+PLGS")]
            x0, y0 = b["TAR [%]"] - a["TAR [%]"], b["TBR [%]"] - a["TBR [%]"]
            x1, y1 = b["Picco [mg/dL]"] - a["Picco [mg/dL]"], b["TBR tardivo [%]"] - a["TBR tardivo [%]"]
            lab = f"{c} g" if p == POPOLAZIONE[0] else None
            ax[0].scatter(x0, y0, color=colore, label=lab)
            ax[0].annotate(p[-2:], (x0, y0), fontsize=7, xytext=(3, 3), textcoords="offset points")
            ax[1].scatter(x1, y1, color=colore, label=lab)
            ax[1].annotate(p[-2:], (x1, y1), fontsize=7, xytext=(3, 3), textcoords="offset points")
    for a_, xl, yl in [(ax[0], "Δ TAR [punti %] (proposto − PID)", "Δ TBR [punti %] (proposto − PID)"),
                       (ax[1], "Δ picco [mg/dL] (proposto − PID)", "Δ TBR tardivo [punti %] (proposto − PID)")]:
        a_.axhline(0, color="k", lw=0.8); a_.axvline(0, color="k", lw=0.8)
        a_.set_xlabel(xl); a_.set_ylabel(yl); a_.grid(ls=":", alpha=0.5); a_.legend()
    fig.suptitle("Compromesso tra riduzione dell'ipoglicemia e controllo dell'iperglicemia (10 adulti)\n"
                 "quadrante in basso a destra: meno ipoglicemia al prezzo di più iperglicemia")
    plt.tight_layout(); plt.savefig("results/fig_compromesso.png", dpi=200); plt.close()

    print("\nFatto: tabelle e grafici in results/")
    print("File prodotti:", sorted(os.path.basename(f) for f in glob.glob("results/*.csv") + glob.glob("results/*.png")))
