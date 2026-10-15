"""
run_campaign.py - Esegue tutte le simulazioni della tesi e salva tabelle e grafici in results/.

Uso (dalla cartella principale del repository):
    python3 run_campaign.py

Durata: circa 4-5 minuti la prima volta. I risultati intermedi sono salvati in results/_cache.pkl e
riutilizzati nelle esecuzioni successive; la cache si invalida da sola se cambiano i controllori
(dizionario CFG) o la versione di simglucose.
"""
import importlib.metadata as md
import os
import pickle
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from simglucose.patient.t1dpatient import T1DPatient

import aid_sim as A

warnings.filterwarnings("ignore")
os.makedirs("results", exist_ok=True)

CFG = {
    "PID":          dict(),                               # PID di riferimento
    "PID cons.":    dict(ki=2e-6),                        # PID con guadagno integrale ridotto
    "PID+PLGS":     dict(use_plgs=True),                  # PID + sospensione predittiva
    "PID+IOB":      dict(use_iob=True),                   # PID + tetto sull'insulina attiva
    "PID+IOB+PLGS": dict(use_iob=True, use_plgs=True),    # controllore proposto
}

# --- cache dei risultati (si invalida se cambiano CFG o la versione di simglucose) ---------
CACHE = "results/_cache.pkl"
SIGNATURE = repr(sorted((k, sorted(v.items())) for k, v in CFG.items())) + md.version("simglucose")
cache = {}
if os.path.exists(CACHE):
    stored = pickle.load(open(CACHE, "rb"))
    if stored.get("signature") == SIGNATURE:
        cache = stored["data"]


def run(pat, cho, name, seed=1):
    """Una simulazione (con cache). Restituisce (DataFrame, metriche)."""
    key = (pat, cho, name, seed)
    if key not in cache:
        df, _ = A.simulate(pat, cho, CFG[name], seed=seed)
        cache[key] = (df, A.metrics(df))
        pickle.dump({"signature": SIGNATURE, "data": cache}, open(CACHE, "wb"))
    return cache[key]


# --- 1. Campagna principale -----------------------------------------------------------------
jobs = []
for c in [0.0, 30.0, 60.0, 90.0]:
    names = ["PID", "PID cons.", "PID+PLGS", "PID+IOB", "PID+IOB+PLGS"] if c == 60.0 else ["PID", "PID cons.", "PID+IOB+PLGS"]
    jobs += [("adult#001", c, n) for n in names]
for p in ["adult#002", "adult#003"]:
    for c in [0.0, 60.0]:
        jobs += [(p, c, n) for n in ["PID", "PID cons.", "PID+IOB+PLGS"]]

rows = []
for pat, cho, name in jobs:
    _, m = run(pat, cho, name)
    rows.append({
        "Paziente": pat, "CHO [g]": int(cho), "Controllore": name,
        "TIR [%]": m["TIR"], "TAR [%]": m["TAR"], "TAR>250 [%]": m["TAR>250"],
        "TBR [%]": m["TBR"], "TBR<54 [%]": m["TBR<54"], "TBR tardivo [%]": m["TBR_tardivo"],
        "Picco [mg/dL]": m["picco"], "Nadir [mg/dL]": m["nadir"],
    })
R = pd.DataFrame(rows).round(1)
R.to_csv("results/metriche.csv", index=False)
print(R.to_string(index=False))

# --- 2. Robustezza al rumore del sensore: 5 semi, adult#001, 60 g -----------------------------
srows = []
for name in ["PID", "PID cons.", "PID+IOB+PLGS"]:
    T = pd.DataFrame([run("adult#001", 60.0, name, s)[1] for s in range(1, 6)])
    srows.append({"Controllore": name,
                  **{k: f"{T[k].mean():.1f} ± {T[k].std():.1f}" for k in ["TIR", "TAR", "TBR", "TBR_tardivo", "picco", "nadir"]}})
S = pd.DataFrame(srows)
S.to_csv("results/robustezza_semi.csv", index=False)
print("\n", S.to_string(index=False))

# --- 3. Grafici -------------------------------------------------------------------------------
COL = {"PID": "tab:red", "PID cons.": "tab:orange", "PID+IOB+PLGS": "tab:blue"}


def band(ax):
    ax.axhspan(70, 180, color="green", alpha=0.12)
    ax.axvline(A.MEAL_OFFSET_H, color="k", ls=":", lw=1.2)


# 3.1 tre pannelli (glicemia, infusione, IOB) - adult#001, 60 g
pz = T1DPatient.withName("adult#001")
iob_cap = 1.5 * pz._params["u2ss"] * pz._params["BW"] / 6000.0 * 60.0     # tetto IOB [U]
fig, ax = plt.subplots(3, 1, figsize=(10, 9), sharex=True)
for n in ["PID", "PID+IOB+PLGS"]:
    df = run("adult#001", 60.0, n)[0]
    ax[0].plot(df.t_h, df.BG, color=COL[n], label=n)
    ax[1].plot(df.t_h, df.Insulin_Uh, color=COL[n])
    ax[2].plot(df.t_h, df.IOB, color=COL[n], label=n)
ax[2].axhline(iob_cap, color="gray", ls="-.", label="Tetto IOB (1.5 x basale oraria)")
band(ax[0])
ax[0].set_ylabel("Glicemia plasmatica [mg/dL]")
ax[0].legend()
ax[0].set_title("adult#001, pasto non annunciato 60 g alle 12:00 (linea punteggiata)")
ax[1].set_ylabel("Infusione [U/h]")
ax[2].set_ylabel("IOB stimato [U]")
ax[2].set_xlabel("Tempo dall'inizio della simulazione [h] (0 = ore 08:00)")
ax[2].legend()
for a in ax:
    a.grid(ls=":", alpha=0.5)
ax[2].set_xlim(0, 16)
plt.tight_layout()
plt.savefig("results/fig_3pannelli_adult001_60g.png", dpi=200)
plt.close()

# 3.2 sensibilità al carico (adult#001)
fig, ax = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
for i, c in enumerate([30.0, 60.0, 90.0]):
    for n in ["PID", "PID cons.", "PID+IOB+PLGS"]:
        d = run("adult#001", c, n)[0]
        ax[i].plot(d.t_h, d.BG, color=COL[n], label=n)
    band(ax[i])
    ax[i].set_ylabel(f"BG [mg/dL] ({int(c)} g)")
    ax[i].grid(ls=":", alpha=0.5)
ax[0].legend()
ax[0].set_title("Sensibilità al carico di carboidrati (adult#001)")
ax[2].set_xlabel("Tempo [h]")
ax[2].set_xlim(0, 16)
plt.tight_layout()
plt.savefig("results/fig_sensibilita_carico.png", dpi=200)
plt.close()

# 3.3 confronto tra pazienti (60 g)
fig, ax = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
for i, p in enumerate(["adult#001", "adult#002", "adult#003"]):
    for n in ["PID", "PID cons.", "PID+IOB+PLGS"]:
        d = run(p, 60.0, n)[0]
        ax[i].plot(d.t_h, d.BG, color=COL[n], label=n)
    band(ax[i])
    ax[i].set_ylabel(f"BG [mg/dL] ({p})")
    ax[i].grid(ls=":", alpha=0.5)
ax[0].legend()
ax[0].set_title("Confronto tra pazienti virtuali, pasto non annunciato 60 g")
ax[2].set_xlabel("Tempo [h]")
ax[2].set_xlim(0, 16)
plt.tight_layout()
plt.savefig("results/fig_multipaziente_60g.png", dpi=200)
plt.close()

print("\nFatto: tabelle e grafici in results/")
