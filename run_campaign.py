"""
run_campaign.py - Esegue tutte le simulazioni della tesi e salva tabelle e grafici in results/.
Uso: python3 run_campaign.py  (circa 4-5 min la prima volta; istantaneo se usa la cache).
"""
import os
import pickle
import warnings
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import aid_sim as A

warnings.filterwarnings("ignore")
os.makedirs("results", exist_ok=True)
CACHE = "results/_cache.pkl"
cache = pickle.load(open(CACHE, "rb")) if os.path.exists(CACHE) else {}

CFG = {
    "PID":          dict(),                                   # PID nominale di riferimento
    "PID cons.":    dict(ki=2e-6),                            # PID conservativo (Ki ridotto)
    "PID+PLGS":     dict(use_plgs=True),                      # PID + sospensione predittiva
    "PID+IOB":      dict(use_iob=True),                       # PID + tetto su insulina attiva
    "PID+IOB+PLGS": dict(use_iob=True, use_plgs=True),        # Controllore proposto
}

def run(pat, cho, name, seed=1):
    key = (pat, cho, name, seed)
    if key not in cache:
        df, ub = A.simulate(pat, cho, CFG[name], seed=seed)
        m = A.metrics(df)
        m["TBR_tardivo"] = 100 * np.mean(df[df.t_h >= A.MEAL_OFFSET_H + 3].BG.values < 70)
        cache[key] = (df, m)
        pickle.dump(cache, open(CACHE, "wb"))
    return cache[key]

# --- 1. Campagna principale -------------------------------------------------------
jobs = []
for c in [0.0, 30.0, 60.0, 90.0]:
    for n in (["PID", "PID cons.", "PID+PLGS", "PID+IOB", "PID+IOB+PLGS"] if c == 60.0 else ["PID", "PID cons.", "PID+IOB+PLGS"]):
        jobs.append(("adult#001", c, n))
for p in ["adult#002", "adult#003"]:
    for c in [0.0, 60.0]:
        for n in ["PID", "PID cons.", "PID+IOB+PLGS"]:
            jobs.append((p, c, n))

rows = []
for pat, cho, name in jobs:
    df, m = run(pat, cho, name)
    rows.append({
        "Paziente": pat, 
        "CHO [g]": int(cho), 
        "Controllore": name,
        "TIR [%]": m["TIR"], 
        "TAR [%]": m["TAR"], 
        "TAR>250 [%]": m["TAR>250"],
        "TBR [%]": m["TBR"], 
        "TBR<54 [%]": m["TBR<54"],
        "TBR tardivo [%]": m["TBR_tardivo"], 
        "Picco [mg/dL]": m["picco"], 
        "Nadir [mg/dL]": m["nadir"]
    })

R = pd.DataFrame(rows).round(1)
R.to_csv("results/metriche.csv", index=False)
R.to_csv("metriche.csv", index=False)  # Allinea anche il file nella root
print(R.to_string(index=False))

# --- 2. Robustezza al rumore del sensore (5 semi) su adult#001, 60 g ----------------------
srows = []
for name in ["PID", "PID cons.", "PID+IOB+PLGS"]:
    T = pd.DataFrame([run("adult#001", 60.0, name, s)[1] for s in range(1, 6)])
    srows.append({
        "Controllore": name, 
        **{k: f"{T[k].mean():.1f} ± {T[k].std():.1f}" for k in ["TIR", "TAR", "TBR", "TBR_tardivo", "picco", "nadir"]}
    })

S = pd.DataFrame(srows)
S.to_csv("results/robustezza_semi.csv", index=False)
S.to_csv("robustezza_semi.csv", index=False)  # Allinea anche il file nella root
print("\n", S.to_string(index=False))

# --- 3. Generazione Grafici -------------------------------------------------------------
COL = {"PID": "tab:red", "PID cons.": "tab:orange", "PID+IOB+PLGS": "tab:blue"}

def band(ax):
    ax.axhspan(70, 180, color="green", alpha=0.12)
    ax.axvline(A.MEAL_OFFSET_H, color="k", ls=":", lw=1.2)

# 3.1 Grafico a tre pannelli (Glicemia, Infusione, IOB) - adult#001, 60 g
fig, ax = plt.subplots(3, 1, figsize=(10, 9), sharex=True)
for n in ["PID", "PID+IOB+PLGS"]:
    df = run("adult#001", 60.0, n)[0]
    ax[0].plot(df.t_h, df.BG, color=COL[n], label=n)
    ax[1].plot(df.t_h, df.Insulin_Uh, color=COL[n])
    ax[2].plot(df.t_h, df.IOB, color=COL[n], label=n)

# Tetto IOB nominale per adult#001 (1.5 x fabbisogno basale orario = 1.5 * 1.267 U/h)
ax[2].axhline(1.5 * 1.267, color="gray", ls="-.", label="Tetto IOB (1.5 x basale oraria)")
band(ax[0])
ax[0].set_ylabel("Glicemia plasmatica [mg/dL]")
ax[0].legend()
ax[0].set_title("adult#001, pasto non annunciato 60 g alle 12:00 (linea tratteggiata)")
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

# 3.2 Sensibilità al carico prandiale (30g, 60g, 90g)
fig, ax = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
for i, c in enumerate([30.0, 60.0, 90.0]):
    for n in ["PID", "PID cons.", "PID+IOB+PLGS"]:
        ax[i].plot(run("adult#001", c, n)[0].t_h, run("adult#001", c, n)[0].BG, color=COL[n], label=n)
    band(ax[i])
    ax[i].set_ylabel(f"BG [mg/dL] ({int(c)} g)")
    ax[i].grid(ls=":", alpha=0.5)

ax[0].legend()
ax[0].set_title("Sensibilita' al carico di carboidrati (adult#001)")
ax[2].set_xlabel("Tempo [h]")
ax[2].set_xlim(0, 16)
plt.tight_layout()
plt.savefig("results/fig_sensibilita_carico.png", dpi=200)
plt.close()

# 3.3 Confronto Multipaziente (adult#001, adult#002, adult#003 a 60g)
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
