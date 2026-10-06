import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

from simglucose.simulation.env import T1DSimEnv
from simglucose.actuator.pump import InsulinPump
from simglucose.sensor.cgm import CGMSensor
from simglucose.patient.t1dpatient import T1DPatient
from simglucose.simulation.scenario import CustomScenario
from simglucose.controller.base import Controller, Action

# -----------------------------------------------------------------------------
# 1. CONTROLLORE AID DI RIFERIMENTO (PID CLASSICO CON CLAMPING STATICO)
# -----------------------------------------------------------------------------
class BaselinePIDController(Controller):
    """
    Controllore discreto standard Proporzionale-Integrale-Derivativo.
    Opera in pura retroazione dall'errore CGM rispetto al setpoint Gtarget.
    Include anti-windup classico con saturazione simmetrica dell'accumulatore integrale.
    """
    def __init__(self, target=115.0, kp=0.0006, ki=0.00001, kd=0.004, basal_rate=0.025):
        self.target = target
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.basal_rate = basal_rate
        self.integral_error = 0.0
        self.prev_cgm = None

    def policy(self, observation, reward, done, **info):
        cgm = observation.CGM
        error = cgm - self.target
        d_error = 0.0 if self.prev_cgm is None else (self.prev_cgm - cgm)
        self.prev_cgm = cgm

        # Anti-windup statico convenzionale
        self.integral_error = np.clip(self.integral_error + error, -5000.0, 15000.0)

        # Azione di controllo: basale + PID
        correction = (self.kp * error) + (self.ki * self.integral_error) + (self.kd * d_error)
        u = max(0.0, self.basal_rate + correction)
        
        # Nessun bolo annunciato (Full Closed-Loop puro)
        return Action(basal=u, bolus=0.0)

    def reset(self):
        self.integral_error = 0.0
        self.prev_cgm = None

# -----------------------------------------------------------------------------
# 2. RUNNER SIMULAZIONE SU SIMGSLUCOSE
# -----------------------------------------------------------------------------
def run_simulation(scenario_tuples, sim_hours=16, patient_name="adult#001"):
    start_time = datetime(2026, 10, 6, 8, 0, 0)
    scenario = CustomScenario(start_time=start_time, scenario=scenario_tuples)

    patient = T1DPatient.withName(patient_name)
    sensor = CGMSensor.withName("Dexcom", seed=10)
    pump = InsulinPump.withName("Insulet")
    env = T1DSimEnv(patient, sensor, pump, scenario)

    # Calcolo del fabbisogno basale esatto da parametri fisiologici stazionari
    try:
        u2ss = patient._params["u2ss"]
        bw = patient._params["BW"]
        basal_nom = (u2ss * bw) / 6000.0
    except Exception:
        basal_nom = 0.0248

    ctrl = BaselinePIDController(target=115.0, basal_rate=basal_nom)
    obs, reward, done, info = env.reset()
    ctrl.reset()

    history = {"time": [], "BG": [], "CGM": [], "insulin": []}
    for step in range(sim_hours * 60):
        t = start_time + timedelta(minutes=step)
        action = ctrl.policy(obs, reward, done, **info)
        obs, reward, done, info = env.step(action)
        history["time"].append(t)
        history["BG"].append(env.patient.observation.Gsub)
        history["CGM"].append(obs.CGM)
        history["insulin"].append(action.basal)

    df = pd.DataFrame(history)
    bg = df["BG"]
    
    # Calcolo Metriche Cliniche Internazionali
    tir = (np.sum((bg >= 70) & (bg <= 180)) / len(bg)) * 100.0
    tar = (np.sum(bg > 180) / len(bg)) * 100.0
    tbr = (np.sum(bg < 70) / len(bg)) * 100.0
    g_max = bg.max()
    g_min = bg.min()
    
    # Istante del picco
    peak_idx = bg.idxmax()
    time_to_peak = (df.loc[peak_idx, "time"] - start_time).total_seconds() / 60.0

    metrics = {
        "TIR [%]": tir,
        "TAR [%]": tar,
        "TBR [%]": tbr,
        "Picco [mg/dL]": g_max,
        "Nadir [mg/dL]": g_min,
        "Tempo Picco [min]": time_to_peak
    }
    return df, metrics

# =============================================================================
# ESECUZIONE DELLA SUITE DI TEST RICHIESTA DA GIADA
# =============================================================================
print("=====================================================================")
print("  AVVIO CARATTERIZZAZIONE BASELINE PID - PIATTAFORMA SIMGSLUCOSE     ")
print("=====================================================================")

sim_data = {}
summary_metrics = []

# --- TEST 1: Condizioni Nominali a Digiuno (Nessun Pasto) ---
print("\n[1/3] Simulazione a digiuno (Condizioni Nominali)...")
df_fasting, met_fasting = run_simulation(scenario_tuples=[])
sim_data["Fasting"] = df_fasting
met_fasting["Scenario"] = "Condizioni Nominali (Digiuno)"
summary_metrics.append(met_fasting)

# --- TEST 2: Sensibilità alla Quantità di Carboidrati (Pasto alle 12:00, t = 4h) ---
carbs = [30, 60, 90]
for c in carbs:
    print(f"\n[2/3] Simulazione Pasto non Annunciato da {c}g CHO...")
    scen = [(timedelta(hours=4), c)]
    df_carb, met_carb = run_simulation(scenario_tuples=scen)
    sim_data[f"Carb_{c}"] = df_carb
    met_carb["Scenario"] = f"Pasto {c}g CHO (Rapido)"
    summary_metrics.append(met_carb)

# --- TEST 3: Sensibilità alla Velocità di Assorbimento (Pasto Dilazionato 60g) ---
# Un pasto misto prolungato viene modellato distribuendo l'immissione (es. 20g + 20g + 20g)
print("\n[3/3] Simulazione Pasto non Annunciato da 60g CHO (Assorbimento Lento/Dilazionato)...")
scen_slow = [
    (timedelta(hours=4), 20),
    (timedelta(hours=4, minutes=30), 20),
    (timedelta(hours=5), 20)
]
df_slow, met_slow = run_simulation(scenario_tuples=scen_slow)
sim_data["Carb_60_slow"] = df_slow
met_slow["Scenario"] = "Pasto 60g CHO (Assorbimento Lento)"
summary_metrics.append(met_slow)

# --- SALVATAGGIO TABELLA ---
df_summary = pd.DataFrame(summary_metrics)
cols = ["Scenario", "TIR [%]", "TAR [%]", "TBR [%]", "Picco [mg/dL]", "Nadir [mg/dL]", "Tempo Picco [min]"]
df_summary = df_summary[cols]

print("\n" + "="*80)
print("             TABELLA RIASSUNTIVA RISULTATI BASELINE PID")
print("="*80)
print(df_summary.round(2).to_string(index=False))
df_summary.to_csv("benchmark_baseline_pid.csv", index=False)
print("\n-> File CSV esportato: benchmark_baseline_pid.csv")

# -----------------------------------------------------------------------------
# GENERAZIONE GRAFICI COMPARATIVI
# -----------------------------------------------------------------------------
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 8), sharex=True)

# Grafico A: Sensibilità al carico (30g vs 60g vs 90g)
ax1.plot(sim_data["Fasting"]["time"], sim_data["Fasting"]["BG"], label="Digiuno (Nominale)", color="gray", ls="--")
ax1.plot(sim_data["Carb_30"]["time"], sim_data["Carb_30"]["BG"], label="30g CHO", color="#2ca02c")
ax1.plot(sim_data["Carb_60"]["time"], sim_data["Carb_60"]["BG"], label="60g CHO", color="#ff7f0e")
ax1.plot(sim_data["Carb_90"]["time"], sim_data["Carb_90"]["BG"], label="90g CHO", color="#d62728")
ax1.axhspan(70, 180, color="green", alpha=0.12, label="Fascia Target [70-180]")
ax1.axhline(70, color="red", ls=":", alpha=0.8)
ax1.axhline(180, color="darkorange", ls=":", alpha=0.8)
ax1.set_ylabel("Glicemia [mg/dL]", fontweight="bold")
ax1.set_title("Risposta del Controllore PID al variare del Carico di Carboidrati (adult#001)", fontweight="bold")
ax1.grid(True, ls=":", alpha=0.6)
ax1.legend(loc="upper right", framealpha=0.9)

# Grafico B: Impatto della velocità di assorbimento (60g rapido vs 60g lento)
ax2.plot(sim_data["Carb_60"]["time"], sim_data["Carb_60"]["BG"], label="60g Rapido (Carboidrati Semplici)", color="#ff7f0e", lw=2)
ax2.plot(sim_data["Carb_60_slow"]["time"], sim_data["Carb_60_slow"]["BG"], label="60g Lento / Dilazionato (Pasto Misto)", color="#9467bd", lw=2)
ax2.axhspan(70, 180, color="green", alpha=0.12)
ax2.axhline(70, color="red", ls=":", alpha=0.8)
ax2.axhline(180, color="darkorange", ls=":", alpha=0.8)
ax2.set_ylabel("Glicemia [mg/dL]", fontweight="bold")
ax2.set_xlabel("Orario della Simulazione", fontweight="bold")
ax2.set_title("Confronto Cinetica di Assorbimento: Pasto Rapido vs Dilazionato (60g CHO)", fontweight="bold")
ax2.grid(True, ls=":", alpha=0.6)
ax2.legend(loc="upper right", framealpha=0.9)

plt.tight_layout()
plt.savefig("grafico_benchmark_baseline.png", dpi=300)
print("-> Grafico esportato: grafico_benchmark_baseline.png")
print("=====================================================================")
print("Studio preliminare completato con successo!")