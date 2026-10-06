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

# ----------------------------------------------------------------------
# 1. CONTROLLORE PID STANDARD
# ----------------------------------------------------------------------
class StandardPIDController(Controller):
    def __init__(self, target=115, kp=0.0006, ki=0.00001, kd=0.004, basal_rate=0.025):
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
        self.integral_error = np.clip(self.integral_error + error, -5000, 15000)
        correction = (self.kp * error) + (self.ki * self.integral_error) + (self.kd * d_error)
        u = max(0.0, self.basal_rate + correction)
        return Action(basal=u, bolus=0.0)

    def reset(self):
        self.integral_error = 0.0
        self.prev_cgm = None

# ----------------------------------------------------------------------
# 2. CONTROLLORE PROPOSTO VINCOLATO ALL'IOB
# ----------------------------------------------------------------------
class IOBConstrainedController(Controller):
    def __init__(self, target=115, kp=0.0007, ki=0.000015, kd=0.005, 
                 basal_rate=0.025, tau_s=50.0, iob_max_factor=3.5):
        self.target = target
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.basal_rate = basal_rate
        self.tau_s = tau_s                  
        self.iob_max_factor = iob_max_factor 
        self.s1 = 0.0
        self.s2 = 0.0
        self.integral_error = 0.0
        self.prev_cgm = None

    def policy(self, observation, reward, done, **info):
        cgm = observation.CGM
        error = cgm - self.target
        d_cgm = 0.0 if self.prev_cgm is None else (cgm - self.prev_cgm)
        self.prev_cgm = cgm
        
        # Integrator Clamping subordinato alla derivata
        if d_cgm < 0 and error > 0:
            pass
        else:
            self.integral_error = np.clip(self.integral_error + error, -2000, 8000)
            
        correction = (self.kp * error) + (self.ki * self.integral_error) - (self.kd * d_cgm)
        desired_u = max(0.0, self.basal_rate + correction)
        
        # Saturazione dinamica basata su osservatore IOB
        iob_current = self.s1 + self.s2
        iob_upper_bound = self.basal_rate * 60.0 * self.iob_max_factor
        
        if iob_current > iob_upper_bound:
            actual_u = min(desired_u, self.basal_rate * 0.5)
        else:
            actual_u = desired_u
            
        u_active = max(0.0, actual_u - self.basal_rate)
        ds1 = u_active - (self.s1 / self.tau_s)
        ds2 = (self.s1 / self.tau_s) - (self.s2 / self.tau_s)
        self.s1 += ds1
        self.s2 += ds2

        # Arresto predittivo preventivo PLGS
        cgm_projected = cgm + (d_cgm * 20.0)
        if cgm_projected < 75 or cgm < 80:
            actual_u = 0.0

        return Action(basal=actual_u, bolus=0.0)

    def reset(self):
        self.s1 = 0.0
        self.s2 = 0.0
        self.integral_error = 0.0
        self.prev_cgm = None

# ----------------------------------------------------------------------
# 3. FUNZIONE DI SIMULAZIONE GENERICA
# ----------------------------------------------------------------------
def simulate_single(patient_name, controller_type, cho_grams):
    start_time = datetime(2026, 10, 4, 8, 0, 0)
    scenario_tuples = [(timedelta(hours=4), cho_grams)]
    scenario = CustomScenario(start_time=start_time, scenario=scenario_tuples)

    patient = T1DPatient.withName(patient_name)
    sensor = CGMSensor.withName("Dexcom", seed=10)
    pump = InsulinPump.withName("Insulet")
    env = T1DSimEnv(patient, sensor, pump, scenario)

    try:
        u2ss = patient._params["u2ss"]
        bw = patient._params["BW"]
        basal_nom = (u2ss * bw) / 6000.0
    except Exception:
        basal_nom = 0.025

    if controller_type == "PID":
        ctrl = StandardPIDController(basal_rate=basal_nom)
    else:
        ctrl = IOBConstrainedController(basal_rate=basal_nom)

    obs, reward, done, info = env.reset()
    ctrl.reset()

    history = {"time": [], "BG": [], "insulin": []}
    for step in range(16 * 60):
        t = start_time + timedelta(minutes=step)
        action = ctrl.policy(obs, reward, done, **info)
        obs, reward, done, info = env.step(action)
        history["time"].append(t)
        history["BG"].append(env.patient.observation.Gsub)
        history["insulin"].append(action.basal)

    df = pd.DataFrame(history)
    bg = df["BG"]
    metrics = {
        "Paziente": patient_name,
        "Controllore": controller_type,
        "CHO [g]": cho_grams,
        "TIR [%]": (np.sum((bg >= 70) & (bg <= 180)) / len(bg)) * 100,
        "TAR [%]": (np.sum(bg > 180) / len(bg)) * 100,
        "TBR [%]": (np.sum(bg < 70) / len(bg)) * 100,
        "Picco [mg/dL]": bg.max(),
        "Nadir [mg/dL]": bg.min()
    }
    return df, metrics

# ----------------------------------------------------------------------
# 4. CAMPAGNA 1: ANALISI DI SENSIBILITA' AL CARICO (30, 60, 90 g)
# ----------------------------------------------------------------------
print("=== AVVIO CAMPAGNA 1: STUDIO DI SENSIBILITA' AL CARICO (adult#001) ===")
meals = [30, 60, 90]
results_list = []
curves_sens = {}

for m in meals:
    print(f"-> Esecuzione pasto {m}g CHO...")
    df_pid, met_pid = simulate_single("adult#001", "PID", m)
    df_iob, met_iob = simulate_single("adult#001", "IOB", m)
    results_list.extend([met_pid, met_iob])
    curves_sens[f"PID_{m}"] = df_pid
    curves_sens[f"IOB_{m}"] = df_iob

# ----------------------------------------------------------------------
# 5. CAMPAGNA 2: ROBUSTEZZA MULTI-PAZIENTE (adult#001, adult#002, adult#003)
# ----------------------------------------------------------------------
print("\n=== AVVIO CAMPAGNA 2: VALIDAZIONE SU POPOLAZIONE VIRTUALE ===")
patients = ["adult#001", "adult#002", "adult#003"]
curves_pat = {}

for p in patients:
    print(f"-> Esecuzione su {p} (pasto 60g CHO)...")
    df_p_pid, met_p_pid = simulate_single(p, "PID", 60)
    df_p_iob, met_p_iob = simulate_single(p, "IOB", 60)
    if p != "adult#001": # adult#001 a 60g e' gia' in lista
        results_list.extend([met_p_pid, met_p_iob])
    curves_pat[f"PID_{p}"] = df_p_pid
    curves_pat[f"IOB_{p}"] = df_p_iob

# Salva tabella riassuntiva
df_results = pd.DataFrame(results_list)
print("\n=================== TABELLA METRICHE CLINICHE ===================")
print(df_results.round(2).to_string(index=False))
df_results.to_csv("metriche_estese.csv", index=False)
print("Salvato: metriche_estese.csv")

# ----------------------------------------------------------------------
# 6. GENERAZIONE DEI GRAFICI COMPARATIVI PER LA TESI
# ----------------------------------------------------------------------
# Grafico 1: Sensibilita' al pasto
fig, axes = plt.subplots(3, 1, figsize=(10, 9), sharex=True)
colors = {"PID": "#d62728", "IOB": "#1f77b4"}

for idx, m in enumerate(meals):
    ax = axes[idx]
    ax.plot(curves_sens[f"PID_{m}"]["time"], curves_sens[f"PID_{m}"]["BG"], 
            label="PID Standard", color=colors["PID"], linestyle="--", lw=2)
    ax.plot(curves_sens[f"IOB_{m}"]["time"], curves_sens[f"IOB_{m}"]["BG"], 
            label="IOB Proposto", color=colors["IOB"], lw=2.2)
    ax.axhspan(70, 180, color="green", alpha=0.12)
    ax.axhline(180, color="darkorange", ls=":", alpha=0.7)
    ax.axhline(70, color="red", ls=":", alpha=0.7)
    ax.set_ylabel("Glicemia [mg/dL]")
    ax.set_title(f"Carico Prandiale non Annunciato: {m}g CHO (adult#001)", fontweight="bold", fontsize=10)
    ax.grid(True, ls=":", alpha=0.6)
    if idx == 0:
        ax.legend(loc="upper right")

axes[-1].set_xlabel("Orario")
plt.tight_layout()
plt.savefig("grafico_sensibilita_pasti.png", dpi=300)
print("Salvato grafico: grafico_sensibilita_pasti.png")

# Grafico 2: Robustezza Multi-paziente
fig2, axes2 = plt.subplots(3, 1, figsize=(10, 9), sharex=True)
for idx, p in enumerate(patients):
    ax = axes2[idx]
    ax.plot(curves_pat[f"PID_{p}"]["time"], curves_pat[f"PID_{p}"]["BG"], 
            label="PID Standard", color=colors["PID"], linestyle="--", lw=2)
    ax.plot(curves_pat[f"IOB_{p}"]["time"], curves_pat[f"IOB_{p}"]["BG"], 
            label="IOB Proposto", color=colors["IOB"], lw=2.2)
    ax.axhspan(70, 180, color="green", alpha=0.12)
    ax.axhline(180, color="darkorange", ls=":", alpha=0.7)
    ax.axhline(70, color="red", ls=":", alpha=0.7)
    ax.set_ylabel("Glicemia [mg/dL]")
    ax.set_title(f"Risposta Glicemica su Soggetto Virtuale: {p} (60g CHO)", fontweight="bold", fontsize=10)
    ax.grid(True, ls=":", alpha=0.6)
    if idx == 0:
        ax.legend(loc="upper right")

axes2[-1].set_xlabel("Orario")
plt.tight_layout()
plt.savefig("grafico_multipaziente.png", dpi=300)
print("Salvato grafico: grafico_multipaziente.png")
print("\n--> Campagna sperimentale completata con successo!")
