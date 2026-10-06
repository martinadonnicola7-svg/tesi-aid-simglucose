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
        
        if d_cgm < 0 and error > 0:
            pass
        else:
            self.integral_error = np.clip(self.integral_error + error, -2000, 8000)
            
        correction = (self.kp * error) + (self.ki * self.integral_error) - (self.kd * d_cgm)
        desired_u = max(0.0, self.basal_rate + correction)
        
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

        cgm_projected = cgm + (d_cgm * 20.0)
        if cgm_projected < 75 or cgm < 80:
            actual_u = 0.0

        return Action(basal=actual_u, bolus=0.0)

    def reset(self):
        self.s1 = 0.0
        self.s2 = 0.0
        self.integral_error = 0.0
        self.prev_cgm = None

def simulate_scenario(mode="announced", patient_name="adult#001", cho=60):
    start_time = datetime(2026, 10, 4, 8, 0, 0)
    meal_time = start_time + timedelta(hours=4)
    scenario_tuples = [(timedelta(hours=4), cho)]
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

    if mode == "announced":
        ctrl = StandardPIDController(basal_rate=basal_nom)
    elif mode == "unannounced_pid":
        ctrl = StandardPIDController(basal_rate=basal_nom)
    elif mode == "unannounced_iob":
        ctrl = IOBConstrainedController(basal_rate=basal_nom)

    obs, reward, done, info = env.reset()
    ctrl.reset()

    history = {"time": [], "BG": [], "CGM": [], "insulin": []}
    bolus_given = False
    ic_ratio = 12.0

    for step in range(16 * 60):
        t = start_time + timedelta(minutes=step)
        action = ctrl.policy(obs, reward, done, **info)
        
        if mode == "announced" and not bolus_given and t >= (meal_time - timedelta(minutes=15)):
            bolus_val = cho / ic_ratio
            action = Action(basal=action.basal, bolus=bolus_val)
            bolus_given = True

        obs, reward, done, info = env.step(action)
        
        history["time"].append(t)
        history["BG"].append(env.patient.observation.Gsub)
        history["CGM"].append(obs.CGM)
        tot_u = action.basal + (action.bolus if action.bolus else 0.0)
        history["insulin"].append(tot_u)

    return pd.DataFrame(history)

print("1/3: Esecuzione Pasto Annunciato (Baseline)...")
df_ann = simulate_scenario("announced")

print("2/3: Esecuzione Pasto Non Annunciato (PID Standard)...")
df_pid = simulate_scenario("unannounced_pid")

print("3/3: Esecuzione Pasto Non Annunciato (IOB Proposto)...")
df_iob = simulate_scenario("unannounced_iob")

def get_metrics(df):
    bg = df["BG"]
    tir = np.sum((bg >= 70) & (bg <= 180)) / len(bg) * 100
    tar = np.sum(bg > 180) / len(bg) * 100
    tbr = np.sum(bg < 70) / len(bg) * 100
    return {
        "TIR [%] (70-180)": tir,
        "TAR [%] (>180)": tar,
        "TBR [%] (<70)": tbr,
        "Picco [mg/dL]": bg.max(),
        "Nadir [mg/dL]": bg.min()
    }

results = pd.DataFrame({
    "Pasto Annunciato (FF)": get_metrics(df_ann),
    "Non Annunciato (PID Standard)": get_metrics(df_pid),
    "Non Annunciato (IOB Proposto)": get_metrics(df_iob)
}).T

print("\n================== RISULTATI COMPARATIVI ==================")
print(results.round(2))
results.to_csv("metriche_confronto.csv")

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 7), sharex=True, gridspec_kw={"height_ratios": [2.5, 1]})

ax1.plot(df_ann["time"], df_ann["BG"], label="Annunciato (Bolo Feedforward)", color="#2ca02c", linewidth=2)
ax1.plot(df_pid["time"], df_pid["BG"], label="Non Annunciato (PID Standard)", color="#d62728", linewidth=2, linestyle="--")
ax1.plot(df_iob["time"], df_iob["BG"], label="Non Annunciato (IOB Proposto)", color="#1f77b4", linewidth=2.2)

ax1.axhspan(70, 180, color="green", alpha=0.12, label="Target Clinico TIR (70-180 mg/dL)")
ax1.axhline(180, color="darkorange", linestyle=":", alpha=0.8, label="Soglia Iperglicemia (180)")
ax1.axhline(70, color="red", linestyle=":", alpha=0.8, label="Soglia Ipoglicemia (70)")

ax1.set_ylabel("Glicemia Sottocutanea [mg/dL]", fontsize=11)
ax1.set_title("Risposta del Sistema AID a un Pasto di 60g CHO (Paziente Adult#001)", fontsize=12, fontweight="bold")
ax1.grid(True, linestyle=":", alpha=0.6)
ax1.legend(loc="upper right", framealpha=0.9)

ax2.plot(df_ann["time"], df_ann["insulin"], color="#2ca02c", alpha=0.8, label="Insulina (Annunciato)")
ax2.plot(df_pid["time"], df_pid["insulin"], color="#d62728", linestyle="--", alpha=0.8, label="Insulina (PID)")
ax2.plot(df_iob["time"], df_iob["insulin"], color="#1f77b4", alpha=0.9, label="Insulina (IOB Proposto)")

ax2.set_ylabel("Erogazione [U/min]", fontsize=11)
ax2.set_xlabel("Orario", fontsize=11)
ax2.grid(True, linestyle=":", alpha=0.6)
ax2.legend(loc="upper right")

plt.tight_layout()
plt.savefig("risultati_simulazione_aid.png", dpi=300)
print("\nSimulazione completata con successo! File salvati: metriche_confronto.csv e risultati_simulazione_aid.png")
