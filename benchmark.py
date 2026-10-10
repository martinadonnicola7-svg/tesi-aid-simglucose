"""
benchmark.py - Modulo di benchmark e algoritmi di controllo AID
Implementazione e validazione in silico su modello metabolico UVA/Padova (simglucose).
Correzioni integrate:
  - Durata: 16 ore esatte (08:00 - 00:00)
  - Campionamento: dt = 3 min (Dexcom sensor, 320 passi)
  - Integrazione temporale corretta per integrale PID e osservatore IOB
  - Metriche cliniche estese (TIR, TAR L1/L2, TBR L1/L2, TBR tardivo postprandiale 3-6h)
"""

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


# ==============================================================================
# 1. ALGORITMI DI CONTROLLO
# ==============================================================================

class StandardPIDController(Controller):
    """
    Controllore PID classico a tempo discreto con anti-windup statico (clamping).
    """
    def __init__(self, target=110.0, kp=0.0006, ki=0.000015, kd=0.004, basal_rate=0.021, dt=3.0):
        self.target = target
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.basal_rate = basal_rate
        self.dt = dt
        self.integral_error = 0.0
        self.prev_cgm = None

    def policy(self, observation, reward, done, **info):
        cgm = observation.CGM
        error = cgm - self.target
        
        # Integrazione numerica riscalata per dt = 3 min
        self.integral_error += error * self.dt
        self.integral_error = float(np.clip(self.integral_error, -3000.0, 10000.0))
        
        # Derivata temporale (variazione di CGM al minuto)
        if self.prev_cgm is None:
            d_cgm_dt = 0.0
        else:
            d_cgm_dt = (cgm - self.prev_cgm) / self.dt
        self.prev_cgm = cgm
        
        # Legge di controllo
        correction = (self.kp * error) + (self.ki * self.integral_error) + (self.kd * d_cgm_dt)
        u = self.basal_rate + correction
        u = max(0.0, min(u, 0.25))  # Saturazione attuatore [0, 0.25] U/min
        
        return Action(basal=u, bolus=0.0)

    def reset(self):
        self.integral_error = 0.0
        self.prev_cgm = None


class IOBConstrainedController(Controller):
    """
    Controllore PID con saturazione dinamica basata su stima dell'insulina attiva (IOB)
    e modulo di sicurezza PLGS (Predictive Low-Glucose Suspend).
    """
    def __init__(self, target=110.0, kp=0.0006, ki=0.000015, kd=0.004, basal_rate=0.021,
                 tau_s=50.0, iob_cap=2.2, plgs_threshold=90.0, dt=3.0):
        self.target = target
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.basal_rate = basal_rate
        self.tau_s = tau_s
        self.iob_cap = iob_cap
        self.plgs_threshold = plgs_threshold
        self.dt = dt
        
        self.integral_error = 0.0
        self.prev_cgm = None
        self.s1 = 0.0
        self.s2 = 0.0
        self.current_iob = 0.0

    def policy(self, observation, reward, done, **info):
        cgm = observation.CGM
        error = cgm - self.target
        
        # Stima IOB residua in Unità [U] dai due compartimenti sottocutanei
        self.current_iob = (self.s1 + self.s2) / 60.0
        
        # Calcolo derivata al minuto
        if self.prev_cgm is None:
            d_cgm_dt = 0.0
        else:
            d_cgm_dt = (cgm - self.prev_cgm) / self.dt
            
        # Proiezione lineare predittiva a 20 minuti
        cgm_projected = cgm + (d_cgm_dt * 20.0)
        
        # Anti-windup con scarica asimmetrica
        if cgm > self.target:
            self.integral_error += error * self.dt
        else:
            self.integral_error = max(0.0, self.integral_error - (abs(error) * self.dt * 0.5))
        self.integral_error = float(np.clip(self.integral_error, -1000.0, 6000.0))
        
        # Calcolo correzione PID nominale
        correction = (self.kp * error) + (self.ki * self.integral_error) + (self.kd * d_cgm_dt)
        u = self.basal_rate + correction
        
        # 1. Vincolo IOB: blocca infusione correttiva oltre la basale se IOB supera la soglia
        if self.current_iob > self.iob_cap and u > self.basal_rate:
            u = self.basal_rate
            
        # 2. Modulo PLGS: sospensione totale se la proiezione o la misura corrente scende sotto soglia
        if cgm_projected < self.plgs_threshold or cgm < 75.0:
            u = 0.0
            
        u = max(0.0, min(u, 0.25))
        
        # Aggiornamento dinamica compartimentale sottocutanea per il passo successivo
        u_excess = max(0.0, u - self.basal_rate)
        ds1 = (u_excess - (self.s1 / self.tau_s)) * self.dt
        ds2 = ((self.s1 / self.tau_s) - (self.s2 / self.tau_s)) * self.dt
        self.s1 += ds1
        self.s2 += ds2
        self.prev_cgm = cgm
        
        return Action(basal=u, bolus=0.0)

    def reset(self):
        self.integral_error = 0.0
        self.prev_cgm = None
        self.s1 = 0.0
        self.s2 = 0.0
        self.current_iob = 0.0


# ==============================================================================
# 2. FRAMEWORK DI SIMULAZIONE
# ==============================================================================

def simulate_patient(controller_class, patient_name='adult#001', meal_cho=60.0, 
                     sim_hours=16.0, meal_time_hour=12.0, seed=1):
    """
    Esegue la simulazione in silico per una durata specificata (default 16 ore).
    """
    start_time = datetime(2026, 1, 1, 8, 0, 0)
    meal_scenario = [(meal_time_hour, meal_cho)]
    
    patient = T1DPatient.withName(patient_name)
    sensor = CGMSensor.withName('Dexcom', seed=seed)
    pump = InsulinPump.withName('Insulet')
    scenario = CustomScenario(start_time=start_time, scenario=meal_scenario)
    env = T1DSimEnv(patient, sensor, pump, scenario)
    
    dt = env.sample_time  # 3 minuti
    total_steps = int(sim_hours * 60 / dt)  # 320 passi per 16h
    
    bw = patient._params['BW']
    u2ss = patient._params['u2ss']
    basal_nominal = (u2ss * bw) / 6000.0  # U/min
    
    controller = controller_class(basal_rate=basal_nominal, dt=dt)
    
    time_log, cgm_log, bg_log, ins_log, iob_log = [], [], [], [], []
    obs, reward, done, info = env.reset()
    controller.reset()
    
    for step in range(total_steps):
        current_t = start_time + timedelta(minutes=step * dt)
        action = controller.policy(obs, reward, done, **info)
        
        time_log.append(current_t)
        cgm_log.append(obs.CGM)
        bg_log.append(env.patient.state[0])
        ins_log.append(action.basal * 60.0)  # Erogazione in U/h
        iob_log.append(getattr(controller, 'current_iob', 0.0))
        
        obs, reward, done, info = env.step(action)
        if done:
            break
            
    return pd.DataFrame({
        'Time': time_log,
        'CGM': cgm_log,
        'BG': bg_log,
        'Insulin_Uh': ins_log,
        'IOB': iob_log
    })


# ==============================================================================
# 3. METRICHE CLINICHE E INDICATORI DI CONSENSO
# ==============================================================================

def compute_clinical_metrics(df, controller_label, patient_label='adult#001', cho=60.0):
    """
    Calcola le metriche di consenso internazionale (TIR, TAR L1/L2, TBR L1/L2)
    e l'indicatore di ipoglicemia tardiva specifica (finestra 3-6 ore postprandiale).
    """
    cgm = df['CGM'].values
    n = len(cgm)
    
    tir = np.sum((cgm >= 70.0) & (cgm <= 180.0)) / n * 100.0
    tar_l1 = np.sum((cgm > 180.0) & (cgm <= 250.0)) / n * 100.0
    tar_l2 = np.sum(cgm > 250.0) / n * 100.0
    tar_tot = tar_l1 + tar_l2
    
    tbr_l1 = np.sum((cgm >= 54.0) & (cgm < 70.0)) / n * 100.0
    tbr_l2 = np.sum(cgm < 54.0) / n * 100.0
    tbr_tot = tbr_l1 + tbr_l2
    
    peak = np.max(cgm)
    nadir = np.min(cgm)
    
    # Finestra tardiva: 3-6 ore dopo il pasto delle 12:00 (dalle 15:00 alle 18:00)
    mask_late = (df['Time'] >= datetime(2026, 1, 1, 15, 0, 0)) & \
                (df['Time'] <= datetime(2026, 1, 1, 18, 0, 0))
    cgm_late = df.loc[mask_late, 'CGM'].values
    tbr_late = np.sum(cgm_late < 70.0) / len(cgm_late) * 100.0 if len(cgm_late) > 0 else 0.0
    
    return {
        'Paziente': patient_label,
        'Controllore': controller_label,
        'CHO [g]': cho,
        'TIR [%]': round(tir, 2),
        'TAR [%]': round(tar_tot, 2),
        'TAR_L1 [%]': round(tar_l1, 2),
        'TAR_L2 [%]': round(tar_l2, 2),
        'TBR [%]': round(tbr_tot, 2),
        'TBR_L1 [%]': round(tbr_l1, 2),
        'TBR_L2 [%]': round(tbr_l2, 2),
        'TBR_Tardivo (3-6h) [%]': round(tbr_late, 2),
        'Picco [mg/dL]': round(peak, 2),
        'Nadir [mg/dL]': round(nadir, 2)
    }


# ==============================================================================
# 4. FUNZIONE DI PLOTTING COMPARATIVO A 3 PANNELLI
# ==============================================================================

def plot_three_panel_comparison(df_pid, df_iob, filename='grafico_benchmark_baseline.png',
                                patient_name='adult#001', cho=60.0):
    t_hours = [(t - df_pid['Time'].iloc[0]).total_seconds() / 3600.0 for t in df_pid['Time']]
    fig, axes = plt.subplots(3, 1, figsize=(11, 9), sharex=True)
    
    # 1. Glicemia
    axes[0].axhspan(70, 180, color='green', alpha=0.15, label='Target Euglicemico (70-180 mg/dL)')
    axes[0].axhline(70, color='red', linestyle='--', linewidth=1, alpha=0.7)
    axes[0].axhline(180, color='orange', linestyle='--', linewidth=1, alpha=0.7)
    axes[0].axvline(4.0, color='black', linestyle=':', linewidth=1.5, label=f'Pasto non annunciato ({cho}g @ 12:00)')
    axes[0].plot(t_hours, df_pid['CGM'], 'r--', linewidth=2, label='PID Baseline')
    axes[0].plot(t_hours, df_iob['CGM'], 'b-', linewidth=2, label='PID + Vincolo IOB + PLGS')
    axes[0].set_ylabel('CGM [mg/dL]', fontsize=11)
    axes[0].set_title(f'Confronto Dinamico a Ciclo Chiuso ({patient_name}, CHO = {cho}g)', fontsize=12, fontweight='bold')
    axes[0].legend(loc='upper right')
    axes[0].grid(True, linestyle=':', alpha=0.6)
    
    # 2. Infusione
    axes[1].axvline(4.0, color='black', linestyle=':', linewidth=1.5)
    axes[1].plot(t_hours, df_pid['Insulin_Uh'], 'r--', linewidth=1.8, label='Infusione PID Baseline')
    axes[1].plot(t_hours, df_iob['Insulin_Uh'], 'b-', linewidth=1.8, label='Infusione IOB + PLGS')
    axes[1].set_ylabel('Erogazione [U/h]', fontsize=11)
    axes[1].grid(True, linestyle=':', alpha=0.6)
    axes[1].legend(loc='upper right')
    
    # 3. IOB
    axes[2].axvline(4.0, color='black', linestyle=':', linewidth=1.5)
    axes[2].axhline(2.2, color='darkblue', linestyle='-.', alpha=0.8, label='Tetto Massimo IOB (2.2 U)')
    axes[2].plot(t_hours, df_iob['IOB'], 'b-', linewidth=2, label='IOB Stimato (Controllore Vincolato)')
    axes[2].set_ylabel('IOB [U]', fontsize=11)
    axes[2].set_xlabel('Tempo di Simulazione [ore]', fontsize=11)
    axes[2].set_xlim([0, 16])
    axes[2].grid(True, linestyle=':', alpha=0.6)
    axes[2].legend(loc='upper right')
    
    plt.tight_layout()
    if filename:
        plt.savefig(filename, dpi=300)
    plt.close()


# ==============================================================================
# 5. ESECUZIONE AUTONOMA (STANDALONE RUN)
# ==============================================================================

if __name__ == '__main__':
    print("Esecuzione benchmark nominale su adult#001 (CHO = 60g, 16 ore, dt = 3 min)...")
    res_pid = simulate_patient(StandardPIDController, 'adult#001', 60.0)
    res_iob = simulate_patient(IOBConstrainedController, 'adult#001', 60.0)
    
    m1 = compute_clinical_metrics(res_pid, 'PID', 'adult#001', 60.0)
    m2 = compute_clinical_metrics(res_iob, 'IOB', 'adult#001', 60.0)
    
    df_metrics = pd.DataFrame([m1, m2])
    df_metrics.to_csv('metriche_confronto.csv', index=False)
    print("\nMetriche calcolate ed esportate in metriche_confronto.csv:")
    print(df_metrics[['Paziente', 'Controllore', 'TIR [%]', 'TAR [%]', 'TBR [%]', 'TBR_Tardivo (3-6h) [%]', 'Picco [mg/dL]', 'Nadir [mg/dL]']])
    
    plot_three_panel_comparison(res_pid, res_iob, 'grafico_benchmark_baseline.png', 'adult#001', 60.0)
    print("\nGrafico a 3 pannelli generato con successo: grafico_benchmark_baseline.png")
