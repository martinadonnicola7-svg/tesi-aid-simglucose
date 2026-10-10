"""
run_extended_experiments.py
Campagna estesa di simulazione multipaziente e multicarico per sistemi AID.
Configurazione:
  - Pazienti: adult#001, adult#002, adult#003
  - Carichi: 30g, 60g, 90g (non annunciati, ore 12:00)
  - Durata: 16 ore esatte (320 passi con dt = 3 min)
  - Output: metriche_estese.csv, grafico_multipaziente.png, grafico_sensibilita_pasti.png
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


class StandardPIDController(Controller):
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
        self.integral_error += error * self.dt
        self.integral_error = float(np.clip(self.integral_error, -3000.0, 10000.0))
        
        d_cgm_dt = 0.0 if self.prev_cgm is None else (cgm - self.prev_cgm) / self.dt
        self.prev_cgm = cgm
        
        correction = (self.kp * error) + (self.ki * self.integral_error) + (self.kd * d_cgm_dt)
        u = max(0.0, min(self.basal_rate + correction, 0.25))
        return Action(basal=u, bolus=0.0)

    def reset(self):
        self.integral_error = 0.0
        self.prev_cgm = None


class IOBConstrainedController(Controller):
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
        
        self.current_iob = (self.s1 + self.s2) / 60.0
        d_cgm_dt = 0.0 if self.prev_cgm is None else (cgm - self.prev_cgm) / self.dt
        cgm_projected = cgm + (d_cgm_dt * 20.0)
        
        if cgm > self.target:
            self.integral_error += error * self.dt
        else:
            self.integral_error = max(0.0, self.integral_error - (abs(error) * self.dt * 0.5))
        self.integral_error = float(np.clip(self.integral_error, -1000.0, 6000.0))
        
        correction = (self.kp * error) + (self.ki * self.integral_error) + (self.kd * d_cgm_dt)
        u = self.basal_rate + correction
        
        if self.current_iob > self.iob_cap and u > self.basal_rate:
            u = self.basal_rate
            
        if cgm_projected < self.plgs_threshold or cgm < 75.0:
            u = 0.0
            
        u = max(0.0, min(u, 0.25))
        
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


def simulate(controller_class, patient_name, meal_cho, sim_hours=16.0):
    start_time = datetime(2026, 1, 1, 8, 0, 0)
    meal_scenario = [(12.0, meal_cho)]
    
    patient = T1DPatient.withName(patient_name)
    sensor = CGMSensor.withName('Dexcom', seed=1)
    pump = InsulinPump.withName('Insulet')
    scenario = CustomScenario(start_time=start_time, scenario=meal_scenario)
    env = T1DSimEnv(patient, sensor, pump, scenario)
    
    dt = env.sample_time
    total_steps = int(sim_hours * 60 / dt)
    
    bw = patient._params['BW']
    u2ss = patient._params['u2ss']
    basal_nominal = (u2ss * bw) / 6000.0
    
    controller = controller_class(basal_rate=basal_nominal, dt=dt)
    
    time_log, cgm_log, ins_log, iob_log = [], [], [], []
    obs, reward, done, info = env.reset()
    controller.reset()
    
    for step in range(total_steps):
        current_t = start_time + timedelta(minutes=step * dt)
        action = controller.policy(obs, reward, done, **info)
        
        time_log.append(current_t)
        cgm_log.append(obs.CGM)
        ins_log.append(action.basal * 60.0)
        iob_log.append(getattr(controller, 'current_iob', 0.0))
        
        obs, reward, done, info = env.step(action)
        if done:
            break
            
    return pd.DataFrame({
        'Time': time_log,
        'CGM': cgm_log,
        'Insulin_Uh': ins_log,
        'IOB': iob_log
    })


def calc_metrics(df, patient, ctrl_name, cho):
    cgm = df['CGM'].values
    n = len(cgm)
    
    tir = np.sum((cgm >= 70.0) & (cgm <= 180.0)) / n * 100.0
    tar = np.sum(cgm > 180.0) / n * 100.0
    tbr = np.sum(cgm < 70.0) / n * 100.0
    
    mask_late = (df['Time'] >= datetime(2026, 1, 1, 15, 0, 0)) & (df['Time'] <= datetime(2026, 1, 1, 18, 0, 0))
    cgm_late = df.loc[mask_late, 'CGM'].values
    tbr_late = np.sum(cgm_late < 70.0) / len(cgm_late) * 100.0 if len(cgm_late) > 0 else 0.0
    
    return {
        'Paziente': patient,
        'Controllore': ctrl_name,
        'CHO [g]': int(cho),
        'TIR [%]': round(tir, 2),
        'TAR [%]': round(tar, 2),
        'TBR [%]': round(tbr, 2),
        'TBR_Tardivo (3-6h) [%]': round(tbr_late, 2),
        'Picco [mg/dL]': round(float(np.max(cgm)), 1),
        'Nadir [mg/dL]': round(float(np.min(cgm)), 1)
    }


def main():
    print("Avvio campagna estesa multipaziente e multicarico...")
    records = []
    
    # 1. Analisi di Sensibilità al Carico su adult#001 (30g, 60g, 90g)
    cho_list = [30.0, 60.0, 90.0]
    cho_dfs = {}
    for cho in cho_list:
        print(f"  -> adult#001 con pasto {int(cho)}g...")
        df_p = simulate(StandardPIDController, 'adult#001', cho)
        df_i = simulate(IOBConstrainedController, 'adult#001', cho)
        records.append(calc_metrics(df_p, 'adult#001', 'PID', cho))
        records.append(calc_metrics(df_i, 'adult#001', 'IOB', cho))
        cho_dfs[cho] = (df_p, df_i)
        
    # 2. Analisi Multipaziente a carico nominale (60g) su adult#002 e adult#003
    multi_dfs = {'adult#001': cho_dfs[60.0]}
    for pat in ['adult#002', 'adult#003']:
        print(f"  -> {pat} con pasto 60g...")
        df_p = simulate(StandardPIDController, pat, 60.0)
        df_i = simulate(IOBConstrainedController, pat, 60.0)
        records.append(calc_metrics(df_p, pat, 'PID', 60.0))
        records.append(calc_metrics(df_i, pat, 'IOB', 60.0))
        multi_dfs[pat] = (df_p, df_i)
        
    # Salvataggio dataset metriche_estese.csv
    df_metrics = pd.DataFrame(records)
    df_metrics.to_csv('metriche_estese.csv', index=False)
    print("\nDataset completato e salvato in metriche_estese.csv:")
    print(df_metrics[['Paziente', 'Controllore', 'CHO [g]', 'TIR [%]', 'TAR [%]', 'TBR [%]', 'TBR_Tardivo (3-6h) [%]', 'Picco [mg/dL]', 'Nadir [mg/dL]']])
    
    # Generazione grafico_sensibilita_pasti.png
    fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
    t_h = [(t - cho_dfs[60.0][0]['Time'].iloc[0]).total_seconds() / 3600.0 for t in cho_dfs[60.0][0]['Time']]
    for idx, cho in enumerate(cho_list):
        ax = axes[idx]
        ax.axhspan(70, 180, color='green', alpha=0.15)
        ax.axvline(4.0, color='black', linestyle=':')
        ax.plot(t_h, cho_dfs[cho][0]['CGM'], 'r--', label='PID')
        ax.plot(t_h, cho_dfs[cho][1]['CGM'], 'b-', label='PID + Vincolo IOB')
        ax.set_ylabel(f'CGM ({int(cho)}g) [mg/dL]')
        ax.grid(True, linestyle=':', alpha=0.6)
        if idx == 0:
            ax.set_title('Sensibilità ai Carichi di Carboidrati (adult#001)')
            ax.legend(loc='upper right')
    axes[2].set_xlabel('Tempo [ore]')
    axes[2].set_xlim([0, 16])
    plt.tight_layout()
    plt.savefig('grafico_sensibilita_pasti.png', dpi=300)
    plt.close()
    
    # Generazione grafico_multipaziente.png
    fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
    for idx, pat in enumerate(['adult#001', 'adult#002', 'adult#003']):
        ax = axes[idx]
        ax.axhspan(70, 180, color='green', alpha=0.15)
        ax.axvline(4.0, color='black', linestyle=':')
        ax.plot(t_h, multi_dfs[pat][0]['CGM'], 'r--', label='PID')
        ax.plot(t_h, multi_dfs[pat][1]['CGM'], 'b-', label='PID + Vincolo IOB')
        ax.set_ylabel(f'CGM ({pat}) [mg/dL]')
        ax.grid(True, linestyle=':', alpha=0.6)
        if idx == 0:
            ax.set_title('Confronto Multipaziente su Disturbo Prandiale Nominale (60g)')
            ax.legend(loc='upper right')
    axes[2].set_xlabel('Tempo [ore]')
    axes[2].set_xlim([0, 16])
    plt.tight_layout()
    plt.savefig('grafico_multipaziente.png', dpi=300)
    plt.close()
    
    print("\nGrafici esportati con successo in:")
    print("  - grafico_sensibilita_pasti.png")
    print("  - grafico_multipaziente.png")


if __name__ == '__main__':
    main()
