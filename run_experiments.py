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
        
        # Integrazione numerica corretta riscalata per dt = 3 min
        self.integral_error += error * self.dt
        self.integral_error = float(np.clip(self.integral_error, -3000.0, 10000.0))
        
        # Derivata corretta (variazione al minuto)
        if self.prev_cgm is None:
            d_cgm_dt = 0.0
        else:
            d_cgm_dt = (cgm - self.prev_cgm) / self.dt
        self.prev_cgm = cgm
        
        correction = (self.kp * error) + (self.ki * self.integral_error) + (self.kd * d_cgm_dt)
        u = self.basal_rate + correction
        u = max(0.0, min(u, 0.25))  # Saturazione fisica attuatore [0, 0.25] U/min
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
        
        self.current_iob = (self.s1 + self.s2) / 60.0  # IOB in Unità [U]
        
        if self.prev_cgm is None:
            d_cgm_dt = 0.0
        else:
            d_cgm_dt = (cgm - self.prev_cgm) / self.dt
            
        cgm_projected = cgm + (d_cgm_dt * 20.0)  # Proiezione PLGS a 20 minuti esatti
        
        # Anti-windup selettivo
        if cgm > self.target:
            self.integral_error += error * self.dt
        else:
            self.integral_error = max(0.0, self.integral_error - (abs(error) * self.dt * 0.5))
        self.integral_error = float(np.clip(self.integral_error, -1000.0, 6000.0))
        
        correction = (self.kp * error) + (self.ki * self.integral_error) + (self.kd * d_cgm_dt)
        u = self.basal_rate + correction
        
        # Vincolo IOB per prevenire insulin stacking
        if self.current_iob > self.iob_cap and u > self.basal_rate:
            u = self.basal_rate
            
        # Sospensione preventiva PLGS
        if cgm_projected < self.plgs_threshold or cgm < 75.0:
            u = 0.0
            
        u = max(0.0, min(u, 0.25))
        
        # Aggiornamento compartimenti IOB per passo dt
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


def run_single_simulation(controller_class, patient_name='adult#001', meal_cho=60.0):
    start_time = datetime(2026, 1, 1, 8, 0, 0)
    sim_hours = 16
    meal_scenario = [(12.0, meal_cho)]  # Pasto alle 12:00
    
    patient = T1DPatient.withName(patient_name)
    sensor = CGMSensor.withName('Dexcom', seed=1)
    pump = InsulinPump.withName('Insulet')
    scenario = CustomScenario(start_time=start_time, scenario=meal_scenario)
    env = T1DSimEnv(patient, sensor, pump, scenario)
    
    dt = env.sample_time  # 3 minuti
    total_steps = int(sim_hours * 60 / dt)  # 320 passi = 16 ore esatte
    
    bw = patient._params['BW']
    u2ss = patient._params['u2ss']
    basal_nominal = (u2ss * bw) / 6000.0
    
    controller = controller_class(basal_rate=basal_nominal, dt=dt)
    
    time_arr, cgm_arr, bg_arr, ins_arr, iob_arr = [], [], [], [], []
    obs, reward, done, info = env.reset()
    controller.reset()
    
    for step in range(total_steps):
        current_t = start_time + timedelta(minutes=step * dt)
        action = controller.policy(obs, reward, done, **info)
        
        time_arr.append(current_t)
        cgm_arr.append(obs.CGM)
        bg_arr.append(env.patient.state[0])
        ins_arr.append(action.basal * 60.0)
        iob_arr.append(getattr(controller, 'current_iob', 0.0))
        
        obs, reward, done, info = env.step(action)
        if done:
            break
            
    return pd.DataFrame({
        'Time': time_arr,
        'CGM': cgm_arr,
        'BG': bg_arr,
        'Insulin_Uh': ins_arr,
        'IOB': iob_arr
    })


def compute_metrics(df, name):
    cgm = df['CGM'].values
    n = len(cgm)
    
    tir = np.sum((cgm >= 70.0) & (cgm <= 180.0)) / n * 100.0
    tar_tot = np.sum(cgm > 180.0) / n * 100.0
    tar_l1 = np.sum((cgm > 180.0) & (cgm <= 250.0)) / n * 100.0
    tar_l2 = np.sum(cgm > 250.0) / n * 100.0
    
    tbr_tot = np.sum(cgm < 70.0) / n * 100.0
    tbr_l1 = np.sum((cgm >= 54.0) & (cgm < 70.0)) / n * 100.0
    tbr_l2 = np.sum(cgm < 54.0) / n * 100.0
    
    mask_late = (df['Time'] >= datetime(2026, 1, 1, 15, 0, 0)) & \
                (df['Time'] <= datetime(2026, 1, 1, 18, 0, 0))
    cgm_late = df.loc[mask_late, 'CGM'].values
    tbr_late = np.sum(cgm_late < 70.0) / len(cgm_late) * 100.0 if len(cgm_late) > 0 else 0.0
    
    return {
        'Controllore': name,
        'TIR [%]': round(tir, 2),
        'TAR Tot [%]': round(tar_tot, 2),
        'TAR L1 [%]': round(tar_l1, 2),
        'TAR L2 [%]': round(tar_l2, 2),
        'TBR Tot [%]': round(tbr_tot, 2),
        'TBR L1 [%]': round(tbr_l1, 2),
        'TBR L2 [%]': round(tbr_l2, 2),
        'TBR Tardivo (3-6h) [%]': round(tbr_late, 2),
        'Picco [mg/dL]': round(np.max(cgm), 1),
        'Nadir [mg/dL]': round(np.min(cgm), 1)
    }


if __name__ == '__main__':
    print("Avvio simulazione nominale a 16 ore su adult#001 (Pasto 60g @ 12:00)...")
    df_pid = run_single_simulation(StandardPIDController, 'adult#001', 60.0)
    df_iob = run_single_simulation(IOBConstrainedController, 'adult#001', 60.0)
    
    m_pid = compute_metrics(df_pid, 'PID Baseline')
    m_iob = compute_metrics(df_iob, 'PID + Vincolo IOB + PLGS')
    
    df_metrics = pd.DataFrame([m_pid, m_iob])
    df_metrics.to_csv('metriche_confronto.csv', index=False)
    print("\nMetriche nominali salvate in metriche_confronto.csv:")
    print(df_metrics[['Controllore', 'TIR [%]', 'TAR Tot [%]', 'TBR Tot [%]', 'TBR Tardivo (3-6h) [%]', 'Picco [mg/dL]', 'Nadir [mg/dL]']])
    
    # Generazione grafico a 3 pannelli
    t_h = [(t - df_pid['Time'].iloc[0]).total_seconds() / 3600.0 for t in df_pid['Time']]
    fig, axes = plt.subplots(3, 1, figsize=(11, 9), sharex=True)
    
    axes[0].axhspan(70, 180, color='green', alpha=0.15, label='Target (70-180 mg/dL)')
    axes[0].axvline(4.0, color='black', linestyle=':', label='Pasto 60g @ 12:00')
    axes[0].plot(t_h, df_pid['CGM'], 'r--', label='PID Baseline', linewidth=2)
    axes[0].plot(t_h, df_iob['CGM'], 'b-', label='PID + Vincolo IOB + PLGS', linewidth=2)
    axes[0].set_ylabel('CGM [mg/dL]')
    axes[0].set_title('Risposta a Disturbo Prandiale Non Annunciato (16h, dt=3 min)')
    axes[0].legend(loc='upper right')
    axes[0].grid(True, linestyle=':', alpha=0.6)
    
    axes[1].plot(t_h, df_pid['Insulin_Uh'], 'r--', linewidth=1.8)
    axes[1].plot(t_h, df_iob['Insulin_Uh'], 'b-', linewidth=1.8)
    axes[1].set_ylabel('Erogazione [U/h]')
    axes[1].grid(True, linestyle=':', alpha=0.6)
    
    axes[2].axhline(2.2, color='darkblue', linestyle='-.', label='Cap IOB (2.2 U)')
    axes[2].plot(t_h, df_iob['IOB'], 'b-', linewidth=2, label='IOB Stimato')
    axes[2].set_ylabel('IOB [U]')
    axes[2].set_xlabel('Tempo [ore]')
    axes[2].set_xlim([0, 16])
    axes[2].legend(loc='upper right')
    axes[2].grid(True, linestyle=':', alpha=0.6)
    
    plt.tight_layout()
    plt.savefig('grafico_benchmark_baseline.png', dpi=300)
    print("Grafico sincronizzato a 3 pannelli esportato in grafico_benchmark_baseline.png")
