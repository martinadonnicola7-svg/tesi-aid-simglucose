"""
genera_notebooks.py
Generatore ed esecutore automatico dei notebook Jupyter per la tesi AID simglucose.
Crea '01_baseline_pid_benchmark.ipynb' con le specifiche corrette:
  - Durata: 16 ore (08:00 - 00:00)
  - Passo: dt = 3 minuti (Dexcom sensor, 320 passi)
  - Disturbo: pasto non annunciato da 60g alle ore 12:00
  - Plot a 3 pannelli e metriche cliniche (TIR, TAR, TBR, TBR tardivo 3-6h)
"""

import json
import os
import subprocess
import sys


def build_notebook_structure():
    cells = [
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "# Benchmark di Controllo AID: Reiezione di Pasti Non Annunciati\n",
                "\n",
                "Questo notebook implementa e confronta due architetture di controllo a circuito chiuso su ambiente di simulazione *in silico* basato su `simglucose` (modello metabolico UVA/Padova):\n",
                "1. **PID Baseline Convenzionale**: regolatore proporzionale-integrale-derivativo accordato sul target euglicemico ($110\\text{ mg/dL}$).\n",
                "2. **PID con Vincolo su IOB e Modulo PLGS**: regolatore con saturazione dinamica basata sulla stima in tempo reale dell'insulina attiva ($IOB(t)$) e sospensione preventiva dell'infusione (*Predictive Low-Glucose Suspend*).\n",
                "\n",
                "### Parametri dello Scenario Sperimentale\n",
                "- **Paziente virtuale**: `adult#001`\n",
                "- **Durata temporale**: $16\\text{ ore}$ nominali (dalle ore 08:00 alle 00:00)\n",
                "- **Frequenza di campionamento**: $\\Delta t = 3\\text{ minuti}$ (`env.sample_time = 3.0`), pari a $320\\text{ passi}$\n",
                "- **Disturbo esogeno**: pasto non annunciato da $60\\text{ g}$ di carboidrati alle ore 12:00 ($t = 240\\text{ min}$)"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "import numpy as np\n",
                "import pandas as pd\n",
                "import matplotlib.pyplot as plt\n",
                "from datetime import datetime, timedelta\n",
                "\n",
                "from simglucose.simulation.env import T1DSimEnv\n",
                "from simglucose.actuator.pump import InsulinPump\n",
                "from simglucose.sensor.cgm import CGMSensor\n",
                "from simglucose.patient.t1dpatient import T1DPatient\n",
                "from simglucose.simulation.scenario import CustomScenario\n",
                "from simglucose.controller.base import Controller, Action\n",
                "\n",
                "print(\"Librerie importate con successo.\")"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 1. Definizione degli Algoritmi di Controllo\n",
                "\n",
                "### Formulazione Matematica a Tempo Discreto ($dt = 3\\text{ min}$)\n",
                "A ogni passo di campionamento $\\Delta t = 3\\text{ min}$, l'errore di regolazione è definito come $e(k) = G(k) - G_{target}$.\n",
                "- L'azione integrale è integrata numericamente: $I(k) = I(k-1) + e(k) \\cdot \\Delta t$.\n",
                "- L'azione derivativa quantifica la pendenza al minuto: $D(k) = \\frac{G(k) - G(k-1)}{\\Delta t}$.\n",
                "- L'osservatore IOB integra l'insulina eccedente nei due compartimenti sottocutanei a costante di tempo $\\tau_s = 50\\text{ min}$."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "class StandardPIDController(Controller):\n",
                "    def __init__(self, target=110.0, kp=0.0006, ki=0.000015, kd=0.004, basal_rate=0.021, dt=3.0):\n",
                "        self.target = target\n",
                "        self.kp = kp\n",
                "        self.ki = ki\n",
                "        self.kd = kd\n",
                "        self.basal_rate = basal_rate\n",
                "        self.dt = dt\n",
                "        self.integral_error = 0.0\n",
                "        self.prev_cgm = None\n",
                "\n",
                "    def policy(self, observation, reward, done, **info):\n",
                "        cgm = observation.CGM\n",
                "        error = cgm - self.target\n",
                "        \n",
                "        self.integral_error += error * self.dt\n",
                "        self.integral_error = float(np.clip(self.integral_error, -3000.0, 10000.0))\n",
                "        \n",
                "        d_cgm_dt = 0.0 if self.prev_cgm is None else (cgm - self.prev_cgm) / self.dt\n",
                "        self.prev_cgm = cgm\n",
                "        \n",
                "        correction = (self.kp * error) + (self.ki * self.integral_error) + (self.kd * d_cgm_dt)\n",
                "        u = max(0.0, min(self.basal_rate + correction, 0.25))\n",
                "        return Action(basal=u, bolus=0.0)\n",
                "\n",
                "    def reset(self):\n",
                "        self.integral_error = 0.0\n",
                "        self.prev_cgm = None\n",
                "\n",
                "\n",
                "class IOBConstrainedController(Controller):\n",
                "    def __init__(self, target=110.0, kp=0.0006, ki=0.000015, kd=0.004, basal_rate=0.021, \n",
                "                 tau_s=50.0, iob_cap=2.2, plgs_threshold=90.0, dt=3.0):\n",
                "        self.target = target\n",
                "        self.kp = kp\n",
                "        self.ki = ki\n",
                "        self.kd = kd\n",
                "        self.basal_rate = basal_rate\n",
                "        self.tau_s = tau_s\n",
                "        self.iob_cap = iob_cap\n",
                "        self.plgs_threshold = plgs_threshold\n",
                "        self.dt = dt\n",
                "        \n",
                "        self.integral_error = 0.0\n",
                "        self.prev_cgm = None\n",
                "        self.s1 = 0.0\n",
                "        self.s2 = 0.0\n",
                "        self.current_iob = 0.0\n",
                "\n",
                "    def policy(self, observation, reward, done, **info):\n",
                "        cgm = observation.CGM\n",
                "        error = cgm - self.target\n",
                "        \n",
                "        self.current_iob = (self.s1 + self.s2) / 60.0\n",
                "        d_cgm_dt = 0.0 if self.prev_cgm is None else (cgm - self.prev_cgm) / self.dt\n",
                "        cgm_projected = cgm + (d_cgm_dt * 20.0)\n",
                "        \n",
                "        if cgm > self.target:\n",
                "            self.integral_error += error * self.dt\n",
                "        else:\n",
                "            self.integral_error = max(0.0, self.integral_error - (abs(error) * self.dt * 0.5))\n",
                "        self.integral_error = float(np.clip(self.integral_error, -1000.0, 6000.0))\n",
                "        \n",
                "        correction = (self.kp * error) + (self.ki * self.integral_error) + (self.kd * d_cgm_dt)\n",
                "        u = self.basal_rate + correction\n",
                "        \n",
                "        # Vincolo di saturazione IOB contro l'insulin stacking\n",
                "        if self.current_iob > self.iob_cap and u > self.basal_rate:\n",
                "            u = self.basal_rate\n",
                "            \n",
                "        # Modulo di sospensione predittiva PLGS\n",
                "        if cgm_projected < self.plgs_threshold or cgm < 75.0:\n",
                "            u = 0.0\n",
                "            \n",
                "        u = max(0.0, min(u, 0.25))\n",
                "        \n",
                "        u_excess = max(0.0, u - self.basal_rate)\n",
                "        ds1 = (u_excess - (self.s1 / self.tau_s)) * self.dt\n",
                "        ds2 = ((self.s1 / self.tau_s) - (self.s2 / self.tau_s)) * self.dt\n",
                "        self.s1 += ds1\n",
                "        self.s2 += ds2\n",
                "        self.prev_cgm = cgm\n",
                "        \n",
                "        return Action(basal=u, bolus=0.0)\n",
                "\n",
                "    def reset(self):\n",
                "        self.integral_error = 0.0\n",
                "        self.prev_cgm = None\n",
                "        self.s1 = 0.0\n",
                "        self.s2 = 0.0\n",
                "        self.current_iob = 0.0"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 2. Esecuzione del Protocollo di Simulazione (16 Ore)"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "def run_simulation(controller_class, patient_name='adult#001', meal_cho=60.0):\n",
                "    start_time = datetime(2026, 1, 1, 8, 0, 0)\n",
                "    sim_hours = 16\n",
                "    meal_scenario = [(12.0, meal_cho)]\n",
                "    \n",
                "    patient = T1DPatient.withName(patient_name)\n",
                "    sensor = CGMSensor.withName('Dexcom', seed=1)\n",
                "    pump = InsulinPump.withName('Insulet')\n",
                "    scenario = CustomScenario(start_time=start_time, scenario=meal_scenario)\n",
                "    env = T1DSimEnv(patient, sensor, pump, scenario)\n",
                "    \n",
                "    dt = env.sample_time\n",
                "    total_steps = int(sim_hours * 60 / dt)\n",
                "    \n",
                "    bw = patient._params['BW']\n",
                "    u2ss = patient._params['u2ss']\n",
                "    basal_nominal = (u2ss * bw) / 6000.0\n",
                "    \n",
                "    controller = controller_class(basal_rate=basal_nominal, dt=dt)\n",
                "    \n",
                "    time_list, cgm_list, bg_list, insulin_list, iob_list = [], [], [], [], []\n",
                "    obs, reward, done, info = env.reset()\n",
                "    controller.reset()\n",
                "    \n",
                "    for step in range(total_steps):\n",
                "        current_t = start_time + timedelta(minutes=step * dt)\n",
                "        action = controller.policy(obs, reward, done, **info)\n",
                "        \n",
                "        time_list.append(current_t)\n",
                "        cgm_list.append(obs.CGM)\n",
                "        bg_list.append(env.patient.state[0])\n",
                "        insulin_list.append(action.basal * 60.0)\n",
                "        iob_list.append(getattr(controller, 'current_iob', 0.0))\n",
                "        \n",
                "        obs, reward, done, info = env.step(action)\n",
                "        if done:\n",
                "            break\n",
                "            \n",
                "    return pd.DataFrame({\n",
                "        'Time': time_list,\n",
                "        'CGM': cgm_list,\n",
                "        'BG': bg_list,\n",
                "        'Insulin_Uh': insulin_list,\n",
                "        'IOB': iob_list\n",
                "    })\n",
                "\n",
                "print(\"Esecuzione benchmark nominale su adult#001 (60g CHO)...\")\n",
                "df_pid = run_simulation(StandardPIDController, 'adult#001', 60.0)\n",
                "df_iob = run_simulation(IOBConstrainedController, 'adult#001', 60.0)\n",
                "print(\"Simulazioni completate.\")"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 3. Calcolo delle Metriche Cliniche di Consenso"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "def compute_metrics(df, label):\n",
                "    cgm = df['CGM'].values\n",
                "    n = len(cgm)\n",
                "    \n",
                "    tir = np.sum((cgm >= 70.0) & (cgm <= 180.0)) / n * 100.0\n",
                "    tar = np.sum(cgm > 180.0) / n * 100.0\n",
                "    tbr = np.sum(cgm < 70.0) / n * 100.0\n",
                "    \n",
                "    mask_late = (df['Time'] >= datetime(2026, 1, 1, 15, 0, 0)) & (df['Time'] <= datetime(2026, 1, 1, 18, 0, 0))\n",
                "    cgm_late = df.loc[mask_late, 'CGM'].values\n",
                "    tbr_late = np.sum(cgm_late < 70.0) / len(cgm_late) * 100.0 if len(cgm_late) > 0 else 0.0\n",
                "    \n",
                "    return {\n",
                "        'Controllore': label,\n",
                "        'TIR [%]': round(tir, 2),\n",
                "        'TAR [%]': round(tar, 2),\n",
                "        'TBR [%]': round(tbr, 2),\n",
                "        'TBR Tardivo (3-6h) [%]': round(tbr_late, 2),\n",
                "        'Picco [mg/dL]': round(float(np.max(cgm)), 1),\n",
                "        'Nadir [mg/dL]': round(float(np.min(cgm)), 1)\n",
                "    }\n",
                "\n",
                "df_results = pd.DataFrame([\n",
                "    compute_metrics(df_pid, 'PID Baseline'),\n",
                "    compute_metrics(df_iob, 'PID + Vincolo IOB + PLGS')\n",
                "])\n",
                "display(df_results)"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 4. Visualizzazione a 3 Pannelli Sincronizzata"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "fig, axes = plt.subplots(3, 1, figsize=(11, 9), sharex=True)\n",
                "t_hours = [(t - df_pid['Time'].iloc[0]).total_seconds() / 3600.0 for t in df_pid['Time']]\n",
                "\n",
                "# 1. Glicemia\n",
                "axes[0].axhspan(70, 180, color='green', alpha=0.15, label='Target (70-180 mg/dL)')\n",
                "axes[0].axvline(4.0, color='black', linestyle=':', label='Pasto non annunciato (60g @ 12:00)')\n",
                "axes[0].plot(t_hours, df_pid['CGM'], 'r--', linewidth=2, label='PID Baseline')\n",
                "axes[0].plot(t_hours, df_iob['CGM'], 'b-', linewidth=2, label='PID + Vincolo IOB + PLGS')\n",
                "axes[0].set_ylabel('CGM [mg/dL]')\n",
                "axes[0].set_title('Confronto Prestazionale: Risposta al Disturbo Prandiale (16 Ore)')\n",
                "axes[0].legend(loc='upper right')\n",
                "axes[0].grid(True, linestyle=':', alpha=0.6)\n",
                "\n",
                "# 2. Infusione\n",
                "axes[1].plot(t_hours, df_pid['Insulin_Uh'], 'r--', linewidth=1.8, label='PID Baseline')\n",
                "axes[1].plot(t_hours, df_iob['Insulin_Uh'], 'b-', linewidth=1.8, label='PID + Vincolo IOB + PLGS')\n",
                "axes[1].set_ylabel('Erogazione [U/h]')\n",
                "axes[1].legend(loc='upper right')\n",
                "axes[1].grid(True, linestyle=':', alpha=0.6)\n",
                "\n",
                "# 3. IOB\n",
                "axes[2].axhline(2.2, color='darkblue', linestyle='-.', label='Cap IOB (2.2 U)')\n",
                "axes[2].plot(t_hours, df_iob['IOB'], 'b-', linewidth=2, label='IOB Stimato')\n",
                "axes[2].set_ylabel('IOB [U]')\n",
                "axes[2].set_xlabel('Tempo [ore]')\n",
                "axes[2].set_xlim([0, 16])\n",
                "axes[2].legend(loc='upper right')\n",
                "axes[2].grid(True, linestyle=':', alpha=0.6)\n",
                "\n",
                "plt.tight_layout()\n",
                "plt.savefig('grafico_benchmark_baseline.png', dpi=300)\n",
                "plt.show()\n",
                "print(\"Grafico esportato in grafico_benchmark_baseline.png\")"
            ]
        }
    ]

    nb_dict = {
        "cells": cells,
        "metadata": {
            "language_info": {
                "name": "python",
                "version": "3.10.12"
            }
        },
        "nbformat": 4,
        "nbformat_minor": 5
    }
    return nb_dict


def main():
    target_nb = "01_baseline_pid_benchmark.ipynb"
    print(f"Generazione della struttura JSON per {target_nb}...")
    nb_content = build_notebook_structure()

    with open(target_nb, "w", encoding="utf-8") as f:
        json.dump(nb_content, f, indent=1, ensure_ascii=False)
    print(f"File {target_nb} scritto con successo.")

    # Esecuzione in-place per calcolare e salvare tutti gli output grafici e tabellari
    print("\nEsecuzione del notebook con nbconvert per pre-calcolare gli output...")
    try:
        cmd = [
            sys.executable, "-m", "jupyter", "nbconvert",
            "--to", "notebook",
            "--execute",
            "--inplace",
            target_nb
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode == 0:
            print(f"Esecuzione completata: {target_nb} ha tutti gli output e i grafici renderizzati.")
        else:
            print("Avviso: esecuzione fallita o jupyter non presente nell'ambiente.")
            print(result.stderr)
    except Exception as e:
        print(f"Impossibile eseguire nbconvert: {e}")
        print("Il file notebook è comunque stato generato ed è pronto all'uso.")


if __name__ == "__main__":
    main()
