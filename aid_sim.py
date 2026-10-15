"""
aid_sim.py - Controllori AID e simulazione in silico con simglucose.

Scelte metodologiche:
  * Campionamento: dt = 3 min (sensore Dexcom di simglucose).
  * Assestamento: la simulazione parte alle 00:00 con il controllore già attivo. Le prime
    WARMUP_H ore servono solo a portare l'anello all'equilibrio sul target e NON entrano nelle metriche.
  * Finestra di analisi: dalle 08:00 alle 24:00 (16 h, 320 passi), come nella versione precedente.
  * Disturbo prandiale: pasto non annunciato alle MEAL_CLOCK_H (default 12:00), nessun bolo.
  * Glicemia: metriche sulla glicemia PLASMATICA del modello UVa/Padova (Gp/Vg). In simglucose
    info['bg'] è invece la glicemia sottocutanea senza rumore (Gsub): viene salvata a parte.
    Il controllore vede solo la misura CGM.
  * Allineamento: ogni riga del DataFrame contiene grandezze riferite allo stesso istante t_k.
  * IOB stimato dal controllore: compartimenti S1 + S2 alimentati dall'insulina in eccesso rispetto
    al basale (IOB "netto", in U). Per confronto si salva anche l'IOB netto vero del modello
    (compartimenti sottocutanei Isc1 + Isc2 meno il loro valore di equilibrio basale).
  * Ipoglicemia tardiva: TBR dalle 3 ore dopo il pasto fino alle 24:00.
"""

from collections import deque
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

from simglucose.simulation.env import T1DSimEnv
from simglucose.actuator.pump import InsulinPump
from simglucose.sensor.cgm import CGMSensor
from simglucose.patient.t1dpatient import T1DPatient
from simglucose.simulation.scenario import CustomScenario
from simglucose.controller.base import Controller, Action

SIM_START = datetime(2026, 1, 1, 0, 0, 0)   # inizio simulazione (00:00)
WARMUP_H = 8.0                              # assestamento escluso dalle metriche (00:00-08:00)
ANALYSIS_END_H = 24.0                       # fine finestra di analisi (24:00)
MEAL_CLOCK_H = 12.0                         # ora del pasto non annunciato (12:00)
U_MAX = 0.25                                # portata massima imposta dal controllore [U/min] (15 U/h)

# Compatibilità con i notebook: START è l'inizio della finestra di analisi (08:00)
START = SIM_START + timedelta(hours=WARMUP_H)


class AIDController(Controller):
    """PID a tempo discreto (uscita in U/min) con anti-windup, vincolo IOB e sospensione predittiva (PLGS).

    Anti-windup (sempre attivo):
      * limiti dell'integrale legati al basale del paziente:
            Ki * I_max =  aw_frac     * u_b   (l'azione integrale aggiunge al massimo aw_frac volte il basale)
            Ki * I_min = -aw_frac_low * u_b   (l'azione integrale toglie al massimo aw_frac_low volte il basale)
        Se i_min / i_max vengono passati esplicitamente, si usano quei valori (per esempio il clamp largo
        [-2000, 10000] della versione precedente, che non interviene mai e mostra il windup).
      * integrazione condizionata (se freeze=True): l'integrale non viene aggiornato quando l'uscita è
        limitata (saturazione dell'attuatore, tetto IOB o PLGS) e l'errore spingerebbe ancora nella
        direzione del limite.
    """

    def __init__(
        self,
        basal_rate,
        dt,
        target=120.0,
        kp=1e-4,
        ki=5e-6,
        kd=2e-3,
        aw_frac=1.0,
        aw_frac_low=0.25,
        i_min=None,
        i_max=None,
        freeze=True,
        n_slope=5,
        use_iob=False,
        use_plgs=False,
        iob_alpha=1.5,
        tau_s=50.0,
        t_pred=20.0,
        plgs_pred=90.0,
        plgs_now=75.0,
    ):
        self.ub = basal_rate
        self.dt = dt
        self.target = target
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.aw_frac = aw_frac
        self.aw_frac_low = aw_frac_low
        self.i_max = aw_frac * basal_rate / ki if i_max is None else float(i_max)
        self.i_min = -aw_frac_low * basal_rate / ki if i_min is None else float(i_min)
        self.freeze = freeze
        self.n_slope = n_slope
        self.use_iob = use_iob
        self.use_plgs = use_plgs
        self.iob_max = iob_alpha * basal_rate * 60.0  # tetto IOB [U] = iob_alpha * basale orario
        self.tau_s = tau_s
        self.t_pred = t_pred
        self.plgs_pred = plgs_pred
        self.plgs_now = plgs_now
        self.reset()

    def reset(self):
        self.I = 0.0
        self.hist = deque(maxlen=self.n_slope)
        self.s1 = 0.0
        self.s2 = 0.0
        self.iob = 0.0
        self.slope = 0.0
        self.limit = ""  # vincolo attivo nell'ultimo passo: "", "sat_alta", "sat_bassa", "iob", "plgs"

    def _slope(self, y):
        """Pendenza CGM [mg/dL/min] stimata ai minimi quadrati sugli ultimi n_slope campioni."""
        self.hist.append(y)
        if len(self.hist) < 2:
            return 0.0
        t = np.arange(len(self.hist)) * self.dt
        return float(np.polyfit(t, np.asarray(self.hist), 1)[0])

    def policy(self, observation, reward, done, **info):
        y = observation.CGM
        e = y - self.target
        slope = self._slope(y)
        self.slope = slope

        # Legge PID con l'integrale candidato (integrazione rettangolare in avanti)
        I_new = float(np.clip(self.I + e * self.dt, self.i_min, self.i_max))
        u_pid = self.ub + self.kp * e + self.ki * I_new + self.kd * slope

        # IOB stimato all'istante corrente
        self.iob = self.s1 + self.s2

        u = u_pid
        limit = ""
        # 1. Vincolo IOB (anti insulin stacking): oltre il tetto non si infonde più del basale
        if self.use_iob and self.iob > self.iob_max and u > self.ub:
            u, limit = self.ub, "iob"
        # 2. Sospensione predittiva (PLGS)
        if self.use_plgs and (y + slope * self.t_pred < self.plgs_pred or y < self.plgs_now):
            u, limit = 0.0, "plgs"
        # 3. Saturazione dell'attuatore
        if u > U_MAX:
            u, limit = U_MAX, limit or "sat_alta"
        elif u < 0.0:
            u, limit = 0.0, limit or "sat_bassa"
        self.limit = limit

        # Anti-windup per integrazione condizionata
        if not self.freeze:
            self.I = I_new
        else:
            spinge_su = e > 0 and u < u_pid    # uscita limitata verso l'alto e errore positivo
            spinge_giu = e < 0 and u > u_pid   # uscita limitata verso il basso e errore negativo
            if not (spinge_su or spinge_giu):
                self.I = I_new

        # Dinamica a due compartimenti dell'insulina in eccesso rispetto al basale (Eulero in avanti)
        ex = max(0.0, u - self.ub)
        d1 = (ex - self.s1 / self.tau_s) * self.dt
        d2 = (self.s1 / self.tau_s - self.s2 / self.tau_s) * self.dt
        self.s1 += d1
        self.s2 += d2

        return Action(basal=u, bolus=0.0)


# Configurazioni dei controllori usate in run_campaign.py e nei notebook (argomenti di AIDController)
CFG = {
    "PID senza AW": dict(i_min=-2000.0, i_max=10000.0, freeze=False),  # prima versione: clamp largo, windup
    "PID":          dict(),                                            # PID di riferimento con anti-windup
    "PID cons.":    dict(aw_frac=0.5),                                 # sicurezza ottenuta riducendo l'azione integrale
    "PID+PLGS":     dict(use_plgs=True),                               # PID + sospensione predittiva
    "PID+IOB":      dict(use_iob=True),                                # PID + tetto sull'insulina attiva
    "PID+IOB+PLGS": dict(use_iob=True, use_plgs=True),                 # controllore proposto
}


def basal_rate(patient):
    """Fabbisogno basale di equilibrio del paziente [U/min]."""
    return patient._params["u2ss"] * patient._params["BW"] / 6000.0


def make_patient(patient_name, assorb_scale=1.0):
    """Paziente virtuale. assorb_scale scala insieme le costanti dello svuotamento gastrico (kmax, kmin) e
    dell'assorbimento intestinale (kabs): < 1 pasto lento, > 1 pasto rapido. Scalare solo kabs non basta,
    perché nel modello UVa/Padova la dinamica del pasto è dominata dallo svuotamento gastrico."""
    p = T1DPatient.withName(patient_name)
    if assorb_scale != 1.0:
        params = p._params.copy()
        for k in ("kmax", "kmin", "kabs"):
            params[k] = params[k] * assorb_scale
        p = T1DPatient(params)
    return p


def _plasma_bg(patient):
    """Glicemia plasmatica [mg/dL] = Gp / Vg (stato x4 del modello UVa/Padova)."""
    return float(patient.state[3] / patient._params["Vg"])


def _true_net_iob(patient):
    """IOB netto vero del modello [U]: (Isc1 + Isc2 - valori di equilibrio basale) convertito da pmol/kg."""
    pr = patient._params
    isc = patient.state[10] + patient.state[11] - pr["isc1ss"] - pr["isc2ss"]   # x11, x12 del modello
    return float(isc * pr["BW"] / 6000.0)


def simulate(patient_name, cho, ctrl_kwargs, seed=1, meal_clock_h=MEAL_CLOCK_H,
             assorb_scale=1.0, end_h=ANALYSIS_END_H):
    """Simula da 00:00 a end_h con un pasto non annunciato di `cho` grammi alle meal_clock_h.

    Restituisce (DataFrame, basale [U/min]). Colonna t_h = ora del giorno [h]; righe da 00:00 (assestamento incluso).
    """
    p = make_patient(patient_name, assorb_scale)
    s = CGMSensor.withName("Dexcom", seed=seed)
    pu = InsulinPump.withName("Insulet")
    meals = [(meal_clock_h, cho)] if cho > 0 else []   # in CustomScenario un numero è un offset in ore
    env = T1DSimEnv(p, s, pu, CustomScenario(start_time=SIM_START, scenario=meals))
    dt = env.sample_time

    ub = basal_rate(p)
    c = AIDController(ub, dt, **ctrl_kwargs)

    obs, r, d, info = env.reset()
    c.reset()
    rows = []
    for k in range(int(end_h * 60 / dt)):
        bg = _plasma_bg(env.patient)
        gsub = env.patient.observation.Gsub
        iob_true = _true_net_iob(env.patient)
        a = c.policy(obs, r, d, **info)
        rows.append((k * dt / 60.0, obs.CGM, bg, gsub, a.basal * 60.0, c.iob, iob_true, c.limit))
        obs, r, d, info = env.step(a)

    df = pd.DataFrame(rows, columns=["t_h", "CGM", "BG", "Gsub", "Insulin_Uh", "IOB", "IOB_vero", "Vincolo"])
    return df, ub


def metrics(df, signal="BG", meal_h=MEAL_CLOCK_H, t0=WARMUP_H):
    """Metriche cliniche sulla finestra di analisi (t_h >= t0), più il TBR tardivo (da meal_h + 3 h)."""
    w = df[df.t_h >= t0]
    g = w[signal].values
    late = w[w.t_h >= meal_h + 3][signal].values
    post = w[(w.t_h >= meal_h) & (w.t_h < meal_h + 6)][signal].values
    return {
        "TIR": 100 * np.mean((g >= 70) & (g <= 180)),
        "TAR": 100 * np.mean(g > 180),
        "TAR>250": 100 * np.mean(g > 250),
        "TBR": 100 * np.mean(g < 70),
        "TBR<54": 100 * np.mean(g < 54),
        "TBR_tardivo": 100 * np.mean(late < 70) if len(late) else 0.0,
        "TIR_post6h": 100 * np.mean((post >= 70) & (post <= 180)) if len(post) else 0.0,
        "picco": float(g.max()),
        "nadir": float(g.min()),
        "insulina_tot": float(w.Insulin_Uh.sum() * (df.t_h.iloc[1] - df.t_h.iloc[0])),  # U nella finestra
    }
