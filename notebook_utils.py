"""
notebook_utils.py - Tabelle di documentazione per i notebook (versioni, controllori, scenario).
Non contiene logica di simulazione: stampa in modo uniforme parametri, unità di misura, condizioni iniziali,
durata, passo, pasto, seme casuale e versioni delle librerie.
"""
import importlib.metadata as md
import inspect
import platform

import pandas as pd

import aid_sim as A
from simglucose.actuator.pump import InsulinPump
from simglucose.sensor.cgm import CGMSensor
from simglucose.simulation.env import T1DSimEnv
from simglucose.simulation.scenario import CustomScenario
from simglucose.patient.t1dpatient import T1DPatient


def library_versions():
    """Versioni dell'interprete e dei pacchetti usati."""
    rows = [{"Pacchetto": "Python", "Versione": platform.python_version()}]
    for n in ["simglucose", "numpy", "scipy", "pandas", "matplotlib"]:
        try:
            rows.append({"Pacchetto": n, "Versione": md.version(n)})
        except md.PackageNotFoundError:
            rows.append({"Pacchetto": n, "Versione": "non installato"})
    return pd.DataFrame(rows)


# Unità di misura e significato dei parametri di AIDController
_DOC = {
    "target":      ("mg/dL", "Glicemia obiettivo del regolatore"),
    "kp":          ("U/min per mg/dL", "Guadagno proporzionale"),
    "ki":          ("U/min per (mg/dL*min)", "Guadagno integrale"),
    "kd":          ("U/min per (mg/dL/min)", "Guadagno derivativo sulla pendenza CGM"),
    "n_slope":     ("campioni", "Campioni CGM usati per stimare la pendenza ai minimi quadrati"),
    "aw_frac":     ("adimensionale", "Anti-windup: Ki*I_max = aw_frac * basale"),
    "aw_frac_low": ("adimensionale", "Anti-windup: Ki*I_min = -aw_frac_low * basale"),
    "freeze":      ("-", "Integrazione condizionata quando l'uscita è limitata"),
    "use_iob":     ("-", "Attiva il tetto sull'insulina attiva (IOB)"),
    "iob_alpha":   ("adimensionale", "Tetto IOB = iob_alpha * basale orario"),
    "tau_s":       ("min", "Costante di tempo dei compartimenti dello stimatore IOB"),
    "use_plgs":    ("-", "Attiva la sospensione predittiva dell'infusione (PLGS)"),
    "t_pred":      ("min", "Orizzonte di predizione lineare della glicemia"),
    "plgs_pred":   ("mg/dL", "Soglia sulla glicemia proiettata per sospendere l'infusione"),
    "plgs_now":    ("mg/dL", "Soglia sulla misura corrente per sospendere l'infusione"),
}


def controller_table(ctrl_kwargs=None, basal_u_per_min=None):
    """Tabella dei parametri effettivi e dei vincoli del controllore, con unità di misura."""
    sig = inspect.signature(A.AIDController.__init__)
    vals = {k: v.default for k, v in sig.parameters.items() if v.default is not inspect.Parameter.empty}
    vals.update(ctrl_kwargs or {})
    rows = [{"Parametro": k, "Valore": vals.get(k, "-"), "Unita'": unit, "Significato": desc}
            for k, (unit, desc) in _DOC.items()]
    if basal_u_per_min is not None:
        c = A.AIDController(basal_u_per_min, 3.0, **(ctrl_kwargs or {}))
        rows += [
            {"Parametro": "I_min, I_max (derivati)", "Valore": f"{c.i_min:.0f} ; {c.i_max:.0f}",
             "Unita'": "mg*min/dL", "Significato": "Limiti effettivi dell'integrale dell'errore"},
            {"Parametro": "Ki*I_min, Ki*I_max", "Valore": f"{c.ki * c.i_min * 60:.2f} ; {c.ki * c.i_max * 60:.2f}",
             "Unita'": "U/h", "Significato": "Contributo minimo e massimo dell'azione integrale"},
            {"Parametro": "iob_max (derivato)", "Valore": round(c.iob_max, 2), "Unita'": "U",
             "Significato": "Tetto IOB effettivo (attivo solo se use_iob=True)"},
        ]
    rows.append({"Parametro": "u_min, u_max (attuatore)", "Valore": f"0.0 ; {A.U_MAX}", "Unita'": "U/min",
                 "Significato": f"Saturazione dell'infusione (0 <= u <= {A.U_MAX * 60:g} U/h)"})
    return pd.DataFrame(rows)


def initial_conditions(patient_name, seed=1):
    """Parametri fisiologici e condizioni iniziali del paziente (all'inizio della simulazione, 00:00)."""
    p = T1DPatient.withName(patient_name)
    s = CGMSensor.withName("Dexcom", seed=seed)
    pu = InsulinPump.withName("Insulet")
    env = T1DSimEnv(p, s, pu, CustomScenario(start_time=A.SIM_START, scenario=[]))
    obs, r, d, info = env.reset()
    return dict(
        BW=float(p._params["BW"]),
        Gb=float(p._params["Gb"]),
        ub=float(A.basal_rate(p)),
        bg0=float(p.state[3] / p._params["Vg"]),
        cgm0=float(obs.CGM),
        dt=float(env.sample_time),
        eat_rate=float(T1DPatient.EAT_RATE),
        kabs=float(p._params["kabs"]),
    )


def scenario_table(patient_name, cho, seed=1, meal_clock_h=A.MEAL_CLOCK_H):
    """Tabella del protocollo di simulazione."""
    ic = initial_conditions(patient_name, seed)
    n_tot = int(A.ANALYSIS_END_H * 60 / ic["dt"])
    n_win = int((A.ANALYSIS_END_H - A.WARMUP_H) * 60 / ic["dt"])
    rows = [
        ("Paziente virtuale", patient_name, "-"),
        ("Peso corporeo BW", round(ic["BW"], 1), "kg"),
        ("Glicemia basale di equilibrio Gb", round(ic["Gb"], 1), "mg/dL"),
        ("Fabbisogno basale nominale u_b", round(ic["ub"] * 60, 3), "U/h"),
        ("Sensore CGM", "Dexcom (rumore di simglucose)", "-"),
        ("Pompa di infusione", "Insulet", "-"),
        ("Seme casuale del sensore", seed, "-"),
        ("Inizio simulazione", A.SIM_START.strftime("%H:%M"), "ora del giorno"),
        ("Assestamento escluso dalle metriche", f"00:00-{int(A.WARMUP_H):02d}:00", "ora del giorno"),
        ("Finestra di analisi", f"{int(A.WARMUP_H):02d}:00-{int(A.ANALYSIS_END_H):02d}:00", "ora del giorno"),
        ("Passo di campionamento dt", ic["dt"], "min"),
        ("Passi totali / nella finestra", f"{n_tot} / {n_win}", "-"),
        ("Glicemia plasmatica iniziale (00:00)", round(ic["bg0"], 1), "mg/dL"),
        ("Misura CGM iniziale (00:00)", round(ic["cgm0"], 1), "mg/dL"),
        ("Stati iniziali stimatore IOB (S1, S2)", "0.0, 0.0", "U"),
        ("Orario del pasto", f"{int(meal_clock_h):02d}:{int(round(meal_clock_h % 1 * 60)):02d}", "ora del giorno"),
        ("Carico di carboidrati (CHO)", cho, "g"),
        ("Velocità di ingestione del pasto", ic["eat_rate"], "g/min"),
        ("Costante di assorbimento intestinale kabs", round(ic["kabs"], 4), "1/min"),
        ("Annuncio / bolo manuale", "Nessuno (disturbo non annunciato)", "-"),
        ("Glicemia usata per le metriche", "plasmatica Gp/Vg del modello", "mg/dL"),
    ]
    return pd.DataFrame(rows, columns=["Voce", "Valore", "Unita'"])
