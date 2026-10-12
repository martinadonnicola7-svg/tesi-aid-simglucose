"""
notebook_utils.py - Tabelle di documentazione per i notebook (versioni, controllori, scenario).
Non contiene logica di simulazione: serve a stampare in modo uniforme parametri, unità,
condizioni iniziali, durata, passo, pasto, seme casuale e versioni delle librerie.
"""

import importlib.metadata as md
import inspect
import platform
import aid_sim as A
import pandas as pd
from simglucose.actuator.pump import InsulinPump
from simglucose.patient.t1dpatient import T1DPatient
from simglucose.sensor.cgm import CGMSensor
from simglucose.simulation.env import T1DSimEnv
from simglucose.simulation.scenario import CustomScenario


def library_versions():
  """Restituisce le versioni esatte dell'interprete e dei pacchetti usati."""
  rows = [{"Pacchetto": "Python", "Versione": platform.python_version()}]
  for n in ["simglucose", "numpy", "scipy", "pandas", "matplotlib"]:
    try:
      rows.append({"Pacchetto": n, "Versione": md.version(n)})
    except md.PackageNotFoundError:
      rows.append({"Pacchetto": n, "Versione": "non installato"})
  return pd.DataFrame(rows)


# Dizionario con unità di misura e significato dei parametri di AIDController
_DOC = {
    "target": ("mg/dL", "Glicemia obiettivo del regolatore"),
    "kp": ("U/min per mg/dL", "Guadagno proporzionale"),
    "ki": ("U/min per (mg/dL*min)", "Guadagno integrale"),
    "kd": (
        "U/min per (mg/dL/min)",
        "Guadagno derivativo sulla pendenza CGM (segno positivo)",
    ),
    "i_min": ("mg*min/dL", "Limite inferiore anti-windup (clamping)"),
    "i_max": ("mg*min/dL", "Limite superiore anti-windup (clamping)"),
    "use_iob": ("-", "Attiva il vincolo dinamico sull'insulina attiva (IOB)"),
    "iob_alpha": (
        "adimensionale",
        "Fattore di scala: tetto IOB = iob_alpha * basale orario",
    ),
    "tau_s": (
        "min",
        "Costante di tempo dei compartimenti sottocutanei IOB",
    ),
    "use_plgs": (
        "-",
        "Attiva la sospensione predittiva dell'infusione (PLGS)",
    ),
    "t_pred": ("min", "Orizzonte di predizione lineare della glicemia"),
    "plgs_pred": (
        "mg/dL",
        "Soglia su glicemia proiettata per sospensione infusione",
    ),
    "plgs_now": (
        "mg/dL",
        "Soglia su misura corrente per arresto immediato di sicurezza",
    ),
}


def controller_table(ctrl_kwargs=None, basal_u_per_min=None):
  """Genera la tabella riassuntiva dei parametri effettivi e dei vincoli del controllore."""
  sig = inspect.signature(A.AIDController.__init__)
  vals = {
      k: v.default
      for k, v in sig.parameters.items()
      if v.default is not inspect.Parameter.empty
  }
  vals.update(ctrl_kwargs or {})

  rows = []
  for k, (unit, desc) in _DOC.items():
    rows.append({
        "Parametro": k,
        "Valore": vals.get(k, "-"),
        "Unita'": unit,
        "Significato": desc,
    })

  if basal_u_per_min is not None:
    alpha = vals.get("iob_alpha", 1.5)
    cap = alpha * basal_u_per_min * 60.0
    rows.append({
        "Parametro": "iob_max (derivato)",
        "Valore": round(cap, 2),
        "Unita'": "U",
        "Significato": (
            "Tetto IOB effettivo = iob_alpha * basale orario (attivo se"
            " use_iob=True)"
        ),
    })

  rows.append({
      "Parametro": "u_min, u_max (vincolo attuatore)",
      "Valore": "0.0 ; 0.25",
      "Unita'": "U/min",
      "Significato": (
          "Saturazione attuatore pompa (0 <= u <= 15 U/h) per vincolo di non"
          " negatività"
      ),
  })
  return pd.DataFrame(rows)


def initial_conditions(patient_name, seed=1):
  """Estrae i parametri fisiologici e le condizioni al contorno iniziali del paziente."""
  p = T1DPatient.withName(patient_name)
  s = CGMSensor.withName("Dexcom", seed=seed)
  pu = InsulinPump.withName("Insulet")
  env = T1DSimEnv(p, s, pu, CustomScenario(start_time=A.START, scenario=[]))
  obs, r, d, info = env.reset()

  bg0 = info.get("bg")
  if bg0 is None:
    bg0 = env.patient.observation.Gsub

  ub = p._params["u2ss"] * p._params["BW"] / 6000.0
  eat_rate = getattr(T1DPatient, "EAT_RATE", 5.0)

  return dict(
      BW=float(p._params["BW"]),
      Gb=float(p._params["Gb"]),
      ub=float(ub),
      bg0=float(bg0),
      cgm0=float(obs.CGM),
      dt=float(env.sample_time),
      eat_rate=float(eat_rate) if eat_rate else 5.0,
  )


def scenario_table(patient_name, cho, seed=1, hours=16.0):
  """Genera la tabella documentativa del protocollo di simulazione per il notebook."""
  ic = initial_conditions(patient_name, seed)
  n = int(hours * 60 / ic["dt"])
  meal_clock = (A.START + pd.Timedelta(hours=A.MEAL_OFFSET_H)).strftime("%H:%M")

  rows = [
      ("Paziente virtuale", patient_name, "-"),
      ("Peso corporeo BW", round(ic["BW"], 1), "kg"),
      ("Glicemia basale di equilibrio Gb", round(ic["Gb"], 1), "mg/dL"),
      ("Fabbisogno basale nominale u_b", round(ic["ub"] * 60, 3), "U/h"),
      ("Sensore CGM", "Dexcom (rumore autoregressivo UVA/Padova)", "-"),
      ("Pompa di infusione", "Insulet", "-"),
      ("Seme casuale sensore (seed)", seed, "-"),
      ("Inizio simulazione", A.START.strftime("%H:%M"), "ora solare"),
      ("Durata simulazione", hours, "h"),
      ("Passo di campionamento dt", ic["dt"], "min"),
      ("Numero complessivo di passi", n, "-"),
      ("Glicemia plasmatica iniziale (t=0)", round(ic["bg0"], 1), "mg/dL"),
      ("Misura CGM iniziale (t=0)", round(ic["cgm0"], 1), "mg/dL"),
      ("Stati iniziali IOB (S1, S2)", "0.0, 0.0", "U"),
      (
          "Orario somministrazione pasto",
          f"{meal_clock} (offset t = {A.MEAL_OFFSET_H:g} h)",
          "ora solare",
      ),
      ("Carico di carboidrati (CHO)", cho, "g"),
      ("Velocità di assorbimento gastrico", ic["eat_rate"], "g/min"),
      ("Annuncio / Bolo manuale", "Nessuno (disturbo non annunciato)", "-"),
  ]
  return pd.DataFrame(rows, columns=["Voce", "Valore", "Unita'"])
