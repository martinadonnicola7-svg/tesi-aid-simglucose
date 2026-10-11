# Controllo AID con pasti non annunciati (simulazioni in silico)

Codice della tesi triennale *Controllo di un sistema AID con disturbi prandiali non annunciati*
(Politecnico di Bari, Fondamenti di Automatica). Le simulazioni usano
[`simglucose`](https://github.com/jxx123/simglucose), un'implementazione Python del simulatore UVa/Padova.

## Scenario

- Pazienti virtuali: `adult#001` (riferimento), `adult#002`, `adult#003`.
- Sensore Dexcom (campionamento 3 minuti), pompa Insulet.
- Simulazione di **16 ore**: 320 passi da 3 minuti, dalle 08:00 alle 24:00.
- **Un pasto non annunciato alle ore 12:00**, cioè 4 ore dopo l'inizio, di 30, 60 o 90 g di carboidrati.
  (In `CustomScenario` un tempo numerico è un *offset in ore dall'inizio*: per questo nel codice è `MEAL_OFFSET_H = 4.0`.)
- Prova di controllo a digiuno (0 g) per verificare che l'anello sia stabile senza disturbo.

## Controllori (`aid_sim.py`)

| Nome | Descrizione |
|---|---|
| `PID` | PID a tempo discreto, azione diretta, anti-windup con clamping dell'integrale, `0 <= u <= 0.25 U/min`. Target 120 mg/dL, `Kp=1e-4`, `Ki=5e-6`, `Kd=2e-3`. |
| `PID cons.` | Lo stesso PID con guadagno integrale ridotto (`Ki=2e-6`): taratura più prudente. |
| `PID+PLGS` | PID + sospensione predittiva (proiezione a 20 min sotto 90 mg/dL, o misura sotto 75 mg/dL). |
| `PID+IOB` | PID + tetto sull'insulina attiva stimata, `IOB = S1 + S2` (due compartimenti, `tau_s = 50 min`), tetto = 1.5 x fabbisogno basale orario. |
| `PID+IOB+PLGS` | Controllore proposto: PID + tetto IOB + PLGS. |

## Metriche

Calcolate sulla **glicemia plasmatica** (il controllore vede solo il CGM): TIR (70-180 mg/dL), TAR (>180 e >250),
TBR (<70 e <54), picco, nadir e **TBR tardivo**, cioè il tempo sotto 70 mg/dL da 3 ore dopo il pasto fino a fine simulazione.

## Come riprodurre

```bash
pip install -r requirements.txt
python run_campaign.py        # circa 4-5 minuti, scrive tutto in results/
jupyter notebook 01_scenario_riferimento.ipynb
```

## Struttura

```text
aid_sim.py                     controllori, simulazione, metriche
run_campaign.py                tutte le simulazioni, tabelle e grafici
01_scenario_riferimento.ipynb  scenario adult#001, 60 g: descrizione, parametri, grafici, indicatori
results/metriche.csv           tabella completa dei risultati
results/robustezza_semi.csv    media e deviazione standard su 5 semi del rumore del sensore
results/fig_*.png              grafici (3 pannelli, sensibilità al carico, multipaziente)
```

## Limiti

Risultati *in silico*, un solo pasto per simulazione, tre pazienti virtuali, guadagni scelti a mano e non ottimizzati.
Il tetto IOB agisce come un interruttore (l'infusione alterna tra valore basale e valore alto), quindi produce commutazioni dell'attuatore.
