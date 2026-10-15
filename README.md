# Controllo AID con pasti non annunciati (simulazioni in silico)

Codice della tesi triennale *Controllo di un sistema AID con disturbi prandiali non annunciati*
(Politecnico di Bari, Fondamenti di Automatica).

Il repository contiene la simulazione *in silico* di algoritmi di regolazione glicemica a circuito chiuso
(*Automated Insulin Delivery*, AID) soggetti a un pasto non annunciato. Le simulazioni usano la libreria open source
[`simglucose`](https://github.com/jxx123/simglucose), implementazione Python del modello UVa/Padova.

---

## Scenario sperimentale

- **Pazienti virtuali**: `adult#001` (riferimento), `adult#002` e `adult#003`.
- **Sensore CGM**: Dexcom, campionamento ogni 3 minuti, con rumore autoregressivo di simglucose (seme `SEED = 1`).
- **Pompa**: Insulet, con saturazione `0 <= u <= 0.25 U/min` (15 U/h).
- **Durata**: 16 ore (320 passi da 3 minuti, dalle 08:00 alle 24:00).
- **Pasto non annunciato** alle 12:00, cioè 4 ore dopo l'inizio (`MEAL_OFFSET_H = 4.0`). In `CustomScenario` un tempo numerico è un
  *offset in ore dall'inizio*, non un'ora del giorno.
- **Carichi**: 0 g (verifica a digiuno), 30 g, 60 g, 90 g di carboidrati.

## Controllori (`aid_sim.py`)

| Nome | Descrizione |
|---|---|
| `PID` | PID a tempo discreto, azione diretta: `u = u_b + Kp e + Ki I + Kd de/dt`. Target 120 mg/dL, `Kp = 1e-4`, `Ki = 5e-6`, `Kd = 2e-3`, anti-windup per clamping dell'integrale in [-2000, 10000]. |
| `PID cons.` | Lo stesso PID con guadagno integrale ridotto (`Ki = 2e-6`). |
| `PID+PLGS` | PID + sospensione predittiva: `u = 0` se la glicemia proiettata a 20 min scende sotto 90 mg/dL o la misura è sotto 75 mg/dL. |
| `PID+IOB` | PID + tetto sull'insulina attiva stimata: se `IOB = S1 + S2` supera 1.5 x fabbisogno basale orario, l'infusione torna al basale. |
| `PID+IOB+PLGS` | **Controllore proposto**: PID con tetto IOB e PLGS attivi insieme. |

Nei confronti principali il PID standard e il controllore proposto hanno condizioni identiche (stesso paziente, sensore, seme, condizioni iniziali,
pasto, guadagni e limiti dell'attuatore): cambia solo l'attivazione del tetto IOB e del PLGS. Le unità di misura di tutti i parametri sono
elencate nei notebook (tabelle generate da `notebook_utils.py`).

## Metriche

Calcolate sulla **glicemia plasmatica** (`info['bg']`); il controllore vede solo il CGM.

- **TIR**: tempo nell'intervallo 70-180 mg/dL.
- **TAR**: tempo sopra 180 mg/dL (e sopra 250 mg/dL).
- **TBR**: tempo sotto 70 mg/dL (e sotto 54 mg/dL).
- **TBR tardivo**: tempo sotto 70 mg/dL da 3 ore dopo il pasto fino a fine simulazione (dalle 15:00 alle 24:00).
- **Picco** e **nadir** della glicemia.

---

## Come riprodurre

```bash
git clone https://github.com/martinadonnicola7-svg/tesi-aid-simglucose.git
cd tesi-aid-simglucose
python3 -m venv venv
source venv/bin/activate          # su Windows: venv\Scripts\activate
pip install -r requirements.txt
```

Poi, dalla cartella principale del repository:

```bash
python3 run_campaign.py           # tutte le simulazioni, tabelle e grafici in results/ (circa 4-5 minuti)
jupyter notebook                  # per aprire i notebook
```

Ogni notebook stampa all'inizio le **versioni delle librerie** usate (in particolare `simglucose`), i parametri dei controllori con le unità
di misura e le condizioni dello scenario (condizioni iniziali, durata, passo, orario e quantità del pasto, seme casuale).
I risultati salvati sono stati ottenuti con Python 3.12, simglucose 0.2.11, numpy 2.4, scipy 1.17, pandas 3.0 e matplotlib 3.10.

## Notebook

| Notebook | Contenuto |
|---|---|
| `01_scenario_riferimento.ipynb` | Scenario di riferimento su `adult#001`, verifica nominale a digiuno, pasto da 60 g, grafici a tre pannelli (glicemia, infusione, IOB) e indicatori. |
| `02_confronto_carichi.ipynb` | Confronto tra PID standard e controllore proposto su 30, 60 e 90 g, paziente fisso (`adult#001`), con tabella delle condizioni identiche. |
| `03_altri_pazienti.ipynb` | Stesso confronto su `adult#002` e `adult#003` (digiuno e 60 g), senza ritarare i parametri. |

## Struttura del repository

```text
aid_sim.py                       controllori, simulazione, metriche
notebook_utils.py                tabelle di documentazione per i notebook (versioni, parametri, scenario)
run_campaign.py                  tutte le simulazioni, tabelle e grafici
01_scenario_riferimento.ipynb    scenario di riferimento e verifica nominale
02_confronto_carichi.ipynb       confronto sui carichi prandiali (paziente fisso)
03_altri_pazienti.ipynb          estensione agli altri pazienti virtuali
results/metriche.csv             tabella completa dei risultati
results/robustezza_semi.csv      media e deviazione standard su 5 semi del rumore del sensore
results/fig_*.png                grafici (tre pannelli, sensibilità al carico, multipaziente)
requirements.txt                 dipendenze
LICENSE
```

## Limiti

Risultati *in silico*, con un solo pasto per simulazione e tre pazienti virtuali. I guadagni sono scelti a mano e non ottimizzati.
Il tetto IOB agisce come un interruttore: l'infusione alterna tra valore basale e valore alto, quindi l'attuatore commuta.
cd tesi-aid-simglucose
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
