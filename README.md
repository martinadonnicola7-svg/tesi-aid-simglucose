# Controllo AID con pasti non annunciati (simulazioni in silico)

Codice della tesi triennale *Controllo di un sistema AID con disturbi prandiali non annunciati*
(Politecnico di Bari, Fondamenti di Automatica).

Il repository contiene la simulazione *in silico* di algoritmi di regolazione glicemica a circuito chiuso
(*Automated Insulin Delivery*, AID) soggetti a un pasto non annunciato. Le simulazioni usano la libreria open source
[`simglucose`](https://github.com/jxx123/simglucose), implementazione Python del modello UVa/Padova.

---

## Scenario sperimentale

- **Pazienti virtuali**: `adult#001` (riferimento), `adult#002` e `adult#003`; per l'estensione alla popolazione tutti i
  10 adulti di simglucose (`adult#001`-`adult#010`).
- **Sensore CGM**: Dexcom, campionamento ogni 3 minuti, con il rumore di simglucose (seme `SEED = 1`; 10 semi per la
  robustezza su `adult#001`-`adult#003`).
- **Pompa**: Insulet, con saturazione imposta dal controllore `0 <= u <= 0.25 U/min` (15 U/h).
- **Assestamento**: la simulazione parte alle 00:00 con il controllore già attivo. Le ore 00:00-08:00 servono a portare
  l'anello all'equilibrio sul target e **non** entrano nelle metriche.
- **Finestra di analisi**: dalle 08:00 alle 24:00 (16 ore, 320 passi da 3 minuti).
- **Pasto non annunciato** alle 12:00, senza bolo. Carichi: 0 g (verifica a digiuno), 30, 60, 90 g di carboidrati.
- **Sensibilità** (adult#001, 60 g): orario del pasto (10, 12, 14, 16) e velocità di assorbimento del pasto
  (svuotamento gastrico `kmax`, `kmin` e assorbimento intestinale `kabs` scalati insieme di 0.5, 1, 2).
- **Sensibilità ai parametri dei vincoli** (adult#001 e adult#003, 60 e 90 g, controllore proposto, un parametro alla volta):
  tetto IOB `iob_alpha` ∈ {1, **1.5**, 2, 3}, costante di tempo dello stimatore `tau_s` ∈ {30, **50**, 70} min,
  orizzonte del PLGS `t_pred` ∈ {10, **20**, 30} min, soglia del PLGS `plgs_pred` ∈ {80, **90**, 100} mg/dL
  (in grassetto il valore nominale).

## Controllori (`aid_sim.py`, dizionario `CFG`)

Tutti condividono la stessa legge PID a tempo discreto, ad azione diretta, con uscita in U/min:
`u = u_b + Kp e + Ki I + Kd dy/dt`, con target 120 mg/dL, `Kp = 1e-4`, `Ki = 5e-6`, `Kd = 2e-3`.
La pendenza `dy/dt` è stimata ai minimi quadrati sugli ultimi 5 campioni CGM (12 minuti).

| Nome | Descrizione |
|---|---|
| `PID senza AW` | PID della prima versione: integrale limitato in [-2000, 10000], un intervallo così largo che il limite non interviene mai. Serve solo a mostrare il **windup** dell'integratore. |
| `PID` | PID di riferimento con anti-windup: limiti dell'integrale legati al basale (`Ki·I_max = u_b`, `Ki·I_min = -0.25 u_b`) e integrazione condizionata quando l'uscita è limitata. |
| `PID cons.` | Come `PID`, ma con azione integrale massima dimezzata (`Ki·I_max = 0.5 u_b`): la sicurezza si ottiene riducendo l'aggressività. |
| `PID+PLGS` | `PID` + sospensione predittiva: `u = 0` se la glicemia proiettata a 20 min scende sotto 90 mg/dL o la misura è sotto 75 mg/dL. |
| `PID+IOB` | `PID` + tetto sull'insulina attiva stimata: se l'IOB netto supera 1.5 volte il basale orario, l'infusione torna al basale. |
| `PID+IOB+PLGS` | **Controllore proposto**: `PID` con tetto IOB e PLGS attivi insieme. |

Tra `PID` e il controllore proposto le condizioni sono identiche (stesso paziente, sensore, seme, condizioni iniziali,
assestamento, pasto, guadagni e limiti dell'attuatore): cambia solo l'attivazione del tetto IOB e del PLGS.

L'**IOB stimato** dal controllore è l'insulina in eccesso rispetto al basale filtrata da due compartimenti in serie con
τ = 50 min. Per confronto si salva anche l'**IOB netto del modello** (compartimenti sottocutanei `Isc1 + Isc2` meno il
loro valore di equilibrio basale).

## Metriche

Calcolate sulla **glicemia plasmatica** del modello (`Gp/Vg`) nella finestra 08:00-24:00. Il controllore vede solo il CGM.
Attenzione: in simglucose `info['bg']` è la glicemia **sottocutanea** senza rumore, non quella plasmatica; è salvata a parte
nella colonna `Gsub`.

- **TIR**: tempo nell'intervallo 70-180 mg/dL; **TIR 6h post**: lo stesso nelle 6 ore dopo il pasto.
- **TAR**: tempo sopra 180 mg/dL (e sopra 250 mg/dL).
- **TBR**: tempo sotto 70 mg/dL (e sotto 54 mg/dL).
- **TBR tardivo**: tempo sotto 70 mg/dL da 3 ore dopo il pasto fino alle 24:00.
- **Picco** e **nadir** della glicemia; **insulina totale** infusa nella finestra.

Indici dell'errore e del transitorio:

- **IAE**: integrale dell'errore assoluto rispetto al target, `∫|BG − 120| dt` [mg/dL·h], sulla finestra e nelle 6 ore dopo il pasto.
- **IAE fuori range**: integrale della distanza dall'intervallo 70-180 mg/dL (zero dentro il range) [mg/dL·h]; penalizza solo
  le escursioni clinicamente rilevanti.
- **Tempo di rientro nel range**: ore dal pasto al rientro definitivo in 70-180 mg/dL (0 se la glicemia non esce dal range,
  vuoto/NaN se a mezzanotte è ancora fuori).

Variabilità dell'infusione:

- **Commutazioni**: numero di cambi del vincolo attivo (attivazioni e disattivazioni di tetto IOB, PLGS e saturazioni).
- **Sospensioni**: numero di passaggi da infusione positiva a infusione nulla.
- **Variazione totale** `Σ|u_k − u_(k−1)|` e **deviazione standard** dell'infusione [U/h]. Sono diverse da zero anche senza
  vincoli, perché il PID reagisce al rumore del sensore.

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
python3 run_campaign.py           # tutte le simulazioni, tabelle e grafici in results/
jupyter notebook                  # per aprire i notebook (Kernel > Restart & Run All)
```

`run_campaign.py` usa tutti i core disponibili e salva una cache in `results/_cache.pkl` (esclusa da git), che si invalida
da sola se cambiano i controllori, `aid_sim.py` o la versione di simglucose. Prima di scrivere i risultati svuota `results/`
di tutte le tabelle e i grafici, quindi la cartella contiene solo i file prodotti dalla versione corrente dello script.
I notebook 05 e 06 usano le stesse funzioni e la stessa cache di `run_campaign.py`: conviene eseguire prima lo script.

Ogni notebook stampa all'inizio le **versioni delle librerie** usate (in particolare `simglucose`), i parametri dei controllori
con le unità di misura e le condizioni dello scenario (condizioni iniziali, assestamento, durata, passo, orario e quantità del
pasto, seme casuale). Le versioni usate per i risultati salvati sono fissate in `requirements.txt`.

## Notebook

| Notebook | Contenuto |
|---|---|
| `01_scenario_riferimento.ipynb` | Scenario di riferimento su `adult#001`: verifica nominale a digiuno, pasto da 60 g, grafici a tre pannelli (glicemia, infusione, IOB stimato e del modello), effetto del windup. |
| `02_confronto_carichi.ipynb` | Confronto sui carichi da 30, 60 e 90 g con paziente fisso (`adult#001`), tabella delle condizioni identiche e contributo dei singoli moduli (IOB, PLGS). |
| `03_altri_pazienti.ipynb` | Stesso confronto su `adult#001`, `adult#002`, `adult#003` (digiuno, 60 g, 90 g), senza ritarare i parametri. |
| `04_orario_assorbimento.ipynb` | Sensibilità all'orario del pasto e alla velocità di assorbimento (`adult#001`, 60 g). |
| `05_commutazioni_sensibilita_iob_plgs.ipynb` | IAE, tempo di rientro, commutazioni dell'infusione per modulo (`adult#001`, `adult#003`, 60 e 90 g) e sensibilità ai parametri del tetto IOB e del PLGS. |
| `06_popolazione_semi.ipynb` | Estensione ai 10 adulti virtuali (digiuno, 60 g, 90 g, moduli separati e insieme), compromesso paziente per paziente tra ipoglicemia e iperglicemia, robustezza su 10 semi. |

## Struttura del repository

```text
aid_sim.py                       controllori, configurazioni (CFG), simulazione, metriche
notebook_utils.py                tabelle di documentazione per i notebook (versioni, parametri, scenario)
run_campaign.py                  tutte le simulazioni, tabelle e grafici
01_scenario_riferimento.ipynb    scenario di riferimento e verifica nominale
02_confronto_carichi.ipynb       confronto sui carichi prandiali (paziente fisso)
03_altri_pazienti.ipynb          estensione agli altri pazienti virtuali
04_orario_assorbimento.ipynb     sensibilità all'orario e alla velocità del pasto
05_commutazioni_sensibilita_iob_plgs.ipynb   indici aggiunti, commutazioni, sensibilità tetto IOB / PLGS
06_popolazione_semi.ipynb        10 adulti virtuali e 10 semi del rumore
results/metriche.csv             campagna principale (adult#001-#003)
results/popolazione.csv          10 adulti, digiuno / 60 g / 90 g, tutti i moduli
results/popolazione_sintesi.csv  media ± deviazione standard sui 10 adulti
results/robustezza_semi.csv      media ± deviazione standard su 10 semi del rumore del sensore
results/sensibilita_pasto.csv    orario del pasto e velocità di assorbimento
results/sensibilita_iob_plgs.csv parametri del tetto IOB e del PLGS
results/fig_*.png                grafici (3 pannelli, carichi, pazienti, orario, assorbimento, commutazioni,
                                 sensibilità IOB/PLGS, popolazione, compromesso)
requirements.txt                 dipendenze con le versioni usate
LICENSE
```

## Limiti

Risultati *in silico*, con un solo pasto per simulazione e dieci pazienti virtuali adulti. Guadagni e soglie sono scelti a mano,
uguali per tutti i pazienti e non ottimizzati. Il tetto IOB agisce come un interruttore (l'infusione torna di colpo al basale),
quindi l'attuatore può commutare. Lo stimatore IOB a due compartimenti con τ fisso è un'approssimazione della cinetica
sottocutanea del modello.
