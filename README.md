# Modellistica e Controllo AID per pasti non annunciati

Repository per la simulazione e validazione *in silico* di algoritmi di controllo a retroazione per sistemi di erogazione automatica di insulina (*Automated Insulin Delivery*, AID) soggetti a disturbi prandiali non annunciati.

La piattaforma impiega il modello metabolico compartimentale UVA/Padova implementato nella libreria open-source `simglucose`.

[![Visualizza Notebook su nbviewer](https://img.shields.io/badge/render-nbviewer-orange.svg)](https://nbviewer.org/github/martinadonnicola7-svg/tesi-aid-simglucose/blob/main/01_baseline_pid_benchmark.ipynb)

---

## Obiettivi e Strategie di Controllo

Lo studio analizza il trade-off tra reattività postprandiale e prevenzione dell'ipoglicemia tardiva dovuta al fenomeno dell'*insulin stacking*, confrontando due leggi di controllo:

1. **PID Baseline**: Regolatore classico a retroazione proporzionale-integrale-derivativo tarato sul target euglicemico ($110\text{ mg/dL}$).
2. **PID con Vincolo su IOB (IOB-Constrained)**: Regolatore con saturazione dinamica della massima velocità di infusione erogabile, determinata dalla stima dello stato di insulina attiva residua ($\widehat{IOB}(t)$) per bloccare erogazioni eccessive e prevenire ipoglicemie iatrogene tardive.

---

## Campagna di Simulazione e Scenari

La sperimentazione valuta la reiezione del disturbo esogeno prandiale su una coorte virtuale eterogenea:
- **Carichi glucidici considerati**: 
  - $30\text{ g}$ (spuntino / pasto leggero)
  - $60\text{ g}$ (pasto nominale di riferimento)
  - $90\text{ g}$ (pasto abbondante / iperglucidico)
- **Soggetti virtuali**: pazienti `adult#001`, `adult#002` e `adult#003` estratti dal simulatore UVA/Padova.
- **Metriche cliniche estratte**:
  - **TIR** (*Time In Range*, $70 - 180\text{ mg/dL}$)
  - **TAR** (*Time Above Range*, $> 180\text{ mg/dL}$)
  - **TBR** (*Time Below Range*, $< 70\text{ mg/dL}$ sia totale sia nella finestra tardiva postprandiale)
  - **Picco glicemico massimo** e **Nadir**

---

## Struttura del Repository

- `01_baseline_pid_benchmark.ipynb`: Notebook Jupyter principale con simulazioni eseguite e grafici salvati cella per cella. Include scenario nominale, parametri, grafici comparativi a 3 pannelli (Glicemia, Velocità di infusione, IOB) e tabelle riassuntive.
- `benchmark.py`: Modulo con le funzioni di supporto, calcolo del decadimento farmacocinetico dell'IOB e metriche cliniche.
- `run_experiments.py`: Script per l'esecuzione batch dello scenario nominale sul paziente `adult#001`.
- `run_extended_experiments.py`: Script per la campagna estesa multipaziente (`adult#001`, `adult#002`, `adult#003`) sui carichi da 30 g, 60 g e 90 g.
- `genera_notebooks.py`: Script di utilità per l'aggiornamento e la generazione automatica dei notebook.
- `metriche_confronto.csv`: Tabella riassuntiva dei benchmark nominali.
- `metriche_estese.csv`: Dataset completo con le 11 combinazioni di test (Paziente, Controllore, CHO, TIR, TAR, TBR, Picco, Nadir).
- `grafico_*.png`: Esportazioni ad alta risoluzione dei grafici a 3 pannelli per la documentazione della tesi.
- `requirements.txt`: Elenco delle dipendenze con versioni compatibili.

---

## Requisiti e Riproducibilità

### Requisiti di Sistema
- **Python**: versione `3.9` o `3.10`

### Installazione delle Dipendenze
```bash
git clone [https://github.com/martinadonnicola7-svg/tesi-aid-simglucose.git](https://github.com/martinadonnicola7-svg/tesi-aid-simglucose.git)
cd tesi-aid-simglucose
pip install -r requirements.txt
