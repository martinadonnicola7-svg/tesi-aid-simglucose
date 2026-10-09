# Modellistica e Controllo AID per pasti non annunciati
Repository per la simulazione e validazione in silico di algoritmi di controllo a retroazione per sistemi di erogazione automatica di insulina (AID) soggetti a disturbi prandiali non annunciati.

Il simulatore impiegato è basato sul modello metabolico UVA/Padova implementato nella libreria open-source `simglucose`.

## Struttura del Repository
- `01_baseline_pid_benchmark.ipynb`: Notebook principale contenente la simulazione nominale con controllore PID baseline, analisi a carichi crescenti (30g, 50g, 70g) e confronto multipaziente. Include grafici a 3 pannelli (Glicemia, Insulina infusa, IOB) e calcolo degli indicatori clinici (TIR, TAR, TBR, Picco).
- `run_experiments.py` / `benchmark.py`: Script Python per l'esecuzione in batch delle simulazioni.
- `metriche_confronto.csv` / `metriche_estese.csv`: Risultati tabellari e indicatori statistici estratti.
- `grafico_*.png`: Esportazioni ad alta risoluzione dei grafici per la tesi.

## Requisiti e Riproducibilità
Per replicare l'ambiente di simulazione:
```bash
pip install -r requirements.txt
