# Modellistica e Controllo AID per Pasti Non Annunciati

[![Visualizza Notebook su nbviewer](https://img.shields.io/badge/render-nbviewer-orange.svg)](https://nbviewer.org/github/martinadonnicola7-svg/tesi-aid-simglucose/blob/main/01_baseline_pid_benchmark.ipynb)
[![Licenza MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Python Version](https://img.shields.io/badge/python-3.9%20%7C%203.10-brightgreen.svg)](requirements.txt)

Repository per la modellistica, simulazione e validazione *in silico* di algoritmi di regolazione a retroazione per sistemi di erogazione automatica di insulina (*Automated Insulin Delivery*, AID) soggetti a disturbi prandiali non annunciati.

La sperimentazione è condotta mediante la libreria open-source `simglucose`, reimplementazione in ambiente Python del modello metabolico compartimentale non lineare UVA/Padova (Dalla Man et al., Kovatchev et al.).

---

## Obiettivi della Ricerca

Nei sistemi commerciali a circuito chiuso ibrido (*Hybrid Closed-Loop*, HCL), la compensazione dei pasti richiede l'annuncio manuale da parte dell'utente per l'erogazione di un bolo prandiale anticipato. L'omissione dell'annuncio configura il pasto come un **disturbo esogeno impulsivo e non misurato**.

A causa del ritardo di misura del sensore CGM ($10-15\text{ min}$) e della lenta farmacocinetica dell'insulina sottocutanea ($60-90\text{ min}$ al picco d'azione), un controllore puramente reattivo non vincolato rischia di provocare il fenomeno dell'**insulin stacking** (accumulo tardivo di farmaco attivo non ancora metabolizzato), innescando severe ipoglicemie iatrogene secondarie a distanza di ore dal pasto.

Il presente progetto analizza questo trade-off dinamico e valida una strategia di mitigazione basata su vincoli dinamici sullo stato stimato dell'insulina attiva residua.

---

## Strategie di Controllo Implementate

1. **PID Baseline Convenzionale**:
   - Regolatore a retroazione proporzionale-integrale-derivativo accordato sul setpoint euglicemico ($110\text{ mg/dL}$).
   - Dotato di saturazione statica dell'integratore (anti-windup clamping) e vincolo di non negatività sull'attuatore ($u(t) \ge 0$).
2. **PID con Vincolo su IOB e Modulo PLGS (IOB-Constrained)**:
   - **Stima dello stato IOB**: integrazione in linea dello stato di insulina attiva residua ($\widehat{IOB}(t)$) mediante modello farmacocinetico a due compartimenti sottocutanei in serie ($\tau_s = 50.0\text{ min}$).
   - **Saturazione dinamica IOB**: limitazione automatica delle erogazioni correttive sovrabasali qualora l'insulina già attiva superi la soglia massima consentita ($IOB_{cap} = 2.2\text{ U}$), prevenendo matematicamente l'insulin stacking.
   - **Modulo di sicurezza PLGS (*Predictive Low-Glucose Suspend*)**: arresto preventivo dell'erogazione ormonale ($u = 0$) qualora la proiezione lineare della traiettoria glicemica a 20 minuti scenda al di sotto della soglia di guardia ($90\text{ mg/dL}$) o il CGM rilevi una discesa rapida verso l'ipoglicemia.

---

## Setup di Simulazione e Protocollo di Benchmark

- **Piattaforma**: `simglucose` con sensore continuo Dexcom e pompa sottocutanea Insulet.
- **Passo di campionamento**: $\Delta t = 3\text{ minuti}$ (`env.sample_time = 3.0`), coerente con la frequenza di campionamento del sensore.
- **Durata della simulazione**: $16\text{ ore}$ nominali ($320\text{ passi}$ da 3 minuti ciascuno), dalle ore 08:00 alle 00:00.
- **Profilo del disturbo prandiale**: pasto non annunciato somministrato alle ore 12:00 ($t = 240\text{ min}$, passo 80).
  - $30\text{ g}$ di carboidrati (snack / pasto leggero);
  - $60\text{ g}$ di carboidrati (pasto nominale di riferimento);
  - $90\text{ g}$ di carboidrati (pasto abbondante / iperglucidico).
- **Coorte virtuale di test**:
  - `adult#001` (paziente nominale di benchmark);
  - `adult#002` e `adult#003` (analisi multipaziente e robustezza parametrica).

---

## Metriche Cliniche di Valutazione

Le prestazioni sono quantificate secondo le linee guida internazionali di consenso clinico:
- **TIR** (*Time In Range*): percentuale di tempo nell'intervallo target $70 - 180\text{ mg/dL}$;
- **TAR Livello 1**: tempo in iperglicemia moderata ($181 - 250\text{ mg/dL}$);
- **TAR Livello 2**: tempo in iperglicemia severa ($> 250\text{ mg/dL}$);
- **TBR Livello 1**: tempo in ipoglicemia moderata ($54 - 69\text{ mg/dL}$);
- **TBR Livello 2**: tempo in ipoglicemia severa ($< 54\text{ mg/dL}$);
- **TBR Tardivo ($3 - 6\text{ ore}$ postprandiale)**: percentuale di ipoglicemia registrata nella finestra critica $15:00 - 18:00$, metrica specifica per l'identificazione quantitativa dell'insulin stacking;
- **Picco postprandiale** ($C_{max}$) e **Nadir** (valore minimo assoluto registrato).

---

## Struttura del Repository

```text
tesi-aid-simglucose/
├── 01_baseline_pid_benchmark.ipynb   # Notebook con simulazioni eseguite, plot a 3 pannelli e metriche
├── benchmark.py                      # Modulo di supporto: osservatore IOB e calcolo metriche cliniche
├── run_experiments.py                # Script batch per la simulazione nominale su adult#001
├── run_extended_experiments.py       # Campagna estesa multipaziente (adult#001, #002, #003 su 30/60/90g)
├── genera_notebooks.py               # Utility per la generazione automatica dei notebook
├── metriche_confronto.csv            # Risultati tabellari dello scenario nominale
├── metriche_estese.csv               # Dataset completo con le 11 configurazioni di test
├── grafico_benchmark_baseline.png    # Grafico a 3 pannelli sincronizzato (Glicemia, Infusione, IOB)
├── grafico_multipaziente.png         # Confronto delle risposte dinamiche su adult#001, #002 e #003
├── grafico_sensibilita_pasti.png     # Risposta dinamica al variare del carico (30g, 60g, 90g)
├── requirements.txt                  # Dipendenze software con versioni fissate
├── LICENSE                           # Licenza MIT
└── README.md                         # Documentazione del progetto
