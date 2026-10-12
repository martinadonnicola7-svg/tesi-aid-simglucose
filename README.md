# Controllo AID con pasti non annunciati (simulazioni in silico)

Repository per la modellistica, simulazione e validazione *in silico* di algoritmi di regolazione glicemica a circuito chiuso (*Automated Insulin Delivery*, AID) soggetti a disturbi prandiali non annunciati.

Le simulazioni sono basate sulla libreria open-source [`simglucose`](https://github.com/jxx123/simglucose), implementazione in ambiente Python del modello metabolico compartimentale non lineare UVA/Padova.

---

## Scenario Sperimentale

- **Pazienti virtuali**: `adult#001` (paziente nominale di riferimento), `adult#002` e `adult#003` (coorte di validazione multipaziente).
- **Sensore CGM**: Dexcom a campionamento discreto ($\Delta t = 3\text{ min}$) con modello di rumore autoregressivo integrato.
- **Pompa di infusione**: Insulet (saturazione fisica $0 \le u \le 0.25\text{ U/min}$, pari a $15\text{ U/h}$).
- **Orizzonte temporale**: **16 ore** nominali ($320\text{ passi}$ da $3\text{ minuti}$, dalle ore 08:00 alle 24:00).
- **Disturbo prandiale**: pasto non annunciato somministrato alle ore 12:00 solari ($4\text{ ore}$ dopo l'avvio, `MEAL_OFFSET_H = 4.0`).
  - Carichi analizzati: $0\text{ g}$ (controllo a digiuno per verifica stabilità), $30\text{ g}$ (snack), $60\text{ g}$ (pasto standard), $90\text{ g}$ (pasto abbondante).

---

## Architettura dei Controllori (`aid_sim.py`)

| Sigla | Tipologia | Parametri & Logica di Controllo |
|---|---|---|
| `PID` | PID nominale (azione aggressiva) | $u = u_b + K_p e + K_i I + K_d \frac{de}{dt}$, target $120\text{ mg/dL}$, $K_p = 10^{-4}$, $K_i = 5\cdot 10^{-6}$, $K_d = 2\cdot 10^{-3}$, clamping $[-2000, 10000]$. |
| `PID cons.` | PID conservativo | Stessa struttura, con guadagno integrale prudente ($K_i = 2\cdot 10^{-6}$) per limitare l'accumulo. |
| `PID+PLGS` | PID con sospensione predittiva | PID nominale + azzeramento dell'infusione ($u = 0$) se la proiezione a $20\text{ min}$ scende sotto $90\text{ mg/dL}$ o la misura scende sotto $75\text{ mg/dL}$. |
| `PID+IOB` | PID vincolato su IOB | PID nominale + saturazione sovrabasale ($u = u_b$) se l'insulina attiva stimata ($IOB = S_1 + S_2$) supera il tetto massimo ($IOB_{cap} = 1.5 \times u_b \times 60$). |
| `PID+IOB+PLGS` | **Controllore proposto** | PID nominale con **tetto IOB dinamico e modulo di sicurezza PLGS attivi congiuntamente**. |

---

## Metriche Cliniche di Consenso

Le metriche sono calcolate sulla **glicemia plasmatica** (`info['bg']`), garantendo una valutazione non distorta dal ritardo del sensore:
- **TIR** (*Time In Range*): percentuale di tempo nell'intervallo euglicemico $70 - 180\text{ mg/dL}$.
- **TAR** (*Time Above Range*): tempo in iperglicemia $> 180\text{ mg/dL}$ e $> 250\text{ mg/dL}$.
- **TBR** (*Time Below Range*): tempo in ipoglicemia $< 70\text{ mg/dL}$ e $< 54\text{ mg/dL}$.
- **TBR tardivo**: percentuale di ipoglicemia registrata a partire da 3 ore dopo il pasto fino al termine della simulazione ($15:00 - 24:00$), metrica ideata per isolare l'*insulin stacking*.
- **Picco postprandiale** ($C_{max}$) e **Nadir** (minimo assoluto).

---

## Istruzioni per l'Installazione e Riproducibilità

### 1. Clonazione del repository e creazione ambiente virtuale
```bash
git clone [https://github.com/martinadonnicola7-svg/tesi-aid-simglucose.git](https://github.com/martinadonnicola7-svg/tesi-aid-simglucose.git)
cd tesi-aid-simglucose
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
