# A320neo Engine Removal & Cycle-Life Forecast

End-to-end engine-removal-forecast pipeline for the LEAP-1A engine fleet on A320neo aircraft.
Predicts Remaining Cycle Life (RCL) using a sequence ensemble (BiLSTM + GRU + 1D CNN)
served behind a FastAPI inference endpoint.

## Headline numbers (latest run, May 2026)

- **Fleet:** 48 LEAP-1A engines (combined LEAP-1A26 and LEAP-1A26E1 variants).
- **Train / val / test split:** 38 / 5 / 5 engines, stratified by life quintile and type.
- **Test MAE:** **210 cycles** (R^2 = 0.960) on 5 held-out engines spanning 1,557 - 5,462 cycles.
- **Per-engine R^2:** uniformly **[0.936, 0.983]** - no engine is a failure mode.
- **LSTM lift over baseline:** +93 cycles MAE on test, +96 on validation.

## Pipeline at a glance

```
Data/A320NEOMSRMSC_masked.csv             (raw operator dump)
        |
        v
engine_removal_forecast_cleaning.ipynb     (column rename, phase filter,
        |                                  flight_cycle assignment,
        |                                  per-engine *_norm baseline,
        |                                  censoring-aware RCL,
        |                                  eng_type one-hot)
        v
Data/LEAB_engines_data_cleaned.csv        (49 engines, 36 columns)
        |
        +-> Engine_Removal_Forecast_EDA_v2.ipynb         (24 sections + Final Conclusion)
        |
        +-> Lstm_PM_py.ipynb                         (BiLSTM + GRU + CNN training)
        |       |
        |       v
        |   best_lstm2_model.keras
        |   best_gru_model.keras                     (artifacts logged to MLflow)
        |   best_cnn_model.keras
        |       |
        v       v
api_inference.py                          (FastAPI / sequence input / 4 model_choice modes)
        |
        v
index.html                                (CSV-upload frontend, live API health pill)
```

## Key methodology decisions

### 1. Combined-fleet training (vs single-variant)
EDA section 6 ran a 2-sample KS test on every sensor x phase combination between
`LEAP-1A26` and `LEAP-1A26E1` engines. 11 of 12 distributions overlap within 5%
of each other; only `fuel_flw` in TAKEOFF diverges by ~24% (the known E1 fuel-burn
improvement). Combining the two types triples the training pool from 15 to 49
engines, and per-engine baseline normalisation (`*_norm` columns) neutralises
the type offset.

### 2. Per-engine baseline normalisation
For each engine, the cleaning notebook subtracts the mean of the first 50 flight
cycles per phase from each of the 6 core sensors, producing `*_norm` drift
features. This removes engine-to-engine and type-to-type level offsets so the
model sees the actual degradation signal directly. Both raw and `*_norm`
features are kept in the input vector; the model decides which to use.

### 3. Censoring-aware RCL labels
Only 6 of 49 engines are off-wing (>=21 days since last observation). The
remaining 43 are still flying as of the dataset cutoff (2026-05-02), so their
`max_cycle` is "what we have so far", not retirement life. Using
`RCL = max_cycle - flight_cycle` for these engines under-labels them by
1,000-3,000 cycles each.

The cleaning notebook now classifies engines as "removed" or "active" based on
days-since-last-observation, and assigns active engines an `expected_max_cycle`
equal to the empirical mean `max_cycle` of the off-wing engines (stratified by
type when at least 3 samples exist, falling back to fleet-wide mean otherwise).
Current priors:

| Type | n removed | Prior (cycles) |
|---|---|---|
| LEAP-1A26 | 4 | 4,141 |
| LEAP-1A26E1 | 2 | 4,380 (falls back to fleet mean) |

Without this fix, per-engine R^2 on short-life (i.e. recently-installed) engines
was -1.5 to -6.9. With it, every per-engine R^2 lands in [0.936, 0.983].

**Definitive finding (Option D):** we trained the BiLSTM with `flight_cycle` exposed as a feature and ran group permutation importance. Test MAE 72.6 cycles, R² 0.97-0.99 per engine — but permuting `flight_cycle_norm` alone pushes MAE to 1,568 cycles (+1,496), while permuting ALL 20 sensor + time + phase features together pushes MAE only to 115.7 (+43). **Sensors contribute ~3% of what cycle count contributes.** The dataset's RCL labels are structurally dominated by `flight_cycle` arithmetic; sequence models cannot extract additional signal. Sensors DO carry signal — for the degradation classifier (Random Forest, 99.9% CV accuracy on 4-class severity), where the task is structurally different. Under the 5,000-cycle policy, 13 of 49 engines are effectively removed; Cox-PH C-index 0.761 (moderately predictive; window-calibrated against the inference 40-flight feature definition). Survival analysis remains v3 scope as more retirements accumulate.

### 4. Group-aware splits
Engines never appear in more than one set. The split is deterministic and
stratified by life quintile + engine type so val and test each cover the
short / short-medium / medium / medium-long / long life range and contain
both LEAP-1A26 and LEAP-1A26E1.

### 5. Residual-RCL target
Inside the LSTM notebook, the model predicts a residual on top of a fleet
baseline (`baseline_RCL = FLEET_MAX_LIFE - flight_cycle`), not the absolute RCL.
This avoids the per-engine arithmetic shortcut that would otherwise make
`flight_cycle` a perfect predictor.

## Project structure

```
A320NEO_ERF_RCL/
|-- Data/
|   |-- A320NEOMSRMSC_masked.csv
|   |-- LEAB_engines_data_cleaned.csv      <- canonical 49-engine input
|   `-- LEAP_Engine_Data_Masked_Final_no_outliers.csv
|
|-- engine_removal_forecast_cleaning.ipynb  cleaning + censoring-aware RCL
|-- Engine_Removal_Forecast_EDA_v2.ipynb       EDA (24 sections, ~530K rows)
|-- Engine_Removal_Forecast_EDA_v2.ipynb      parallel EDA notebook
|
|-- Lstm_PM_py.ipynb                       BiLSTM + GRU + CNN training pipeline
|-- best_lstm2_model.keras                 trained BiLSTM (window=40, n_features=20)
|-- best_gru_model.keras                   trained GRU
|-- best_cnn_model.keras                   trained 1D CNN
|-- best_lstm2_model_all.keras             production LSTM (full-data retrain)
|-- best_lstm2_history.json                training curves
|
|-- train_pipeline.py                      MLflow-logged retrain wrapper (executes the LSTM notebook)
|-- api_inference.py                       FastAPI sequence-input service (port 8001)
|-- index.html                             CSV-upload frontend (port 8000)
|-- start_servers.py                       launches both servers
|-- Run local host.bat                     Windows one-click launcher
|-- Dockerfile                             packages FastAPI as :8000
|-- prometheus.yml                         metrics scrape config
|-- data_drift.py                          Evidently 0.7+ drift detector
|-- retrain_dag.py                         Airflow weekly drift-gated retrain DAG
|-- mlruns/                                MLflow tracking directory
|-- test_*.py                              pytest suites
|
`-- archive/legacy/                        pre-LEAP tree-ensemble notebooks + .pkl artifacts
```

## Setup

### Prerequisites
- Python 3.10+ (3.13 tested locally; Colab T4 uses 3.12 with TF 2.20)
- ~4 GB free disk for `.keras` model files + MLflow runs
- T4 GPU (Colab) or any CUDA GPU recommended for retraining; CPU works but is ~30x slower

### Install
```bash
pip install -r requirements.txt
```

### Data files (not committed to git)

The CSVs in `Data/` are not in the repository — they exceed GitHub's 100 MB per-file
hard limit and contain operator-sensitive fleet data. **The inference stack does not
need them**; the trained `.keras` models and `degradation_rf.pkl` are committed.

You only need the data files to (re)run the cleaning notebook, EDA, or retraining:

| File | Size | Produced by |
|---|---|---|
| `Data/A320NEOMSRMSC_masked.csv` | ~131 MB | raw operator dump (masked) |
| `Data/LEAB_engines_data_cleaned.csv` | ~181 MB | `engine_removal_forecast_cleaning.ipynb` |
| `Data/LEAP_Engine_Data_Masked_Final_no_outliers.csv` | ~165 MB | cleaning + outlier removal |

To obtain them, contact the project owner. <!-- TODO: replace with a download link (Google Drive / S3 / HF Hub) -->

### Run the inference stack locally
```bash
python start_servers.py
```
This launches:
- FastAPI backend on `http://127.0.0.1:8001`
- Static frontend on `http://127.0.0.1:8000`

Open `http://127.0.0.1:8000/index.html` in a browser. The page polls `/health`
and `/schema` on load; the green pill confirms the API is ready.

### Use the prediction tool
The frontend expects a CSV upload of the **last 40 flights of one engine**
(oldest -> newest). Required columns:

- `flight_cycle`, `flight_datetime_c`
- 6 raw sensors: `egt_probe_average`, `fuel_flw`, `core_spd`, `oil_temp`, `oil_pres`, `pt2_(fan_inlet_tot_pres)`
- 6 `*_norm` variants of the same sensors
- Either `eng_type` or pre-computed `engtype_LEAP-1A26` / `engtype_LEAP-1A26E1`
- Either `flight_phase` or pre-computed `flight_phase_CRUISE` / `flight_phase_TAKEOFF`

Cyclical time encodings (`month_sin`, `month_cos`, `dayofweek_sin`, `dayofweek_cos`)
are auto-derived from `flight_datetime_c` if not pre-computed. Rows from
`Data/LEAB_engines_data_cleaned.csv` already match this layout.

The model returns `RCL_prediction`, the per-model residuals, and the baseline
breakdown so the prediction is interpretable.

### Retrain
```bash
# end-to-end via the canonical script (executes Lstm_PM_py.ipynb, logs to MLflow)
python train_pipeline.py

# or directly in the notebook (recommended on a GPU machine / Colab)
jupyter notebook Lstm_PM_py.ipynb
```

In Colab, mount Drive and `cd` into the project folder before Runtime -> Run All.
With `FORCE_RETRAIN=False` in cell 10 the cached `.keras` files are loaded in
~10 seconds; flip to `True` to retrain (~25-35 minutes on T4 for all three
architectures).

## Testing

```bash
pytest                                    # full suite
pytest test_data_drift.py -v              # 6 drift detection tests
```

## MLOps

- **MLflow:** every retrain via `train_pipeline.py` logs hyperparameters, per-epoch loss curves, summary metrics (best_val_loss, epochs_run), and the `.keras` artifacts under the `models/` artifact path. A `last_train_run.json` marker file is written so the Airflow DAG can promote the new run.
- **Airflow (`retrain_dag.py`):** weekly DAG with three tasks - `check_drift_and_retrain` (calls `data_drift.check_data_drift()` against an env-controlled current-data CSV; runs `train_pipeline.py` only if drift is detected), `register_model` (promotes each `.keras` to its own MLflow Model Registry name), `notify` (logs a summary line, ready for Slack/email).
- **Drift detection (`data_drift.py`):** Evidently 0.7+ `DataDriftPreset` over the 6 core sensors. Returns True when more than 50% of sensors drift on the K-S test (configurable via `DRIFT_SHARE_THRESHOLD`).
- **Prometheus:** `prometheus.yml` scrapes the FastAPI on port 8000 every 15s. Add a `/metrics` endpoint via `prometheus-fastapi-instrumentator` to feed it.
- **Docker:** `Dockerfile` builds a slim Python 3.10 image, installs `requirements.txt`, exposes 8000, and runs `uvicorn api_inference:app`.


## Next-generation direction — borescope-finding progression

The current pipeline ships three working capabilities: an RCL forecast (per-engine
cycles-to-policy-removal, Test MAE 72 cycles, R² 0.97-0.99), a degradation
classifier (Random Forest, 99.9% CV accuracy on stable/mild/elevated/severe),
and survival curves (Kaplan-Meier + Cox-PH on 13 effective removal events).

The natural next capability is **modelling borescope-finding progression**.
Engine removal events are physical inspection decisions: engines come off-wing
when a borescope reveals HPT blade tip rub, coking, crack initiation,
hot-section distress, or an LLP cycle-limit approach. These are step-function
events triggered by inspection findings, not sensor thresholds — and they're
what maintenance teams actually optimise around.

**v3 direction (additive, not replacement):**

- **Inputs**: sensor trends + previous borescope-finding history + cycle count
- **Target**: probability the next borescope reveals a finding of concern
- **Output for ops**: "flag this engine for early borescope at flight_cycle X"
- **Operational value**: reduces unnecessary inspections (cost) while catching
  issues earlier (safety)
- **Reframe**: today we answer *when will this engine be removed*; tomorrow we
  answer *when should this engine be inspected, and what should we look for*

This is incremental work on top of the existing pipeline — sensor processing,
training infrastructure, and serving stack are all reusable. The only new
ingredient is borescope finding records.

## Future work



- **Full 48-fold leave-one-engine-out CV** (cell 15, ~3.5 hours on T4) for the canonical fleet-wide R^2 distribution.
- **Tuned residual head** to close the gap between baseline R^2 (0.92-0.95) and LSTM R^2 (0.96-0.98).
- **Survival analysis** (Cox-PH or DeepSurv) once enough engines are off-wing to learn the actual retirement distribution rather than the empirical shop-visit prior.
- **Per-engine calibration** layer that learns each engine's bias from its first-N-cycle trajectory and refines the prediction at inference time.
- **Real-time data ingestion** via Kafka / scheduled CSV drop into `Data/` + Airflow DAG triggering retrain on drift.

