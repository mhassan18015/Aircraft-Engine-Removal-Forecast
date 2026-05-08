# A320neo Engine RCL Forecast — System Flowcharts

Six diagrams covering every layer of the pipeline. All rendered with Mermaid;
view in GitHub, VS Code (Mermaid Preview extension), or any modern markdown viewer.

1. [System Overview](#1-system-overview)
2. [Data Cleaning Pipeline](#2-data-cleaning-pipeline)
3. [Model Training Pipeline](#3-model-training-pipeline)
4. [Inference Request Lifecycle](#4-inference-request-lifecycle)
5. [Retrain DAG (Airflow + Drift Gate)](#5-retrain-dag-airflow--drift-gate)
6. [Test Suite Coverage](#6-test-suite-coverage)

---

## 1. System Overview

End-to-end view: where data flows, where models are trained, where they're served, and how MLOps monitors and retrains them.

```mermaid
flowchart TD
    classDef rawData fill:#fff4e6,stroke:#f0ad4e,color:#000
    classDef cleanedData fill:#e6f6ec,stroke:#28a745,color:#000
    classDef notebook fill:#e8f3ff,stroke:#007BFF,color:#000
    classDef artifact fill:#f0e6ff,stroke:#6f42c1,color:#000
    classDef service fill:#fde8ea,stroke:#dc3545,color:#000
    classDef mlops fill:#fff3d6,stroke:#b8860b,color:#000

    Raw["📁 Data/A320NEOMSRMSC_masked.csv<br/>raw operator dump"]:::rawData

    subgraph DataPrep [" "]
        direction TB
        Clean["📓 engine_removal_forecast_cleaning.ipynb<br/>• column rename + phase filter<br/>• per-engine baseline norm (*_norm)<br/>• one-hot eng_type<br/>• <b>censoring-aware RCL</b>"]:::notebook
    end

    Cleaned["📁 Data/LEAB_engines_data_cleaned.csv<br/>49 engines, 36 columns"]:::cleanedData

    subgraph Analysis [" "]
        direction LR
        EDA["📓 Engine_Removal_Forecast_EDA_v2.ipynb<br/>24 sections, dist + STL + corr"]:::notebook
        ERF["📓 Engine_Removal_Forecast_EDA_v2.ipynb<br/>parallel deeper EDA"]:::notebook
    end

    subgraph Training [" "]
        direction TB
        LSTM["📓 Lstm_PM_py.ipynb<br/>• stratified 5/5/38 split<br/>• 40-step sliding windows<br/>• residual-RCL target<br/>• 3 architectures"]:::notebook
        Wrapper["🐍 train_pipeline.py<br/>nbconvert + MLflow logging"]:::notebook
    end

    Models["💾 Model artifacts<br/>best_lstm2_model.keras<br/>best_gru_model.keras<br/>best_cnn_model.keras<br/>best_lstm2_history.json"]:::artifact

    subgraph Serving [" "]
        direction TB
        API["⚙️ api_inference.py<br/>FastAPI on :8001 / :8000<br/>/predict /health /schema /metrics"]:::service
        Frontend["🌐 index.html<br/>CSV upload + live API status"]:::service
    end

    User(("👤 User<br/>maintenance team")):::service

    subgraph MLOps [" "]
        direction TB
        Drift["🔍 data_drift.py<br/>Evidently 0.7+ K-S test"]:::mlops
        DAG["⏰ retrain_dag.py<br/>Airflow weekly DAG"]:::mlops
        Registry["📊 MLflow Registry<br/>mlruns/"]:::mlops
        Prom["📈 Prometheus<br/>:9090 scrapes /metrics"]:::mlops
    end

    Raw --> Clean
    Clean --> Cleaned
    Cleaned --> EDA
    Cleaned --> ERF
    Cleaned --> LSTM
    LSTM --> Models
    Wrapper --> LSTM
    Wrapper -.logs.-> Registry
    Models --> API
    Frontend --> API
    User --> Frontend
    API -.exposes.-> Prom
    DAG --> Drift
    Drift -- "drift detected" --> Wrapper
    Wrapper --> Registry
    DAG -- "promote" --> Registry
```

**Legend:**
- 🟧 raw data
- 🟩 cleaned canonical data
- 🟦 notebook / Python script
- 🟪 model artifact
- 🟥 user-facing service
- 🟨 MLOps component

---

## 2. Data Cleaning Pipeline

Inside `engine_removal_forecast_cleaning.ipynb`. The big new piece is the **censoring-aware RCL** in cell 28 — without it the model labels were systematically wrong by 1,000-3,000 cycles for the 43 engines still actively flying.

```mermaid
flowchart TD
    classDef step fill:#e8f3ff,stroke:#007BFF,color:#000
    classDef decision fill:#fff3d6,stroke:#b8860b,color:#000
    classDef data fill:#f0f7ff,stroke:#666,color:#000

    Raw[("Data/A320NEOMSRMSC_masked.csv")]:::data

    Load[/"Cell 4: read CSV<br/>parse_dates flight_datetime_c"/]:::step
    Drop[/"Cell 5: drop CARRIER, AIRCRAFT_ID, EPOSITION"/]:::step
    Rename[/"Cell 12: snake_case columns<br/>strip special chars"/]:::step
    PhaseFilter[/"Cell 20: keep only CRUISE + TAKEOFF<br/>phases ENGINE_START, CEOD_*, CLIMB dropped"/]:::step
    Sort[/"Cell 23: sort by esn, flight_datetime_c"/]:::step
    Cycle[/"Cell 25: assign flight_cycle<br/>increment on each TAKEOFF row"/]:::step

    subgraph CensoringFix ["Cell 28 — censoring-aware RCL"]
        direction TB
        D1{"days_since_last >= 21d ?"}:::decision
        Removed[/"engine treated as removed<br/>expected_max = observed max_cycle"/]:::step
        Active[/"engine still flying<br/>expected_max = empirical prior"/]:::step
        Prior[("Empirical prior<br/>LEAP-1A26: 4141 (n=4)<br/>LEAP-1A26E1: 4380 (fallback)")]:::data
        D1 -- yes --> Removed
        D1 -- no --> Active
        Active --> Prior
        Prior --> Active
        Removed --> RCL
        Active --> RCL
        RCL[/"RCL = max(expected_max - flight_cycle, 0)<br/>+ is_engine_active flag<br/>+ expected_max_cycle column"/]:::step
    end

    NormStep[/"Cell 33: per-engine baseline normalisation<br/>for each esn × phase, subtract first-50-cycle mean<br/>produces 6 *_norm drift columns"/]:::step
    OneHot[/"Cell 34: one-hot eng_type<br/>engtype_LEAP-1A26, engtype_LEAP-1A26E1"/]:::step
    Save[/"Cell 35: write Data/LEAB_engines_data_cleaned.csv<br/>49 engines × 36 columns × ~546k rows"/]:::step

    Cleaned[("Data/LEAB_engines_data_cleaned.csv")]:::data

    Raw --> Load --> Drop --> Rename --> PhaseFilter --> Sort --> Cycle --> CensoringFix
    CensoringFix --> NormStep --> OneHot --> Save --> Cleaned
```

---Three pending plumbing items

## 3. Model Training Pipeline

Inside `Lstm_PM_py.ipynb`. Three architectures train on the same data and split, each cached to its own `.keras` file. The residual-RCL target is the key trick that prevents the model from short-circuiting via `flight_cycle`.

```mermaid
flowchart TD
    classDef cell fill:#e8f3ff,stroke:#007BFF,color:#000
    classDef tensor fill:#f0e6ff,stroke:#6f42c1,color:#000
    classDef metric fill:#e6f6ec,stroke:#28a745,color:#000
    classDef cache fill:#fff3d6,stroke:#b8860b,color:#000

    Cleaned[("LEAB_engines_data_cleaned.csv")]:::tensor

    C0[/"Cell 0: Colab auto-mount + cd"/]:::cell
    C2[/"Cell 2: feature engineering<br/>add month_sin/cos, dayofweek_sin/cos<br/>get_dummies flight_phase"/]:::cell
    C3[/"Cell 3: feature selection (sensor_cols)<br/>raw 6 + norm 6 + engtype 2 + time 4 + phase 2 = 20"/]:::cell
    C4[/"Cell 4: stratified 5/5/38 engine split<br/>VAL = ESN18, ESN15, ESN17, ESN20, ESN10<br/>TEST = ESN21, ESN41, ESN13, ESN24, ESN14<br/>TRAIN = the other 38"/]:::cell
    C6[/"Cell 6: per-engine ffill/bfill<br/>MinMax scale features<br/>compute residual = RCL - baseline_RCL<br/>scale residual to 0..1"/]:::cell
    C7[/"Cell 7: create_lstm_windows()<br/>40-step sliding windows, stride 2<br/>label = residual at window-end"/]:::cell

    Xtrain[("X_train, y_train<br/>~250K windows × 40 × 20")]:::tensor
    Xval[("X_val<br/>~21K windows")]:::tensor
    Xtest[("X_test<br/>~22K windows")]:::tensor

    subgraph TrainLoop ["Training (cells 11, 18, 19) — same data, three architectures"]
        direction LR
        BiLSTM["BiLSTM 64-32-16<br/>+ Dense(32) + Dense(1)<br/>Huber(0.1), Adam, EarlyStopping(10)"]:::cell
        GRU["GRU 64-32 + Dense"]:::cell
        CNN["1D CNN 64-32 + Dense"]:::cell
    end

    LSTM_F{"FORCE_RETRAIN<br/>flag?"}:::cache
    GRU_F{"FORCE_RETRAIN_GRU<br/>flag?"}:::cache
    CNN_F{"FORCE_RETRAIN_CNN<br/>flag?"}:::cache

    LSTMart[("best_lstm2_model.keras<br/>+ best_lstm2_history.json")]:::tensor
    GRUart[("best_gru_model.keras")]:::tensor
    CNNart[("best_cnn_model.keras")]:::tensor

    subgraph Eval ["Evaluation (cells 12 - 15)"]
        direction TB
        C12[/"Cell 12: aggregate eval<br/>val_pred_rcl = baseline + scaler.inverse(residual)<br/>Val MAE 149.91, R² 0.979<br/>Test MAE 210.62, R² 0.960"/]:::metric
        C13[/"Cell 13: per-engine breakdown<br/>all engines R² 0.936-0.983"/]:::metric
        C14[/"Cell 14: sensor KDE overlay<br/>covariate-shift detection"/]:::metric
        C15[/"Cell 15: RCL trajectory<br/>label-shift detection"/]:::metric
        C16[/"Cell 16: LOEO preview<br/>6 engines × ~12 epochs each"/]:::metric
    end

    C19[/"Cell 19: full-data retrain<br/>writes best_lstm2_model_all.keras"/]:::cell
    AllArt[("best_lstm2_model_all.keras<br/>production artifact")]:::tensor

    Cleaned --> C0 --> C2 --> C3 --> C4 --> C6 --> C7
    C7 --> Xtrain & Xval & Xtest

    Xtrain --> LSTM_F
    LSTM_F -- "False + cache exists" --> LSTMart
    LSTM_F -- "True or no cache" --> BiLSTM --> LSTMart

    Xtrain --> GRU_F
    GRU_F -- "False + cache" --> GRUart
    GRU_F -- "True or no cache" --> GRU --> GRUart

    Xtrain --> CNN_F
    CNN_F -- "False + cache" --> CNNart
    CNN_F -- "True or no cache" --> CNN --> CNNart

    LSTMart --> C12
    Xval --> C12
    Xtest --> C12
    C12 --> C13 --> C14 --> C15 --> C16

    LSTMart --> C19 --> AllArt
```

---

## 4. Inference Request Lifecycle

What happens when a maintenance engineer uploads a CSV through the frontend.

```mermaid
sequenceDiagram
    autonumber
    actor User as Maintenance Engineer
    participant Browser as index.html<br/>(static :8000)
    participant API as api_inference.py<br/>FastAPI :8001
    participant Models as 3× Keras models
    participant Log as predictions.log

    User->>Browser: Open page (loads on form submit)
    Browser->>API: GET /health
    API-->>Browser: {status: "healthy", schema_match: all true}
    Browser->>API: GET /schema
    API-->>Browser: {feature_order: [20], window_size: 40, fleet_max_life: 4000}
    Note over Browser: Renders green "healthy" pill,<br/>enables submit button.

    User->>Browser: Upload last-40-flights CSV<br/>+ pick model_choice
    activate Browser
    Browser->>Browser: parseCSV(text)<br/>auto-derive month_sin/cos,<br/>dayofweek_sin/cos from<br/>flight_datetime_c if missing
    Browser->>Browser: rowToFlight() ×40<br/>derive engtype_* and<br/>flight_phase_* one-hots
    deactivate Browser

    Browser->>+API: POST /predict<br/>{flights: [40 rows], model_choice}
    API->>API: Pydantic validates<br/>FlightRow schema +<br/>flights.length == 40
    API->>API: SCHEMA_MATCH check<br/>against loaded models

    alt Schema mismatch
        API-->>Browser: 400 — "retrain via Lstm_PM_py.ipynb"
    else Valid
        API->>API: _request_to_tensor()<br/>reshape to (1, 40, 20)
        API->>+Models: model.predict(X)<br/>(LSTM / GRU / CNN / all 3)
        Models-->>-API: residual (scaled)
        API->>API: rcl = (FLEET_MAX_LIFE - latest_cycle)<br/>+ residual
        API->>API: clip rcl ≥ 0,<br/>add warning if clipped
        API->>Log: append prediction line
        API-->>-Browser: 200<br/>{RCL_prediction, baseline_rcl,<br/>residual, per_model_residuals,<br/>warnings}
    end

    activate Browser
    Browser->>Browser: showResult():<br/>render cycles + breakdown +<br/>health-bar gradient
    deactivate Browser
    Browser-->>User: predicted RCL on screen
```

---

## 5. Retrain DAG (Airflow + Drift Gate)

The weekly retrain pipeline. Runs `data_drift.check_data_drift()` first; only retrains when drift is detected.

```mermaid
flowchart TD
    classDef trigger fill:#fff3d6,stroke:#b8860b,color:#000
    classDef task fill:#e8f3ff,stroke:#007BFF,color:#000
    classDef decision fill:#fde8ea,stroke:#dc3545,color:#000
    classDef artifact fill:#f0e6ff,stroke:#6f42c1,color:#000
    classDef sink fill:#e6f6ec,stroke:#28a745,color:#000

    Sched["⏰ Airflow scheduler<br/>schedule = timedelta(days=7)"]:::trigger

    T1[/"check_drift_and_retrain<br/>calls data_drift.check_data_drift()"/]:::task
    Decision{"More than 50% of<br/>sensors drifted<br/>(K-S p<0.05)?"}:::decision

    Skip[/"log 'no drift; skipping retrain'<br/>delete stale last_train_run.json"/]:::sink

    Train[/"subprocess.run(python train_pipeline.py)<br/>• executes Lstm_PM_py.ipynb via nbconvert<br/>• logs hyperparams + metrics + artifacts<br/>• writes last_train_run.json"/]:::task

    MLflow[("MLflow run<br/>mlruns/<exp>/<run_id>/<br/>artifacts/models/*.keras")]:::artifact
    Marker[("last_train_run.json<br/>{run_id, metrics, artifacts}")]:::artifact

    T3[/"register_model<br/>reads marker; calls<br/>mlflow.register_model() per .keras"/]:::task

    Registry[("MLflow Model Registry<br/>leap1a_rcl_ensemble__best_lstm2_model<br/>leap1a_rcl_ensemble__best_gru_model<br/>leap1a_rcl_ensemble__best_cnn_model")]:::artifact

    T4[/"notify (trigger_rule=all_done)<br/>logs run_id + best_val_loss + epoch<br/>hook ready for Slack/email"/]:::task

    Sink["📋 Operations team"]:::sink

    Sched --> T1 --> Decision
    Decision -- no --> Skip --> T4
    Decision -- yes --> Train
    Train --> MLflow
    Train --> Marker
    Marker --> T3
    MLflow --> T3
    T3 --> Registry --> T4
    T4 --> Sink
```

---

## 6. Test Suite Coverage

29 tests across 4 files, all passing in 45-60 seconds. Each layer has its own test file matching its responsibility.

```mermaid
flowchart LR
    classDef testfile fill:#e8f3ff,stroke:#007BFF,color:#000
    classDef target fill:#fff3d6,stroke:#b8860b,color:#000
    classDef pass fill:#e6f6ec,stroke:#28a745,color:#000

    subgraph Tests
        direction TB
        T1[/"test_api_inference.py<br/>13 tests"/]:::testfile
        T2[/"test_data_drift.py<br/>6 tests"/]:::testfile
        T3[/"test_train_pipeline.py<br/>4 tests"/]:::testfile
        T4[/"test_infrastructure.py<br/>6 tests"/]:::testfile
    end

    subgraph Targets
        direction TB
        A1["api_inference.py<br/>FlightRow schema, /health, /schema,<br/>/predict × 4 model_choices,<br/>shape mismatch handling, /metrics"]:::target
        A2["data_drift.py<br/>two-arg form, zero-arg DAG form,<br/>edge cases (None/empty/disjoint)"]:::target
        A3["train_pipeline.py<br/>side-effect-free import,<br/>SKIP_TRAINING=1 path,<br/>marker file written"]:::target
        A4["Dockerfile + .dockerignore<br/>+ prometheus.yml + requirements.txt"]:::target
    end

    R[("pytest -v<br/>29 passed in ~50s")]:::pass

    T1 --> A1
    T2 --> A2
    T3 --> A3
    T4 --> A4

    A1 --> R
    A2 --> R
    A3 --> R
    A4 --> R
```

---

## How to view these diagrams

| Tool | What you do |
|---|---|
| **GitHub** | Push the file to a repo. Mermaid renders inline in the rendered markdown view. |
| **VS Code** | Install the [Markdown Preview Mermaid Support](https://marketplace.visualstudio.com/items?itemName=bierner.markdown-mermaid) extension, then open the preview pane. |
| **JupyterLab** | The built-in markdown renderer (3.0+) supports Mermaid out of the box. |
| **Browser export** | `npx @mermaid-js/mermaid-cli -i FLOWCHARTS.md -o flowcharts.pdf` produces a single PDF. |
| **Standalone images** | Each `mermaid` block can be exported individually via [mermaid.live](https://mermaid.live) — paste, screenshot, save as PNG/SVG. |
