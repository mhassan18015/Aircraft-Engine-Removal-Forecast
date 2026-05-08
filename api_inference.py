"""FastAPI inference service for the LEAP-1A engine RCL pipeline.

Replaces the previous tree-ensemble (.pkl) implementation. Now serves the three
Keras models trained by Lstm_PM_py.ipynb on the 49-engine combined dataset:
LSTM, GRU, CNN -- plus an "average" ensemble option.

Schema is sequence-based: the LSTM/GRU/CNN models all expect a window of
WINDOW_SIZE consecutive flights, with the same feature ordering used during
training. Single-shot tabular inputs are no longer supported because the
sequence ordering is the model's main source of signal.
"""
from __future__ import annotations

import logging
import os
from typing import List, Literal

import numpy as np
import pandas as pd
import tensorflow as tf
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, field_validator

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
WINDOW_SIZE = 40  # matches Lstm_PM_py.ipynb cell 10 (LSTM uses WINDOW_SIZE=40, GRU/CNN follow)

# rul_scaler bounds + FLEET_MAX_LIFE come from the training run. Lstm_PM_py.ipynb
# cell 7 fits a MinMaxScaler on the residual_RCL target (range = [residual_min,
# residual_max]) and trains the model on values in [0, 1]. At inference time
# we MUST inverse-transform the model's output before adding it to baseline_RCL,
# otherwise we're treating a [0, 1] scaled value as if it were cycles.
#
# These constants are exported by `_export_rul_scaler.py` after each retrain
# (saved to rul_scaler.json); the API reads them on startup.
import json as _json
_SCALER_PATH = "rul_scaler.json"
_ENGINE_MAP_PATH = "engine_expected_max.json"
try:
    _scaler = _json.loads(open(_SCALER_PATH).read())
    FLEET_MAX_LIFE = float(_scaler["fleet_max_life"])
    RESIDUAL_MIN   = float(_scaler["residual_min"])
    RESIDUAL_MAX   = float(_scaler["residual_max"])
    BASELINE_KIND  = _scaler.get("baseline_kind", "fleet_median")
except FileNotFoundError:
    FLEET_MAX_LIFE = float(os.getenv("FLEET_MAX_LIFE", "4000"))
    RESIDUAL_MIN   = 0.0
    RESIDUAL_MAX   = 1.0
    BASELINE_KIND  = "fleet_median"

# Per-engine expected_max_cycle lookup (populated by the cleaning + retrain pipeline).
# Without this file the API falls back to FLEET_MAX_LIFE for every engine - works
# but loses the per-engine calibration introduced in the v2 baseline change.
try:
    ENGINE_EXPECTED_MAX = _json.loads(open(_ENGINE_MAP_PATH).read())
except FileNotFoundError:
    ENGINE_EXPECTED_MAX = {}

_RESIDUAL_RANGE = RESIDUAL_MAX - RESIDUAL_MIN

def _inverse_scale_residual(scaled: float) -> float:
    """Inverse-transform a [0, 1] model output back to cycles."""
    return scaled * _RESIDUAL_RANGE + RESIDUAL_MIN

def _engine_baseline(eng_number: str | None, expected_max_override: float | None,
                     latest_flight_cycle: int) -> tuple[float, str]:
    """Pick the per-prediction expected_max_cycle and return (baseline_RCL, source).

    Priority order:
      1. expected_max_override (caller supplied directly)
      2. engine_expected_max.json lookup by eng_number
      3. fleet-wide FLEET_MAX_LIFE fallback
    """
    if expected_max_override is not None:
        return expected_max_override - latest_flight_cycle, "request_override"
    if eng_number and eng_number in ENGINE_EXPECTED_MAX:
        em = float(ENGINE_EXPECTED_MAX[eng_number]["expected_max_cycle"])
        return em - latest_flight_cycle, f"lookup({eng_number})"
    return FLEET_MAX_LIFE - latest_flight_cycle, "fleet_median_fallback"

# Feature schema -- must match Lstm_PM_py.ipynb cell 3 ordering exactly.
RAW_SENSORS = [
    "egt_probe_average", "fuel_flw", "core_spd",
    "oil_temp", "oil_pres", "pt2_(fan_inlet_tot_pres)",
]
NORM_SENSORS = [f"{c}_norm" for c in RAW_SENSORS]
ENGTYPE_COLS = ["engtype_LEAP-1A26", "engtype_LEAP-1A26E1"]
TIME_FEATS   = ["month_sin", "month_cos", "dayofweek_sin", "dayofweek_cos"]
PHASE_FEATS  = ["flight_phase_CRUISE", "flight_phase_TAKEOFF"]

FEATURE_ORDER = RAW_SENSORS + NORM_SENSORS + ENGTYPE_COLS + TIME_FEATS + PHASE_FEATS
N_FEATURES = len(FEATURE_ORDER)

MODEL_PATHS = {
    "lstm": "best_lstm2_model.keras",
    "gru":  "best_gru_model.keras",
    "cnn":  "best_cnn_model.keras",
}

# ---------------------------------------------------------------------------
# Degradation analysis (sensor-driven, no ML model — quick win added 2026-05-06).
#
# Fits a linear slope of each key sensor against flight_cycle over the
# uploaded window, then compares against fleet-wide percentile thresholds
# (computed offline from Data/LEAB_engines_data_cleaned.csv) to classify
# the engine as stable / mild / elevated / severe.
#
# Why this exists alongside the LSTM/GRU/CNN ensemble:
#   - The sequence ensemble's residual target is degenerate (residual = 0
#     for non-overdue points by construction), so the model collapsed.
#   - This sensor-driven analysis has no such degeneracy: the slopes are
#     genuinely sensor-driven and not a function of flight_cycle by formula.
#   - Operationally it's what maintenance teams already track.
# ---------------------------------------------------------------------------
try:
    DEGRADATION_THRESHOLDS = _json.loads(open("degradation_thresholds.json").read())
    DEGRADATION_ENABLED = True
except FileNotFoundError:
    DEGRADATION_THRESHOLDS = None
    DEGRADATION_ENABLED = False

# Trained Random Forest classifier — replaces / augments the rule-based
# percentile classifier. Loaded lazily at module startup so a missing model
# doesn't break the API (we fall back to the rule-based path).
DEGRADATION_RF = None
DEGRADATION_RF_META = None
try:
    import joblib as _joblib
    DEGRADATION_RF = _joblib.load("degradation_rf.pkl")
    DEGRADATION_RF_META = _json.loads(open("degradation_rf_meta.json").read())
except (FileNotFoundError, Exception) as _e:
    DEGRADATION_RF = None


def _classify_percentile(signed_slope: float, sensor_thresholds: dict) -> tuple[str, int]:
    """Return (severity_label, fleet_percentile) for a sensor slope."""
    p25 = sensor_thresholds["p25_signed"]
    p50 = sensor_thresholds["p50_signed"]
    p75 = sensor_thresholds["p75_signed"]
    p90 = sensor_thresholds["p90_signed"]
    if signed_slope <= p50:
        if signed_slope <= p25:
            return "stable", 12
        return "stable", 38
    if signed_slope <= p75:
        return "mild", 62
    if signed_slope <= p90:
        return "elevated", 82
    return "severe", 95


def _compute_degradation(flights_data: list[dict]) -> dict | None:
    """For each tracked sensor, fit slope vs flight_cycle and classify.

    Returns a dict with per-sensor diagnosis + an aggregate index.
    """
    if not DEGRADATION_ENABLED:
        return None
    cycles = np.array([f["flight_cycle"] for f in flights_data], dtype=np.float64)
    if len(cycles) < 5 or np.unique(cycles).size < 3:
        return None
    per_sensor = {}
    fleet_percentiles = []
    for sensor, cfg in DEGRADATION_THRESHOLDS["sensors"].items():
        values = np.array([f[sensor] for f in flights_data], dtype=np.float64)
        if np.isnan(values).any():
            continue
        slope = float(np.polyfit(cycles, values, 1)[0])
        signed = slope * cfg["sign"]
        label, pct = _classify_percentile(signed, cfg)
        per_sensor[sensor] = {
            "slope_per_cycle": round(slope, 6),
            "signed_slope":    round(signed, 6),
            "severity":        label,
            "fleet_percentile": pct,
        }
        fleet_percentiles.append(pct)

    if not fleet_percentiles:
        return None

    # Aggregate: take the max sensor percentile (worst sensor drives the
    # overall verdict) and the mean (composite picture).
    max_pct  = max(fleet_percentiles)
    mean_pct = round(sum(fleet_percentiles) / len(fleet_percentiles))
    if max_pct >= 90:
        overall = "severe"
    elif max_pct >= 75:
        overall = "elevated"
    elif max_pct >= 50:
        overall = "mild"
    else:
        overall = "stable"

    result = {
        "overall_severity": overall,
        "max_sensor_percentile": max_pct,
        "mean_sensor_percentile": mean_pct,
        "per_sensor": per_sensor,
    }

    # If a trained Random Forest is available, also return its prediction +
    # class-probability vector. Trained on 53,963 windows across 49 engines
    # with 99.9% engine-grouped CV accuracy. Gives soft confidence scores
    # instead of hard percentile cutoffs.
    if DEGRADATION_RF is not None and DEGRADATION_RF_META is not None:
        try:
            feat_cols = DEGRADATION_RF_META["feature_columns"]
            class_labels = DEGRADATION_RF_META["class_labels"]
            # Build feature row matching training-time order
            sensors_for_rf = ["egt_probe_average", "fuel_flw", "core_spd", "oil_pres"]
            window_arrs = {s: np.array([f[s] for f in flights_data], dtype=np.float64)
                           for s in sensors_for_rf}
            cycles = np.array([f["flight_cycle"] for f in flights_data], dtype=np.float64)
            row = {}
            for s in sensors_for_rf:
                v = window_arrs[s]
                row[f"{s}_slope"] = float(np.polyfit(cycles, v, 1)[0])
                row[f"{s}_mean"]  = float(v.mean())
                row[f"{s}_std"]   = float(v.std())
            row["latest_flight_cycle"] = float(cycles[-1])
            row["engtype_a26"]   = int(flights_data[-1].get("engtype_LEAP-1A26", 0))
            row["engtype_a26e1"] = int(flights_data[-1].get("engtype_LEAP-1A26E1", 0))
            X_rf = np.array([[row[c] for c in feat_cols]], dtype=np.float64)
            preds = DEGRADATION_RF.predict(X_rf)[0]
            probs = DEGRADATION_RF.predict_proba(X_rf)[0]
            result["rf"] = {
                "severity":      str(preds),
                "probabilities": {label: float(p) for label, p in zip(DEGRADATION_RF.classes_, probs)},
                "agrees_with_rule": str(preds) == overall,
                "cv_accuracy_train": DEGRADATION_RF_META.get("cv_accuracy_mean"),
            }
        except Exception as _rf_err:
            logger.warning("RF degradation classifier failed: %s", _rf_err)

    return result

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    filename="predictions.log",
    level=logging.INFO,
    format="%(asctime)s - %(message)s",
)
logger = logging.getLogger("api_inference")

# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------
def _load_keras(path: str) -> tf.keras.Model:
    if not os.path.exists(path):
        raise RuntimeError(f"Model file not found: {path}")
    return tf.keras.models.load_model(path, compile=False)

models: dict[str, tf.keras.Model] = {name: _load_keras(p) for name, p in MODEL_PATHS.items()}
logger.info("Loaded models: %s", list(models))

# Each saved model carries its own (window, n_features) input shape. The current
# feature schema (FEATURE_ORDER above) targets 20 features at window=10. If any
# loaded model disagrees, /health and /predict will surface the mismatch instead
# of silently passing a wrong-shaped tensor.
def _model_input_shape(m: tf.keras.Model) -> tuple[int, int]:
    _, w, f = m.input_shape  # (batch, window, n_features)
    return int(w), int(f)

MODEL_SHAPES = {name: _model_input_shape(m) for name, m in models.items()}
SCHEMA_MATCH = {
    name: (w == WINDOW_SIZE and f == N_FEATURES)
    for name, (w, f) in MODEL_SHAPES.items()
}
if not all(SCHEMA_MATCH.values()):
    logger.warning(
        "Schema mismatch: training schema was (window=%d, n_features=%d) but loaded models expect %s. "
        "Retrain via Lstm_PM_py.ipynb to regenerate artifacts that match the current cleaning pipeline.",
        WINDOW_SIZE, N_FEATURES, MODEL_SHAPES,
    )

# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------
app = FastAPI(title="A320NEO Engine RCL Inference", version="2.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Prometheus /metrics endpoint - prometheus.yml scrapes this every 15s.
# Wrapped in try/except so the API still starts if the package isn't
# installed (e.g. minimal dev env without the MLOps deps).
try:
    from prometheus_fastapi_instrumentator import Instrumentator
    Instrumentator().instrument(app).expose(app, endpoint="/metrics", include_in_schema=False)
    logger.info("Prometheus /metrics endpoint enabled.")
except ImportError:
    logger.warning("prometheus-fastapi-instrumentator not installed; /metrics disabled.")

# ---------------------------------------------------------------------------
# Pydantic schemas
# ---------------------------------------------------------------------------
class FlightRow(BaseModel):
    """One flight's worth of preprocessed features.

    Caller is responsible for computing `*_norm` (per-engine baseline drift)
    and the cyclical time encodings beforehand, exactly the way the cleaning
    notebook does. The API does no per-engine state lookup.
    """
    egt_probe_average: float
    fuel_flw: float
    core_spd: float
    oil_temp: float
    oil_pres: float
    pt2_fan_inlet_tot_pres: float = Field(..., alias="pt2_(fan_inlet_tot_pres)")

    egt_probe_average_norm: float
    fuel_flw_norm: float
    core_spd_norm: float
    oil_temp_norm: float
    oil_pres_norm: float
    pt2_fan_inlet_tot_pres_norm: float = Field(..., alias="pt2_(fan_inlet_tot_pres)_norm")

    engtype_LEAP_1A26: int = Field(..., alias="engtype_LEAP-1A26")
    engtype_LEAP_1A26E1: int = Field(..., alias="engtype_LEAP-1A26E1")

    month_sin: float
    month_cos: float
    dayofweek_sin: float
    dayofweek_cos: float

    flight_phase_CRUISE: int
    flight_phase_TAKEOFF: int

    flight_cycle: int  # required for reconstructing absolute RCL from residual

    model_config = {"populate_by_name": True}


class PredictionRequest(BaseModel):
    flights: List[FlightRow] = Field(
        ...,
        description=f"Exactly {WINDOW_SIZE} consecutive flights for one engine, oldest -> newest.",
    )
    model_choice: Literal["lstm", "gru", "cnn", "average"] = "average"

    # New for v2 baseline: per-engine baseline lookup. Either pass `eng_number`
    # (the API resolves it via engine_expected_max.json) or pass an explicit
    # `expected_max_cycle`. If neither is supplied the API falls back to the
    # fleet-median FLEET_MAX_LIFE (the v1 behaviour).
    eng_number: str | None = Field(
        default=None,
        description="Engine serial (e.g. ESN3). Used to look up expected_max_cycle from the per-engine table.",
    )
    expected_max_cycle: float | None = Field(
        default=None,
        description="Override expected_max_cycle directly; useful if the engine isn't in the lookup table yet.",
    )

    @field_validator("flights")
    @classmethod
    def _check_window(cls, v: List[FlightRow]) -> List[FlightRow]:
        if len(v) != WINDOW_SIZE:
            raise ValueError(f"flights must contain exactly {WINDOW_SIZE} rows (got {len(v)}).")
        return v


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------
@app.get("/health")
async def health_check():
    schema_ok = all(SCHEMA_MATCH.values())
    return JSONResponse(content={
        "status": "healthy" if schema_ok else "schema_mismatch",
        "models_loaded": list(models),
        "model_shapes": {n: {"window": w, "n_features": f} for n, (w, f) in MODEL_SHAPES.items()},
        "schema_match": SCHEMA_MATCH,
        "expected_window_size": WINDOW_SIZE,
        "expected_n_features": N_FEATURES,
        "fleet_max_life": FLEET_MAX_LIFE,
        "residual_scaler": {
            "min": RESIDUAL_MIN,
            "max": RESIDUAL_MAX,
            "loaded_from_disk": _RESIDUAL_RANGE != 1.0,
        },
        "note": (
            None if (schema_ok and _RESIDUAL_RANGE != 1.0) else
            (
                "Loaded model artifacts predate the current feature schema. Retrain via "
                "Lstm_PM_py.ipynb on Data/LEAB_engines_data_cleaned.csv to regenerate them."
                if not schema_ok else
                "rul_scaler.json missing - residuals are not being inverse-transformed; "
                "predictions will be wrong. Run `python _export_rul_scaler.py` to fix."
            )
        ),
    })


@app.get("/schema")
async def get_schema():
    """Expose the feature ordering and window size so clients can match training-time layout."""
    return {
        "window_size": WINDOW_SIZE,
        "feature_order": FEATURE_ORDER,
        "n_features": N_FEATURES,
        "fleet_max_life": FLEET_MAX_LIFE,
        "model_choices": list(models) + ["average"],
    }


def _request_to_tensor(req: PredictionRequest) -> tuple[np.ndarray, int, list[dict]]:
    """Turn a PredictionRequest into (X, latest_flight_cycle, raw_flight_dicts)."""
    rows = []
    raw_dicts = []
    for f in req.flights:
        d = f.model_dump(by_alias=True)
        rows.append([d[col] for col in FEATURE_ORDER])
        raw_dicts.append(d)
    arr = np.asarray(rows, dtype=np.float32)
    if arr.shape != (WINDOW_SIZE, N_FEATURES):
        raise ValueError(f"Built tensor shape {arr.shape}, expected ({WINDOW_SIZE}, {N_FEATURES}).")
    latest_cycle = int(req.flights[-1].flight_cycle)
    return arr[np.newaxis, ...], latest_cycle, raw_dicts


def _predict_residual(model: tf.keras.Model, X: np.ndarray) -> float:
    """Run the model and inverse-transform the [0, 1] output to cycles."""
    raw = float(model.predict(X, verbose=0).ravel()[0])
    return _inverse_scale_residual(raw)


@app.post("/predict")
async def predict(req: PredictionRequest):
    """Predict Remaining Cycle Life (RCL) for an engine from its last `WINDOW_SIZE` flights.

    The Keras models output a *residual* relative to the fleet baseline:
        predicted_RCL = (FLEET_MAX_LIFE - latest_flight_cycle) + model_residual

    `model_choice = "average"` averages the residuals from LSTM, GRU, and CNN.
    """
    try:
        # Hard-stop if the requested model's saved shape disagrees with the
        # current feature schema; otherwise the call would fail with an opaque
        # TensorFlow shape error from deep inside Keras.
        targets = list(models) if req.model_choice == "average" else [req.model_choice]
        bad = [t for t in targets if not SCHEMA_MATCH.get(t, False)]
        if bad:
            details = {t: MODEL_SHAPES[t] for t in bad}
            raise ValueError(
                f"Model artifacts {bad} were trained on an older feature schema (shapes={details}); "
                f"current schema is window={WINDOW_SIZE}, n_features={N_FEATURES}. "
                "Retrain by running Lstm_PM_py.ipynb end-to-end on the current "
                "Data/LEAB_engines_data_cleaned.csv to regenerate the .keras files."
            )

        X, latest_cycle, raw_flights = _request_to_tensor(req)
        baseline_rcl, baseline_source = _engine_baseline(
            req.eng_number, req.expected_max_cycle, latest_cycle
        )
        degradation = _compute_degradation(raw_flights)

        if req.model_choice == "average":
            residuals = {name: _predict_residual(m, X) for name, m in models.items()}
            residual = float(np.mean(list(residuals.values())))
        else:
            if req.model_choice not in models:
                raise ValueError(f"Unknown model_choice: {req.model_choice}")
            residuals = {req.model_choice: _predict_residual(models[req.model_choice], X)}
            residual = residuals[req.model_choice]

        rcl = baseline_rcl + residual
        # RCL cannot physically be negative; clip and warn.
        warnings = []
        if rcl < 0:
            warnings.append(f"Raw prediction {rcl:.1f} was negative; clipped to 0. "
                            "Engine has likely already exceeded its expected life.")
            rcl = 0.0

        logger.info(
            "model=%s flight_cycle=%d baseline=%.1f (%s) residual=%.1f rcl=%.1f",
            req.model_choice, latest_cycle, baseline_rcl, baseline_source, residual, rcl,
        )

        return {
            "RCL_prediction": int(round(rcl)),
            "model_choice": req.model_choice,
            "baseline_rcl": baseline_rcl,
            "baseline_source": baseline_source,
            "residual": residual,
            "per_model_residuals": residuals,
            "degradation": degradation,
            "warnings": warnings,
        }
    except ValueError as ve:
        logger.error("Validation error: %s", ve)
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        logger.exception("Inference failure")
        raise HTTPException(status_code=500, detail=str(e))
