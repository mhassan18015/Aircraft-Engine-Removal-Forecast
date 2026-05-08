"""Tests for the sequence-based FastAPI inference service.

Targets api_inference.py v2 (LSTM/GRU/CNN ensemble served from the 49-engine
combined-fleet pipeline). Replaces the previous tree-ensemble single-row tests.

The tests load a real 40-flight window from Data/LEAB_engines_data_cleaned.csv
so the assertions exercise the full Pydantic schema + sequence-tensor path,
not just stub data.
"""
from __future__ import annotations

import warnings

warnings.filterwarnings("ignore")
import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

import api_inference
from api_inference import app

PROJECT = Path(__file__).resolve().parent
CSV = PROJECT / "Data" / "LEAB_engines_data_cleaned.csv"

client = TestClient(app)


# --------------------------------------------------------------------------- #
# Fixtures
# --------------------------------------------------------------------------- #
@pytest.fixture(scope="module")
def real_window():
    """Build one engine's last WINDOW_SIZE rows into the FlightRow JSON shape."""
    if not CSV.exists():
        pytest.skip(f"Test fixture missing: {CSV}")

    df = pd.read_csv(CSV, parse_dates=["flight_datetime_c"])
    df["dayofweek"]    = df["flight_datetime_c"].dt.dayofweek
    df["month"]        = df["flight_datetime_c"].dt.month
    df["dayofweek_sin"] = np.sin(2 * np.pi * df["dayofweek"] / 7)
    df["dayofweek_cos"] = np.cos(2 * np.pi * df["dayofweek"] / 7)
    df["month_sin"]    = np.sin(2 * np.pi * df["month"] / 12)
    df["month_cos"]    = np.cos(2 * np.pi * df["month"] / 12)
    df = pd.get_dummies(df, columns=["flight_phase"])
    for c in ["flight_phase_CRUISE", "flight_phase_TAKEOFF"]:
        if c in df.columns:
            df[c] = df[c].astype(int)

    engine = df["esn"].unique()[0]
    sub = (
        df[df["esn"] == engine]
        .sort_values("flight_cycle")
        .tail(api_inference.WINDOW_SIZE)
    )
    if len(sub) < api_inference.WINDOW_SIZE:
        pytest.skip(f"Engine {engine} has only {len(sub)} rows; need {api_inference.WINDOW_SIZE}.")

    flights = []
    for _, row in sub.iterrows():
        flights.append(
            {
                "egt_probe_average":  float(row["egt_probe_average"]),
                "fuel_flw":           float(row["fuel_flw"]),
                "core_spd":           float(row["core_spd"]),
                "oil_temp":           float(row["oil_temp"]),
                "oil_pres":           float(row["oil_pres"]),
                "pt2_(fan_inlet_tot_pres)":      float(row["pt2_(fan_inlet_tot_pres)"]),
                "egt_probe_average_norm":        float(row["egt_probe_average_norm"]),
                "fuel_flw_norm":                 float(row["fuel_flw_norm"]),
                "core_spd_norm":                 float(row["core_spd_norm"]),
                "oil_temp_norm":                 float(row["oil_temp_norm"]),
                "oil_pres_norm":                 float(row["oil_pres_norm"]),
                "pt2_(fan_inlet_tot_pres)_norm": float(row["pt2_(fan_inlet_tot_pres)_norm"]),
                "engtype_LEAP-1A26":   int(row.get("engtype_LEAP-1A26", 0)),
                "engtype_LEAP-1A26E1": int(row.get("engtype_LEAP-1A26E1", 0)),
                "month_sin":     float(row["month_sin"]),
                "month_cos":     float(row["month_cos"]),
                "dayofweek_sin": float(row["dayofweek_sin"]),
                "dayofweek_cos": float(row["dayofweek_cos"]),
                "flight_phase_CRUISE":  int(row.get("flight_phase_CRUISE", 0)),
                "flight_phase_TAKEOFF": int(row.get("flight_phase_TAKEOFF", 0)),
                "flight_cycle": int(row["flight_cycle"]),
            }
        )
    return flights


# --------------------------------------------------------------------------- #
# /health and /schema
# --------------------------------------------------------------------------- #
def test_health_status_healthy():
    """Models on disk match the current feature schema."""
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "healthy", body.get("note")
    assert set(body["models_loaded"]) == {"lstm", "gru", "cnn"}
    assert all(body["schema_match"].values())


def test_health_reports_window_and_features():
    body = client.get("/health").json()
    assert body["expected_window_size"] == api_inference.WINDOW_SIZE
    assert body["expected_n_features"] == api_inference.N_FEATURES
    for name, shape in body["model_shapes"].items():
        assert shape["window"] == api_inference.WINDOW_SIZE
        assert shape["n_features"] == api_inference.N_FEATURES


def test_schema_endpoint_lists_features():
    r = client.get("/schema")
    assert r.status_code == 200
    body = r.json()
    assert body["window_size"] == api_inference.WINDOW_SIZE
    assert body["n_features"] == api_inference.N_FEATURES
    assert len(body["feature_order"]) == api_inference.N_FEATURES
    assert "average" in body["model_choices"]


# --------------------------------------------------------------------------- #
# /predict — happy paths
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("model_choice", ["lstm", "gru", "cnn", "average"])
def test_predict_each_model_choice(real_window, model_choice):
    payload = {"flights": real_window, "model_choice": model_choice}
    r = client.post("/predict", json=payload)
    assert r.status_code == 200, r.text
    body = r.json()
    assert "RCL_prediction" in body
    assert isinstance(body["RCL_prediction"], int)
    assert 0 <= body["RCL_prediction"] <= 10000
    assert body["model_choice"] == model_choice
    assert "baseline_rcl" in body and "residual" in body
    if model_choice == "average":
        assert set(body["per_model_residuals"]) == {"lstm", "gru", "cnn"}
    else:
        assert set(body["per_model_residuals"]) == {model_choice}


def test_predict_returns_warning_when_overdue(real_window):
    """If the residual pushes the prediction below zero we expect a clipped
    result + a warning string, not a crash."""
    # Force a high latest_flight_cycle so baseline_rcl is small/negative.
    payload = {"flights": real_window, "model_choice": "average"}
    payload["flights"][-1]["flight_cycle"] = 99999  # past any plausible engine life
    r = client.post("/predict", json=payload)
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["RCL_prediction"] >= 0
    if body["RCL_prediction"] == 0:
        assert any("clipped to 0" in w for w in body.get("warnings", []))


# --------------------------------------------------------------------------- #
# /predict — validation errors
# --------------------------------------------------------------------------- #
def test_predict_wrong_window_size_rejected(real_window):
    payload = {"flights": real_window[:-1], "model_choice": "lstm"}
    r = client.post("/predict", json=payload)
    assert r.status_code == 422
    assert "exactly" in r.text.lower()


def test_predict_unknown_model_choice_rejected(real_window):
    payload = {"flights": real_window, "model_choice": "totally_made_up"}
    r = client.post("/predict", json=payload)
    assert r.status_code == 422  # Pydantic Literal validation


def test_predict_missing_required_field_rejected(real_window):
    bad = [dict(f) for f in real_window]
    del bad[0]["egt_probe_average"]  # required by FlightRow
    r = client.post("/predict", json={"flights": bad, "model_choice": "lstm"})
    assert r.status_code == 422


# --------------------------------------------------------------------------- #
# Module-level imports & schema invariants
# --------------------------------------------------------------------------- #
def test_feature_order_size_matches_n_features():
    assert len(api_inference.FEATURE_ORDER) == api_inference.N_FEATURES


def test_three_models_loaded_at_import_time():
    assert set(api_inference.models) == {"lstm", "gru", "cnn"}
