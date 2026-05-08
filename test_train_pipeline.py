"""Smoke tests for the training pipeline.

Does NOT execute Lstm_PM_py.ipynb (that takes 25-35 minutes on T4 and would
make the test suite unrunnable in CI). Instead it asserts the pipeline:
  - imports cleanly
  - reads the canonical CSV
  - finds the expected on-disk artifacts
  - logs an MLflow run via the SKIP_TRAINING=1 path

For a real end-to-end retrain run train_pipeline.py manually, or run it under
Airflow.
"""
from __future__ import annotations

import importlib
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

PROJECT = Path(__file__).resolve().parent
SCRIPT  = PROJECT / "train_pipeline.py"
CSV     = PROJECT / "Data" / "LEAB_engines_data_cleaned.csv"
NB      = PROJECT / "Lstm_PM_py.ipynb"


def test_train_script_exists():
    assert SCRIPT.exists(), f"train_pipeline.py missing at {SCRIPT}"


def test_canonical_inputs_present():
    """The script's required inputs must be on disk."""
    assert CSV.exists(),  f"Cleaned CSV missing: {CSV}"
    assert NB.exists(),   f"LSTM notebook missing: {NB}"


def test_train_script_imports_cleanly():
    """Source must parse and the dependencies must be installed."""
    if "train_pipeline" in sys.modules:
        del sys.modules["train_pipeline"]
    # Avoid actually executing main() at import time.
    spec = importlib.util.spec_from_file_location("train_pipeline", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    assert callable(mod.main)
    assert mod.NOTEBOOK_PATH.name == "Lstm_PM_py.ipynb"
    assert "LEAB_engines_data_cleaned.csv" in str(mod.DATA_PATH)


def test_skip_training_path_logs_mlflow_run(tmp_path):
    """SKIP_TRAINING=1 should run the pipeline end-to-end without retraining,
    log artifacts to a local file-store MLflow tracking dir, and write the
    last_train_run.json marker."""
    env = {
        **os.environ,
        "SKIP_TRAINING": "1",
        "MLFLOW_TRACKING_URI": f"file:{tmp_path / 'mlruns'}",
        "MLFLOW_EXPERIMENT_NAME": "PM_Test_Smoke",
        # Belt-and-braces: keep TF quiet.
        "TF_CPP_MIN_LOG_LEVEL": "2",
    }
    res = subprocess.run(
        [sys.executable, str(SCRIPT)],
        cwd=str(PROJECT),
        env=env,
        capture_output=True,
        text=True,
        timeout=300,
    )
    assert res.returncode == 0, f"train_pipeline.py exited {res.returncode}\nstdout:\n{res.stdout}\nstderr:\n{res.stderr}"

    marker = PROJECT / "last_train_run.json"
    assert marker.exists(), "Expected last_train_run.json marker not written."
    info = json.loads(marker.read_text())
    assert info["run_id"]
    assert "best_lstm2_history.json" in info["artifacts"]
    # Metrics dict should at least contain the val/train summary keys we log.
    expected_keys = {"final_train_loss", "best_val_loss", "best_val_epoch", "epochs_run"}
    assert expected_keys.issubset(info["metrics"].keys())
