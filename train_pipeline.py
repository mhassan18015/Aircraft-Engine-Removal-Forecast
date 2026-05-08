"""Training pipeline for the LEAP-1A engine RCL ensemble.

Replaces the previous tree-ensemble pipeline (RF/XGBoost/Extra Trees on the
pre-LEAP dataset, those notebooks/artifacts now live in archive/legacy/).

This script is the canonical retrain entry point - the Airflow DAG calls it
via subprocess on a weekly schedule and after drift is detected. It:

  1. Executes Lstm_PM_py.ipynb end-to-end via nbconvert against the current
     Data/LEAB_engines_data_cleaned.csv (49-engine combined fleet).
  2. Reads the resulting training history from best_lstm2_history.json.
  3. Logs hyperparameters, per-epoch losses, and final artifacts to MLflow.
  4. Returns exit code 0 on success, non-zero on any failure (so Airflow's
     PythonOperator fails its task and triggers retries).

Why "execute the notebook" instead of duplicating training code here:
  - The notebook is the source of truth for architecture and split logic.
  - Architectural tweaks land in one place (the notebook); this script picks
     them up automatically on the next run.
  - Removes ~300 lines of duplicated Keras code from this file.

Environment variables:
  MLFLOW_TRACKING_URI       (default: http://localhost:5000)
  MLFLOW_EXPERIMENT_NAME    (default: Engine_Removal_Forecast)
  TRAIN_NOTEBOOK            (default: Lstm_PM_py.ipynb)
  TRAIN_TIMEOUT_SECONDS     (default: 5400 - 90 minutes for CPU runs)
  SKIP_TRAINING             (default: 0; set to 1 to skip notebook execution
                             and only re-log the existing on-disk artifacts -
                             useful for verifying MLflow plumbing without
                             paying the training cost.)
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

import mlflow

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
PROJECT_ROOT     = Path(__file__).resolve().parent
NOTEBOOK_PATH    = PROJECT_ROOT / os.getenv("TRAIN_NOTEBOOK", "Lstm_PM_py.ipynb")
HISTORY_PATH     = PROJECT_ROOT / "best_lstm2_history.json"
DATA_PATH        = PROJECT_ROOT / "Data" / "LEAB_engines_data_cleaned.csv"

ARTIFACT_FILES = [
    "best_lstm2_model.keras",
    "best_gru_model.keras",
    "best_cnn_model.keras",
    "best_lstm2_model_all.keras",
    "best_lstm2_history.json",
]

TRAIN_TIMEOUT = int(os.getenv("TRAIN_TIMEOUT_SECONDS", "5400"))
SKIP_TRAINING = os.getenv("SKIP_TRAINING", "0") == "1"


def _configure_mlflow() -> None:
    """Idempotent MLflow setup. Deferred from module import so `import
    train_pipeline` is side-effect-free (lets tests load the module without
    requiring an MLflow server on localhost:5000)."""
    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000"))
    mlflow.set_experiment(os.getenv("MLFLOW_EXPERIMENT_NAME", "Engine_Removal_Forecast"))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _run_notebook() -> None:
    """Execute the LSTM notebook in-place via nbconvert."""
    if not NOTEBOOK_PATH.exists():
        raise FileNotFoundError(f"Training notebook not found: {NOTEBOOK_PATH}")
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Cleaned dataset not found: {DATA_PATH}. "
            "Run engine_removal_forecast_cleaning.ipynb first."
        )
    cmd = [
        sys.executable, "-m", "nbconvert",
        "--to", "notebook",
        "--execute", str(NOTEBOOK_PATH),
        "--output", str(NOTEBOOK_PATH.name),
        "--output-dir", str(NOTEBOOK_PATH.parent),
        f"--ExecutePreprocessor.timeout={TRAIN_TIMEOUT}",
    ]
    print(f"[train_pipeline] executing: {' '.join(cmd)}", flush=True)
    t0 = time.time()
    res = subprocess.run(cmd, cwd=str(PROJECT_ROOT))
    dt = time.time() - t0
    if res.returncode != 0:
        raise RuntimeError(
            f"Notebook execution failed (exit {res.returncode}) after {dt:.0f}s. "
            "Inspect the notebook in-place for the cell that errored."
        )
    print(f"[train_pipeline] notebook finished in {dt:.0f}s.", flush=True)


def _load_history() -> dict:
    if not HISTORY_PATH.exists():
        raise FileNotFoundError(
            f"Expected history file not found: {HISTORY_PATH}. "
            "The training cell may not have completed."
        )
    with open(HISTORY_PATH) as fh:
        return json.load(fh)


def _summary_metrics(history: dict) -> dict:
    """Squash a Keras-style history dict into a few headline metrics."""
    metrics: dict[str, float] = {}
    if "loss" in history:
        metrics["final_train_loss"] = float(history["loss"][-1])
        metrics["min_train_loss"]   = float(min(history["loss"]))
    if "val_loss" in history:
        vl = history["val_loss"]
        metrics["final_val_loss"] = float(vl[-1])
        metrics["best_val_loss"]  = float(min(vl))
        metrics["best_val_epoch"] = float(vl.index(min(vl)) + 1)
        metrics["epochs_run"]     = float(len(vl))
    return metrics


def _log_artifacts(run) -> list[str]:
    """Log all training artifacts that exist in the project root."""
    logged: list[str] = []
    for name in ARTIFACT_FILES:
        p = PROJECT_ROOT / name
        if p.exists():
            artifact_path = "models" if name.endswith(".keras") else None
            mlflow.log_artifact(str(p), artifact_path=artifact_path)
            logged.append(name)
    return logged


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> int:
    _configure_mlflow()
    with mlflow.start_run() as run:
        mlflow.log_param("notebook", NOTEBOOK_PATH.name)
        mlflow.log_param("data_path", str(DATA_PATH.relative_to(PROJECT_ROOT)))
        mlflow.log_param("skip_training", SKIP_TRAINING)
        mlflow.log_param("train_timeout_seconds", TRAIN_TIMEOUT)

        if SKIP_TRAINING:
            print("[train_pipeline] SKIP_TRAINING=1; logging existing artifacts only.")
        else:
            _run_notebook()

        history = _load_history()
        metrics = _summary_metrics(history)
        for k, v in metrics.items():
            mlflow.log_metric(k, v)
        print(f"[train_pipeline] summary metrics: {metrics}")

        # Log per-epoch curves so MLflow renders them nicely.
        for epoch, val in enumerate(history.get("loss", []), start=1):
            mlflow.log_metric("train_loss", float(val), step=epoch)
        for epoch, val in enumerate(history.get("val_loss", []), start=1):
            mlflow.log_metric("val_loss", float(val), step=epoch)

        logged = _log_artifacts(run)
        mlflow.log_param("artifacts_logged", ",".join(logged))
        print(f"[train_pipeline] logged artifacts: {logged}")

        # Marker file so retrain_dag.register_model can pick the run up.
        marker = PROJECT_ROOT / "last_train_run.json"
        marker.write_text(json.dumps({
            "run_id":         run.info.run_id,
            "experiment_id":  run.info.experiment_id,
            "metrics":        metrics,
            "artifacts":      logged,
        }, indent=2))
        print(f"[train_pipeline] wrote {marker.name} (run_id={run.info.run_id}).")

    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as e:
        print(f"[train_pipeline] FAILED: {e}", file=sys.stderr)
        sys.exit(1)
