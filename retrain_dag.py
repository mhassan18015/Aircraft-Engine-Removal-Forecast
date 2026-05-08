"""Airflow DAG: weekly drift-driven retrain of the LEAP-1A RCL ensemble.

Pipeline:
    check_drift_and_retrain --> retrain_model --> register_model --> notify

`train_pipeline.py` is the canonical retrain script. It executes
Lstm_PM_py.ipynb, logs metrics/artifacts to MLflow, and writes
`last_train_run.json` that downstream tasks consume.
"""
from __future__ import annotations

import json
import logging
import os
import subprocess
from datetime import datetime, timedelta
from pathlib import Path

import mlflow
from airflow import DAG
from airflow.operators.python import PythonOperator

from data_drift import check_data_drift

PROJECT_ROOT = Path(__file__).resolve().parent
TRAIN_SCRIPT = PROJECT_ROOT / "train_pipeline.py"
MARKER_PATH  = PROJECT_ROOT / "last_train_run.json"

REGISTERED_MODEL_NAME = os.getenv("MODEL_REGISTRY_NAME", "leap1a_rcl_ensemble")
MLFLOW_TRACKING_URI   = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")

log = logging.getLogger(__name__)


def _run_train_pipeline() -> None:
    """Invoke train_pipeline.py and propagate its exit code."""
    cmd = ["python", str(TRAIN_SCRIPT)]
    log.info("Running %s", " ".join(cmd))
    res = subprocess.run(cmd, cwd=str(PROJECT_ROOT))
    if res.returncode != 0:
        raise RuntimeError(f"train_pipeline.py exited {res.returncode}")


def _read_marker() -> dict:
    if not MARKER_PATH.exists():
        raise FileNotFoundError(f"Marker file missing: {MARKER_PATH}. Did train_pipeline.py finish?")
    return json.loads(MARKER_PATH.read_text())


# --- Task callables -------------------------------------------------------

def check_drift_and_retrain():
    """Decide whether to retrain. Skips downstream cleanly when no drift."""
    if check_data_drift():
        log.info("Drift detected - triggering retrain.")
        _run_train_pipeline()
    else:
        log.info("No data drift detected. Retraining skipped.")
        # The pipeline didn't run, so erase any stale marker to avoid
        # register_model picking up an old run as if it were fresh.
        if MARKER_PATH.exists():
            MARKER_PATH.unlink()


def retrain_model():
    """Unconditional retrain entry point (e.g. for the weekly cadence)."""
    _run_train_pipeline()


def register_model():
    """Promote the freshly-logged MLflow artifacts to the Model Registry."""
    if not MARKER_PATH.exists():
        log.info("No marker file - nothing to register (likely no drift this run).")
        return
    info = _read_marker()
    run_id = info["run_id"]
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    log.info("Registering models from run %s under '%s'", run_id, REGISTERED_MODEL_NAME)

    for artifact in info.get("artifacts", []):
        if not artifact.endswith(".keras"):
            continue
        # MLflow artifacts of .keras files are logged under "models/" by train_pipeline.py.
        source = f"runs:/{run_id}/models/{artifact}"
        # Each architecture gets its own registered name to keep promotion granular.
        registered_name = f"{REGISTERED_MODEL_NAME}__{artifact.replace('.keras','')}"
        try:
            mlflow.register_model(source, registered_name)
            log.info("Registered %s -> %s", artifact, registered_name)
        except Exception as e:
            log.warning("Failed to register %s: %s", artifact, e)


def notify():
    """Hook for Slack/email notifications - currently logs the summary line."""
    if not MARKER_PATH.exists():
        log.info("No marker file - no retraining happened, nothing to notify about.")
        return
    info = _read_marker()
    msg = (
        f"[retrain_model_dag] run_id={info['run_id']} "
        f"best_val_loss={info['metrics'].get('best_val_loss', 'n/a'):.5f} "
        f"@ epoch {int(info['metrics'].get('best_val_epoch', 0))}"
    )
    log.info(msg)
    print(msg)


# --- DAG definition --------------------------------------------------------

default_args = {
    "owner": "airflow",
    "depends_on_past": False,
    "email_on_failure": False,
    "email_on_retry": False,
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
}

dag = DAG(
    "retrain_model_dag",
    default_args=default_args,
    description="Weekly drift-gated retrain of the LEAP-1A RCL ensemble",
    schedule=timedelta(days=7),
    start_date=datetime(2026, 1, 1),
    catchup=False,
    tags=["leap1a", "rcl", "engine-removal-forecast"],
)

check_drift_task = PythonOperator(
    task_id="check_drift_and_retrain",
    python_callable=check_drift_and_retrain,
    dag=dag,
)
register_task = PythonOperator(
    task_id="register_model",
    python_callable=register_model,
    dag=dag,
)
notify_task = PythonOperator(
    task_id="notify",
    python_callable=notify,
    dag=dag,
    trigger_rule="all_done",  # always notify, even if upstream failed
)

check_drift_task >> register_task >> notify_task
