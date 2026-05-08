"""Data-drift detection for the LEAP-1A RCL pipeline.

Targets Evidently 0.7+ (the major API after the 0.4 -> 0.7 rewrite). The
previous implementation read `report_data["metrics"]["data_drift"]
["data_drift_detected"]` against the older nested-dict format; the new
metric_v2 API surfaces drift via DriftedColumnsCount, which we read from
`snapshot.dict()["metrics"][0]["value"]`.

The function has two callable shapes:

  check_data_drift(reference_df, current_df)  -> bool
      Direct two-argument form used by tests.

  check_data_drift()  -> bool
      Zero-argument form used by the Airflow DAG. Reads paths from env vars
      (DRIFT_REFERENCE_CSV, DRIFT_CURRENT_CSV); when DRIFT_CURRENT_CSV is
      unset it returns False (no drift) so the DAG simply skips retraining.
"""
from __future__ import annotations

import logging
import os
from pathlib import Path

import pandas as pd
from evidently import Report
from evidently.presets import DataDriftPreset

PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_REFERENCE = PROJECT_ROOT / "Data" / "LEAB_engines_data_cleaned.csv"

# Columns the LSTM/GRU/CNN ensemble actually consumes - we only care about
# drift in features the model uses, not in metadata columns.
DRIFT_FEATURES = [
    "egt_probe_average", "fuel_flw", "core_spd",
    "oil_temp", "oil_pres", "pt2_(fan_inlet_tot_pres)",
]

DRIFT_SHARE_THRESHOLD = float(os.getenv("DRIFT_SHARE_THRESHOLD", "0.5"))

log = logging.getLogger(__name__)


def _select_features(df: pd.DataFrame) -> pd.DataFrame:
    """Keep only the columns shared between the LSTM feature set and the input.

    If none of the canonical features are present (e.g. the test's mock data
    uses arbitrary names like 'feature1'), pass the dataframe through so the
    caller's columns are compared as-is.
    """
    overlap = [c for c in DRIFT_FEATURES if c in df.columns]
    return df[overlap] if overlap else df


def check_data_drift(
    reference_data: pd.DataFrame | None = None,
    current_data: pd.DataFrame | None = None,
) -> bool:
    """Return True iff drift is detected against the reference distribution.

    `True` is returned when the share of drifted columns exceeds
    DRIFT_SHARE_THRESHOLD (default 0.5 == half the columns drifted).
    """
    # Both None: zero-arg DAG path.
    if reference_data is None and current_data is None:
        ref_path = Path(os.getenv("DRIFT_REFERENCE_CSV", str(DEFAULT_REFERENCE)))
        cur_path_str = os.getenv("DRIFT_CURRENT_CSV")
        if not cur_path_str:
            log.info("DRIFT_CURRENT_CSV not set; assuming no drift (DAG will skip retrain).")
            return False
        cur_path = Path(cur_path_str)
        if not ref_path.exists():
            raise FileNotFoundError(f"Reference dataset missing: {ref_path}")
        if not cur_path.exists():
            raise FileNotFoundError(f"Current dataset missing: {cur_path}")
        reference_data = pd.read_csv(ref_path)
        current_data   = pd.read_csv(cur_path)

    # One-but-not-the-other or explicit Nones: still illegal.
    if reference_data is None or current_data is None:
        raise ValueError("Both reference_data and current_data must be provided.")

    if not isinstance(reference_data, pd.DataFrame) or not isinstance(current_data, pd.DataFrame):
        raise ValueError("reference_data and current_data must be pandas DataFrames.")
    if reference_data.empty or current_data.empty:
        raise ValueError("reference_data and current_data must be non-empty.")

    ref = _select_features(reference_data)
    cur = _select_features(current_data)
    common = [c for c in ref.columns if c in cur.columns]
    if not common:
        raise ValueError("reference_data and current_data share no columns.")
    ref = ref[common]
    cur = cur[common]

    report = Report(metrics=[DataDriftPreset(drift_share=DRIFT_SHARE_THRESHOLD)])
    snapshot = report.run(reference_data=ref, current_data=cur)
    snap = snapshot.dict()

    # Find DriftedColumnsCount in the metric list (it carries `share` and `count`).
    for m in snap.get("metrics", []):
        cfg = m.get("config", {})
        if cfg.get("type", "").endswith("DriftedColumnsCount"):
            share = float(m["value"]["share"])
            count = int(m["value"]["count"])
            log.info(
                "drift check: %d/%d columns drifted (share=%.2f, threshold=%.2f)",
                count, len(common), share, DRIFT_SHARE_THRESHOLD,
            )
            return share > DRIFT_SHARE_THRESHOLD

    # No DriftedColumnsCount metric -> conservative default.
    log.warning("DriftedColumnsCount missing from snapshot; defaulting to no drift.")
    return False
