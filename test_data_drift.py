"""Drift-detection tests targeting the Evidently 0.7+ implementation."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from data_drift import check_data_drift


def test_check_data_drift_no_drift():
    """Identical reference and current data => no drift."""
    rng = np.random.default_rng(42)
    ref = pd.DataFrame({
        "f1": rng.normal(0, 1, 1000),
        "f2": rng.normal(5, 2, 1000),
    })
    assert check_data_drift(ref, ref.copy()) is False


def test_check_data_drift_with_mock_data():
    """Both columns shifted by ~3 sigma => drift on >= 50% of columns."""
    rng = np.random.default_rng(0)
    ref = pd.DataFrame({
        "f1": rng.normal(0, 1, 1000),
        "f2": rng.normal(5, 2, 1000),
    })
    cur = pd.DataFrame({
        "f1": rng.normal(3, 1, 1000),  # shifted
        "f2": rng.normal(8, 2, 1000),  # shifted
    })
    assert check_data_drift(ref, cur) is True


def test_check_data_drift_invalid_input():
    """Passing None for either side raises ValueError."""
    with pytest.raises(ValueError):
        check_data_drift(None, pd.DataFrame({"a": [1, 2, 3]}))
    with pytest.raises(ValueError):
        check_data_drift(pd.DataFrame({"a": [1, 2, 3]}), None)


def test_check_data_drift_empty_input():
    """Empty dataframes raise ValueError."""
    with pytest.raises(ValueError):
        check_data_drift(pd.DataFrame(), pd.DataFrame())


def test_check_data_drift_no_common_columns():
    """No shared columns between reference and current raises ValueError."""
    ref = pd.DataFrame({"a": [1, 2, 3]})
    cur = pd.DataFrame({"b": [1, 2, 3]})
    with pytest.raises(ValueError):
        check_data_drift(ref, cur)


def test_zero_arg_path_no_env_var(monkeypatch):
    """Zero-arg call with DRIFT_CURRENT_CSV unset returns False (DAG skips retrain)."""
    monkeypatch.delenv("DRIFT_CURRENT_CSV", raising=False)
    assert check_data_drift() is False
