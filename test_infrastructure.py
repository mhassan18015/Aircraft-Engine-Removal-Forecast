"""Infrastructure smoke tests: prometheus.yml, requirements.txt, and the
API's /metrics endpoint.

These don't actually launch a Prometheus daemon (which would need root + 1+ GB
downloads in CI). They lint the configs and exercise the parts of the runtime
that end-users see: that the API exposes /metrics, that prometheus.yml parses
to YAML and points at the right endpoint, and that requirements.txt covers
everything the API imports at boot.
"""
from __future__ import annotations

import re
import warnings
from pathlib import Path

import pytest

warnings.filterwarnings("ignore")
import os  # noqa: E402

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

from fastapi.testclient import TestClient  # noqa: E402

PROJECT = Path(__file__).resolve().parent

PROM_YML     = PROJECT / "prometheus.yml"
REQ_TXT      = PROJECT / "requirements.txt"


# --------------------------------------------------------------------------- #
# requirements.txt
# --------------------------------------------------------------------------- #
def test_requirements_lists_every_runtime_import():
    """Every package the API imports at module load must be in requirements.txt.

    Streamlit-Cloud deployment uses a slim requirements set; uvicorn and
    prometheus-fastapi-instrumentator are launch-time / optional deps and
    aren't required for the API module to import or for /predict to work.
    """
    text = REQ_TXT.read_text(encoding="utf-8")
    required = [
        "fastapi", "pydantic",
        "tensorflow", "numpy", "pandas",
        "scikit-learn", "joblib",
    ]
    missing = [p for p in required if not re.search(rf"^\s*{re.escape(p)}\b", text, re.MULTILINE | re.IGNORECASE)]
    assert not missing, f"requirements.txt is missing: {missing}"


# --------------------------------------------------------------------------- #
# prometheus.yml
# --------------------------------------------------------------------------- #
def test_prometheus_yml_parses_and_targets_metrics():
    yaml = pytest.importorskip("yaml", reason="PyYAML required for prometheus.yml lint")
    cfg = yaml.safe_load(PROM_YML.read_text(encoding="utf-8"))
    assert "scrape_configs" in cfg
    fastapi_jobs = [j for j in cfg["scrape_configs"] if "fastapi" in j["job_name"].lower()]
    assert fastapi_jobs, "No FastAPI scrape job defined."
    job = fastapi_jobs[0]
    assert job.get("metrics_path", "/metrics") == "/metrics"
    targets = [t for sc in job["static_configs"] for t in sc["targets"]]
    assert any(":8000" in t or ":8001" in t for t in targets), \
        f"FastAPI scrape job targets unexpected ports: {targets}"


# --------------------------------------------------------------------------- #
# /metrics endpoint
# --------------------------------------------------------------------------- #
def test_metrics_endpoint_returns_prometheus_text():
    """/metrics should be reachable and return Prometheus text format."""
    import api_inference
    client = TestClient(api_inference.app)
    r = client.get("/metrics")
    if r.status_code == 404:
        pytest.skip("prometheus-fastapi-instrumentator not installed; /metrics disabled.")
    assert r.status_code == 200
    body = r.text
    assert "# HELP" in body or "# TYPE" in body, \
        f"/metrics did not return Prometheus text format. First 200 chars:\n{body[:200]}"
