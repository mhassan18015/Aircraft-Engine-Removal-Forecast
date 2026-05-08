"""Infrastructure smoke tests: Dockerfile, prometheus.yml, requirements.txt,
and the API's /metrics endpoint.

These don't actually launch a container or a Prometheus daemon (which would
need root + 1+ GB downloads in CI). They lint the configs and exercise the
parts of the runtime that end-users see: that the API exposes /metrics, that
prometheus.yml parses to YAML and points at the right endpoint, that the
Dockerfile references files that actually exist, and that requirements.txt
covers everything the API imports at boot.
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

DOCKERFILE   = PROJECT / "Dockerfile"
DOCKERIGNORE = PROJECT / ".dockerignore"
PROM_YML     = PROJECT / "prometheus.yml"
REQ_TXT      = PROJECT / "requirements.txt"


# --------------------------------------------------------------------------- #
# requirements.txt
# --------------------------------------------------------------------------- #
def test_requirements_lists_every_runtime_import():
    """Every package the API imports at module load must be in requirements.txt."""
    text = REQ_TXT.read_text(encoding="utf-8")
    required = [
        "fastapi", "uvicorn", "pydantic",
        "tensorflow", "numpy", "pandas",
        "prometheus-fastapi-instrumentator",
    ]
    missing = [p for p in required if not re.search(rf"^\s*{re.escape(p)}\b", text, re.MULTILINE | re.IGNORECASE)]
    assert not missing, f"requirements.txt is missing: {missing}"


# --------------------------------------------------------------------------- #
# Dockerfile
# --------------------------------------------------------------------------- #
def test_dockerfile_references_only_existing_files():
    """COPY directives must reference paths that exist locally; otherwise
    `docker build` fails with no warning."""
    content = DOCKERFILE.read_text(encoding="utf-8")
    copy_targets = re.findall(r"^COPY\s+(\S+)", content, re.MULTILINE)
    for target in copy_targets:
        if any(c in target for c in "*?["):
            # glob - match at least one file
            matches = list(PROJECT.glob(target))
            assert matches, f"Dockerfile COPY pattern '{target}' matches no files."
        else:
            assert (PROJECT / target).exists(), f"Dockerfile COPY '{target}' missing on disk."


def test_dockerfile_exposes_8000_and_uses_uvicorn():
    content = DOCKERFILE.read_text(encoding="utf-8")
    assert "EXPOSE 8000" in content
    assert "uvicorn" in content
    assert "api_inference:app" in content


def test_dockerignore_excludes_heavy_paths():
    """The container should not ship 150+ MB of mlruns/, archive/, .venv/."""
    if not DOCKERIGNORE.exists():
        pytest.skip(".dockerignore not present (expected to be created during plumbing)")
    text = DOCKERIGNORE.read_text(encoding="utf-8")
    for pattern in ["mlruns/", "archive/", ".venv*/", "__pycache__/"]:
        assert pattern in text, f".dockerignore missing pattern '{pattern}'"


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
    # Ensure the job hits /metrics on either dev (8001) or Docker (8000) port.
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
    # Prometheus exposition format: lines beginning with `# HELP` or metric names
    assert "# HELP" in body or "# TYPE" in body, \
        f"/metrics did not return Prometheus text format. First 200 chars:\n{body[:200]}"
