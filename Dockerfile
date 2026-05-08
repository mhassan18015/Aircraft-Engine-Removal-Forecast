# Inference-only image for the LEAP-1A engine RCL FastAPI service.
# Loads three Keras models at startup, exposes /predict, /health, /schema, and
# /metrics for Prometheus on port 8000.
FROM python:3.10-slim

# System packages that TF + scientific Python pull in. Keep this list short -
# tensorflow already bundles its own libstdc++ via wheels.
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy only requirements first so dependency layer is cached across code edits.
COPY requirements.txt /app/
RUN pip install --no-cache-dir -r requirements.txt

# Now copy the inference-side files (.dockerignore filters out notebooks,
# raw data, archive/, mlruns/, virtualenvs, etc.).
COPY api_inference.py        /app/
COPY data_drift.py           /app/
COPY index.html              /app/
COPY start_servers.py        /app/
COPY train_pipeline.py       /app/
COPY retrain_dag.py          /app/
COPY *.keras                 /app/
COPY best_lstm2_history.json /app/
COPY Data/LEAB_engines_data_cleaned.csv /app/Data/

# Health check uses the FastAPI /health endpoint - confirms TF actually
# managed to load the three .keras files at boot.
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -fsS http://localhost:8000/health || exit 1

ENV PYTHONUNBUFFERED=1
ENV TF_CPP_MIN_LOG_LEVEL=2

EXPOSE 8000

CMD ["uvicorn", "api_inference:app", "--host", "0.0.0.0", "--port", "8000"]
