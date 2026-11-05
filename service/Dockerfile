# syntax=docker/dockerfile:1
# FinBERT scoring service: ONNX (fp16-stored weights, fp32 compute) on CPU, no torch at runtime.
#   docker build -f service/Dockerfile -t t2a-service .     (from the repo root)
#   docker run --rm -p 8080:8080 t2a-service

ARG PYTHON=python:3.12-slim

# ---- export: download pinned weights, FinBERT -> ONNX fp32 -> fp16-stored weights -------
FROM ${PYTHON} AS export
ENV PIP_DISABLE_PIP_VERSION_CHECK=1 PIP_DEFAULT_TIMEOUT=120 PIP_RETRIES=10
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --index-url https://download.pytorch.org/whl/cpu torch==2.14.0
COPY service/requirements-export.txt /tmp/
RUN --mount=type=cache,target=/root/.cache/pip pip install -r /tmp/requirements-export.txt
ARG MODEL_ID=ProsusAI/finbert
ARG MODEL_REVISION=0bc2ac10dda6929bcb1f77ccc43c7ec5c424f031
COPY service/scripts/export_onnx.py /tmp/
RUN --mount=type=cache,target=/root/.cache/huggingface \
    python /tmp/export_onnx.py --model ${MODEL_ID} --revision ${MODEL_REVISION} --out /models/finbert-onnx \
 && rm /models/finbert-onnx/model.fp32.onnx

# ---- runtime deps ----------------------------------------------------------------------
FROM ${PYTHON} AS deps
ENV PIP_DISABLE_PIP_VERSION_CHECK=1 PIP_DEFAULT_TIMEOUT=120 PIP_RETRIES=10
RUN python -m venv /opt/venv
COPY service/requirements.txt /tmp/
RUN --mount=type=cache,target=/root/.cache/pip /opt/venv/bin/pip install -r /tmp/requirements.txt \
 && find /opt/venv -name __pycache__ -prune -exec rm -rf {} + \
 && rm -rf /opt/venv/lib/python3.12/site-packages/pip /opt/venv/bin/pip*

# ---- runtime ---------------------------------------------------------------------------
FROM ${PYTHON}
RUN useradd --create-home --uid 1000 app
ENV PATH=/opt/venv/bin:$PATH PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 \
    T2A_MODEL_DIR=/models/finbert-onnx T2A_RESULTS_DIR=/srv/results PORT=8080
COPY --from=deps /opt/venv /opt/venv
COPY --from=export /models /models
COPY results/*.json /srv/results/
WORKDIR /srv/service
COPY service/app ./app
USER app
EXPOSE 8080
HEALTHCHECK --interval=30s --timeout=5s --start-period=30s \
  CMD python -c "import os, urllib.request; urllib.request.urlopen(f'http://127.0.0.1:{os.environ[\"PORT\"]}/health', timeout=4)"
# Cloud Run sets $PORT (8080). sh expands it; exec hands PID 1 to uvicorn so it gets SIGTERM.
CMD ["sh", "-c", "exec uvicorn app.main:app --host 0.0.0.0 --port ${PORT} --no-access-log"]
