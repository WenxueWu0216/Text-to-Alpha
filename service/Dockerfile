# syntax=docker/dockerfile:1
# FinBERT scoring service, CPU only.
#   docker build -f service/Dockerfile -t t2a-service .     (from the repo root)
#   docker run --rm -p 7860:7860 t2a-service

ARG PYTHON=python:3.12-slim

# ---- build: deps into a venv + model weights ------------------------------------------
FROM ${PYTHON} AS build
ENV PIP_DISABLE_PIP_VERSION_CHECK=1 PIP_DEFAULT_TIMEOUT=120 PIP_RETRIES=10 PATH=/opt/venv/bin:$PATH
RUN python -m venv /opt/venv
# CPU-only wheel: no CUDA libraries (several GB) in the image.
# The pip cache lives in a BuildKit cache mount, not in an image layer.
RUN --mount=type=cache,target=/root/.cache/pip pip install --index-url https://download.pytorch.org/whl/cpu torch==2.14.0
COPY service/requirements.txt /tmp/requirements.txt
RUN --mount=type=cache,target=/root/.cache/pip pip install -r /tmp/requirements.txt \
 && rm -rf /opt/venv/lib/python3.12/site-packages/torch/include \
 && find /opt/venv -name __pycache__ -prune -exec rm -rf {} +

# Bake the weights in at a pinned revision, so cold starts never hit the network.
ARG MODEL_ID=ProsusAI/finbert
ARG MODEL_REVISION=0bc2ac10dda6929bcb1f77ccc43c7ec5c424f031
RUN python -c "from huggingface_hub import snapshot_download as d; d('${MODEL_ID}', revision='${MODEL_REVISION}', \
local_dir='/models/finbert', allow_patterns=['config.json', 'model.safetensors', 'vocab.txt', \
'tokenizer_config.json', 'special_tokens_map.json'])" \
 && rm -rf /models/finbert/.cache

# ---- runtime ---------------------------------------------------------------------------
FROM ${PYTHON}
# uid 1000: the user Hugging Face Spaces runs containers as.
RUN useradd --create-home --uid 1000 app
ENV PATH=/opt/venv/bin:$PATH \
    PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 \
    HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
    T2A_MODEL_PATH=/models/finbert T2A_RESULTS_DIR=/srv/results
COPY --from=build /opt/venv /opt/venv
COPY --from=build /models /models
COPY results/*.json /srv/results/
WORKDIR /srv/service
COPY service/app ./app
USER app
EXPOSE 7860
HEALTHCHECK --interval=30s --timeout=5s --start-period=60s \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:7860/health', timeout=4)"
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "7860"]
