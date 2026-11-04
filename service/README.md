# Model service

FastAPI app serving FinBERT (`ProsusAI/finbert`) on CPU. Weights are baked into the image at a
pinned revision; the benchmark numbers in `results/*.json` are baked in too and served read-only.

| Endpoint | |
|---|---|
| `POST /score` | `{"headlines": [...]}` (1 to 64, each 1 to 1000 chars after trimming) -> per-headline `score` = P(pos) - P(neg) in [-1, 1], argmax `label`, and `probabilities`. Response order = request order. Invalid input -> 422. |
| `GET /health` | `{"status": "ok", "model": ..., "device": "cpu"}` |
| `GET /metrics` | headline metrics per method and split, the same-subset comparison (`results/val_same_subset.json`), and service latency |
| `GET /openapi.json` | the schema the Next.js frontend mirrors |

## Run

```bash
# local (from service/)
uvicorn app.main:app --port 7860

# docker (from the repo root; the build context is the root so results/*.json can be copied in)
docker build -f service/Dockerfile -t t2a-service .
docker run --rm -p 7860:7860 t2a-service
curl -s localhost:7860/score -H 'content-type: application/json' \
  -d '{"headlines": ["Apple beats earnings estimates", "Intel cuts guidance"]}'
```

Environment: `T2A_MODEL_PATH` (weights dir or hub id), `T2A_RESULTS_DIR`, `T2A_MAX_LENGTH` (default 64,
same as the benchmark), `TORCH_NUM_THREADS`.

## Tests and latency

```bash
pytest -q                                        # from the repo root; the model is mocked
T2A_MODEL_TESTS=1 pytest -q service/tests        # also runs one test against the real weights
python service/scripts/bench_latency.py          # p50/p95 for 1 and 32 headlines -> results/service_latency.json
```
