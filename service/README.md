# Model service

FastAPI app serving FinBERT (`ProsusAI/finbert`, pinned revision) as an **int8 ONNX** model on CPU.
There's no torch or transformers at runtime, only onnxruntime + tokenizers. The image bakes in the
model and the benchmark numbers from `results/*.json`, so a cold start downloads nothing.

| Endpoint | |
|---|---|
| `POST /score` | `{"headlines": [...]}` (1 to 64, each 1 to 1000 chars after trimming) -> per-headline `score` = P(pos) - P(neg) in [-1, 1], argmax `label`, and `probabilities`. Response order = request order. Invalid input -> 422. |
| `GET /health` | model, variant (`int8`), device, `uptime_s`, `model_load_s` |
| `GET /metrics` | headline metrics per method and split, the same-subset comparison, the eval gate, service latency |
| `GET /openapi.json` | the schema the Next.js frontend mirrors |

## Model: FinBERT -> ONNX -> int8

`service/scripts/export_onnx.py` exports fp32 ONNX (it matches torch to 1e-5 on the fixture), then
applies **dynamic int8 quantization with per-channel weights**. The Docker build runs the export in
a throwaway stage, so torch never reaches the runtime image.

Each headline gets its own unpadded forward pass. Dynamic int8 computes one activation scale per
tensor, so in a padded batch a headline's score would depend on the other headlines in the request.
Single passes are also faster on CPU.

Measured on the frozen fixture (`service/eval_fixture/gate.json` has the full numbers):

| | fixture IC | vs fp32 |
|---|---:|---|
| fp32 ONNX | -0.02900 | (reference) |
| int8 per-channel (served) | -0.02746 | dIC +0.00154; score Spearman 0.996, mean abs diff 0.017 |

**Known risk: x86 without VNNI.** Under Docker's amd64 emulation (AVX2, no VNNI), int8 scores
drifted far from fp32 (Spearman 0.25), while fp32 was exact. Real x86 hardware may behave
differently. So the deploy pipeline runs the eval gate twice: on the GitHub runner, and against the
Cloud Run candidate revision before it gets any traffic.

## Eval gate

`scripts/eval_gate.py` scores the frozen fixture (1,508 val headlines on 58 days, 1 per stock-day,
`scripts/08_build_eval_fixture.py`, fixed seed) with the served model and fails if:
- fixture IC < `min_ic` = reference IC - 0.005, or
- the scores disagree with the frozen fp32 reference scores (Spearman < 0.98 or mean abs diff > 0.035).

The fixture IC is negative (FinBERT on 58 days, t = -1.1), so the IC check alone can't catch a sign
flip. The agreement check does. The rationale is in `gate.json`.

## Run

```bash
python service/scripts/export_onnx.py --out models/finbert-onnx    # needs torch + transformers + onnx
cd service && uvicorn app.main:app --port 8080

docker build -f service/Dockerfile -t t2a-service .                # from the repo root
docker run --rm -p 8080:8080 t2a-service
curl -s localhost:8080/score -H 'content-type: application/json' \
  -d '{"headlines": ["Apple beats earnings estimates", "Intel cuts guidance"]}'
```

Environment: `PORT` (8080), `T2A_MODEL_DIR`, `T2A_MODEL_FILE` (`model.int8.onnx`), `T2A_RESULTS_DIR`,
`T2A_NUM_THREADS` (1 on Cloud Run).

## Deployment

Cloud Run, us-central1: 1 vCPU, 1 GiB, concurrency 8, timeout 30 s, 0 to 2 instances, public.
One-time setup is in `infra/gcp_setup.sh`, with Workload Identity Federation and no JSON keys.
On every push to `main`, `.github/workflows/deploy.yml` does:
build -> gate on the runner -> push -> deploy a 0%-traffic candidate -> gate on Cloud Run -> promote
-> latency benchmark (uploaded as an artifact, never committed).

## Tests and latency

```bash
pytest -q                                          # from the repo root; the model is mocked
T2A_MODEL_TESTS=1 pytest -q service/tests          # also runs against models/finbert-onnx
python service/scripts/bench_latency.py --url URL --cold-start --label NAME
python service/scripts/cold_start_docker.py IMAGE@PORT --runs 3
```
