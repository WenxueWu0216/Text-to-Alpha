# Model service

FastAPI app serving FinBERT (`ProsusAI/finbert`, pinned revision) as ONNX on CPU, with
**fp16-stored weights and fp32 compute**. There's no torch or transformers at runtime, only
onnxruntime + tokenizers. The image bakes in the model and the benchmark numbers from
`results/*.json`, so a cold start downloads nothing.

| Endpoint | |
|---|---|
| `POST /score` | `{"headlines": [...]}` (1 to 64, each 1 to 1000 chars after trimming) -> per-headline `score` = P(pos) - P(neg) in [-1, 1], argmax `label`, and `probabilities`. Response order = request order. Invalid input -> 422. |
| `GET /health` | model, variant (`fp16w`), device, `uptime_s`, `model_load_s` |
| `GET /metrics` | headline metrics per method and split, the same-subset comparison, the eval gate, service latency |
| `GET /openapi.json` | the schema the Next.js frontend mirrors |

## Model: FinBERT -> ONNX -> fp16-stored weights

`service/scripts/export_onnx.py` exports fp32 ONNX (it matches torch to 1e-5 on the fixture). It
then stores every large weight in fp16 behind a `Cast(to=float)`. onnxruntime folds those Casts at
load time, so inference is ordinary fp32 and the file is half the size (219 MB vs 438 MB). The
Docker build runs the export in a throwaway stage, so torch never reaches the runtime image.

**Why not int8?** Dynamic int8 (111 MB, about 2x faster) was the first choice, but its accuracy
depends on the CPU. On the frozen fixture (`service/scripts/quant_sweep.py`, run on real runners by
`.github/workflows/quant-sweep.yml`):

| int8 dynamic, per-channel | Spearman vs fp32 | IC - fp32 IC |
|---|---:|---:|
| Apple M4 (ARM) | 0.996 | +0.0015 |
| x86 AMD EPYC 9V74 (AVX-512 VNNI) | 0.928 | +0.0073 |
| x86 AMD EPYC 7763 (AVX2, no VNNI) | 0.146 | +0.0053 |

Cloud Run doesn't promise a CPU model. On the AVX2 runner, every int8 variant we tried
(per-tensor, reduce_range, excluding layers) missed both the 0.002 IC budget and the agreement check. The fp16-stored model gives the same scores on
all three CPUs:

| | fixture IC | vs fp32 |
|---|---:|---|
| fp32 ONNX | -0.02901 | (reference) |
| fp16 weights, fp32 compute (served) | -0.02882 | dIC +0.00019; Spearman 0.999999, mean abs diff 0.00008 |

Headlines run in length-sorted micro-batches of 8, which keeps padding small. For 32 headlines on
1 thread that took 377 ms, against 601 ms one at a time and 715 ms as one padded batch.

## Eval gate

`scripts/eval_gate.py` scores the frozen fixture (1,508 val headlines on 58 days, 1 per stock-day,
`scripts/08_build_eval_fixture.py`, fixed seed) with the served model and fails if:
- fixture IC < `min_ic` = reference IC (-0.02882) - 0.002 = -0.03082, or
- the scores disagree with the frozen fp32 reference scores (Spearman < 0.999 or mean abs diff > 0.005).

The fixture IC is negative (FinBERT on 58 days, t = -1.1), so the IC check alone can't catch a sign
flip. The agreement check does. The rationale is in `service/eval_fixture/gate.json`.

## Run

```bash
python service/scripts/export_onnx.py --out models/finbert-onnx    # needs torch + transformers + onnx
cd service && uvicorn app.main:app --port 8080

docker build -f service/Dockerfile -t t2a-service .                # from the repo root
docker run --rm -p 8080:8080 t2a-service
curl -s localhost:8080/score -H 'content-type: application/json' \
  -d '{"headlines": ["Apple beats earnings estimates", "Intel cuts guidance"]}'
```

Environment: `PORT` (8080), `T2A_MODEL_DIR`, `T2A_MODEL_FILE` (`model.fp16w.onnx`), `T2A_RESULTS_DIR`,
`T2A_NUM_THREADS` (1 on Cloud Run), `T2A_MICRO_BATCH` (8).

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
