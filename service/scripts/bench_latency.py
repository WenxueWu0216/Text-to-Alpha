"""End-to-end /score latency, sequential requests, client-side HTTP round trip.

  python service/scripts/bench_latency.py                                    # start uvicorn locally
  python service/scripts/bench_latency.py --url https://... --label cloud-run --cold-start

--cold-start sends ONE /score request before anything else and records its latency together
with the instance's uptime_s from /health right after. If uptime_s is about the request's
latency (or less), that request started the instance: a real cold start. If uptime_s is much
larger, the request hit an already running instance and is reported as warm, not as a cold start.

Headlines come from the frozen fixture (service/eval_fixture), so CI needs no data/ directory.
Writes/merges results into --out under measurements[--label].
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

import httpx
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
SERVICE = ROOT / "service"
FIXTURE = SERVICE / "eval_fixture" / "val_fixture.parquet"


def headlines() -> list[str]:
    import pandas as pd

    return pd.read_parquet(FIXTURE, columns=["title"])["title"].drop_duplicates().tolist()


def wait_healthy(c: httpx.Client, timeout: float = 180) -> dict:
    t0 = time.time()
    while time.time() - t0 < timeout:
        try:
            r = c.get("/health")
            if r.status_code == 200:
                return r.json()
        except httpx.TransportError:
            pass
        time.sleep(0.5)
    raise TimeoutError("service did not become healthy")


def cold_start_probe(c: httpx.Client, headline: str) -> dict:
    t0 = time.perf_counter()
    while True:  # a local server may not be listening yet; time from the first attempt
        try:
            r = c.post("/score", json={"headlines": [headline]})
            break
        except httpx.TransportError:
            if time.perf_counter() - t0 > 180:
                raise
            time.sleep(0.05)
    latency = time.perf_counter() - t0
    r.raise_for_status()
    h = c.get("/health").json()
    # Cold if the instance is no older than this request (+2 s slack for the /health round trip).
    cold = h["uptime_s"] <= latency + 2.0
    return {"first_request_latency_s": round(latency, 3), "instance_uptime_after_s": h["uptime_s"],
            "instance_model_load_s": h["model_load_s"], "was_cold_start": bool(cold)}


def bench(c: httpx.Client, pool: list[str], batch: int, n: int, warmup: int) -> dict:
    rng = np.random.default_rng(batch)
    lat = []
    for i in range(warmup + n):
        heads = [pool[j] for j in rng.choice(len(pool), size=batch, replace=False)]
        t0 = time.perf_counter()
        r = c.post("/score", json={"headlines": heads})
        dt = time.perf_counter() - t0
        r.raise_for_status()
        assert len(r.json()["scores"]) == batch
        if i >= warmup:
            lat.append(dt * 1000)
    a = np.array(lat)
    return {"batch_size": batch, "n_requests": n, "warmup_requests": warmup,
            "p50_ms": float(np.percentile(a, 50)), "p95_ms": float(np.percentile(a, 95)),
            "mean_ms": float(a.mean()), "max_ms": float(a.max()),
            "ms_per_headline_p50": float(np.percentile(a, 50) / batch)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", help="benchmark an already running server instead of starting one")
    ap.add_argument("--label", default="native")
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--port", type=int, default=7861)
    ap.add_argument("--cold-start", action="store_true", help="time the very first request (see docstring)")
    ap.add_argument("--out", default=str(ROOT / "results/service_latency.json"))
    args = ap.parse_args()

    proc = None
    url = args.url or f"http://127.0.0.1:{args.port}"
    if not args.url:
        proc = subprocess.Popen([sys.executable, "-m", "uvicorn", "app.main:app", "--port", str(args.port),
                                 "--log-level", "warning"], cwd=SERVICE)
    pool = headlines()
    cold = None
    try:
        with httpx.Client(base_url=url, timeout=60) as c:
            if args.cold_start:
                cold = cold_start_probe(c, pool[0])
            health = wait_healthy(c)
            runs = [bench(c, pool, b, args.n, args.warmup) for b in (1, 32)] if args.n > 0 else []
    finally:
        if proc:
            proc.terminate()
            proc.wait(timeout=30)

    entry = {
        "label": args.label,
        "url": args.url,
        "created_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        "git_commit": os.environ.get("GITHUB_SHA") or subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True).stdout.strip(),
        "model": health["model"], "variant": health.get("variant"), "device": health["device"],
        "client": {"os": platform.platform(), "python": platform.python_version(),
                   "runner": os.environ.get("RUNNER_NAME")},
        "method": "client-side HTTP round trip, sequential requests, fixture headlines, max_length 64",
        "cold_start": cold,
        "runs": runs,
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    doc = json.loads(out.read_text()) if out.exists() else {"method": "service_latency", "measurements": {}}
    doc["measurements"][args.label] = entry
    out.write_text(json.dumps(doc, indent=2))
    if cold:
        print(f"[{args.label}] first request {cold['first_request_latency_s']:.2f}s "
              f"({'cold start' if cold['was_cold_start'] else 'warm instance'}, "
              f"instance uptime {cold['instance_uptime_after_s']:.1f}s)")
    for r in runs:
        print(f"[{args.label}] batch {r['batch_size']:>2}: p50 {r['p50_ms']:.1f} ms, p95 {r['p95_ms']:.1f} ms "
              f"({r['n_requests']} requests)")
    print(f"saved {out}")


if __name__ == "__main__":
    main()
