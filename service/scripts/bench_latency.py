"""End-to-end /score latency: starts the service with uvicorn, sends sequential requests, writes JSON.

  python service/scripts/bench_latency.py                      # native, this machine's CPU
  python service/scripts/bench_latency.py --url http://127.0.0.1:7860 --label docker   # running server

Latency is measured client-side over HTTP (JSON encode -> tokenize -> forward pass -> response),
one request at a time. Headlines are real validation headlines, sampled with a fixed seed.
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import httpx
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
SERVICE = ROOT / "service"


def headlines(n: int, seed: int = 0) -> list[str]:
    import pandas as pd

    ev = pd.read_parquet(ROOT / "data/processed/events.parquet", columns=["title", "split"])
    titles = ev.loc[ev["split"] == "val", "title"].dropna().drop_duplicates()
    return titles.sample(n, random_state=seed).tolist()


def cpu_name() -> str:
    try:
        return subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"], capture_output=True,
                              text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return platform.processor()


def wait_healthy(c: httpx.Client, timeout: float = 120) -> dict:
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
    ap.add_argument("--out", default=str(ROOT / "results/service_latency.json"))
    args = ap.parse_args()

    proc = None
    url = args.url or f"http://127.0.0.1:{args.port}"
    if not args.url:
        proc = subprocess.Popen([sys.executable, "-m", "uvicorn", "app.main:app", "--port", str(args.port),
                                 "--log-level", "warning"], cwd=SERVICE,
                                env={**os.environ, "HF_HUB_OFFLINE": "1"})
    try:
        with httpx.Client(base_url=url, timeout=60) as c:
            t0 = time.time()
            health = wait_healthy(c)
            startup_s = time.time() - t0
            pool = headlines(2000)
            runs = [bench(c, pool, b, args.n, args.warmup) for b in (1, 32)]
    finally:
        if proc:
            proc.terminate()
            proc.wait(timeout=30)

    import torch

    entry = {
        "label": args.label,
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "git_commit": subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True,
                                     text=True).stdout.strip(),
        "model": health["model"], "device": health["device"],
        "machine": {"cpu": cpu_name(), "os": platform.platform(), "python": platform.python_version(),
                    "torch": torch.__version__, "torch_threads": torch.get_num_threads(),
                    "load_avg_1m_after_run": round(os.getloadavg()[0], 2)},
        "method": "client-side HTTP round trip, sequential requests, real val headlines, max_length 64",
        "startup_to_healthy_s": round(startup_s, 2) if proc else None,
        "runs": runs,
    }
    out = Path(args.out)
    doc = json.loads(out.read_text()) if out.exists() else {"method": "service_latency", "measurements": {}}
    doc["measurements"][args.label] = entry
    out.write_text(json.dumps(doc, indent=2))
    for r in runs:
        print(f"[{args.label}] batch {r['batch_size']:>2}: p50 {r['p50_ms']:.1f} ms, p95 {r['p95_ms']:.1f} ms "
              f"({r['n_requests']} requests)")
    print(f"saved {out}")


if __name__ == "__main__":
    main()
