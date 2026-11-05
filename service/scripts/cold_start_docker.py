"""Container cold start: time from `docker run` to the first 200 from /health (model loaded).

  python service/scripts/cold_start_docker.py IMAGE[:CONTAINER_PORT] [IMAGE[:PORT] ...] --runs 3

Prints JSON with the image size and per-run cold-start seconds.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import time
import urllib.request


def sh(*args: str) -> str:
    return subprocess.run(args, capture_output=True, text=True, check=True).stdout.strip()


def cold_start(image: str, port: int, host_port: int = 18080, timeout: float = 300) -> float:
    name = f"t2a-cold-{int(time.time() * 1000)}"
    t0 = time.perf_counter()
    sh("docker", "run", "-d", "--rm", "--name", name, "-p", f"{host_port}:{port}", image)
    try:
        while time.perf_counter() - t0 < timeout:
            try:
                with urllib.request.urlopen(f"http://127.0.0.1:{host_port}/health", timeout=2) as r:
                    if r.status == 200:
                        return time.perf_counter() - t0
            except OSError:
                pass
            time.sleep(0.05)
        raise TimeoutError(image)
    finally:
        subprocess.run(["docker", "rm", "-f", name], capture_output=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("images", nargs="+", help="image or image:tag@port (default port 8080)")
    ap.add_argument("--runs", type=int, default=3)
    a = ap.parse_args()
    out = {}
    for spec in a.images:
        image, _, port = spec.partition("@")
        runs = [cold_start(image, int(port or 8080)) for _ in range(a.runs)]
        info = json.loads(sh("docker", "image", "inspect", image))[0]
        out[image] = {"image_bytes": info["Size"], "architecture": info["Architecture"],
                      "cold_start_s": [round(r, 2) for r in runs],
                      "cold_start_median_s": round(sorted(runs)[len(runs) // 2], 2)}
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
