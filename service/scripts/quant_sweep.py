"""Compare compression variants of the fp32 FinBERT ONNX on THIS machine's CPU.

  python service/scripts/quant_sweep.py --model-dir models/finbert-onnx --out sweep.json

int8 kernels differ by CPU (ARM vs x86, VNNI or not), so a variant has to be judged on the CPU
that serves it. For each variant: file size, fixture IC vs fp32, agreement with the frozen fp32
reference scores, and single-thread latency per headline.
"""
from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "service"), str(ROOT / "scripts")]

import numpy as np
import onnx
import pandas as pd
from onnxruntime.quantization import QuantType, quantize_dynamic
from scipy.stats import spearmanr

from app.model import OnnxFinBert
from eval_gate import FIXTURE_DIR, fixture_metrics, score_local
from export_onnx import fp16_weights  # same function that builds the served file
from t2a.config import load_config


def act_x_act_matmuls(path: Path) -> list[str]:
    m = onnx.load(str(path), load_external_data=False)
    inits = {i.name for i in m.graph.initializer}
    return [n.name for n in m.graph.node if n.op_type == "MatMul" and not any(i in inits for i in n.input)]


def ffn_down(path: Path) -> list[str]:
    m = onnx.load(str(path), load_external_data=False)
    return [n.name for n in m.graph.node if n.op_type == "MatMul" and n.name.endswith("/output/dense/MatMul")
            and "/attention/" not in n.name]


def latency_ms(model_dir: Path, f: str, texts: list[str]) -> float:
    m = OnnxFinBert(model_dir, f, num_threads=1)
    m.predict(texts[:8])
    t0 = time.perf_counter()
    m.predict(texts)
    return (time.perf_counter() - t0) / len(texts) * 1000


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", type=Path, default=ROOT / "models" / "finbert-onnx")
    ap.add_argument("--out", type=Path, default=Path("sweep.json"))
    a = ap.parse_args()
    d = a.model_dir
    fp32 = d / "model.fp32.onnx"
    aa, down = act_x_act_matmuls(fp32), ffn_down(fp32)

    variants = {
        "int8_pc_all": dict(per_channel=True),
        "int8_pc_weights": dict(per_channel=True, nodes_to_exclude=aa),
        "int8_pc_weights_noffn": dict(per_channel=True, nodes_to_exclude=aa + down),
        "int8_pc_weights_rr": dict(per_channel=True, reduce_range=True, nodes_to_exclude=aa),
        "int8_pt_weights": dict(nodes_to_exclude=aa),
    }
    files = {"fp32": "model.fp32.onnx", "fp16_weights": "model.v_fp16_weights.onnx"}
    fp16_weights(fp32, d / files["fp16_weights"])
    for k, kw in variants.items():
        files[k] = f"model.v_{k}.onnx"
        quantize_dynamic(str(fp32), str(d / files[k]), weight_type=QuantType.QInt8, **kw)

    cfg = load_config()
    fx = pd.read_parquet(FIXTURE_DIR / "val_fixture.parquet")
    texts = fx["title"].tolist()
    gate = json.loads((FIXTURE_DIR / "gate.json").read_text())
    ref = pd.read_parquet(FIXTURE_DIR / gate["reference_scores"]).set_index("news_id")["score"]
    ref = ref.reindex(fx["news_id"]).to_numpy()
    ic_fp32 = None
    rows = {}
    for k, f in files.items():
        s = score_local(d, f, texts)
        ic = fixture_metrics(fx, s, cfg)["ic_mean"]
        ic_fp32 = ic if k == "fp32" else ic_fp32
        rows[k] = {"file_mb": round((d / f).stat().st_size / 1e6, 1), "ic": round(float(ic), 5),
                   "ic_minus_fp32": round(float(ic - ic_fp32), 5),
                   "spearman_vs_ref": round(float(spearmanr(s, ref).statistic), 5),
                   "mean_abs_diff_vs_ref": round(float(np.abs(s - ref).mean()), 5),
                   "ms_per_headline_1_thread": round(latency_ms(d, f, texts[:200]), 2)}
        print(k, rows[k], flush=True)
    cpu = platform.processor()
    try:
        cpu = next(line.split(":", 1)[1].strip() for line in open("/proc/cpuinfo") if line.startswith("model name"))
    except OSError:
        pass
    a.out.write_text(json.dumps({"cpu": cpu, "machine": platform.machine(), "variants": rows}, indent=2))
    print("| variant | MB | IC | dIC vs fp32 | Spearman | mean abs diff | ms/headline |")
    print("|---|---:|---:|---:|---:|---:|---:|")
    for k, r in rows.items():
        print(f"| {k} | {r['file_mb']} | {r['ic']:+.5f} | {r['ic_minus_fp32']:+.5f} | {r['spearman_vs_ref']:.4f} "
              f"| {r['mean_abs_diff_vs_ref']:.4f} | {r['ms_per_headline_1_thread']:.1f} |")


if __name__ == "__main__":
    main()
