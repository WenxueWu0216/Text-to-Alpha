"""Eval gate: score the frozen fixture with the SERVED model and fail if quality regressed.

  python scripts/eval_gate.py --url http://127.0.0.1:7860      # a running service (the image that ships)
  python scripts/eval_gate.py                                  # in-process, models/finbert-onnx/model.fp16w.onnx
  python scripts/eval_gate.py --model-file model.fp32.onnx --no-gate --out results/fixture_fp32.json

Two checks, both against numbers frozen in service/eval_fixture/gate.json:
  1. cross-sectional IC (shared evaluator) >= min_ic
  2. agreement with the frozen reference scores (catches label-order / tokenizer bugs that a
     noisy IC on 58 days could miss by luck): Spearman >= min_score_spearman and
     mean |diff| <= max_mean_abs_score_diff
Exit code 1 if any check fails.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "service"))

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from t2a.config import load_config
from t2a.eval.metrics import evaluate
from t2a.provenance import write_result
from t2a.signals.aggregate import to_stock_day

FIXTURE_DIR = ROOT / "service" / "eval_fixture"
BATCH = 64  # the service's request limit


def score_http(url: str, texts: list[str]) -> np.ndarray:
    import httpx

    out = []
    with httpx.Client(base_url=url, timeout=120) as c:
        for s in range(0, len(texts), BATCH):
            r = c.post("/score", json={"headlines": texts[s:s + BATCH]})
            r.raise_for_status()
            out += [x["score"] for x in r.json()["scores"]]
    return np.array(out)


def score_local(model_dir: Path, model_file: str, texts: list[str]) -> np.ndarray:
    from app.model import OnnxFinBert

    m = OnnxFinBert(model_dir, model_file)
    return np.concatenate([np.array([p.score for p in m.predict(texts[s:s + BATCH])])
                           for s in range(0, len(texts), BATCH)])


def fixture_metrics(fx: pd.DataFrame, scores: np.ndarray, cfg: dict) -> dict:
    e = cfg["eval"]
    m = evaluate(to_stock_day(fx.assign(score=scores), "score"), min_names=e["min_names_per_day"],
                 cost_bps_per_side=e["cost_bps_per_side"], horizon=cfg["labels"]["horizon_days"],
                 top_q=cfg["labels"]["top_quantile"], cost_grid=e["cost_grid_bps_per_side"])
    return {k: m[k] for k in ("ic_mean", "ic_tstat", "ic_hit_rate", "n_days", "n_obs", "ls_sharpe_gross",
                              "ls_sharpe_net", "macro_f1")}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", help="score through a running service instead of in-process")
    ap.add_argument("--model-dir", type=Path, default=ROOT / "models" / "finbert-onnx")
    ap.add_argument("--model-file", default="model.fp16w.onnx")
    ap.add_argument("--out", type=Path, default=ROOT / "results" / "gate_metrics.json")
    ap.add_argument("--no-gate", action="store_true", help="report only, never fail")
    ap.add_argument("--save-scores", type=Path, help="write the scores (to freeze a new reference)")
    a = ap.parse_args()

    cfg = load_config()
    fx = pd.read_parquet(FIXTURE_DIR / "val_fixture.parquet")
    texts = fx["title"].tolist()
    path = (a.model_dir / a.model_file).resolve()
    source = a.url or str(path.relative_to(ROOT) if path.is_relative_to(ROOT) else path)
    scores = score_http(a.url, texts) if a.url else score_local(a.model_dir, a.model_file, texts)
    assert len(scores) == len(fx) and np.isfinite(scores).all()
    if a.save_scores:
        fx[["news_id"]].assign(score=scores).to_parquet(a.save_scores, index=False)

    metrics = fixture_metrics(fx, scores, cfg)
    gate = json.loads((FIXTURE_DIR / "gate.json").read_text())
    ref = pd.read_parquet(FIXTURE_DIR / gate["reference_scores"]).set_index("news_id")["score"]
    ref = ref.reindex(fx["news_id"]).to_numpy()
    agreement = {"score_spearman_vs_reference": float(spearmanr(scores, ref).statistic),
                 "mean_abs_score_diff_vs_reference": float(np.abs(scores - ref).mean()),
                 "max_abs_score_diff_vs_reference": float(np.abs(scores - ref).max())}
    checks = {
        "ic_mean": bool(metrics["ic_mean"] >= gate["min_ic"]),
        "score_spearman": bool(agreement["score_spearman_vs_reference"] >= gate["min_score_spearman"]),
        "mean_abs_score_diff": bool(agreement["mean_abs_score_diff_vs_reference"] <= gate["max_mean_abs_score_diff"]),
    }
    passed = all(checks.values())
    write_result(a.out.stem, {"source": source, "fixture_rows": len(fx), "metrics": metrics,
                              "agreement": agreement, "thresholds": gate, "checks": checks,
                              "passed": passed}, cfg, a.out.parent)

    print(f"source: {source}")
    print(f"fixture IC   {metrics['ic_mean']:+.5f}  (t {metrics['ic_tstat']:.2f}, {metrics['n_days']} days)")
    print(f"threshold    {gate['min_ic']:+.5f}  = reference {gate['reference_ic']:+.5f} - margin {gate['ic_margin']}")
    print(f"agreement    spearman {agreement['score_spearman_vs_reference']:.5f} (min {gate['min_score_spearman']}), "
          f"mean |diff| {agreement['mean_abs_score_diff_vs_reference']:.5f} (max {gate['max_mean_abs_score_diff']})")
    out = a.out.resolve()
    shown = out.relative_to(ROOT) if out.is_relative_to(ROOT) else out
    print(f"gate: {'PASS' if passed else 'FAIL'} {checks} -> {shown}")
    return 0 if passed or a.no_gate else 1


if __name__ == "__main__":
    sys.exit(main())
