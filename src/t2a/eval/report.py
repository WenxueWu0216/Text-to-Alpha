"""Evaluate a stock-day signal per split and save results as JSON with provenance."""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from t2a.data.io import DEFAULT_SPLITS, check_splits
from t2a.eval.metrics import evaluate
from t2a.provenance import RESULTS_DIR, write_result


def evaluate_splits(stock_day: pd.DataFrame, cfg: dict, splits=DEFAULT_SPLITS,
                    unlock_test: bool = False) -> dict[str, dict]:
    check_splits(splits, unlock_test)
    rows = {}
    for s in splits:
        part = stock_day[stock_day["split"] == s]
        if len(part):
            rows[s] = evaluate(part, min_names=cfg["eval"]["min_names_per_day"],
                               cost_bps=cfg["eval"]["cost_bps_round_trip"],
                               horizon=cfg["labels"]["horizon_days"],
                               top_q=cfg["labels"]["top_quantile"])
    return rows


def report(stock_day: pd.DataFrame, cfg: dict, name: str, splits=DEFAULT_SPLITS,
           out_dir: Path | None = RESULTS_DIR, extra: dict | None = None,
           unlock_test: bool = False) -> pd.DataFrame:
    rows = evaluate_splits(stock_day, cfg, splits, unlock_test)
    table = pd.DataFrame(rows).T
    print(f"\n=== {name} ===\n{table.round(4).to_string()}")
    if out_dir is not None:
        path = write_result(name, {"metrics": rows, **(extra or {})}, cfg, out_dir)
        print(f"saved {path}")
    return table
