"""Evaluate a stock-day signal per split and save results as JSON."""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from t2a.eval.metrics import evaluate


def report(stock_day: pd.DataFrame, cfg: dict, name: str, splits=("train", "val"),
           out_dir: Path | None = None) -> pd.DataFrame:
    # "test" is deliberately excluded by default: it is only opened once, at the very end,
    # after every design choice is frozen. Otherwise we would be tuning on the test set.
    rows = {}
    for s in splits:
        part = stock_day[stock_day["split"] == s]
        if len(part):
            rows[s] = evaluate(part, min_names=cfg["eval"]["min_names_per_day"],
                               cost_bps=cfg["eval"]["cost_bps_round_trip"],
                               horizon=cfg["labels"]["horizon_days"])
    table = pd.DataFrame(rows).T
    print(f"\n=== {name} ===\n{table.round(4).to_string()}")
    if out_dir is not None:
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / f"{name}_metrics.json").write_text(json.dumps(rows, indent=2, default=float))
    return table
