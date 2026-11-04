"""Evaluate a stock-day signal per split and save results as JSON with provenance."""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from t2a.data.io import DEFAULT_SPLITS, check_splits
from t2a.data.market import VIX_BUCKETS, load_or_download_vix, vix_regime
from t2a.eval.metrics import daily_series, evaluate
from t2a.provenance import RESULTS_DIR, write_result

HEADLINE = ["ic_mean", "ic_tstat", "n_days", "ls_sharpe_gross", "ls_sharpe_net",
            "breakeven_cost_bps_per_side", "turnover_mean", "macro_f1", "n_obs"]


def regime_labels(cfg: dict, dates: pd.DatetimeIndex) -> tuple[dict[str, pd.Series], dict]:
    vix = load_or_download_vix(cfg["paths"]["vix"], cfg["dates"]["start"],
                               (pd.Timestamp(cfg["dates"]["end"]) + pd.Timedelta(days=15)).strftime("%Y-%m-%d"))
    if vix is None:
        return {}, {}
    info = {"vix_regime": {"rule": "prior-day VIX close bucketed at fixed levels",
                           "buckets": [[u, n] for u, n in VIX_BUCKETS]}}
    return {"vix_regime": vix_regime(vix, dates)}, info


def evaluate_splits(stock_day: pd.DataFrame, cfg: dict, splits=DEFAULT_SPLITS,
                    unlock_test: bool = False, regimes: dict[str, pd.Series] | None = None) -> dict[str, dict]:
    check_splits(splits, unlock_test)
    e = cfg["eval"]
    rows = {}
    for s in splits:
        part = stock_day[stock_day["split"] == s]
        if len(part):
            rows[s] = evaluate(part, min_names=e["min_names_per_day"],
                               cost_bps_per_side=e["cost_bps_per_side"],
                               horizon=cfg["labels"]["horizon_days"],
                               top_q=cfg["labels"]["top_quantile"],
                               cost_grid=e["cost_grid_bps_per_side"], regimes=regimes)
    return rows


def report(stock_day: pd.DataFrame, cfg: dict, name: str, splits=DEFAULT_SPLITS,
           out_dir: Path | None = RESULTS_DIR, extra: dict | None = None,
           unlock_test: bool = False, use_regimes: bool = True) -> pd.DataFrame:
    check_splits(splits, unlock_test)
    regimes, regime_info = (regime_labels(cfg, pd.DatetimeIndex(stock_day["date"].unique()))
                            if use_regimes else ({}, {}))
    rows = evaluate_splits(stock_day, cfg, splits, unlock_test, regimes)
    table = pd.DataFrame({s: {k: m[k] for k in HEADLINE} for s, m in rows.items()}).T
    print(f"\n=== {name} ===\n{table.round(4).to_string()}")
    if out_dir is not None:
        path = write_result(name, {"metrics": rows, **regime_info, **(extra or {})}, cfg, out_dir)
        daily_dir = out_dir / "daily"
        daily_dir.mkdir(parents=True, exist_ok=True)
        for s in rows:
            part = stock_day[stock_day["split"] == s]
            daily_series(part, cfg["eval"]["min_names_per_day"], cfg["eval"]["cost_bps_per_side"]) \
                .to_csv(daily_dir / f"{name}_{s}.csv", index=False, float_format="%.6g")
        print(f"saved {path}")
    return table
