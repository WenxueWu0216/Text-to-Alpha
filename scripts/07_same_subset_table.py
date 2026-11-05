"""Compare Claude zero-shot, FinBERT and the LM lexicon on the SAME validation articles.

Uses cached signals only (no API calls, no model inference):
  data/signals/llm_zero_shot_events.parquet   (Claude zero-shot, 1 article per stock-day)
  data/signals/finbert_events.parquet
  data/signals/lexicon_events.parquet

  python scripts/07_same_subset_table.py      # writes results/val_same_subset.json
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd

from t2a.config import load_config
from t2a.data.io import load_events
from t2a.eval.report import evaluate_splits
from t2a.provenance import write_result
from t2a.signals.aggregate import to_stock_day

SPLIT = "val"
METHODS = [  # (result key, display name, score column, cached parquet)
    ("llm_zero_shot", "Claude zero-shot (direction x confidence)", "llm_signal", "llm_zero_shot_events.parquet"),
    ("finbert", "FinBERT (headline)", "finbert", "finbert_events.parquet"),
    ("lexicon_lm", "Loughran-McDonald lexicon", "lexicon", "lexicon_events.parquet"),
]
KEYS = ["n_obs", "n_days", "ic_mean", "ic_tstat", "ic_hit_rate", "ls_sharpe_gross", "ls_sharpe_net",
        "macro_f1"]


def fmt(x, nd=3):
    return "n/a" if x is None or pd.isna(x) else f"{x:.{nd}f}"


def main():
    cfg = load_config()
    sig = cfg["paths"]["signals_dir"]
    events = load_events(cfg, splits=(SPLIT,))

    llm = pd.read_parquet(sig / METHODS[0][3])
    llm = llm.loc[llm["status"] == "ok", ["news_id", "llm_signal"]]
    sub = events.merge(llm, on="news_id", how="inner")
    for _, _, col, fname in METHODS[1:]:
        sub = sub.merge(pd.read_parquet(sig / fname)[["news_id", col]], on="news_id", how="inner")
    # Every method must be scored on exactly the same articles, or the comparison is unfair.
    assert sub[[m[2] for m in METHODS]].notna().all().all(), "missing scores on the shared subset"
    assert sub["news_id"].is_unique

    metrics = {}
    for key, _, col, _ in METHODS:
        metrics[key] = evaluate_splits(to_stock_day(sub, col), cfg, (SPLIT,))[SPLIT]

    payload = {
        "split": SPLIT,
        "n_articles": len(sub),
        "n_stock_days": int(sub.groupby(["trade_date", "ticker"]).ngroups),
        "n_trade_days": int(sub["trade_date"].nunique()),
        "subset_rule": "articles Claude zero-shot scored with status ok (1 random article per stock-day, "
                       "every val day); FinBERT and lexicon scores looked up for the same news_ids",
        "summary": {k: {m: metrics[k][m] for m in KEYS} for k in metrics},
        "metrics": metrics,
    }
    path = write_result("val_same_subset", payload, cfg)

    print(f"### Same {len(sub):,} {SPLIT} articles ({payload['n_stock_days']:,} stock-days, "
          f"{payload['n_trade_days']} days)\n")
    print("| Method | IC mean | IC t-stat | IC hit rate | Sharpe gross | Sharpe net (5 bps/side) | Macro-F1 |")
    print("|---|---:|---:|---:|---:|---:|---:|")
    for key, label, _, _ in METHODS:
        m = metrics[key]
        print(f"| {label} | {fmt(m['ic_mean'], 4)} | {fmt(m['ic_tstat'], 2)} | {fmt(m['ic_hit_rate'] * 100, 1)}% "
              f"| {fmt(m['ls_sharpe_gross'], 2)} | {fmt(m['ls_sharpe_net'], 2)} | {fmt(m['macro_f1'])} |")
    print(f"\nsaved {path}")


if __name__ == "__main__":
    main()
