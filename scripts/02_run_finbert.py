"""Step 2: FinBERT baseline scores + shared evaluation.

Scores are cached in data/signals/finbert_events.parquet; only articles without a
cached score are run through the model (a full rescore takes ~85 min on an M4).
"""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd

from t2a.config import load_config
from t2a.data.io import load_events
from t2a.eval.report import report
from t2a.signals.aggregate import to_stock_day


def main():
    cfg = load_config()
    P, fb = cfg["paths"], cfg["finbert"]
    events = load_events(cfg)  # train + val only; test stays locked
    out = P["signals_dir"]
    out.mkdir(parents=True, exist_ok=True)
    cache_path = out / "finbert_events.parquet"

    cached = pd.read_parquet(cache_path) if cache_path.exists() else pd.DataFrame(columns=["news_id", "finbert"])
    todo = events[~events["news_id"].isin(cached["news_id"])]
    latency = None
    if len(todo):
        from t2a.signals.finbert import FinBertScorer  # torch import only when needed

        scorer = FinBertScorer(fb["model"], fb["max_length"])
        print(f"scoring {len(todo):,} uncached headlines on {scorer.device} ...")
        t0 = time.perf_counter()
        new = todo[["news_id"]].assign(finbert=scorer.score(todo["title"].tolist(), fb["batch_size"]))
        latency = {"device": str(scorer.device), "n": len(todo),
                   "seconds": time.perf_counter() - t0}
        cached = pd.concat([cached, new], ignore_index=True)
        cached.to_parquet(cache_path)
    print(f"FinBERT scores: {len(events) - len(todo):,} from cache, {len(todo):,} newly scored")

    events = events.merge(cached, on="news_id", how="left")
    extra = {"scored_text": "title", "n_articles": len(events),
             "cost_usd_per_1k_articles": 0.0,  # local model: no API cost
             "cache": {"hits": len(events) - len(todo), "misses": len(todo)}}
    if latency:
        extra["batch_throughput"] = latency
    report(to_stock_day(events, "finbert"), cfg, "finbert", extra=extra)


if __name__ == "__main__":
    main()
