"""Step 1: raw news + prices -> aligned, labeled, split event table."""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd

from t2a.config import ensure_parent, load_config
from t2a.data.align import align_news, build_panel
from t2a.data.news import load_news
from t2a.data.prices import download_prices
from t2a.data.split import assign_split


def main():
    cfg = load_config()
    P = cfg["paths"]
    start, end = cfg["dates"]["start"], cfg["dates"]["end"]

    print("[1/4] loading news ...")
    news, raw_stats = load_news(P["raw_news_csv"], cfg["tickers"], start, end)
    news.to_parquet(ensure_parent(P["news"]))
    # Raw-file counts can only be measured here (the raw CSV is ~22 GB and not kept),
    # so they are saved for the data quality report.
    ensure_parent(P["data_quality"]).write_text(json.dumps(raw_stats, indent=2))
    print("raw filtering counts:", json.dumps(raw_stats))

    print("[2/4] downloading prices ...")
    prices = download_prices(cfg["tickers"], start, end)
    prices.to_parquet(ensure_parent(P["prices"]))
    trading_days = pd.DatetimeIndex(prices["date"].unique())

    print("[3/4] building labels and aligning news ...")
    panel = build_panel(prices, cfg["labels"]["horizon_days"], cfg["labels"]["top_quantile"])
    s = cfg["split"]
    panel["split"] = assign_split(panel["date"], trading_days, s["train_end"], s["val_end"], s["embargo_days"])
    events = align_news(news, panel, trading_days)
    events = events[events["split"] != "embargo"]

    print("[4/4] saving ...")
    panel.to_parquet(P["panel"])
    events.to_parquet(P["events"])

    # Sanity report: read these numbers before trusting anything downstream.
    print(f"\nnews items: {len(news):,}  |  date-only timestamps: {news['date_only'].mean():.1%}")
    print(f"aligned events: {len(events):,}")
    print(events.groupby("split").agg(events=("news_id", "size"),
                                      days=("trade_date", "nunique"),
                                      tickers=("ticker", "nunique")))
    per_day = events.groupby("trade_date")["ticker"].nunique()
    k = cfg["eval"]["min_names_per_day"]
    print(f"names with news per day: median {per_day.median():.0f}, days with >= {k}: {(per_day >= k).mean():.1%}")
    print("label distribution:", events["label"].value_counts(normalize=True).round(3).to_dict())


if __name__ == "__main__":
    main()
