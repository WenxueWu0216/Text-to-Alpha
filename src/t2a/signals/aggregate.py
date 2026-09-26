"""Turn news-level scores into one signal per (trade_date, ticker).

A stock can have 0..N news items before one open. The portfolio trades stocks, not
articles, so we collapse to one number per stock-day before evaluating.
"""
from __future__ import annotations

import pandas as pd


def to_stock_day(events: pd.DataFrame, score_col: str) -> pd.DataFrame:
    """Mean score across the day's news; mean (not sum) so heavily covered names
    don't get a large signal just for being popular. n_news is kept as a feature."""
    g = events.groupby(["trade_date", "ticker"])
    out = g.agg(
        signal=(score_col, "mean"),
        n_news=(score_col, "size"),
        fwd_ret=("fwd_ret", "first"),
        excess_ret=("excess_ret", "first"),
        label=("label", "first"),
        split=("split", "first"),
    ).reset_index()
    return out.rename(columns={"trade_date": "date"})
