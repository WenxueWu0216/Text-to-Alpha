"""Sample news events for LLM scoring within an API budget."""
from __future__ import annotations

import numpy as np
import pandas as pd


def cap_per_stock_day(events: pd.DataFrame, k: int, seed: int) -> pd.DataFrame:
    """Keep at most k random news per (trade_date, ticker).

    The signal is averaged per stock-day anyway, so the 3rd..Nth article on the same
    stock-day adds little. Capping spreads the API budget over MORE days, which is
    what the cross-sectional IC's t-stat depends on.
    """
    shuffled = events.sample(frac=1.0, random_state=seed)
    return shuffled.groupby(["trade_date", "ticker"]).head(k)


def sample_by_day(events: pd.DataFrame, target: int | None, seed: int) -> pd.DataFrame:
    """Sample whole trading DAYS, not individual articles.

    IC and the long-short book are cross-sectional: they need many stocks on the same day.
    Sampling random articles would leave ~1-2 names per day and make every day unusable.
    """
    if target is None or len(events) <= target:
        return events
    rng = np.random.default_rng(seed)
    days = np.array(events["trade_date"].unique())
    rng.shuffle(days)
    counts = events.groupby("trade_date").size().reindex(days)
    keep = counts.index[counts.cumsum() <= target]
    return events[events["trade_date"].isin(keep)]


def sample_days(events: pd.DataFrame, n_days: int | None, seed: int) -> pd.DataFrame:
    """Keep all articles on n_days randomly chosen trading days (whole cross-sections)."""
    days = np.array(sorted(events["trade_date"].unique()))
    if n_days is None or n_days >= len(days):
        return events
    keep = np.random.default_rng(seed).choice(days, size=n_days, replace=False)
    return events[events["trade_date"].isin(keep)]


OUTCOME_TEXT = {1: "outperformed (top 30% of the universe)",
                -1: "underperformed (bottom 30% of the universe)",
                0: "in line (middle 40%)"}


def select_fewshot_examples(train: pd.DataFrame, per_class: dict[int, int], seed: int) -> list[dict]:
    """Few-shot demonstrations drawn ONLY from the train split.

    Restricted to stock-days with exactly one article, so the realised move can be
    attributed to that article rather than to a cluster of news.
    """
    if set(train["split"].unique()) != {"train"}:
        raise ValueError("few-shot examples must come from the train split only")
    solo = train.groupby(["trade_date", "ticker"])["news_id"].transform("size") == 1
    pool = train[solo & (train["body"].str.len() > 200)]
    picks = [pool[pool["label"] == c].sample(n, random_state=seed + i)
             for i, (c, n) in enumerate(sorted(per_class.items()))]
    ex = pd.concat(picks).sample(frac=1.0, random_state=seed)  # interleave classes
    return [{"news_id": r.news_id, "ticker": r.ticker, "title": r.title, "body": r.body,
             "label": int(r.label), "outcome": OUTCOME_TEXT[int(r.label)],
             "trade_date": str(pd.Timestamp(r.trade_date).date())} for r in ex.itertuples()]
