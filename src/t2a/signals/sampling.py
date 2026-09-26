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
