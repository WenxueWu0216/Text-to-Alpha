"""Chronological train/val/test split with an embargo gap.

Why not random split: news for the same event is spread over several days and
returns are autocorrelated, so a random split leaks the future into training.

Why the embargo: a label on the last train day D uses prices up to D+h. If validation
starts at D+1, train labels and val labels overlap in time. Dropping `embargo` trading
days after each boundary removes that overlap.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def assign_split(dates: pd.Series, trading_days: pd.DatetimeIndex,
                 train_end: str, val_end: str, embargo: int) -> pd.Series:
    days = trading_days.sort_values()
    # boundaries are defined in trading days, not calendar days (holidays/weekends)
    def first_after(boundary: str, skip: int) -> pd.Timestamp:
        idx = days.searchsorted(pd.Timestamp(boundary), side="right") + skip
        return days[min(idx, len(days) - 1)]

    val_start = first_after(train_end, embargo)
    test_start = first_after(val_end, embargo)

    d = pd.to_datetime(dates)
    split = np.select(
        [d <= pd.Timestamp(train_end),
         (d >= val_start) & (d <= pd.Timestamp(val_end)),
         d >= test_start],
        ["train", "val", "test"],
        default="embargo",
    )
    return pd.Series(split, index=dates.index)
