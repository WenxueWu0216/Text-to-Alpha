"""Market-regime labels for performance breakdowns (high vs low VIX)."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def download_vix(start: str, end: str) -> pd.Series:
    import yfinance as yf  # lazy: tests and CI never touch the network

    raw = yf.download("^VIX", start=start, end=end, auto_adjust=False, progress=False)
    close = raw["Close"]
    if isinstance(close, pd.DataFrame):  # yfinance returns (field, ticker) columns
        close = close.iloc[:, 0]
    close.index = pd.to_datetime(close.index).tz_localize(None).normalize()
    return close.dropna().rename("vix")


def load_or_download_vix(path: Path, start: str, end: str) -> pd.Series | None:
    if path.exists():
        return pd.read_parquet(path)["vix"]
    try:
        vix = download_vix(start, end)
    except Exception as e:  # offline: regime breakdown is optional, the rest still runs
        print(f"VIX unavailable ({e!r}); skipping the VIX regime breakdown")
        return None
    path.parent.mkdir(parents=True, exist_ok=True)
    vix.to_frame().to_parquet(path)
    return vix


def prior_close(vix: pd.Series, dates: pd.DatetimeIndex) -> pd.Series:
    """VIX close of the last day STRICTLY before each date.

    We trade at the open of D, so D's own close is not yet known.
    """
    v = vix.sort_index()
    pos = np.searchsorted(v.index.values, pd.DatetimeIndex(dates).values, side="left") - 1
    vals = np.where(pos >= 0, v.values[np.clip(pos, 0, None)], np.nan)
    return pd.Series(vals, index=pd.DatetimeIndex(dates))


# Conventional levels fixed a priori (no sample statistic), so the split leaks nothing.
# A train-median split was tried first: in 2022 (val) the prior-day VIX sat above the
# 2018-2021 median on 245 of 246 days, leaving the "low" bucket empty.
VIX_BUCKETS = ((20.0, "lt20"), (30.0, "20to30"), (np.inf, "ge30"))


def vix_regime(vix: pd.Series, dates: pd.DatetimeIndex, buckets=VIX_BUCKETS) -> pd.Series:
    prev = prior_close(vix, dates)
    labels = pd.Series(None, index=prev.index, dtype=object)
    for upper, name in reversed(buckets):
        labels[prev < upper] = name
    return labels
