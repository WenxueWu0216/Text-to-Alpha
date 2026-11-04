"""Map each news item to the first moment we could actually trade on it, and build labels.

Rule: a news item is tradable at the first market OPEN (09:30 New York time) strictly
after it was published. The label is the return from that open to the open h days later.

  published Tue 08:00 ET  -> trade at Tue open
  published Tue 11:00 ET  -> trade at Wed open  (Tue open already happened)
  published Fri 18:00 ET  -> trade at Mon open  (or Tue if Mon is a holiday)
"""
from __future__ import annotations

import numpy as np
import pandas as pd

NY = "America/New_York"
MARKET_OPEN = pd.Timedelta(hours=9, minutes=30)


def to_effective_local_time(ts_utc: pd.Series, date_only: pd.Series) -> pd.Series:
    """Convert to naive New York time.

    Date-only rows (no real publish time) are pushed to 23:59 of that date: we don't know
    WHEN that day it came out, so we assume the worst case (after the close). Assuming
    midnight instead would let us 'trade' at that morning's open before the news existed.
    """
    local = ts_utc.dt.tz_convert(NY).dt.tz_localize(None)
    end_of_day = ts_utc.dt.tz_localize(None).dt.normalize() + pd.Timedelta(hours=23, minutes=59)
    return local.where(~date_only, end_of_day)


def assign_trade_date(local_time: pd.Series, trading_days: pd.DatetimeIndex) -> pd.Series:
    """First trading day whose 09:30 open is strictly after the publish time."""
    day = local_time.dt.normalize()
    before_open = (local_time - day) < MARKET_OPEN
    candidate = day.where(before_open, day + pd.Timedelta(days=1))
    days = trading_days.sort_values().values
    pos = np.searchsorted(days, candidate.values, side="left")  # first trading day >= candidate
    out = pd.Series(pd.NaT, index=local_time.index, dtype="datetime64[ns]")
    ok = pos < len(days)
    out[ok] = days[pos[ok]]
    return out


def build_panel(prices: pd.DataFrame, horizon: int, top_q: float) -> pd.DataFrame:
    """One row per (date, ticker): forward return, market-excess return, 3-class label."""
    px = prices.sort_values(["ticker", "date"]).copy()
    px["fwd_ret"] = px.groupby("ticker")["open"].shift(-horizon) / px["open"] - 1.0
    px = px.dropna(subset=["fwd_ret"])

    # Subtract the equal-weight universe mean each day -> we predict RELATIVE moves,
    # which is what a market-neutral long/short book actually earns.
    px["excess_ret"] = px["fwd_ret"] - px.groupby("date")["fwd_ret"].transform("mean")

    # Rank within each day across the WHOLE universe (not only names with news),
    # so labels are comparable across days with different news coverage.
    # Symmetric percentile (rank-1)/(n-1) in [0, 1]. pandas' rank(pct=True) is rank/n,
    # which is asymmetric (worst stock gets 1/n, best gets 1) and biases the labels.
    g = px.groupby("date")["excess_ret"]
    pct = (g.rank() - 1) / (g.transform("size") - 1)
    px["label"] = np.select([pct >= 1 - top_q, pct <= top_q], [1, -1], default=0)
    return px[["date", "ticker", "fwd_ret", "excess_ret", "label"]].reset_index(drop=True)


def align_news(news: pd.DataFrame, panel: pd.DataFrame, trading_days: pd.DatetimeIndex) -> pd.DataFrame:
    """Attach trade_date + labels to each news item. Drops items with no future price.

    trading_days must come from the raw price table (not the panel): the panel drops the
    last h days, and mapping news onto a *later* day than the true next open would be wrong.
    """
    local = to_effective_local_time(news["ts_utc"], news["date_only"])
    events = news.assign(local_time=local, trade_date=assign_trade_date(local, trading_days))
    events = events.merge(
        panel.rename(columns={"date": "trade_date"}), on=["trade_date", "ticker"], how="inner"
    )
    return events.sort_values(["trade_date", "ticker"]).reset_index(drop=True)
