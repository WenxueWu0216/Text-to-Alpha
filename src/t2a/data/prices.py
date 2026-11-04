"""Download daily OHLC prices and return a tidy long table: (date, ticker, open, close)."""
from __future__ import annotations

import pandas as pd


def download_prices(tickers: list[str], start: str, end: str, buffer_days: int = 15) -> pd.DataFrame:
    import yfinance as yf  # imported lazily so tests don't need network deps

    # Pad the end date: labels at the end of the sample need FUTURE prices (open at D+h).
    end_padded = (pd.Timestamp(end) + pd.Timedelta(days=buffer_days)).strftime("%Y-%m-%d")
    raw = yf.download(
        tickers, start=start, end=end_padded,
        auto_adjust=True,      # split/dividend-adjusted; otherwise a 4:1 split looks like a -75% return
        group_by="column", progress=False, threads=True,
    )
    return to_long(raw[["Open", "Close"]])


def to_long(wide: pd.DataFrame) -> pd.DataFrame:
    """yfinance gives columns (field, ticker); flatten to one row per (date, ticker)."""
    long = wide.stack(level=1, future_stack=True)
    long.columns = [str(c).lower() for c in long.columns]
    long.index.names = ["date", "ticker"]
    long = long.reset_index().dropna(subset=["open", "close"])
    long["date"] = pd.to_datetime(long["date"]).dt.tz_localize(None).dt.normalize()
    return long[["date", "ticker", "open", "close"]].sort_values(["ticker", "date"]).reset_index(drop=True)
