"""Load and clean FNSPID news.

The raw FNSPID news CSV is several GB, so we stream it in chunks and keep only
the tickers and dates we need. Output: one row per unique news item.
"""
from __future__ import annotations

import hashlib
import re
from collections import Counter
from pathlib import Path

import pandas as pd

RAW_COLS = ["Date", "Article_title", "Stock_symbol", "Publisher", "Article"]
ARTICLE_CHARS = 1500  # keep only the lead; headlines + first paragraphs carry most signal


def parse_utc_timestamps(raw: pd.Series) -> tuple[pd.Series, pd.Series]:
    """Parse FNSPID timestamps (e.g. '2023-12-16 22:00:00 UTC').

    Returns (timestamp_utc, date_only). date_only marks rows whose time is exactly
    00:00:00 UTC, which almost always means the source only had a date, not a time.
    """
    cleaned = raw.astype(str).str.replace(r"\s*UTC$", "", regex=True)
    ts = pd.to_datetime(cleaned, utc=True, errors="coerce", format="mixed")
    date_only = (ts.dt.hour == 0) & (ts.dt.minute == 0) & (ts.dt.second == 0)
    return ts, date_only


def normalize_title(title: pd.Series) -> pd.Series:
    """Lowercase + strip punctuation/whitespace, so syndicated copies collide."""
    return (
        title.str.lower()
        .str.replace(r"[^a-z0-9 ]", " ", regex=True)
        .str.replace(r"\s+", " ", regex=True)
        .str.strip()
    )


def make_news_id(df: pd.DataFrame) -> pd.Series:
    """Deterministic id, so caches and joins stay stable across reruns."""
    keys = df["ticker"] + "|" + df["title_norm"] + "|" + df["ts_utc"].astype(str)
    return keys.map(lambda k: hashlib.sha1(k.encode()).hexdigest()[:16])


def clean_chunk(chunk: pd.DataFrame, tickers: set[str], start: str, end: str,
                stats: Counter | None = None) -> pd.DataFrame:
    """Filter one raw chunk. If `stats` is given, every dropped row is counted by reason."""
    stats = stats if stats is not None else Counter()
    stats["raw_rows"] += len(chunk)
    chunk = chunk.rename(
        columns={"Date": "raw_date", "Article_title": "title", "Stock_symbol": "ticker",
                 "Publisher": "publisher", "Article": "body"}
    )
    sym = chunk["ticker"].astype("string").str.upper().str.strip()
    stats["missing_ticker"] += int(sym.isna().sum() + (sym == "").sum())
    chunk["ticker"] = sym.fillna("")
    chunk = chunk[chunk["ticker"].isin(tickers)]
    stats["in_universe"] += len(chunk)
    if chunk.empty:
        return chunk

    chunk["title"] = chunk["title"].fillna("").astype(str).str.strip()
    empty = chunk["title"].str.len() == 0
    stats["dropped_empty_title"] += int(empty.sum())
    chunk = chunk[~empty]

    ts, date_only = parse_utc_timestamps(chunk["raw_date"])
    stats["dropped_bad_timestamp"] += int(ts.isna().sum())
    chunk = chunk.assign(ts_utc=ts, date_only=date_only).dropna(subset=["ts_utc"])
    lo, hi = pd.Timestamp(start, tz="UTC"), pd.Timestamp(end, tz="UTC") + pd.Timedelta(days=1)
    in_range = (chunk["ts_utc"] >= lo) & (chunk["ts_utc"] < hi)
    stats["dropped_out_of_range"] += int((~in_range).sum())
    chunk = chunk[in_range]

    chunk["body"] = chunk["body"].fillna("").astype(str).str.slice(0, ARTICLE_CHARS)
    return chunk[["ts_utc", "date_only", "ticker", "title", "body", "publisher"]]


def load_news(csv_path: Path, tickers: list[str], start: str, end: str,
              chunksize: int = 200_000) -> tuple[pd.DataFrame, dict]:
    """Returns (deduplicated news, counts of every filtering step)."""
    tickers_set = set(tickers)
    parts = []
    stats: Counter = Counter()
    reader = pd.read_csv(csv_path, usecols=lambda c: c in RAW_COLS, chunksize=chunksize,
                         dtype=str, on_bad_lines="skip")
    for i, chunk in enumerate(reader):
        for col in RAW_COLS:  # some FNSPID shards lack columns; keep the schema fixed
            if col not in chunk.columns:
                chunk[col] = None
        part = clean_chunk(chunk, tickers_set, start, end, stats)
        if not part.empty:
            parts.append(part)
        if i % 10 == 0:
            print(f"  chunk {i}: kept {sum(len(p) for p in parts):,} rows so far")

    if not parts:
        raise ValueError("No news rows matched the universe/date range. Check the CSV path and tickers.")
    news = pd.concat(parts, ignore_index=True)
    stats["kept_before_dedup"] = len(news)
    news = deduplicate(news)
    stats["duplicates_removed"] = stats["kept_before_dedup"] - len(news)
    stats["kept_after_dedup"] = len(news)
    stats["date_only_timestamps"] = int(news["date_only"].sum())
    # pandas' on_bad_lines="skip" drops malformed CSV lines silently; they are invisible here.
    return news, dict(stats)


def deduplicate(news: pd.DataFrame, window: str = "3D") -> pd.DataFrame:
    """Same story from several publishers -> keep the EARLIEST copy.

    The earliest timestamp is when the information actually hit the market; a later
    copy would make stale information look fresh. We only collapse copies within a
    short window, because generic titles ("Apple stock falls") recur on different days
    and those are genuinely different news.
    """
    news = news.copy()
    news["title_norm"] = normalize_title(news["title"])
    news = news.sort_values(["ticker", "title_norm", "ts_utc"])
    gap = news.groupby(["ticker", "title_norm"])["ts_utc"].diff()
    news = news[gap.isna() | (gap > pd.Timedelta(window))]
    news["news_id"] = make_news_id(news)
    return news.sort_values("ts_utc").reset_index(drop=True)


_WS = re.compile(r"\s+")


def mask_entities(text: str, ticker: str, aliases: list[str]) -> str:
    """Replace the company's names/ticker with a placeholder.

    Why: an LLM trained on data after the news date may 'remember' how the stock
    moved. Hiding identity forces it to judge the text itself (reduces look-ahead bias).
    """
    patterns = sorted(aliases + [ticker], key=len, reverse=True)  # longest first: "JP Morgan" before "JP"
    for p in patterns:
        text = re.sub(rf"(?<![A-Za-z]){re.escape(p)}(?![A-Za-z])", "the Company", text,
                      flags=re.IGNORECASE if len(p) > 3 else 0)  # short tickers: exact case only
    return _WS.sub(" ", text).strip()
