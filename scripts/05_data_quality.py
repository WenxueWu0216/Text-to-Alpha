"""Step A5: data quality report from the processed data (+ raw counts when step 1 saved them).

Writes results/data_quality.json and docs/data_quality.md. The markdown is generated from
the JSON, so every number in it comes from this run.
"""
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd

from t2a.config import REPO_ROOT, load_config
from t2a.data.align import assign_trade_date, to_effective_local_time
from t2a.data.news import normalize_title
from t2a.provenance import write_result

RAW_KEYS = ["raw_rows", "missing_ticker", "in_universe", "dropped_empty_title",
            "dropped_bad_timestamp", "dropped_out_of_range", "duplicates_removed", "kept_after_dedup"]


def mentions_company(df: pd.DataFrame, aliases: dict[str, list[str]]) -> pd.Series:
    """Does the headline or lead name the tagged company at all? A proxy for ticker-tagging
    errors in FNSPID (an article about another firm filed under this symbol)."""
    out = pd.Series(False, index=df.index)
    for t, g in df.groupby("ticker"):
        names = sorted(aliases[t] + [t], key=len, reverse=True)
        pat = re.compile("|".join(rf"(?<![A-Za-z]){re.escape(n)}(?![A-Za-z])" for n in names),
                         flags=0)
        pat_ci = re.compile("|".join(rf"(?<![A-Za-z]){re.escape(n)}(?![A-Za-z])"
                                     for n in names if len(n) > 3), flags=re.IGNORECASE)
        text = g["title"] + " " + g["body"]
        hit = text.str.contains(pat)
        if pat_ci.pattern:
            hit |= text.str.contains(pat_ci)
        out[g.index] = hit
    return out


def main():
    cfg = load_config()
    P = cfg["paths"]
    news = pd.read_parquet(P["news"])
    prices = pd.read_parquet(P["prices"], columns=["date", "ticker"])
    # Counts only (no labels or returns): reading the split column of events is safe.
    ev = pd.read_parquet(P["events"], columns=["news_id", "split"])
    trading_days = pd.DatetimeIndex(prices["date"].unique())

    n = len(news)
    local = to_effective_local_time(news["ts_utc"], news["date_only"])
    trade = assign_trade_date(local, trading_days)
    timed = news[~news["date_only"]]
    et_hour = timed["ts_utc"].dt.tz_convert("America/New_York").dt.hour

    tn = normalize_title(news["title"])
    cross_ticker = news.assign(tn=tn).groupby("tn")["ticker"].nunique()
    same_day_dupes = news.assign(tn=tn, d=news["ts_utc"].dt.date).duplicated(["ticker", "tn", "d"]).sum()
    names = mentions_company(news, cfg["aliases"])

    per_ticker = news["ticker"].value_counts()
    zero_news = sorted(set(cfg["tickers"]) - set(per_ticker.index))
    missing_px = sorted(set(cfg["tickers"]) - set(prices["ticker"].unique()))

    raw_path = P["data_quality"]
    raw = json.loads(raw_path.read_text()) if raw_path.exists() else None

    q = {
        "raw_counts": raw if raw else {
            "status": "unavailable",
            "reason": "The raw FNSPID CSV (21.6 GB) is not present in this session; "
                      "scripts/01_build_dataset.py saves these counts on its next run from raw",
            "fields": RAW_KEYS},
        "news_items": n,
        "date_range_utc": [str(news["ts_utc"].min()), str(news["ts_utc"].max())],
        "per_year": {str(k): int(v) for k, v in news["ts_utc"].dt.year.value_counts().sort_index().items()},
        "timestamps": {
            "date_only": int(news["date_only"].sum()), "date_only_share": float(news["date_only"].mean()),
            "with_time_of_day": len(timed),
            "with_time_premarket_share": float((et_hour < 9).mean()) if len(timed) else None,
            "unparseable_in_processed": int(news["ts_utc"].isna().sum()),
            "handling": "date-only rows are treated as published 23:59 New York time (after the "
                        "close), so they trade at the NEXT open; unparseable rows are dropped in step 1",
        },
        "duplicates": {
            "residual_same_ticker_same_title_same_day": int(same_day_dupes),
            "titles_tagged_to_multiple_tickers": int((cross_ticker > 1).sum()),
            "handling": "same ticker + normalised title within 3 days -> keep the earliest copy; "
                        "the same story tagged to several tickers is kept once per ticker",
        },
        "ticker_mapping": {
            "universe_size": len(cfg["tickers"]),
            "tickers_with_zero_news": zero_news,
            "tickers_missing_prices": missing_px,
            "articles_not_naming_tagged_company": int((~names).sum()),
            "articles_not_naming_tagged_company_share": float((~names).mean()),
            "articles_per_ticker_min_median_max": [int(per_ticker.min()), int(per_ticker.median()),
                                                   int(per_ticker.max())],
            "handling": "FNSPID's Stock_symbol is taken as given; articles that never name the company "
                        "are kept (dropping them would use a hand-made filter not applied to all methods) "
                        "and reported here as a known source of label noise",
        },
        "alignment": {
            "no_tradable_open_after_publication": int(trade.isna().sum()),
            "news_without_price_or_in_embargo": int(n - len(ev)),
            "events_by_split": {k: int(v) for k, v in ev["split"].value_counts().items()},
            "body_empty_share": float((news["body"].str.len() == 0).mean()),
        },
    }
    write_result("data_quality", q, cfg)
    write_markdown(q, REPO_ROOT / "docs" / "data_quality.md")
    print(json.dumps(q, indent=1, default=str)[:3000])


def write_markdown(q: dict, path: Path) -> None:
    t, d, m, a = q["timestamps"], q["duplicates"], q["ticker_mapping"], q["alignment"]
    raw = q["raw_counts"]
    years = " | ".join(f"{y}: {c:,}" for y, c in q["per_year"].items())
    raw_md = ("Duplicates removed, unparseable timestamps and missing symbols: **unavailable**. " + raw["reason"] + "." if raw.get("status") == "unavailable"
              else "\n".join(f"- {k}: {raw.get(k, 'n/a'):,}" for k in RAW_KEYS))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(f"""# Data quality report

Generated by `scripts/05_data_quality.py` from the processed data; numbers are also in
`results/data_quality.json` (with git commit and config).

## Raw-file filtering (step 1)
{raw_md}

## Coverage
- News items after cleaning and dedup: **{q['news_items']:,}** ({q['date_range_utc'][0][:10]} to {q['date_range_utc'][1][:10]})
- Per year: {years}
- Articles per ticker (min / median / max): {' / '.join(f'{x:,}' for x in m['articles_per_ticker_min_median_max'])}
- Universe tickers with no news: {', '.join(m['tickers_with_zero_news']) or 'none'}; without prices: {', '.join(m['tickers_missing_prices']) or 'none'}
{'- META has no articles: FNSPID most likely files it under its pre-2022 symbol FB, which is not in the universe. Not verified (needs the raw CSV); fixing it means adding FB as an alias of META and rebuilding from raw.' if 'META' in m['tickers_with_zero_news'] else ''}

## Timestamps
- Date-only (no time of day): **{t['date_only']:,} ({t['date_only_share']:.1%})**; with a real time: {t['with_time_of_day']:,}
- Unparseable timestamps remaining in processed data: {t['unparseable_in_processed']}
- Handling: {t['handling']}.

## Duplicates
- Residual same-ticker, same-title, same-day duplicates after dedup: {d['residual_same_ticker_same_title_same_day']:,}
- Titles tagged to more than one ticker: {d['titles_tagged_to_multiple_tickers']:,}
- Handling: {d['handling']}.

## Ticker mapping
- Articles whose headline + lead never name the tagged company: **{m['articles_not_naming_tagged_company']:,} ({m['articles_not_naming_tagged_company_share']:.1%})**
- Handling: {m['handling']}.

## Alignment to tradable opens
- No tradable open after publication: {a['no_tradable_open_after_publication']:,}
- News dropped at alignment (no price on the trade date) or in an embargo gap: {a['news_without_price_or_in_embargo']:,}
- Events by split: {', '.join(f'{k} {v:,}' for k, v in sorted(a['events_by_split'].items()))}
- Empty article body: {a['body_empty_share']:.1%}
""")


if __name__ == "__main__":
    main()
