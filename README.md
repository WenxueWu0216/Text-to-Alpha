# Text-to-Alpha

Benchmarking NLP/LLM approaches for extracting trading signals from financial news.

**Hypothesis:** LLM-extracted event signals from news predict next-day relative stock
returns beyond classic sentiment baselines (FinBERT).

Every approach is scored by the same evaluator on the same data:
cross-sectional IC, long-short Sharpe (net of costs), classification metrics, and cost/latency.

## Status

| Step | What | Status |
|---|---|---|
| 1 | Data pipeline: FNSPID news + prices, look-ahead-safe alignment, labels, purged split | done |
| 2 | Shared evaluator + FinBERT baseline | done |
| 3 | Zero-shot LLM signal extraction (Claude API, structured output, cached) | done |
| 4 | RAG: fine-tuned retriever over company history | todo |
| 5 | QLoRA fine-tuning of an open-source LLM, distilled from step-3 labels | todo |
| 6 | Tool-calling agent | todo |
| 7 | Serving + final benchmark on the held-out test set | todo |

## Data

- News: [FNSPID](https://github.com/Zdong104/FNSPID_Financial_News_Dataset) (Dong et al., KDD 2024), CC BY-NC 4.0.
  Download `Stock_news/nasdaq_exteral_data.csv` into `data/raw/`.
- Prices: daily adjusted open/close from Yahoo Finance (`yfinance`).

## Key design decisions

- **Tradable time:** news is mapped to the first 09:30 ET open strictly after publication;
  date-only timestamps are treated as after-close. Labels are open(D) -> open(D+1) returns.
- **Market-neutral target:** returns minus the universe mean; labels are cross-sectional terciles.
- **Purged split:** chronological train / val / test with a trading-day embargo. Test is untouched until the end.
- **Entity masking:** company names/tickers are hidden in LLM prompts to reduce look-ahead
  from the LLM's pretraining knowledge.
- **Fair comparison:** the LLM runs on a day-sampled subset; FinBERT is re-scored on the same subset.

### Known data limitations

- 99.7% of FNSPID news timestamps are date-only (no time of day). Under the look-ahead-safe rule,
  these are treated as after-close, so every item trades at the NEXT day's open. The strategy
  therefore cannot capture same-day reaction and only measures post-publication drift. Expect
  small ICs.
- Ticker coverage in FNSPID declines over time (42 tickers with news in train, 29 in val, 33 in
  test), though the daily cross-section stays around 27 names.

## Run

```bash
pip install -r requirements.txt
cp .env.example .env            # add your Anthropic API key
pytest -q                       # look-ahead, split, and metric tests
python scripts/01_build_dataset.py
python scripts/02_run_finbert.py
python scripts/03_run_llm.py
python scripts/07_same_subset_table.py   # all methods on the same val articles, from cache
```

## Layout

```
configs/config.yaml        universe, dates, split, eval and model settings
src/t2a/data/              news cleaning, prices, alignment + labels, purged split
src/t2a/signals/           FinBERT scorer, LLM extractor, stock-day aggregation
src/t2a/eval/              IC / long-short / classification metrics, per-split report
scripts/                   one script per pipeline step
tests/                     unit tests for the parts where bugs are silent
service/                   FastAPI FinBERT service + Dockerfile (see service/README.md)
```
