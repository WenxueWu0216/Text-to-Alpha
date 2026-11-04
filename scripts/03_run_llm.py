"""Step 3: zero-shot LLM signals on a day-sampled subset, compared with FinBERT on the SAME subset."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd
from dotenv import load_dotenv

from t2a.config import load_config
from t2a.data.io import load_events
from t2a.data.news import mask_entities
from t2a.eval.report import report
from t2a.signals.aggregate import to_stock_day
from t2a.signals.llm_zero_shot import LLMSignalExtractor
from t2a.signals.sampling import cap_per_stock_day, sample_by_day


def main():
    load_dotenv()  # ANTHROPIC_API_KEY from .env
    cfg = load_config()
    P, L = cfg["paths"], cfg["llm"]
    events = load_events(cfg)

    # train sample -> future distillation labels; val sample -> benchmark. Never test.
    capped = cap_per_stock_day(events, L["max_news_per_stock_day"], L["seed"])
    sub = pd.concat([sample_by_day(capped[capped["split"] == s], L["sample_size"], L["seed"])
                     for s in ["train", "val"]])
    print(f"LLM subset: {len(sub):,} items on {sub['trade_date'].nunique()} days")
    for s, g in sub.groupby("split"):
        print(f"  {s}: {g['trade_date'].nunique()} distinct trade_dates")

    titles, bodies = sub["title"].tolist(), sub["body"].tolist()
    if L["mask_entities"]:
        titles = [mask_entities(t, k, cfg["aliases"][k]) for t, k in zip(titles, sub["ticker"])]
        bodies = [mask_entities(b, k, cfg["aliases"][k]) for b, k in zip(bodies, sub["ticker"])]

    extractor = LLMSignalExtractor(L["model"], P["llm_cache"], L["max_concurrency"])
    res = extractor.score(titles, bodies)
    res.index = sub.index
    sub = sub.join(res)

    n_err = res["error"].notna().sum() if "error" in res else 0
    print(f"errors: {n_err} | input tokens: {res['input_tokens'].sum():,.0f} | "
          f"output tokens: {res['output_tokens'].sum():,.0f}")

    out = P["signals_dir"]
    out.mkdir(parents=True, exist_ok=True)
    keep = ["news_id", "llm_signal", "expected_impact", "relevance", "surprise", "confidence",
            "event_type", "rationale", "input_tokens", "output_tokens"]
    sub[[c for c in keep if c in sub]].to_parquet(out / "llm_zeroshot_events.parquet")

    ok = sub.dropna(subset=["llm_signal"])
    report(to_stock_day(ok, "llm_signal"), cfg, "llm_zeroshot")

    # Fair comparison: FinBERT on exactly the same articles.
    fb_path = out / "finbert_events.parquet"
    if fb_path.exists():
        fb = ok.merge(pd.read_parquet(fb_path), on="news_id")
        report(to_stock_day(fb, "finbert"), cfg, "finbert_same_subset")


if __name__ == "__main__":
    main()
