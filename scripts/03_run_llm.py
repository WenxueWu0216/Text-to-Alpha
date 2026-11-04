"""Step A3/A4: Claude zero-shot / few-shot signals on validation, with the shared evaluator.

  python scripts/03_run_llm.py --variant zero_shot --estimate-only   # prints cost, no API calls
  python scripts/03_run_llm.py --variant zero_shot                   # runs if within budget

Flow: select articles -> cache lookup -> estimate (exact input tokens via count_tokens) ->
budget gate -> small sync sample (real latency) -> re-estimate with measured usage ->
gate -> Batches API for the rest -> evaluate. Reruns are served from the cache for free.
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import pandas as pd

from t2a.config import load_config
from t2a.data.io import load_events
from t2a.data.news import mask_entities
from t2a.eval.llm_analysis import by_event_type, latency_summary, status_summary
from t2a.eval.report import evaluate_splits, regime_labels, report
from t2a.signals.aggregate import to_stock_day
from t2a.signals.budget import BudgetExceeded, Ledger, check_budget
from t2a.signals.llm import (MAX_TOKENS, PROMPT_VERSION, ClaudeScorer, Prices, build_system,
                             build_user_message, cache_key, records_to_frame)
from t2a.signals.sampling import cap_per_stock_day, sample_days, select_fewshot_examples

N_COUNT_SAMPLE = 25


def masked(df: pd.DataFrame, col: str, cfg: dict) -> list[str]:
    if not cfg["llm"]["mask_entities"]:
        return df[col].tolist()
    return [mask_entities(t, k, cfg["aliases"][k]) for t, k in zip(df[col], df["ticker"])]


def select(cfg: dict, V: dict, seed: int) -> pd.DataFrame:
    events = load_events(cfg, splits=(V["split"],))
    sub = cap_per_stock_day(events, V["max_news_per_stock_day"], seed)
    return sample_days(sub, V.get("n_days"), seed).sort_values(["trade_date", "ticker"])


def fewshot_examples(cfg: dict, V: dict, seed: int) -> list[dict]:
    if not V.get("examples_per_class"):
        return []
    train = load_events(cfg, splits=("train",))
    ex = select_fewshot_examples(train, {int(k): v for k, v in V["examples_per_class"].items()}, seed)
    for e in ex:  # same masking as the scored articles
        e["title"] = mask_entities(e["title"], e["ticker"], cfg["aliases"][e["ticker"]])
        e["body"] = mask_entities(e["body"], e["ticker"], cfg["aliases"][e["ticker"]])
    return ex


def estimate(n_sync: int, n_batch: int, in_tok: float, out_tok: float, prices: Prices) -> float:
    return (prices.cost(n_sync * in_tok, n_sync * out_tok)
            + prices.cost(n_batch * in_tok, n_batch * out_tok, batch=True))


def spend_of(records: list[dict], prices: Prices) -> float:
    return sum(prices.cost(r.get("input_tokens") or 0, r.get("output_tokens") or 0,
                           batch=r.get("transport") == "batch") for r in records)


def gate(label: str, est: float, ledger: Ledger, L: dict, V: dict) -> None:
    B = L["budget"]
    reserve = B["reserve_teacher_usd"] if V.get("respect_teacher_reserve", True) else 0.0
    print(f"[{label}] estimate ${est:.2f} | spent so far ${ledger.spent():.2f} | "
          f"run cap ${V['max_usd']:.2f} | total cap ${B['total_usd']:.2f} (reserve ${reserve:.2f})")
    check_budget(est, ledger.spent(), V["max_usd"], B["total_usd"], reserve)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", choices=["zero_shot", "few_shot"], default="zero_shot")
    ap.add_argument("--estimate-only", action="store_true")
    args = ap.parse_args()

    cfg = load_config()
    P, L = cfg["paths"], cfg["llm"]
    V, seed = L["variants"][args.variant], L["seed"]
    prices = Prices(L["prices_per_mtok"]["input"], L["prices_per_mtok"]["output"])
    ledger = Ledger(P["llm_ledger"])
    scorer = ClaudeScorer(L["model"], P["llm_cache"], L["max_concurrency"])
    name = f"llm_{args.variant}"

    if scorer._pending() and not args.estimate_only:
        print("collecting batches left over from an earlier run ...")
        scorer.collect_batches()

    sub = select(cfg, V, seed)
    examples = fewshot_examples(cfg, V, seed)
    system = build_system(examples)
    messages = [build_user_message(t, b) for t, b in zip(masked(sub, "title", cfg), masked(sub, "body", cfg))]
    keys = [cache_key(L["model"], system, m) for m in messages]
    print(f"{name}: {len(sub):,} articles on {sub['trade_date'].nunique()} {V['split']} days "
          f"(<= {V['max_news_per_stock_day']}/stock-day), {len(examples)} few-shot examples")

    cached_before = scorer.cache.get_many(list(set(keys)))
    todo = list(dict.fromkeys(k for k in keys if k not in cached_before))  # dedupe, keep order
    msg_of = dict(zip(keys, messages))
    print(f"cache: {len(keys) - sum(k not in cached_before for k in keys):,} hits, {len(todo):,} to score")

    if todo:
        rng = np.random.default_rng(seed)
        sample = rng.choice(todo, size=min(N_COUNT_SAMPLE, len(todo)), replace=False)
        in_tok = float(np.mean(scorer.count_input_tokens(system, [msg_of[k] for k in sample])))
        n_sync = min(V["latency_sample"], len(todo))
        n_batch = len(todo) - n_sync
        est = estimate(n_sync, n_batch, in_tok, L["est_output_tokens"], prices)
        worst = estimate(n_sync, n_batch, in_tok, MAX_TOKENS, prices)
        print(f"estimate: {len(todo):,} calls ({n_sync} sync @ list price, {n_batch:,} batch @ 50%), "
              f"~{in_tok:.0f} input tokens/call (count_tokens on {len(sample)}), "
              f"{L['est_output_tokens']} output tokens/call assumed")
        print(f"          expected ${est:.2f}; worst case (every reply hits max_tokens={MAX_TOKENS}) ${worst:.2f}")
        try:
            gate("pre-run", est, ledger, L, V)
        except BudgetExceeded as e:
            sys.exit(f"STOP: {e}. Nothing was sent to the API.")
        if args.estimate_only:
            return

        sync_keys = list(rng.choice(todo, size=n_sync, replace=False)) if n_sync else []
        sync_out = scorer.score_sync(system, [(k, msg_of[k]) for k in sync_keys])
        failed = [r for r in sync_out if r.get("status") == "api_error"]
        if sync_out and len(failed) > len(sync_out) / 2:
            sys.exit(f"STOP: {len(failed)}/{len(sync_out)} sync calls failed, not submitting the batch. "
                     f"First error: {failed[0]['error']}")
        sync_recs = [r for r in scorer.cache.get_many(sync_keys).values()]
        ledger.record(f"{name}:sync", spend_of(sync_recs, prices), calls=len(sync_recs),
                      prompt_version=PROMPT_VERSION)

        rest = [k for k in todo if k not in set(sync_keys)]
        if rest:
            ok = [r for r in sync_recs if r.get("input_tokens")]
            m_in = np.mean([r["input_tokens"] for r in ok]) if ok else in_tok
            m_out = np.mean([r["output_tokens"] for r in ok]) if ok else L["est_output_tokens"]
            est2 = estimate(0, len(rest), m_in, m_out, prices)
            print(f"re-estimate with measured usage ({m_in:.0f} in / {m_out:.0f} out per call): "
                  f"batch of {len(rest):,} = ${est2:.2f}")
            try:
                gate("batch", est2, ledger, L, {**V, "max_usd": V["max_usd"] - spend_of(sync_recs, prices)})
            except BudgetExceeded as e:
                sys.exit(f"STOP before the batch: {e}. The sync sample is cached; rerun after review.")
            scorer.submit_batch(system, [(k, msg_of[k]) for k in rest])
            scorer.collect_batches()
            batch_recs = list(scorer.cache.get_many(rest).values())
            ledger.record(f"{name}:batch", spend_of(batch_recs, prices), calls=len(batch_recs),
                          prompt_version=PROMPT_VERSION)

    # ---- evaluate -----------------------------------------------------------------------
    cache_now = scorer.cache.get_many(list(set(keys)))
    res = records_to_frame([cache_now.get(k) for k in keys])
    res.index = sub.index
    sub = sub.join(res)
    out_dir = P["signals_dir"]
    keep = ["news_id", "trade_date", "ticker", "status", "direction", "confidence", "event_type",
            "rationale", "llm_signal", "llm_direction", "input_tokens", "output_tokens",
            "latency_s", "transport", "validation_error"]
    sub[[c for c in keep if c in sub]].to_parquet(out_dir / f"{name}_events.parquet")

    ok = sub[sub["status"] == "ok"]
    n_all = len(sub)
    tok_in, tok_out = sub["input_tokens"].sum(), sub["output_tokens"].sum()
    hits = sum(k in cached_before for k in keys)
    regimes, regime_info = regime_labels(cfg, pd.DatetimeIndex(ok["trade_date"].unique()))
    stock_day_dir = to_stock_day(ok, "llm_direction")
    baselines = {}
    for col, fname in [("finbert", "finbert_events.parquet"), ("lexicon", "lexicon_events.parquet")]:
        if (out_dir / fname).exists():
            b = ok.merge(pd.read_parquet(out_dir / fname), on="news_id")
            baselines[col] = evaluate_splits(to_stock_day(b, col), cfg, (V["split"],), regimes=regimes)
    extra = {
        "variant": args.variant, "model": L["model"], "prompt_version": PROMPT_VERSION,
        "n_articles": n_all, "n_days": int(sub["trade_date"].nunique()),
        "validity": status_summary(sub),
        "cache": {"hits": hits, "misses": n_all - hits, "hit_rate": hits / n_all},
        "latency": latency_summary(sub.loc[sub["transport"] == "sync", "latency_s"]),
        "tokens": {"input": float(tok_in), "output": float(tok_out),
                   "input_per_article": float(tok_in / n_all), "output_per_article": float(tok_out / n_all)},
        "cost_usd_per_1k_articles": {
            "list_price": prices.cost(tok_in, tok_out) / n_all * 1000,
            "batch_price": prices.cost(tok_in, tok_out, batch=True) / n_all * 1000},
        "spend_usd_total_ledger": ledger.spent(),
        "signal_definition": "direction (+1/0/-1) x confidence, mean per stock-day",
        "ablations": {"direction_only": evaluate_splits(stock_day_dir, cfg, (V["split"],), regimes=regimes)},
        "ic_by_event_type": by_event_type(ok),
        "same_subset_baselines": baselines,
        "fewshot_examples": [{k: e[k] for k in ("news_id", "trade_date", "label")} for e in examples],
    }
    print(json.dumps({k: extra[k] for k in ("validity", "cache", "latency", "cost_usd_per_1k_articles")},
                     indent=1, default=float))
    report(to_stock_day(ok, "llm_signal"), cfg, name, splits=(V["split"],), extra=extra)


if __name__ == "__main__":
    main()
