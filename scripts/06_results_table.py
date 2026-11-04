"""Print markdown result tables straight from results/*.json (no hand-copied numbers).

  python scripts/06_results_table.py            # validation tables
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from t2a.provenance import RESULTS_DIR


def load(name: str) -> dict | None:
    p = RESULTS_DIR / f"{name}.json"
    return json.loads(p.read_text()) if p.exists() else None


def fmt(x, nd=3):
    return "n/a" if x is None else f"{x:.{nd}f}"


def row(label: str, m: dict, cost: str, latency: str) -> str:
    return (f"| {label} | {m['n_obs']:,} | {m['n_days']} | {fmt(m['macro_f1'])} | {fmt(m['ic_mean'])} "
            f"({fmt(m['ic_tstat'], 2)}) | {fmt(m['ls_sharpe_gross'], 2)} | {fmt(m['ls_sharpe_net'], 2)} | "
            f"{fmt(m['breakeven_cost_bps_per_side'], 1)} | {fmt(m['turnover_mean'], 2)} | {cost} | {latency} |")


HEADER = ("| Method | Stock-days | Days | Macro-F1 | IC (t) | Sharpe gross | Sharpe net (5 bps/side) "
          "| Break-even bps/side | Turnover/day | $ per 1K articles | Latency |\n"
          "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|")


def main(split: str = "val"):
    fb, lex, zs = load("finbert"), load("lexicon_lm"), load("llm_zero_shot")
    n_all = (fb or lex or {}).get("n_articles_by_split", {}).get(split)
    print(f"### All {split} articles ({n_all:,} articles, averaged per stock-day)\n\n{HEADER}")
    for label, d in [("Loughran-McDonald lexicon", lex), ("FinBERT (headline)", fb)]:
        if d:
            print(row(label, d["metrics"][split], "0 (local)", "local, see Phase D"))
    if zs:
        m = zs["metrics"][split]
        lat = zs["latency"]
        cost = zs["cost_usd_per_1k_articles"]
        print(f"\n### Same {zs['n_articles']:,} {split} articles (1 random article per stock-day, all "
              f"{zs['n_days']} days)\n\n{HEADER}")
        for label, key in [("Loughran-McDonald lexicon", "lexicon"), ("FinBERT (headline)", "finbert")]:
            if key in zs["same_subset_baselines"]:
                print(row(label, zs["same_subset_baselines"][key][split], "0 (local)", "local"))
        lat_s = f"p50 {lat['p50_s']:.2f}s / p95 {lat['p95_s']:.2f}s"
        cost_s = f"{cost['batch_price']:.2f} batch / {cost['list_price']:.2f} list"
        print(row(f"Claude zero-shot ({zs['model']}), direction x confidence", m, cost_s, lat_s))
        print(row("Claude zero-shot, direction only (ablation)", zs["ablations"]["direction_only"][split],
                  cost_s, lat_s))
        print(f"\nClaude validity: {zs['validity']['valid_share']:.1%} schema-valid "
              f"({zs['validity']['n_invalid_schema']} invalid, {zs['validity']['n_api_error']} API errors); "
              f"latency over {lat['n_sync_calls']} synchronous calls.")
        print("\n### Claude zero-shot: IC by event type (pooled, one article per stock-day)\n")
        print("| Event type | Articles | Share positive / negative | Pooled rank IC | p-value |\n"
              "|---|---:|---:|---:|---:|")
        for et, v in zs["ic_by_event_type"].items():
            print(f"| {et} | {v['n']:,} | {v['share_positive']:.0%} / {v['share_negative']:.0%} | "
                  f"{fmt(v.get('pooled_rank_ic'))} | {fmt(v.get('pooled_rank_ic_p'))} |")


if __name__ == "__main__":
    main(*sys.argv[1:])
