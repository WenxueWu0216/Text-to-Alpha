"""Step A2: Loughran-McDonald lexicon baseline + shared evaluation."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from t2a.config import load_config
from t2a.data.io import load_events
from t2a.eval.report import report
from t2a.signals.aggregate import to_stock_day
from t2a.signals.lexicon import LexiconScorer, load_lm


def main():
    cfg = load_config()
    lex_dir = cfg["paths"]["lexicon_dir"]
    if not (lex_dir / "lm_positive.txt").exists():
        sys.exit(f"LM word lists missing in {lex_dir}; run: python scripts/fetch_lm_lexicon.py")

    events = load_events(cfg)
    scorer = LexiconScorer(*load_lm(lex_dir))
    events["lexicon"] = scorer.score((events["title"] + ". " + events["body"]).tolist())
    events[["news_id", "lexicon"]].to_parquet(cfg["paths"]["signals_dir"] / "lexicon_events.parquet")

    coverage = float((events["lexicon"] != 0).mean())
    print(f"articles with at least one LM word (non-zero score): {coverage:.1%}")
    report(to_stock_day(events, "lexicon"), cfg, "lexicon_lm",
           extra={"scored_text": "title + article lead (1500 chars)",
                  "n_articles_by_split": events["split"].value_counts().to_dict(),
                  "nonzero_score_share": coverage, "cost_usd_per_1k_articles": 0.0,
                  "lexicon": "Loughran-McDonald (pysentiment2 0.1.1 copy)",
                  "n_positive_words": len(scorer.pos), "n_negative_words": len(scorer.neg)})


if __name__ == "__main__":
    main()
