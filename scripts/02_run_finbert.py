"""Step 2: FinBERT baseline scores + shared evaluation."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd

from t2a.config import load_config
from t2a.eval.report import report
from t2a.signals.aggregate import to_stock_day
from t2a.signals.finbert import FinBertScorer


def main():
    cfg = load_config()
    P, fb = cfg["paths"], cfg["finbert"]
    events = pd.read_parquet(P["events"])

    scorer = FinBertScorer(fb["model"], fb["max_length"])
    print(f"scoring {len(events):,} headlines on {scorer.device} ...")
    events["finbert"] = scorer.score(events["title"].tolist(), fb["batch_size"])

    out = P["signals_dir"]
    out.mkdir(parents=True, exist_ok=True)
    events[["news_id", "finbert"]].to_parquet(out / "finbert_events.parquet")

    report(to_stock_day(events, "finbert"), cfg, "finbert", out_dir=out)


if __name__ == "__main__":
    main()
