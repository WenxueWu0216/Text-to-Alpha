"""Frozen eval fixture for CI: ~1,500 val headlines (1 per stock-day) with realised returns.

Whole trading days are sampled (not single articles), so every day keeps its full
cross-section and the daily IC stays computable. Fixed seed: rerunning gives the same file.

  python scripts/08_build_eval_fixture.py      # writes service/eval_fixture/val_fixture.parquet

News: FNSPID (Dong et al., KDD 2024), CC BY-NC 4.0. Returns: Yahoo Finance prices.
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from t2a.config import REPO_ROOT, load_config
from t2a.data.io import load_events
from t2a.signals.sampling import cap_per_stock_day, sample_days

SEED = 20261105
TARGET = 1500
OUT_DIR = REPO_ROOT / "service" / "eval_fixture"
COLS = ["news_id", "trade_date", "ticker", "title", "fwd_ret", "excess_ret", "label", "split"]


def main():
    cfg = load_config()
    val = load_events(cfg, splits=("val",))
    one = cap_per_stock_day(val, 1, SEED)
    per_day = one.groupby("trade_date").size().mean()
    n_days = round(TARGET / per_day)
    fx = sample_days(one, n_days, SEED).sort_values(["trade_date", "ticker"]).reset_index(drop=True)[COLS]
    fx["label"] = fx["label"].astype("int8")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUT_DIR / "val_fixture.parquet"
    fx.to_parquet(path, index=False, compression="zstd")
    size = path.stat().st_size
    assert size < 1_000_000, f"fixture is {size:,} bytes, must stay under 1 MB"
    meta = {"seed": SEED, "split": "val", "rule": "1 random article per stock-day, all names on "
            f"{n_days} randomly sampled val days", "n_rows": len(fx), "n_days": int(fx["trade_date"].nunique()),
            "bytes": size, "columns": COLS,
            "source": "FNSPID (Dong et al., KDD 2024), CC BY-NC 4.0; returns from Yahoo Finance"}
    (OUT_DIR / "README.json").write_text(json.dumps(meta, indent=2))
    print(f"{len(fx):,} headlines on {meta['n_days']} days, {size / 1e3:.0f} KB -> {path}")


if __name__ == "__main__":
    main()
