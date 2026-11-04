import numpy as np
from sklearn.metrics import f1_score
import pandas as pd

from t2a.eval.metrics import daily_ic, evaluate, long_short_book, long_short_returns, portfolio_weights
from t2a.data.news import deduplicate, mask_entities


def make_panel(signal_fn, n_days=50, n_names=10, seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for d in pd.bdate_range("2022-01-03", periods=n_days):
        ret = rng.normal(0, 0.02, n_names)
        for i in range(n_names):
            rows.append({"date": d, "ticker": f"T{i}", "fwd_ret": ret[i],
                         "excess_ret": ret[i] - ret.mean(), "signal": signal_fn(ret[i], rng)})
    df = pd.DataFrame(rows)
    # Symmetric percentile (rank - 1) / (n - 1), same rule build_panel() uses for labels
    # and classification_metrics() now uses for predictions.
    g = df.groupby("date")["excess_ret"]
    pct = (g.rank() - 1) / (g.transform("size") - 1)
    df["label"] = np.select([pct >= 0.7, pct <= 0.3], [1, -1], default=0)
    return df


def test_perfect_signal_has_ic_one():
    ic = daily_ic(make_panel(lambda r, rng: r))
    assert np.allclose(ic, 1.0)


def test_random_signal_ic_near_zero():
    ic = daily_ic(make_panel(lambda r, rng: rng.normal(), n_days=400))
    assert abs(ic.mean()) < 0.03


def test_long_short_is_dollar_neutral_and_costed_on_turnover():
    df = make_panel(lambda r, rng: r)
    book = long_short_book(df)
    gross = long_short_returns(df, cost_bps_per_side=0)
    net = long_short_returns(df, cost_bps_per_side=10)
    assert (gross > 0).all()                       # perfect foresight always wins
    assert np.allclose(gross - net, book["turnover"] * 10 / 1e4)
    w = portfolio_weights(df)
    assert np.allclose(w.groupby("date")["w"].sum(), 0)          # dollar-neutral
    assert np.allclose(w.groupby("date")["w"].apply(lambda x: x.abs().sum()), 1)


def test_days_with_too_few_names_are_skipped():
    df = make_panel(lambda r, rng: r, n_names=3)
    assert len(daily_ic(df, min_names=5)) == 0


def test_evaluate_runs():
    out = evaluate(make_panel(lambda r, rng: r + rng.normal(0, 0.02)))
    assert out["ic_mean"] > 0.3 and out["directional_acc"] > 0.6


def test_perfect_signal_gives_macro_f1_one():
    # signal == fwd_ret is a monotonic transform of excess_ret within each day, so ranking
    # the signal reproduces the exact same buckets used to build the label.
    out = evaluate(make_panel(lambda r, rng: r))
    assert out["macro_f1"] == 1.0


def test_dedup_keeps_earliest_within_window_only():
    news = pd.DataFrame({
        "ticker": ["AAPL"] * 3,
        "title": ["Apple beats earnings!", "Apple Beats Earnings", "Apple beats earnings"],
        "ts_utc": pd.to_datetime(["2023-01-02 10:00", "2023-01-01 09:00", "2023-03-01 09:00"], utc=True),
    })
    out = deduplicate(news)
    assert len(out) == 2                                   # copy within 3 days collapsed, March kept
    assert out["ts_utc"].min() == pd.Timestamp("2023-01-01 09:00", tz="UTC")


def test_mask_entities():
    s = mask_entities("JP Morgan (JPM) upgrades; JPMorgan shares rise", "JPM", ["JPMorgan", "JP Morgan"])
    assert "Morgan" not in s and "JPM" not in s
    # short tickers are case-sensitive so ordinary words survive
    assert mask_entities("Visa (V) says v2 launch", "V", ["Visa"]) == "the Company (the Company) says v2 launch"


def test_f1_uses_cross_sectional_quantiles_not_sign():
    # A signal that is positive for every stock: sign() would call everything "+1".
    # Quantile classes depend only on the within-day ORDER, so shifting or monotonically
    # transforming the signal per day must not change F1.
    df = make_panel(lambda r, rng: r + rng.normal(0, 0.02), n_days=100)
    base = evaluate(df)["macro_f1"]
    shifted = evaluate(df.assign(signal=np.exp(df["signal"] * 50) + 5))["macro_f1"]
    assert shifted == base
    sign_f1 = f1_score(df["label"], np.sign(df["signal"] + 1.0), average="macro")
    assert base > sign_f1


def test_predicted_classes_match_label_proportions():
    from t2a.eval.metrics import classification_metrics
    df = make_panel(lambda r, rng: rng.normal(), n_days=50)
    assert abs(classification_metrics(df)["coverage"] - 0.6) < 0.05


def test_raw_news_filter_counts_every_drop():
    from pathlib import Path
    from t2a.data.news import load_news
    news, st = load_news(Path(__file__).parent / "fixtures" / "raw_news_tiny.csv",
                         ["AAPL", "MSFT"], "2018-01-01", "2023-12-31")
    assert st["raw_rows"] == 8 and st["missing_ticker"] == 1 and st["in_universe"] == 6
    assert st["dropped_empty_title"] == 1 and st["dropped_bad_timestamp"] == 1
    assert st["dropped_out_of_range"] == 1 and st["duplicates_removed"] == 1
    assert st["kept_after_dedup"] == len(news) == 2 and st["date_only_timestamps"] == 1
