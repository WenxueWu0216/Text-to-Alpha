"""Turnover, costs, break-even and regime breakdowns of the shared evaluator."""
import numpy as np
import pandas as pd

from t2a.data.market import prior_close, vix_regime
from t2a.eval.metrics import (breakeven_cost_bps, evaluate, long_short_book, net_returns,
                              trading_summary)
from test_metrics import make_panel


def _frame(days: list[dict[str, float]], rets: list[dict[str, float]]) -> pd.DataFrame:
    rows = []
    for i, (sig, ret) in enumerate(zip(days, rets)):
        d = pd.Timestamp("2022-01-03") + pd.Timedelta(days=i)
        for t in sig:
            rows.append({"date": d, "ticker": t, "signal": sig[t], "fwd_ret": ret[t]})
    return pd.DataFrame(rows)


def test_turnover_open_hold_and_full_flip():
    sig = {"A": 1.0, "B": 0.0}
    flipped = {"A": 0.0, "B": 1.0}
    ret = {"A": 0.0, "B": 0.0}
    book = long_short_book(_frame([sig, sig, flipped], [ret] * 3), min_names=2)
    # open from cash = 1, unchanged book = 0, long<->short flip of every name = 2
    assert np.allclose(book["turnover"].values, [1.0, 0.0, 2.0])


def test_disjoint_names_cost_two():
    a = {"A": 1.0, "B": 0.0}
    b = {"C": 1.0, "D": 0.0}
    ret = {k: 0.0 for k in "ABCD"}
    book = long_short_book(_frame([a, b], [ret, ret]), min_names=2)
    assert np.isclose(book["turnover"].iloc[1], 2.0)


def test_breakeven_cost_zeroes_mean_net_return():
    book = long_short_book(make_panel(lambda r, rng: r + rng.normal(0, 0.03), n_days=200))
    c = breakeven_cost_bps(book)
    assert c > 0
    assert abs(net_returns(book, c).mean()) < 1e-12


def test_sharpe_decreases_with_cost_and_holding_period():
    book = long_short_book(make_panel(lambda r, rng: r + rng.normal(0, 0.03), n_days=200))
    s = trading_summary(book, cost_grid=(0, 5, 10, 20))
    grid = list(s["sharpe_by_cost_bps_per_side"].values())
    assert grid == sorted(grid, reverse=True)
    assert np.isclose(s["holding_period_days"], 2 / book["turnover"].mean())


def test_vix_uses_prior_close_only():
    vix = pd.Series([10.0, 30.0, 12.0], index=pd.to_datetime(["2022-01-03", "2022-01-04", "2022-01-05"]))
    prev = prior_close(vix, pd.DatetimeIndex(["2022-01-03", "2022-01-04", "2022-01-06"]))
    assert np.isnan(prev.iloc[0])           # no earlier close known
    assert prev.iloc[1] == 10.0             # NOT 30.0: Jan-4 close is after the Jan-4 open
    assert prev.iloc[2] == 12.0             # skips the missing day
    reg = vix_regime(vix, pd.DatetimeIndex(["2022-01-03", "2022-01-04", "2022-01-05"]))
    assert pd.isna(reg.iloc[0])             # unknown, not silently "calm"
    assert reg.iloc[1] == "lt20"            # Jan-3 close 10
    assert reg.iloc[2] == "ge30"            # driven by the Jan-4 close of 30


def test_evaluate_reports_regimes_and_cost_grid():
    df = make_panel(lambda r, rng: r + rng.normal(0, 0.03), n_days=300)
    dates = pd.DatetimeIndex(df["date"].unique())
    labels = pd.Series(np.where(np.arange(len(dates)) % 2, "high", "low"), index=dates)
    out = evaluate(df, regimes={"vix_regime": labels})
    assert set(out["by_vix_regime"]) == {"high", "low"}
    assert sum(v["n_days"] for v in out["by_vix_regime"].values()) == out["n_days"]
    assert set(out["by_year"]) == {"2022", "2023"}
    assert set(out["sharpe_by_cost_bps_per_side"]) == {"0", "5", "10", "20"}
    assert np.isclose(out["sharpe_by_cost_bps_per_side"]["5"], out["ls_sharpe_net"])
