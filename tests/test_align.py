"""Look-ahead bugs are silent: the backtest just looks great. So we pin the rules with tests."""
import pandas as pd

from t2a.data.align import assign_trade_date, build_panel, to_effective_local_time

# Mon 2023-07-03 .. Wed 2023-07-05, with Tue 07-04 a holiday (Independence Day)
DAYS = pd.DatetimeIndex(["2023-06-30", "2023-07-03", "2023-07-05", "2023-07-06"])


def trade_date_for(utc_str: str, date_only: bool = False) -> pd.Timestamp:
    ts = pd.Series(pd.to_datetime([utc_str], utc=True))
    local = to_effective_local_time(ts, pd.Series([date_only]))
    return assign_trade_date(local, DAYS).iloc[0]


def test_pre_open_news_trades_same_day():
    # 12:00 UTC = 08:00 ET (summer, UTC-4)
    assert trade_date_for("2023-07-03 12:00") == pd.Timestamp("2023-07-03")


def test_intraday_news_trades_next_day_skipping_holiday():
    # 15:00 UTC = 11:00 ET on Mon; Tue is a holiday -> Wed
    assert trade_date_for("2023-07-03 15:00") == pd.Timestamp("2023-07-05")


def test_exactly_at_open_is_too_late():
    assert trade_date_for("2023-07-03 13:30") == pd.Timestamp("2023-07-05")


def test_weekend_news_trades_monday():
    assert trade_date_for("2023-07-01 18:00") == pd.Timestamp("2023-07-03")


def test_date_only_is_treated_as_after_close():
    # midnight UTC with no real time: must NOT be tradable at that morning's open
    assert trade_date_for("2023-07-03 00:00", date_only=True) == pd.Timestamp("2023-07-05")


def test_dst_is_handled():
    # winter: 14:00 UTC = 09:00 ET (UTC-5) -> still before the open
    days = pd.DatetimeIndex(["2023-01-10", "2023-01-11"])
    ts = pd.Series(pd.to_datetime(["2023-01-10 14:00"], utc=True))
    local = to_effective_local_time(ts, pd.Series([False]))
    assert assign_trade_date(local, days).iloc[0] == pd.Timestamp("2023-01-10")


def test_panel_labels_use_future_open_and_are_cross_sectional():
    prices = pd.DataFrame({
        "date": list(pd.to_datetime(["2023-01-02", "2023-01-03"])) * 3,
        "ticker": ["A", "A", "B", "B", "C", "C"],
        "open": [100, 110, 100, 100, 100, 90],   # A +10%, B 0%, C -10%
        "close": [1] * 6,
    })
    p = build_panel(prices, horizon=1, top_q=0.3).set_index("ticker")
    assert abs(p.loc["A", "fwd_ret"] - 0.10) < 1e-12
    assert abs(p["excess_ret"].sum()) < 1e-12           # excess returns are demeaned
    assert p.loc["A", "label"] == 1 and p.loc["C", "label"] == -1 and p.loc["B", "label"] == 0
    assert len(p) == 3                                   # last day has no future open -> dropped
