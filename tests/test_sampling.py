import pandas as pd

from t2a.signals.sampling import cap_per_stock_day


def _events():
    rows = []
    # AAPL/2021-01-01 has 5 items (> k), MSFT/2021-01-01 has 2 items (<= k)
    for i in range(5):
        rows.append({"trade_date": "2021-01-01", "ticker": "AAPL", "news_id": f"a{i}"})
    for i in range(2):
        rows.append({"trade_date": "2021-01-01", "ticker": "MSFT", "news_id": f"m{i}"})
    return pd.DataFrame(rows)


def test_no_group_exceeds_k():
    out = cap_per_stock_day(_events(), k=2, seed=0)
    counts = out.groupby(["trade_date", "ticker"]).size()
    assert (counts <= 2).all()


def test_small_groups_kept_whole():
    out = cap_per_stock_day(_events(), k=2, seed=0)
    msft = out[out["ticker"] == "MSFT"]
    assert len(msft) == 2
    assert set(msft["news_id"]) == {"m0", "m1"}


def test_deterministic_for_fixed_seed():
    events = _events()
    out1 = cap_per_stock_day(events, k=2, seed=0)
    out2 = cap_per_stock_day(events, k=2, seed=0)
    pd.testing.assert_frame_equal(
        out1.sort_values("news_id").reset_index(drop=True),
        out2.sort_values("news_id").reset_index(drop=True),
    )
