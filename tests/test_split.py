import pandas as pd

from t2a.data.split import assign_split


def test_embargo_creates_gap_and_no_overlap():
    days = pd.bdate_range("2021-12-20", "2022-01-20")
    s = assign_split(pd.Series(days), pd.DatetimeIndex(days), "2021-12-31", "2022-01-12", embargo=3)
    by = dict(zip(days, s))
    assert by[pd.Timestamp("2021-12-31")] == "train"
    # first 3 trading days of 2022 are embargoed
    assert [by[d] for d in days[(days > "2021-12-31")][:4]] == ["embargo"] * 3 + ["val"]
    train_max = max(d for d, v in by.items() if v == "train")
    val_min = min(d for d, v in by.items() if v == "val")
    test_min = min(d for d, v in by.items() if v == "test")
    assert train_max < val_min < test_min
