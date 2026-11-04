"""The test period must stay closed unless it is opened on purpose (rule 1)."""
import pandas as pd
import pytest

import t2a.data.io as io
from t2a.data.io import LockedSplitError, check_splits, load_events
from t2a.eval.report import evaluate_splits


@pytest.fixture
def cfg(tmp_path, monkeypatch):
    ev = pd.DataFrame({"split": ["train", "val", "test"], "x": [1, 2, 3]})
    ev.to_parquet(tmp_path / "events.parquet")
    monkeypatch.setattr(io, "UNLOCK_LOG", tmp_path / "unlock.jsonl")
    monkeypatch.delenv(io.UNLOCK_ENV, raising=False)
    return {"paths": {"events": tmp_path / "events.parquet"}}


def test_default_load_excludes_test(cfg):
    assert set(load_events(cfg)["split"]) == {"train", "val"}


def test_requesting_test_without_flag_raises(cfg):
    with pytest.raises(LockedSplitError):
        load_events(cfg, splits=("val", "test"))


def test_flag_alone_or_env_alone_is_not_enough(cfg, monkeypatch):
    with pytest.raises(LockedSplitError):
        load_events(cfg, splits=("test",), unlock_test=True)
    monkeypatch.setenv(io.UNLOCK_ENV, "1")
    with pytest.raises(LockedSplitError):
        load_events(cfg, splits=("test",))


def test_explicit_unlock_loads_and_logs(cfg, monkeypatch):
    monkeypatch.setenv(io.UNLOCK_ENV, "1")
    out = load_events(cfg, splits=("test",), unlock_test=True)
    assert list(out["x"]) == [3]
    assert io.UNLOCK_LOG.read_text().count("\n") == 1


def test_evaluator_refuses_test_split():
    with pytest.raises(LockedSplitError):
        evaluate_splits(pd.DataFrame({"split": []}), cfg={}, splits=("test",))


def test_check_splits_allows_train_val():
    check_splits(("train", "val"))
