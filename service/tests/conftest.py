import sys
from pathlib import Path

import pytest

SERVICE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SERVICE_DIR))

from app.model import Prediction  # noqa: E402


class FakeScorer:
    """Deterministic stand-in for FinBERT, so CI never needs the weights.

    Score depends on the text (keyword + length), so a reordering bug would show up.
    """
    name = "fake-finbert"
    variant = "fake"
    device = "cpu"

    def __init__(self):
        self.calls: list[list[str]] = []

    def predict(self, texts: list[str]) -> list[Prediction]:
        self.calls.append(list(texts))
        out = []
        for t in texts:
            low = t.lower()
            pos = 0.8 if "beat" in low else 0.1
            neg = 0.8 if "miss" in low else 0.1
            z = pos + neg + 0.1 + (len(t) % 7) / 10
            p = {"positive": pos / z, "negative": neg / z, "neutral": 1 - (pos + neg) / z}
            out.append(Prediction(score=p["positive"] - p["negative"], probabilities=p))
        return out


@pytest.fixture
def fake_scorer():
    return FakeScorer()


@pytest.fixture
def results_dir(tmp_path):
    import json

    (tmp_path / "finbert.json").write_text(json.dumps({
        "method": "finbert", "git_commit": "abc", "created_utc": "2026-01-01T00:00:00+00:00",
        "metrics": {"val": {"n_obs": 10, "n_days": 5, "ic_mean": 0.01, "ic_tstat": None,
                            "ls_sharpe_net": -0.2, "by_year": {"2022": {}}}}}))
    (tmp_path / "val_same_subset.json").write_text(json.dumps({
        "split": "val", "n_articles": 3, "n_trade_days": 2, "git_commit": "abc",
        "metrics": {"finbert": {"ic_mean": 0.02}, "llm_zero_shot": {"ic_mean": 0.03}}}))
    return tmp_path


@pytest.fixture
def client(fake_scorer, results_dir):
    from fastapi.testclient import TestClient

    from app.main import create_app

    with TestClient(create_app(load_scorer=lambda: fake_scorer, results_dir=results_dir)) as c:
        yield c
