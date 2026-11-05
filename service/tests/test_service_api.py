import os

import pytest

from app.schemas import MAX_HEADLINES


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    h = r.json()
    assert {k: h[k] for k in ("status", "model", "variant", "device")} == {
        "status": "ok", "model": "fake-finbert", "variant": "fake", "device": "cpu"}
    assert h["uptime_s"] >= 0 and h["model_load_s"] >= 0


def test_model_loaded_once_at_startup(fake_scorer, results_dir):
    from fastapi.testclient import TestClient

    from app.main import create_app

    loads = []

    def loader():
        loads.append(1)
        return fake_scorer

    with TestClient(create_app(load_scorer=loader, results_dir=results_dir)) as c:
        for _ in range(3):
            assert c.post("/score", json={"headlines": ["x"]}).status_code == 200
    assert loads == [1]


def test_score_shape_range_and_label(client):
    r = client.post("/score", json={"headlines": ["Apple beats estimates", "Intel misses on revenue", "Flat day"]})
    assert r.status_code == 200
    body = r.json()
    assert body["model"] == "fake-finbert"
    assert len(body["scores"]) == 3
    for s in body["scores"]:
        assert -1 <= s["score"] <= 1
        p = s["probabilities"]
        assert set(p) == {"positive", "negative", "neutral"}
        assert all(0 <= v <= 1 for v in p.values())
        assert sum(p.values()) == pytest.approx(1)
        assert s["score"] == pytest.approx(p["positive"] - p["negative"])
        assert s["label"] == max(p, key=p.get)
    assert body["scores"][0]["label"] == "positive"
    assert body["scores"][1]["label"] == "negative"


def test_order_preserved(client):
    heads = [f"headline {i} {'beat' if i % 3 == 0 else 'miss' if i % 3 == 1 else 'flat'}" + "!" * i
             for i in range(20)]
    one_by_one = [client.post("/score", json={"headlines": [h]}).json()["scores"][0]["score"] for h in heads]
    batch = client.post("/score", json={"headlines": heads}).json()["scores"]
    assert [s["headline"] for s in batch] == heads
    assert [s["score"] for s in batch] == pytest.approx(one_by_one)


def test_headlines_are_stripped(client, fake_scorer):
    r = client.post("/score", json={"headlines": ["  beat  "]})
    assert r.json()["scores"][0]["headline"] == "beat"
    assert fake_scorer.calls[-1] == ["beat"]


def test_batch_limit(client, fake_scorer):
    ok = client.post("/score", json={"headlines": ["a"] * MAX_HEADLINES})
    assert ok.status_code == 200 and len(ok.json()["scores"]) == MAX_HEADLINES
    n_calls = len(fake_scorer.calls)
    too_many = client.post("/score", json={"headlines": ["a"] * (MAX_HEADLINES + 1)})
    assert too_many.status_code == 422
    assert len(fake_scorer.calls) == n_calls  # rejected before reaching the model


@pytest.mark.parametrize("payload", [
    {},
    {"headlines": []},
    {"headlines": "Apple beats"},
    {"headlines": [""]},
    {"headlines": ["   "]},
    {"headlines": [123]},
    {"headlines": [None]},
    {"headlines": ["x" * 1001]},
    {"headlines": ["ok"], "extra": 1},
])
def test_validation_errors(client, fake_scorer, payload):
    r = client.post("/score", json=payload)
    assert r.status_code == 422
    assert fake_scorer.calls == []


def test_non_json_body(client):
    r = client.post("/score", content=b"not json", headers={"content-type": "application/json"})
    assert r.status_code == 422


def test_metrics_read_only(client):
    r = client.get("/metrics")
    assert r.status_code == 200
    m = r.json()
    assert m["methods"]["finbert"]["splits"]["val"]["ic_mean"] == 0.01
    assert m["methods"]["finbert"]["splits"]["val"]["ic_tstat"] is None
    assert "by_year" not in m["methods"]["finbert"]["splits"]["val"]
    assert m["same_subset"]["methods"]["llm_zero_shot"]["ic_mean"] == 0.03
    assert client.post("/metrics").status_code == 405


def test_metrics_with_no_results(fake_scorer, tmp_path):
    from fastapi.testclient import TestClient

    from app.main import create_app

    with TestClient(create_app(load_scorer=lambda: fake_scorer, results_dir=tmp_path)) as c:
        assert c.get("/metrics").json() == {"methods": {}, "same_subset": None, "service_latency": None,
                                          "gate": None}


def test_metrics_parses_real_results():
    """The committed results/*.json must fit the response schema."""
    from pathlib import Path

    from app.benchmarks import load_metrics

    m = load_metrics(Path(__file__).resolve().parents[2] / "results")
    assert {"finbert", "lexicon_lm", "llm_zero_shot"} <= set(m.methods)
    assert m.same_subset is not None and m.same_subset.n_articles > 0


@pytest.mark.skipif(os.environ.get("T2A_MODEL_TESTS") != "1",
                    reason="needs models/finbert-onnx (service/scripts/export_onnx.py); set T2A_MODEL_TESTS=1")
def test_real_finbert_end_to_end(results_dir):
    from fastapi.testclient import TestClient

    from app.main import create_app

    heads = ["Company shares plunge after profit warning", "Record quarterly profit beats expectations"]
    with TestClient(create_app(results_dir=results_dir)) as c:
        h = c.get("/health").json()
        assert h["device"] == "cpu" and h["variant"] == "int8"
        s = c.post("/score", json={"headlines": heads}).json()["scores"]
    assert s[0]["score"] < 0 < s[1]["score"]
    assert [x["headline"] for x in s] == heads


@pytest.mark.skipif(os.environ.get("T2A_MODEL_TESTS") != "1", reason="needs models/finbert-onnx; set T2A_MODEL_TESTS=1")
def test_onnx_tokenizer_matches_padding_free_scores():
    """Padding must not change a headline's score: batched == one at a time."""
    from app.model import load_finbert

    m = load_finbert()
    heads = ["Short", "A much longer headline about quarterly earnings beating every analyst estimate"]
    batched = [p.score for p in m.predict(heads)]
    single = [m.predict([h])[0].score for h in heads]
    assert batched == pytest.approx(single, abs=1e-4)
