"""LLM scorer pieces that must work without network access."""
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from t2a.signals.budget import BudgetExceeded, Ledger, check_budget
from t2a.signals.llm import (Prices, ResponseCache, build_system, cache_key, parse_message,
                             records_to_frame, request_params, validate)
from t2a.signals.sampling import select_fewshot_examples

GOOD = {"rationale": "Beat on revenue.", "event_type": "earnings", "direction": "positive",
        "confidence": 0.8}


def test_valid_output_passes():
    clean, err = validate(GOOD)
    assert err is None and clean == GOOD


@pytest.mark.parametrize("bad", [
    {**GOOD, "confidence": 1.4},                 # range is not enforced by the API
    {**GOOD, "direction": "up"},                 # not in enum
    {k: v for k, v in GOOD.items() if k != "event_type"},
    {**GOOD, "extra": 1},                        # additionalProperties: false
    {**GOOD, "rationale": ""},
])
def test_invalid_outputs_are_rejected(bad):
    clean, err = validate(bad)
    assert clean is None and err


def _msg(tool_input):
    usage = SimpleNamespace(input_tokens=500, output_tokens=60)
    content = [SimpleNamespace(type="tool_use", input=tool_input)] if tool_input is not None else \
        [SimpleNamespace(type="text", text="hi")]
    return content, "tool_use", usage


def test_parse_message_records_every_outcome():
    assert parse_message(*_msg(GOOD))["status"] == "ok"
    bad = parse_message(*_msg({**GOOD, "confidence": -1}))
    assert bad["status"] == "invalid" and "confidence" in bad["validation_error"]
    assert parse_message(*_msg(None))["status"] == "no_tool_call"


def test_signal_is_direction_times_confidence_and_invalid_is_nan():
    df = records_to_frame([
        {**GOOD, "status": "ok"},
        {**GOOD, "direction": "negative", "confidence": 0.5, "status": "ok"},
        {**GOOD, "direction": "neutral", "status": "ok"},
        {"status": "invalid"},
        None,
    ])
    assert np.allclose(df["llm_signal"][:3], [0.8, -0.5, 0.0])
    assert np.allclose(df["llm_direction"][:3], [1, -1, 0])
    assert df["llm_signal"][3:].isna().all()


def test_cache_key_separates_prompts_and_models():
    k = cache_key("m", build_system(), "msg")
    assert k != cache_key("m", build_system([{"title": "t", "body": "b", "outcome": "up"}]), "msg")
    assert k != cache_key("m2", build_system(), "msg")
    assert k == cache_key("m", build_system(), "msg")


def test_request_is_forced_strict_tool_use():
    p = request_params("m", "sys", "msg")
    assert p["tool_choice"] == {"type": "tool", "name": "record_signal"}
    assert p["tools"][0]["strict"] is True and p["temperature"] == 0


def test_cache_roundtrip(tmp_path):
    c = ResponseCache(tmp_path / "c.sqlite")
    c.put("a", {"x": 1})
    c.commit()
    assert c.get_many(["a", "b"]) == {"a": {"x": 1}}


def test_batch_is_half_price():
    p = Prices(1.0, 5.0)
    assert p.cost(1e6, 1e6) == 6.0 and p.cost(1e6, 1e6, batch=True) == 3.0


def test_budget_gates(tmp_path):
    check_budget(estimate=3, spent=5, run_cap=9, total_cap=15, reserve=4)
    with pytest.raises(BudgetExceeded):
        check_budget(estimate=10, spent=0, run_cap=9, total_cap=15)
    with pytest.raises(BudgetExceeded):
        check_budget(estimate=7, spent=5, run_cap=9, total_cap=15, reserve=4)
    led = Ledger(tmp_path / "l.jsonl")
    led.record("a", 1.25)
    led.record("b", 0.5)
    assert led.spent() == 1.75


def _train(split="train"):
    rows = []
    for i in range(40):
        rows.append({"news_id": f"n{i}", "ticker": "AAPL", "trade_date": pd.Timestamp("2020-01-01") + pd.Timedelta(days=i),
                     "title": f"t{i}", "body": "x" * 300, "label": [1, -1, 0][i % 3], "split": split})
    return pd.DataFrame(rows)


def test_fewshot_examples_are_train_only_and_balanced():
    ex = select_fewshot_examples(_train(), {1: 2, -1: 2, 0: 1}, seed=0)
    assert sorted(e["label"] for e in ex) == [-1, -1, 0, 1, 1]
    with pytest.raises(ValueError):
        select_fewshot_examples(pd.concat([_train(), _train("val")]), {1: 1}, seed=0)
