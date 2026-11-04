"""Read-only view of the benchmark numbers in results/*.json."""
from __future__ import annotations

import json
from pathlib import Path

from .schemas import HeadlineMetrics, MethodResult, MetricsResponse, SameSubsetResult

METHOD_FILES = ("llm_zero_shot", "finbert", "lexicon_lm")
FIELDS = tuple(HeadlineMetrics.model_fields)


def _headline(m: dict) -> HeadlineMetrics:
    return HeadlineMetrics(**{k: m.get(k) for k in FIELDS})


def _read(path: Path) -> dict | None:
    return json.loads(path.read_text()) if path.exists() else None


def load_metrics(results_dir: Path) -> MetricsResponse:
    methods = {}
    for name in METHOD_FILES:
        d = _read(results_dir / f"{name}.json")
        if d:
            methods[name] = MethodResult(method=name, git_commit=d.get("git_commit"),
                                         created_utc=d.get("created_utc"),
                                         splits={s: _headline(m) for s, m in d["metrics"].items()})
    same = None
    d = _read(results_dir / "val_same_subset.json")
    if d:
        same = SameSubsetResult(split=d["split"], n_articles=d["n_articles"], n_trade_days=d["n_trade_days"],
                                git_commit=d.get("git_commit"),
                                methods={k: _headline(m) for k, m in d["metrics"].items()})
    lat = _read(results_dir / "service_latency.json")
    if lat:
        lat.pop("config", None)
    return MetricsResponse(methods=methods, same_subset=same, service_latency=lat)
