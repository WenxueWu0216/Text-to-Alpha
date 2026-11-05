"""FinBERT scoring service.

  uvicorn app.main:app --port 7860          (from service/)
"""
from __future__ import annotations

import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Callable

from fastapi import FastAPI, Request

from .benchmarks import load_metrics
from .model import Scorer, load_finbert
from .schemas import (HeadlineScore, HealthResponse, LabelProbabilities, MetricsResponse, ScoreRequest,
                      ScoreResponse)

DEFAULT_RESULTS_DIR = Path(__file__).resolve().parents[2] / "results"


def create_app(load_scorer: Callable[[], Scorer] = load_finbert, results_dir: Path | None = None) -> FastAPI:
    results_dir = Path(results_dir or os.environ.get("T2A_RESULTS_DIR", DEFAULT_RESULTS_DIR))

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.scorer = load_scorer()  # once per process, before the first request
        app.state.metrics = load_metrics(results_dir)  # results are baked in; read once
        yield

    app = FastAPI(title="Text-to-Alpha model service", version="0.1.0", lifespan=lifespan)

    @app.get("/health", response_model=HealthResponse)
    def health(request: Request) -> HealthResponse:
        s = request.app.state.scorer
        return HealthResponse(status="ok", model=s.name, device=s.device)

    @app.get("/metrics", response_model=MetricsResponse)
    def metrics(request: Request) -> MetricsResponse:
        return request.app.state.metrics

    # Sync def: FastAPI runs it in a worker thread, so CPU inference doesn't block the event loop.
    @app.post("/score", response_model=ScoreResponse)
    def score(req: ScoreRequest, request: Request) -> ScoreResponse:
        s = request.app.state.scorer
        preds = s.predict(req.headlines)
        return ScoreResponse(model=s.name, scores=[
            HeadlineScore(headline=h, score=p.score, label=max(p.probabilities, key=p.probabilities.get),
                          probabilities=LabelProbabilities(**p.probabilities))
            for h, p in zip(req.headlines, preds, strict=True)])

    return app


app = create_app()
