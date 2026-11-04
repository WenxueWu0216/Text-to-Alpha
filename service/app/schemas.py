"""Request/response models. The Next.js frontend mirrors these types (see /openapi.json)."""
from __future__ import annotations

from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, StringConstraints

MAX_HEADLINES = 64
MAX_HEADLINE_CHARS = 1000

Headline = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=MAX_HEADLINE_CHARS)]


class ScoreRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    headlines: list[Headline] = Field(min_length=1, max_length=MAX_HEADLINES,
                                      description=f"1 to {MAX_HEADLINES} headlines, scored independently")


class LabelProbabilities(BaseModel):
    positive: float = Field(ge=0, le=1)
    negative: float = Field(ge=0, le=1)
    neutral: float = Field(ge=0, le=1)


class HeadlineScore(BaseModel):
    headline: str
    score: float = Field(ge=-1, le=1, description="P(positive) - P(negative)")
    label: Literal["positive", "negative", "neutral"] = Field(description="argmax label")
    probabilities: LabelProbabilities


class ScoreResponse(BaseModel):
    model: str
    scores: list[HeadlineScore] = Field(description="same order as the request headlines")


class HealthResponse(BaseModel):
    status: Literal["ok"]
    model: str
    device: str


class HeadlineMetrics(BaseModel):
    """Headline numbers from the shared evaluator (src/t2a/eval/metrics.py). null = undefined."""
    n_obs: int | None = None
    n_days: int | None = None
    ic_mean: float | None = None
    ic_tstat: float | None = None
    ic_hit_rate: float | None = None
    ls_sharpe_gross: float | None = None
    ls_sharpe_net: float | None = None
    breakeven_cost_bps_per_side: float | None = None
    turnover_mean: float | None = None
    macro_f1: float | None = None


class MethodResult(BaseModel):
    method: str
    git_commit: str | None = None
    created_utc: str | None = None
    splits: dict[str, HeadlineMetrics]


class SameSubsetResult(BaseModel):
    split: str
    n_articles: int
    n_trade_days: int
    git_commit: str | None = None
    methods: dict[str, HeadlineMetrics]


class MetricsResponse(BaseModel):
    methods: dict[str, MethodResult] = Field(description="each method on all articles it scored")
    same_subset: SameSubsetResult | None = Field(
        None, description="all methods on the same validation articles (the fair comparison)")
    service_latency: dict | None = None
