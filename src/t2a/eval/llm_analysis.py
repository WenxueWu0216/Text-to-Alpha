"""Diagnostics specific to structured LLM outputs: validity counts and IC by event type."""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats


def status_summary(df: pd.DataFrame) -> dict:
    counts = df["status"].value_counts().to_dict()
    n = len(df)
    return {"n": n, "counts": counts, "valid_share": counts.get("ok", 0) / n if n else np.nan,
            "n_invalid_schema": counts.get("invalid", 0),
            "n_no_tool_call": counts.get("no_tool_call", 0),
            "n_api_error": counts.get("api_error", 0) + counts.get("missing", 0),
            "n_truncated_max_tokens": int((df["stop_reason"] == "max_tokens").sum())
            if "stop_reason" in df else 0}


def by_event_type(df: pd.DataFrame, signal: str = "llm_signal", min_n: int = 30) -> dict:
    """Which event types carry signal? Pooled across days (excess_ret is already demeaned
    per day). Each row is one stock-day here (one article per stock-day), so rows are not
    clustered within a stock-day; they are still clustered by day, so the t-stats are
    optimistic and only meant to rank event types, not to claim significance."""
    out = {}
    ok = df[df["status"] == "ok"]
    for et, g in ok.groupby("event_type"):
        pos, neg = g.loc[g["dir_num"] > 0, "excess_ret"], g.loc[g["dir_num"] < 0, "excess_ret"]
        row = {"n": len(g), "share": len(g) / len(ok),
               "share_positive": float((g["dir_num"] > 0).mean()),
               "share_negative": float((g["dir_num"] < 0).mean())}
        if len(g) >= min_n and g[signal].nunique() > 1:
            rho, p = stats.spearmanr(g[signal], g["excess_ret"])
            row.update(pooled_rank_ic=float(rho), pooled_rank_ic_p=float(p))
        if len(pos) >= 5 and len(neg) >= 5:
            t, p = stats.ttest_ind(pos, neg, equal_var=False)
            row.update(spread_bps=float((pos.mean() - neg.mean()) * 1e4), spread_t=float(t))
        out[str(et)] = row
    return dict(sorted(out.items(), key=lambda kv: -kv[1]["n"]))


def latency_summary(latencies: pd.Series) -> dict:
    lat = latencies.dropna()
    if lat.empty:
        return {"n_sync_calls": 0}
    return {"n_sync_calls": int(len(lat)), "p50_s": float(lat.quantile(0.5)),
            "p95_s": float(lat.quantile(0.95)), "mean_s": float(lat.mean())}
