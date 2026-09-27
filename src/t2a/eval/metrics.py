"""Evaluation used by EVERY approach (FinBERT, zero-shot LLM, RAG, fine-tuned, agent).

Input: a (date, ticker) frame with a `signal` column plus fwd_ret / excess_ret / label.
Keeping one shared evaluator is what makes the final benchmark table apples-to-apples.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

TRADING_DAYS = 252


def _eligible_days(df: pd.DataFrame, min_names: int) -> pd.DataFrame:
    n = df.groupby("date")["ticker"].transform("size")
    varies = df.groupby("date")["signal"].transform("nunique") > 1  # IC undefined if all equal
    return df[(n >= min_names) & varies]


def daily_ic(df: pd.DataFrame, target: str = "excess_ret", min_names: int = 5) -> pd.Series:
    """Spearman rank correlation between signal and target, computed per day.

    Rank-based: robust to fat-tailed returns and to signals on different scales
    (FinBERT probabilities vs LLM scores), so methods are comparable.
    """
    d = _eligible_days(df, min_names).copy()
    d["r_sig"] = d.groupby("date")["signal"].rank()
    d["r_tgt"] = d.groupby("date")[target].rank()
    return d.groupby("date")[["r_sig", "r_tgt"]].apply(lambda g: g["r_sig"].corr(g["r_tgt"])).dropna()


def ic_summary(ic: pd.Series) -> dict:
    n = len(ic)
    mean, std = ic.mean(), ic.std(ddof=1)
    return {
        "ic_mean": mean,
        "ic_std": std,
        "icir": mean / std if std > 0 else np.nan,
        "ic_tstat": mean / std * np.sqrt(n) if std > 0 else np.nan,  # is the mean IC distinguishable from 0?
        "ic_hit_rate": (ic > 0).mean(),
        "n_days": n,
    }


def long_short_returns(df: pd.DataFrame, min_names: int = 5, cost_bps: float = 10.0) -> pd.Series:
    """Daily P&L of a dollar-neutral book with weights proportional to the demeaned signal.

    Demeaning makes the book long-short (sum of weights = 0), so market moves cancel.
    Normalising sum|w| = 1 fixes gross exposure, so days are comparable.
    The book is rebuilt every day (1-day hold), so we pay one round-trip cost on the full gross.
    """
    d = _eligible_days(df, min_names).copy()
    d["w"] = d["signal"] - d.groupby("date")["signal"].transform("mean")
    d["w"] = d["w"] / d.groupby("date")["w"].transform(lambda w: w.abs().sum())
    gross = (d["w"] * d["fwd_ret"]).groupby(d["date"]).sum()
    return gross - cost_bps / 1e4


def sharpe(daily: pd.Series, horizon: int = 1) -> float:
    if daily.std(ddof=1) == 0 or len(daily) < 2:
        return np.nan
    return daily.mean() / daily.std(ddof=1) * np.sqrt(TRADING_DAYS / horizon)


def classification_metrics(df: pd.DataFrame, top_q: float = 0.3) -> dict:
    """How well does the signal's predicted class match the realised label?

    Predictions are made the SAME way labels are (see build_panel): per date, the
    symmetric percentile (rank - 1) / (n - 1) of the signal decides +1 / -1 / 0.
    A raw sign() rule would never predict 0 for a continuous signal (e.g. FinBERT),
    while an LLM signal can output exactly 0 -- comparing the two on macro-F1 would
    be unfair unless both go through the same percentile-based decision rule.

    directional_acc ignores label==0 (middle 40%): those moves are mostly noise.
    """
    g = df.groupby("date")["signal"]
    n = g.transform("size")
    rank = g.rank(method="average")
    denom = (n - 1).where(n > 1, 1)  # placeholder to avoid /0; overridden below
    pct = (rank - 1) / denom
    pred = np.select([pct >= 1 - top_q, pct <= top_q], [1, -1], default=0)
    pred = np.where(n > 1, pred, 0)  # a single name has no cross-section to rank against

    mask = (df["label"] != 0) & (pred != 0)
    return {
        "macro_f1": f1_score(df["label"], pred, average="macro"),
        "directional_acc": (pred[mask] == df["label"][mask]).mean() if mask.any() else np.nan,
        "coverage": (pred != 0).mean(),
    }


def evaluate(df: pd.DataFrame, min_names: int = 5, cost_bps: float = 10.0, horizon: int = 1,
             top_q: float = 0.3) -> dict:
    ic = daily_ic(df, min_names=min_names)
    ls = long_short_returns(df, min_names=min_names, cost_bps=cost_bps)
    out = ic_summary(ic)
    out.update({
        "ls_sharpe_net": sharpe(ls, horizon),
        "ls_sharpe_gross": sharpe(ls + cost_bps / 1e4, horizon),
        "ls_ann_return_net": ls.mean() * TRADING_DAYS / horizon,
    })
    out.update(classification_metrics(df, top_q=top_q))
    out["n_obs"] = len(df)
    return out
