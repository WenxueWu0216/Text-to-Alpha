"""Evaluation used by EVERY approach (FinBERT, lexicon, zero-/few-shot LLM, distilled student).

Input: a (date, ticker) frame with a `signal` column plus fwd_ret / excess_ret / label.
Keeping one shared evaluator is what makes the final benchmark table apples-to-apples.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

TRADING_DAYS = 252
DEFAULT_COST_GRID = (0.0, 5.0, 10.0, 20.0)


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
    ok = n > 1 and std > 0
    return {
        "ic_mean": mean,
        "ic_std": std,
        "icir": mean / std if ok else np.nan,
        "ic_tstat": mean / std * np.sqrt(n) if ok else np.nan,  # is the mean IC distinguishable from 0?
        "ic_hit_rate": (ic > 0).mean(),
        "n_days": n,
    }


def portfolio_weights(df: pd.DataFrame, min_names: int = 5) -> pd.DataFrame:
    """Dollar-neutral weights proportional to the demeaned signal, sum|w| = 1 per day.

    Demeaning makes the book long-short (sum of weights = 0), so market moves cancel.
    Normalising gross exposure to 1 makes days comparable.
    """
    d = _eligible_days(df, min_names)[["date", "ticker", "signal", "fwd_ret"]].copy()
    d["w"] = d["signal"] - d.groupby("date")["signal"].transform("mean")
    d["w"] = d["w"] / d.groupby("date")["w"].transform(lambda w: w.abs().sum())
    return d


def long_short_book(df: pd.DataFrame, min_names: int = 5) -> pd.DataFrame:
    """Daily gross P&L and turnover of the 1-day-hold book.

    turnover_t = sum_i |w_t,i - w_prev,i| over the union of names, where `prev` is the
    previous traded day (names absent on a day have weight 0). With sum|w| = 1, closing
    the whole book and opening a disjoint one costs turnover 2. The first day opens
    from cash: turnover 1.
    """
    d = portfolio_weights(df, min_names)
    gross = (d["w"] * d["fwd_ret"]).groupby(d["date"]).sum()
    wide = d.pivot_table(index="date", columns="ticker", values="w", aggfunc="sum").fillna(0.0)
    turnover = wide.diff().abs().sum(axis=1)
    if len(wide):
        turnover.iloc[0] = wide.iloc[0].abs().sum()
    return pd.DataFrame({"gross": gross, "turnover": turnover.reindex(gross.index)})


def net_returns(book: pd.DataFrame, cost_bps_per_side: float) -> pd.Series:
    return book["gross"] - book["turnover"] * cost_bps_per_side / 1e4


def long_short_returns(df: pd.DataFrame, min_names: int = 5, cost_bps_per_side: float = 5.0) -> pd.Series:
    return net_returns(long_short_book(df, min_names), cost_bps_per_side)


def sharpe(daily: pd.Series, horizon: int = 1) -> float:
    if len(daily) < 2 or daily.std(ddof=1) == 0:
        return np.nan
    return daily.mean() / daily.std(ddof=1) * np.sqrt(TRADING_DAYS / horizon)


def breakeven_cost_bps(book: pd.DataFrame) -> float:
    """Per-side cost at which mean net P&L (hence net Sharpe) is exactly zero.

    Negative means the book loses money even before costs.
    """
    mean_turnover = book["turnover"].mean()
    if not len(book) or mean_turnover == 0:
        return np.nan
    return book["gross"].mean() / mean_turnover * 1e4


def trading_summary(book: pd.DataFrame, horizon: int = 1, cost_bps_per_side: float = 5.0,
                    cost_grid=DEFAULT_COST_GRID) -> dict:
    net = net_returns(book, cost_bps_per_side)
    mean_turnover = book["turnover"].mean()
    return {
        "ls_sharpe_gross": sharpe(book["gross"], horizon),
        "ls_sharpe_net": sharpe(net, horizon),
        "ls_ann_return_gross": book["gross"].mean() * TRADING_DAYS / horizon,
        "ls_ann_return_net": net.mean() * TRADING_DAYS / horizon,
        "cost_bps_per_side": cost_bps_per_side,
        "turnover_mean": mean_turnover,
        # Average holding period of a dollar in the book: turnover 2 (= full replacement) -> 1 day.
        "holding_period_days": 2.0 / mean_turnover if mean_turnover > 0 else np.nan,
        "sharpe_by_cost_bps_per_side": {f"{c:g}": sharpe(net_returns(book, c), horizon)
                                        for c in cost_grid},
        "breakeven_cost_bps_per_side": breakeven_cost_bps(book),
    }


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


def _period_stats(ic: pd.Series, book: pd.DataFrame, horizon: int, cost_bps_per_side: float) -> dict:
    s = ic_summary(ic)
    return {
        "ic_mean": s["ic_mean"], "ic_tstat": s["ic_tstat"], "n_days": s["n_days"],
        "ls_sharpe_gross": sharpe(book["gross"], horizon),
        "ls_sharpe_net": sharpe(net_returns(book, cost_bps_per_side), horizon),
    }


def breakdown(ic: pd.Series, book: pd.DataFrame, labels: pd.Series, horizon: int = 1,
              cost_bps_per_side: float = 5.0) -> dict:
    """Stats per regime label (e.g. calendar year, high/low VIX). labels: date -> label."""
    out = {}
    lab_ic = labels.reindex(ic.index)
    lab_book = labels.reindex(book.index)
    for lab in sorted(labels.dropna().unique()):
        out[str(lab)] = _period_stats(ic[lab_ic == lab], book[lab_book == lab], horizon,
                                      cost_bps_per_side)
    return out


def evaluate(df: pd.DataFrame, min_names: int = 5, cost_bps_per_side: float = 5.0, horizon: int = 1,
             top_q: float = 0.3, cost_grid=DEFAULT_COST_GRID,
             regimes: dict[str, pd.Series] | None = None) -> dict:
    """regimes: name -> Series(date -> regime label). Must only use information known
    before each date's open (see t2a.data.market.vix_regime)."""
    ic = daily_ic(df, min_names=min_names)
    book = long_short_book(df, min_names=min_names)
    out = ic_summary(ic)
    out.update(trading_summary(book, horizon, cost_bps_per_side, cost_grid))
    # The pre-turnover cost model (full round trip on the whole book every day), kept so
    # numbers reported before the turnover model remain comparable.
    out["ls_sharpe_net_full_rebuild_10bps_rt"] = sharpe(book["gross"] - 10 / 1e4, horizon)
    out.update(classification_metrics(df, top_q=top_q))
    out["n_obs"] = len(df)

    years = pd.Series(pd.DatetimeIndex(book.index).year, index=book.index)
    out["by_year"] = breakdown(ic, book, years, horizon, cost_bps_per_side)
    for name, labels in (regimes or {}).items():
        out[f"by_{name}"] = breakdown(ic, book, labels, horizon, cost_bps_per_side)
    out["ic_by_month"] = {k.strftime("%Y-%m"): v
                          for k, v in ic.groupby(pd.DatetimeIndex(ic.index).to_period("M")).mean()
                          .to_timestamp().items()}
    return out


def daily_series(df: pd.DataFrame, min_names: int = 5, cost_bps_per_side: float = 5.0) -> pd.DataFrame:
    """Per-day IC, gross/net P&L and turnover: what a dashboard plots."""
    book = long_short_book(df, min_names)
    out = book.assign(net=net_returns(book, cost_bps_per_side))
    out["ic"] = daily_ic(df, min_names=min_names)
    out.index.name = "date"
    return out.reset_index()
