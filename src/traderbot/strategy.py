"""Signals -> target weights, with the portfolio-level risk rules.

Stateful (hysteresis) so a coin is kept while it stays in the top `hold_k`,
which cuts turnover; new entries need to be in the top `top_k`.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .config import SETTINGS, Risk


@dataclass(frozen=True)
class PortfolioParams:
    top_k: int = 5
    hold_k: int = 8
    rebalance_every: int = 1      # days between re-optimisations
    exposure_mode: str = "regime"  # "full" | "regime" | "trend"
    min_score: float | None = None  # optional absolute gate on the signal


def _clusters(corr: pd.DataFrame, thr: float) -> list[list[str]]:
    left = list(corr.index)
    groups = []
    while left:
        s = left.pop(0)
        g = [s] + [o for o in left if corr.loc[s, o] > thr]
        left = [o for o in left if o not in g]
        groups.append(g)
    return groups


def size(selected: list[str], t, rets: pd.DataFrame, risk: Risk, exposure: float) -> pd.Series:
    """Inverse-vol weights, scaled to target vol, capped per asset/cluster/gross."""
    if not selected:
        return pd.Series(dtype=float)
    hist = rets.loc[:t, selected].iloc[-90:]
    vol = hist.iloc[-30:].std().replace(0, np.nan).fillna(hist.std()).fillna(0.05).clip(lower=0.005)
    w = (1 / vol) / (1 / vol).sum()
    w = w.clip(upper=risk.max_weight)
    cov = hist.cov().fillna(0) * 365
    port_vol = float(np.sqrt(w.values @ cov.values @ w.values)) if len(hist) > 20 else np.nan
    scale = risk.target_vol / port_vol if port_vol and port_vol > 0 else 1.0
    w = w * min(scale, 10)
    w = w.clip(upper=risk.max_weight)
    if len(selected) > 1 and len(hist) > 20:
        for g in _clusters(hist.corr().fillna(0), risk.cluster_corr):
            tot = w[g].sum()
            if tot > risk.max_cluster_weight:
                w[g] *= risk.max_cluster_weight / tot
    if w.sum() > risk.max_gross:
        w *= risk.max_gross / w.sum()
    return w * exposure


def exposure_series(M: pd.DataFrame, regimes: pd.DataFrame | None, mode: str) -> pd.Series:
    if mode == "full" or (mode == "regime" and regimes is None):
        return pd.Series(1.0, index=M.index)
    if mode == "trend":
        return (M["btc_dist_ma_200"] > 0).astype(float).where(M["btc_dist_ma_200"].notna(), 0.0)
    if mode == "regime":
        r = regimes.reindex(M.index).ffill()
        # cut exposure in proportion to stress probability; small floor stays invested
        e = 1.0 - 0.75 * r["p_stress"].fillna(0)
        return e.clip(0.25, 1.0)
    raise ValueError(mode)


def build_targets(scores: pd.DataFrame, P: dict[str, pd.DataFrame], member: pd.DataFrame,
                  exposure: pd.Series, params: PortfolioParams = PortfolioParams(),
                  risk: Risk | None = None) -> pd.DataFrame:
    risk = risk or SETTINGS.risk
    rets = np.log(P["close"]).diff()
    cols = scores.columns
    out = pd.DataFrame(np.nan, index=scores.index, columns=cols)
    held: list[str] = []
    for i, t in enumerate(scores.index):
        if i % params.rebalance_every:
            continue
        s = scores.loc[t].where(member.loc[t, cols].astype(bool)).dropna()
        if params.min_score is not None:
            s = s[s > params.min_score]
        if s.empty:
            out.loc[t] = 0.0
            held = []
            continue
        order = list(s.sort_values(ascending=False).index)
        keep = [h for h in held if h in order[: params.hold_k]]
        new = [o for o in order if o not in keep][: max(params.top_k - len(keep), 0)]
        held = (keep + new)[: params.top_k]
        w = size(held, t, rets, risk, float(exposure.get(t, 1.0)))
        row = pd.Series(0.0, index=cols)
        row[w.index] = w.values
        out.loc[t] = row
    return out
