"""Shared glue: baselines, signal -> backtest, Monte Carlo tests."""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import backtest as bt
from . import metrics as mt
from .config import SETTINGS
from .strategy import PortfolioParams, build_targets, exposure_series

NO_RISK = dict(use_stops=False, use_breaker=False)


def wide(pred: pd.Series, index, columns) -> pd.DataFrame:
    return pred.unstack("symbol").reindex(index=index, columns=columns)


def baseline_targets(P, member, start, end) -> dict[str, tuple[pd.DataFrame, dict]]:
    idx = P["close"].loc[start:end].index
    cols = P["close"].columns
    out = {}
    # 1. BTC buy & hold
    t = pd.DataFrame(np.nan, index=idx, columns=cols)
    t.iloc[0] = 0.0
    t.iloc[0, cols.get_loc("BTCUSDT")] = 1.0
    out["btc_buy_hold"] = (t, NO_RISK)
    # 2. equal-weight universe, rebalanced monthly (the "market" a passive holder gets)
    t = pd.DataFrame(np.nan, index=idx, columns=cols)
    m = member.reindex(index=idx, columns=cols).fillna(False).astype(bool)
    first_of_month = idx[(idx.day == 1) | (idx == idx[0])]
    for d in first_of_month:
        row = m.loc[d].astype(float)
        t.loc[d] = row / row.sum() if row.sum() else 0.0
    out["ew_universe_monthly"] = (t, NO_RISK)
    # 3. BTC time-series momentum (200d MA filter), decided daily
    c = P["close"]["BTCUSDT"]
    on = (c > c.rolling(200, min_periods=160).mean()).loc[idx].astype(float)
    t = pd.DataFrame(0.0, index=idx, columns=cols)
    t["BTCUSDT"] = on
    out["btc_trend_ma200"] = (t, NO_RISK)
    # 4. BTC scaled to the same 35% vol target as our system (fair risk-matched benchmark)
    vol = np.log(c).diff().rolling(30, min_periods=20).std() * np.sqrt(365)
    t = pd.DataFrame(0.0, index=idx, columns=cols)
    t["BTCUSDT"] = (SETTINGS.risk.target_vol / vol).clip(upper=1.0).loc[idx].fillna(0.0)
    out["btc_vol_target"] = (t, NO_RISK)
    return out


def xs_momentum_scores(P, idx) -> pd.DataFrame:
    """Simple cross-sectional momentum: 30-day return (the 'momentum' baseline)."""
    return np.log(P["close"]).diff(30).loc[idx]


def run_signal(scores: pd.DataFrame, P, member, M, regimes, exposure_mode: str,
               start, end, params: PortfolioParams | None = None, costs=None, **kw) -> bt.BTResult:
    params = params or PortfolioParams(exposure_mode=exposure_mode)
    idx = scores.loc[start:end].index
    expo = exposure_series(M.loc[idx], regimes, exposure_mode)
    tgt = build_targets(scores.loc[idx], P, member.reindex(index=idx), expo, params)
    return bt.run(tgt, P, costs=costs, **kw)


_MC: dict = {}


def _mc_one(seed: int) -> float:
    a = _MC
    rng = np.random.default_rng(seed)
    idx = a["P"]["close"].loc[a["start"]:a["end"]].index
    cols = a["P"]["close"].columns
    s = pd.DataFrame(rng.normal(size=(len(idx), len(cols))), index=idx, columns=cols)
    # persistent random scores (EWMA) so turnover is comparable to a real signal
    s = s.ewm(alpha=0.2).mean()
    r = run_signal(s, a["P"], a["member"], a["M"], a["regimes"], a["mode"], a["start"], a["end"])
    return mt.sharpe(r.returns)


def random_signal_test(P, member, M, regimes, exposure_mode, start, end, actual_sharpe: float,
                       n: int = 200, seed: int = 0, workers: int = 10) -> dict:
    """Same portfolio machinery, random coin selection each day. If our model's Sharpe
    is not well above this distribution, the 'edge' is just the risk overlay / beta."""
    import multiprocessing as mp
    _MC.update(P=P, member=member, M=M, regimes=regimes, mode=exposure_mode, start=start, end=end)
    with mp.get_context("fork").Pool(workers) as pool:   # fork shares the big panels
        sh = np.array(pool.map(_mc_one, range(seed, seed + n)))
    return {"random_sharpe_mean": float(np.nanmean(sh)), "random_sharpe_p95": float(np.nanquantile(sh, 0.95)),
            "p_value": float((np.sum(sh >= actual_sharpe) + 1) / (len(sh) + 1)), "n": n,
            "dist": sh.tolist()}
