"""Performance statistics, all on daily (365-day) crypto returns."""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

ANN = 365


def cagr(r: pd.Series) -> float:
    r = r.dropna()
    if len(r) < 2:
        return np.nan
    growth = float((1 + r).prod())
    return growth ** (ANN / len(r)) - 1 if growth > 0 else -1.0


def sharpe(r: pd.Series) -> float:
    r = r.dropna()
    s = r.std()
    return float(r.mean() / s * np.sqrt(ANN)) if s > 0 else np.nan


def sortino(r: pd.Series) -> float:
    r = r.dropna()
    dd = np.sqrt((r.clip(upper=0) ** 2).mean())
    return float(r.mean() / dd * np.sqrt(ANN)) if dd > 0 else np.nan


def max_drawdown(r: pd.Series) -> float:
    eq = (1 + r.fillna(0)).cumprod()
    return float((eq / eq.cummax() - 1).min())


def summary(r: pd.Series, bt=None, bench: pd.Series | None = None) -> dict:
    r = r.dropna()
    out = {
        "start": str(r.index.min().date()) if len(r) else None,
        "end": str(r.index.max().date()) if len(r) else None,
        "days": len(r),
        "total_return": float((1 + r).prod() - 1),
        "cagr": cagr(r),
        "vol": float(r.std() * np.sqrt(ANN)),
        "sharpe": sharpe(r),
        "sortino": sortino(r),
        "max_dd": max_drawdown(r),
        "calmar": cagr(r) / abs(max_drawdown(r)) if max_drawdown(r) < 0 else np.nan,
        "pos_days": float((r > 0).mean()),
        "exposure": np.nan,
    }
    if bt is not None:
        tr = bt.trades
        out["n_trades"] = int(len(tr))
        out["costs_paid_pct_of_start"] = float(bt.costs_paid / bt.equity.iloc[0]) if len(bt.equity) else np.nan
        years = max(len(r) / ANN, 1e-9)
        out["turnover_per_year"] = bt.turnover / years
        gross = bt.weights.sum(axis=1)
        out["exposure"] = float(gross.mean())
        out["win_rate"] = round_trip_win_rate(bt)
        out["events"] = len(bt.events)
    if bench is not None:
        b = bench.reindex(r.index).fillna(0)
        out["bench_total_return"] = float((1 + b).prod() - 1)
        out["excess_total_return"] = out["total_return"] - out["bench_total_return"]
        out["bench_sharpe"] = sharpe(b)
    return out


def round_trip_win_rate(bt) -> float:
    """Fraction of round trips (first buy -> weight back to 0) whose exit price beats the
    quantity-weighted average entry price by more than ~round-trip costs."""
    tr = bt.trades.sort_values("date", kind="stable")
    wins, n = 0, 0
    book: dict[str, list] = {}   # symbol -> [units bought, cost basis]
    for row in tr.itertuples():
        b = book.setdefault(row.symbol, [0.0, 0.0])
        if row.dw > 0:
            b[0] += row.qty
            b[1] += row.qty * row.price
        if row.w_after <= 1e-9 and b[0] > 0:
            pnl = row.price / (b[1] / b[0]) - 1
            n += 1
            wins += pnl > 0.0025
            book[row.symbol] = [0.0, 0.0]
    return wins / n if n else np.nan


# ---------- statistical robustness -----------------------------------------

def deflated_sharpe(r: pd.Series, n_trials: int, sr_var_trials: float | None = None) -> dict:
    """Bailey & Lopez de Prado (2014). Probability that the true Sharpe > 0
    after accounting for the number of strategy variants tried."""
    r = r.dropna()
    T = len(r)
    sr = r.mean() / r.std()  # per-period
    skew, kurt = stats.skew(r), stats.kurtosis(r, fisher=False)
    if sr_var_trials is None:
        sr_var_trials = 1.0 / T  # variance of SR estimates under H0
    emc = 0.5772156649
    n = max(n_trials, 1)
    sr0 = np.sqrt(sr_var_trials) * ((1 - emc) * stats.norm.ppf(1 - 1 / n) + emc * stats.norm.ppf(1 - 1 / (n * np.e))) if n > 1 else 0.0
    denom = np.sqrt(1 - skew * sr + (kurt - 1) / 4 * sr ** 2)
    z = (sr - sr0) * np.sqrt(T - 1) / denom
    return {"sharpe_ann": float(sr * np.sqrt(ANN)), "sr0_ann": float(sr0 * np.sqrt(ANN)),
            "n_trials": n, "dsr_prob": float(stats.norm.cdf(z))}


def block_bootstrap(r: pd.Series, n: int = 5000, block: int = 20, seed: int = 0) -> pd.DataFrame:
    """Stationary-ish block bootstrap of daily returns -> distribution of metrics."""
    rng = np.random.default_rng(seed)
    x = r.dropna().values
    T = len(x)
    out = np.empty((n, 3))
    nb = int(np.ceil(T / block))
    for i in range(n):
        starts = rng.integers(0, T - block, nb)
        s = np.concatenate([x[j:j + block] for j in starts])[:T]
        eq = np.cumprod(1 + s)
        out[i] = [s.mean() / s.std() * np.sqrt(ANN), eq[-1] ** (ANN / T) - 1, (eq / np.maximum.accumulate(eq) - 1).min()]
    return pd.DataFrame(out, columns=["sharpe", "cagr", "max_dd"])


def paired_bootstrap_diff(a: pd.Series, b: pd.Series, n: int = 5000, block: int = 20, seed: int = 0) -> dict:
    """P(Sharpe(a) > Sharpe(b)) via paired block bootstrap (same days resampled)."""
    df = pd.concat([a, b], axis=1).dropna()
    x = df.values
    T = len(x)
    rng = np.random.default_rng(seed)
    nb = int(np.ceil(T / block))
    d = np.empty(n)
    for i in range(n):
        starts = rng.integers(0, T - block, nb)
        s = np.concatenate([x[j:j + block] for j in starts])[:T]
        sa = s[:, 0].mean() / s[:, 0].std()
        sb = s[:, 1].mean() / s[:, 1].std()
        d[i] = (sa - sb) * np.sqrt(ANN)
    return {"sharpe_diff_mean": float(d.mean()), "p_a_better": float((d > 0).mean()),
            "ci95": [float(np.quantile(d, 0.025)), float(np.quantile(d, 0.975))]}
