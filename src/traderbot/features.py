"""Feature engineering on point-in-time daily panels.

Timing convention (enforced everywhere):
  * daily bar t covers [t 00:00, t+1 00:00) UTC; its close is known at t+1 00:00.
  * features for decision date t use only bars with ts <= t.
  * the trade is executed at exec_px[t+1] = typical price (OHLC/4) of the first
    hourly bar of day t+1 (i.e. ~00:00-01:00 UTC), never at a price known before.
  * labels are forward returns from exec_px[t+1] to exec_px[t+1+h].
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import data

ANN = 365


def build_panels(symbols: list[str]) -> dict[str, pd.DataFrame]:
    """Wide daily panels (date x asset) + hourly-derived daily statistics."""
    d = data.load_long(symbols, "1d")
    piv = lambda f: d.pivot(index="ts", columns="symbol", values=f).sort_index()
    P = {f: piv(f) for f in ("open", "high", "low", "close", "volume", "quote_volume",
                              "taker_buy_quote", "trades")}
    idx = pd.date_range(P["close"].index.min(), P["close"].index.max(), freq="D", tz="UTC")
    P = {k: v.reindex(idx) for k, v in P.items()}

    h = data.load_long(symbols, "1h")
    if len(h):
        h = h.sort_values(["symbol", "ts"])
        h["day"] = h["ts"].dt.floor("D")
        lr = np.log(h["close"]).groupby(h["symbol"]).diff()
        # only use returns between consecutive hours
        gap = h.groupby("symbol")["ts"].diff() == pd.Timedelta(hours=1)
        h["lr"] = lr.where(gap)
        h["lr2"] = h["lr"] ** 2
        h["dn2"] = h["lr"].clip(upper=0) ** 2
        g = h.groupby(["day", "symbol"])
        n = g["lr"].count()
        ok = n >= 12
        P["h_rv"] = np.sqrt(g["lr2"].sum().where(ok)).unstack().reindex(idx)
        P["h_down_rv"] = np.sqrt(g["dn2"].sum().where(ok)).unstack().reindex(idx)
        P["h_n"] = n.unstack().reindex(idx)
        first = h[h["ts"].dt.hour == 0].copy()
        first["tp"] = (first["open"] + first["high"] + first["low"] + first["close"]) / 4
        P["first_hour_px"] = first.pivot(index="day", columns="symbol", values="tp").reindex(idx)
        P["hourly_lr"] = h.pivot(index="ts", columns="symbol", values="lr")
    cols = P["close"].columns
    P = {k: (v.reindex(columns=cols) if k != "hourly_lr" else v) for k, v in P.items()}

    # execution price for a decision taken at the end of day t: first hour of t+1
    px = P.get("first_hour_px", pd.DataFrame(index=idx, columns=cols))
    px = px.where(px.notna(), P["open"])           # fall back to daily open
    P["exec_px"] = px.shift(-1)                      # value at index t = price on t+1
    return P


def _rsi(close: pd.DataFrame, n: int = 14) -> pd.DataFrame:
    d = close.diff()
    up = d.clip(lower=0).ewm(alpha=1 / n, adjust=False, min_periods=n).mean()
    dn = (-d.clip(upper=0)).ewm(alpha=1 / n, adjust=False, min_periods=n).mean()
    return 100 - 100 / (1 + up / dn.replace(0, np.nan))


def _xs_rank(df: pd.DataFrame, mask: pd.DataFrame | None = None) -> pd.DataFrame:
    if mask is not None:
        df = df.where(mask)
    return df.rank(axis=1, pct=True) - 0.5


def asset_features(P: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    c, o, h, l = P["close"], P["open"], P["high"], P["low"]
    qv = P["quote_volume"]
    lc = np.log(c)
    r1 = lc.diff()
    F: dict[str, pd.DataFrame] = {}

    vol30 = r1.rolling(30, min_periods=20).std()
    for k in (1, 3, 7, 14, 30, 60, 90):
        F[f"ret_{k}"] = lc.diff(k)
        if k >= 7:
            F[f"mom_va_{k}"] = lc.diff(k) / (vol30 * np.sqrt(k))
    F["mom_12_1"] = lc.shift(7) - lc.shift(90)  # medium-term momentum skipping last week
    for n in (20, 50, 200):
        F[f"dist_ma_{n}"] = c / c.rolling(n, min_periods=int(n * 0.8)).mean() - 1

    # volatility
    F["vol_7"] = r1.rolling(7, min_periods=5).std()
    F["vol_30"] = vol30
    F["vol_ratio"] = F["vol_7"] / vol30
    hl = np.log(h / l)
    F["parkinson_14"] = np.sqrt((hl ** 2).rolling(14, min_periods=10).mean() / (4 * np.log(2)))
    co = np.log(c / o)
    F["gk_14"] = np.sqrt((0.5 * hl ** 2 - (2 * np.log(2) - 1) * co ** 2).rolling(14, min_periods=10).mean().clip(lower=0))
    pc = c.shift()
    tr = np.maximum(np.maximum(h - l, (h - pc).abs()), (l - pc).abs())
    F["atr_14"] = tr.rolling(14, min_periods=10).mean() / c
    if "h_rv" in P:
        F["hrv_7"] = np.sqrt((P["h_rv"] ** 2).rolling(7, min_periods=5).mean())
        F["hrv_30"] = np.sqrt((P["h_rv"] ** 2).rolling(30, min_periods=20).mean())
        F["hdown_share_7"] = (P["h_down_rv"] ** 2).rolling(7, min_periods=5).sum() / (P["h_rv"] ** 2).rolling(7, min_periods=5).sum()
        hl_r = P["hourly_lr"]
        sk = hl_r.rolling(24 * 14, min_periods=24 * 10).skew()
        ku = hl_r.rolling(24 * 14, min_periods=24 * 10).kurt()
        # take the value at the last hour of each day (23:00 -> belongs to day t)
        day_end = lambda x: x[x.index.hour == 23].set_axis(x[x.index.hour == 23].index.floor("D")).reindex(c.index)
        F["hskew_14"] = day_end(sk).reindex(columns=c.columns)
        F["hkurt_14"] = day_end(ku).reindex(columns=c.columns)
    F["skew_60"] = r1.rolling(60, min_periods=40).skew()
    F["max_ret_30"] = r1.rolling(30, min_periods=20).max()
    F["dd_30"] = c / c.rolling(30, min_periods=20).max() - 1

    # volume / liquidity
    lqv = np.log(qv.replace(0, np.nan))
    F["qv_z_30"] = (lqv - lqv.rolling(30, min_periods=20).mean()) / lqv.rolling(30, min_periods=20).std()
    F["qv_trend"] = lqv.rolling(7, min_periods=5).mean() - lqv.rolling(60, min_periods=40).mean()
    F["taker_buy_7"] = (P["taker_buy_quote"].rolling(7, min_periods=5).sum() / qv.rolling(7, min_periods=5).sum()) - 0.5
    F["amihud_30"] = np.log1p((r1.abs() / qv.replace(0, np.nan)).rolling(30, min_periods=20).mean() * 1e9)
    F["log_adv_30"] = np.log(qv.rolling(30, min_periods=20).mean())
    F["trades_z_30"] = np.log1p(P["trades"]).pipe(lambda x: (x - x.rolling(30, min_periods=20).mean()) / x.rolling(30, min_periods=20).std())

    # oscillators (normalised)
    F["rsi_14"] = _rsi(c, 14) / 100 - 0.5
    F["rsi_2"] = _rsi(c, 2) / 100 - 0.5
    ema12, ema26 = c.ewm(span=12, adjust=False).mean(), c.ewm(span=26, adjust=False).mean()
    macd = ema12 - ema26
    F["macd_hist"] = (macd - macd.ewm(span=9, adjust=False).mean()) / c
    ma20, sd20 = c.rolling(20, min_periods=16).mean(), c.rolling(20, min_periods=16).std()
    F["bb_pctb"] = (c - ma20) / (2 * sd20)
    F["bb_width"] = 4 * sd20 / ma20
    return F


def market_features(P: dict[str, pd.DataFrame], member: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    """Market-wide series (date) and cross-asset per-asset features."""
    c = P["close"]
    r1 = np.log(c).diff()
    m = member.reindex(index=c.index, columns=c.columns).fillna(False).astype(bool)
    mkt_r = r1.where(m).mean(axis=1)                 # equal-weight universe return
    btc = "BTCUSDT"
    btc_r = r1[btc]
    qv = P["quote_volume"].where(m)
    M = pd.DataFrame(index=c.index)
    M["mkt_ret_1"] = mkt_r
    for k in (7, 30):
        M[f"mkt_ret_{k}"] = mkt_r.rolling(k).sum()
        M[f"btc_ret_{k}"] = btc_r.rolling(k).sum()
    M["btc_ret_1"] = btc_r
    M["mkt_vol_30"] = mkt_r.rolling(30, min_periods=20).std()
    M["btc_vol_30"] = btc_r.rolling(30, min_periods=20).std()
    M["btc_dist_ma_200"] = c[btc] / c[btc].rolling(200, min_periods=160).mean() - 1
    M["btc_vol_share"] = P["quote_volume"][btc] / qv.sum(axis=1)  # dominance proxy (by volume)
    M["btc_vol_share_chg_30"] = M["btc_vol_share"].rolling(7).mean() - M["btc_vol_share"].rolling(60, min_periods=40).mean()
    above = (c > c.rolling(50, min_periods=40).mean()).astype(float).where(m)
    M["breadth_50"] = above.mean(axis=1)
    M["dispersion_7"] = np.log(c).diff(7).where(m).std(axis=1)

    X: dict[str, pd.DataFrame] = {}
    cov = r1.rolling(60, min_periods=40).cov(mkt_r)
    X["beta_mkt_60"] = cov.div(mkt_r.rolling(60, min_periods=40).var(), axis=0)
    X["corr_btc_60"] = r1.rolling(60, min_periods=40).corr(btc_r)
    X["corr_mkt_60"] = r1.rolling(60, min_periods=40).corr(mkt_r)
    M["avg_corr_60"] = X["corr_mkt_60"].where(m).mean(axis=1)
    resid = r1.sub(X["beta_mkt_60"].mul(mkt_r, axis=0))
    X["idio_vol_30"] = resid.rolling(30, min_periods=20).std()
    for k in (7, 30):
        X[f"rs_btc_{k}"] = np.log(c).diff(k).sub(np.log(c[btc]).diff(k), axis=0)
        X[f"rs_mkt_{k}"] = np.log(c).diff(k).sub(mkt_r.rolling(k).sum(), axis=0)
    M = M.astype(float).replace([np.inf, -np.inf], np.nan)
    return M, X


def labels(P: dict[str, pd.DataFrame], horizons=(1, 3, 7)) -> dict[str, pd.DataFrame]:
    """Forward log returns from exec_px[t] (= price on t+1) to exec_px[t+h]."""
    lx = np.log(P["exec_px"])
    return {f"fwd_{h}": lx.shift(-h) - lx for h in horizons}


def to_long(F: dict[str, pd.DataFrame], member: pd.DataFrame, M: pd.DataFrame,
            Y: dict[str, pd.DataFrame], rank_cols: list[str]) -> pd.DataFrame:
    """Stack wide features into a long (date, symbol) design matrix, members only."""
    m = member.astype(bool)
    stacked = {}
    for k, v in F.items():
        v = v.reindex(index=m.index, columns=m.columns)
        stacked[k] = v.where(m).stack(future_stack=True)
        if k in rank_cols:
            stacked[f"xr_{k}"] = _xs_rank(v, m).stack(future_stack=True)
    for k, v in Y.items():
        v = v.reindex(index=m.index, columns=m.columns)
        stacked[k] = v.where(m).stack(future_stack=True)
        # cross-sectional demeaned target (excess vs. universe average)
        stacked[f"{k}_xs"] = v.where(m).sub(v.where(m).mean(axis=1), axis=0).stack(future_stack=True)
    df = pd.DataFrame(stacked)
    df.index.names = ["date", "symbol"]
    df = df[m.stack(future_stack=True).reindex(df.index).fillna(False).values]
    df = df.join(M, on="date")
    return df.sort_index()
