"""Event-driven daily portfolio simulator with realistic frictions.

Clock: decision at end of day t (index t). Trades fill at exec_px[t] (first hour of
day t+1). Positions are then exposed to day t+1's path until the next fill at
exec_px[t+1]. Stops/TPs are checked against day t+1's high/low.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .config import SETTINGS, Costs, Risk


def cost_rate(adv: pd.Series, vol: pd.Series, trade_value: pd.Series, costs: Costs) -> pd.Series:
    """One-way cost as a fraction of traded value, per asset."""
    adv = adv.clip(lower=1e5).fillna(1e5)
    vol = vol.fillna(0.05)
    half_spread = np.maximum(costs.half_spread_min, 0.5 * 0.0010 * np.sqrt(1e8 / adv)).clip(upper=0.01)
    impact = costs.impact_k * vol * np.sqrt(trade_value.abs() / adv)
    return costs.fee + half_spread + costs.slip_floor + impact


@dataclass
class BTResult:
    equity: pd.Series
    returns: pd.Series
    weights: pd.DataFrame
    trades: pd.DataFrame
    costs_paid: float
    turnover: float
    events: list = field(default_factory=list)


def run(target: pd.DataFrame, P: dict[str, pd.DataFrame], costs: Costs | None = None,
        risk: Risk | None = None, capital: float = 10_000.0, use_stops: bool = True,
        use_breaker: bool = True, start=None, end=None) -> BTResult:
    """target: desired weights decided at end of day t (rows), NaN row = keep."""
    costs = costs or SETTINGS.costs
    risk = risk or SETTINGS.risk
    px = P["exec_px"]
    nxt_high, nxt_low, nxt_open = P["high"].shift(-1), P["low"].shift(-1), P["open"].shift(-1)
    adv = P["quote_volume"].rolling(30, min_periods=5).mean()
    vol = np.log(P["close"]).diff().rolling(30, min_periods=10).std()
    atr = (P["high"] - P["low"]).rolling(14, min_periods=5).mean()
    last_close = P["close"].ffill()

    dates = target.index
    if start is not None:
        dates = dates[dates >= pd.Timestamp(start, tz="UTC")]
    if end is not None:
        dates = dates[dates <= pd.Timestamp(end, tz="UTC")]
    cols = target.columns

    w = pd.Series(0.0, index=cols)          # current weights (fraction of equity)
    entry = pd.Series(np.nan, index=cols)    # entry price for stop logic
    stop = pd.Series(np.nan, index=cols)
    take = pd.Series(np.nan, index=cols)
    cooldown = pd.Series(0, index=cols)
    eq, peak = capital, capital
    halted_until = None
    risk_scale = 1.0
    equity, rets, W, trades, events = [], [], [], [], []
    total_cost, total_turn = 0.0, 0.0
    prev_ret = 0.0

    for i, t in enumerate(dates):
        p_now = px.loc[t, cols]
        # ---- 1. decide target for this fill
        tgt = target.loc[t, cols].fillna(0.0).clip(lower=0) if target.loc[t].notna().any() else w.copy()
        tgt = tgt.where(p_now.notna(), 0.0)          # cannot trade what has no price
        tgt = tgt.where(cooldown <= 0, 0.0)
        if use_breaker:
            dd = eq / peak - 1
            if halted_until is None and dd < -risk.breaker_dd:
                halted_until = i + 30
                events.append((t, "circuit_breaker", round(dd, 4)))
            if halted_until is not None:
                if i < halted_until:
                    tgt = tgt * 0.0
                else:
                    halted_until, peak, risk_scale = None, eq, 0.5   # resume at half risk
                    events.append((t, "resume_half_risk", None))
            elif risk_scale < 1 and eq >= peak:
                risk_scale = 1.0
            if prev_ret < -risk.daily_loss_limit:     # no new risk after a bad day
                tgt = np.minimum(tgt, w)
                events.append((t, "daily_loss_limit", round(prev_ret, 4)))
        tgt = tgt * risk_scale
        delta = tgt - w
        delta = delta.where(delta.abs() >= risk.min_trade, 0.0)
        delta = delta.where(~((tgt == 0) & (w > 0)), -w)     # always allow full exits
        # ---- 2. execute with costs
        tv = delta.abs() * eq
        cr = cost_rate(adv.loc[t, cols], vol.loc[t, cols], tv, costs)
        c_paid = float((tv * cr).sum())
        eq -= c_paid
        total_cost += c_paid
        total_turn += float(delta.abs().sum())
        for s in delta[delta != 0].index:
            trades.append((t, s, float(delta[s]), float(p_now[s]), float(tv[s] * cr[s]), float(delta[s] * eq / p_now[s]), float(w[s] + delta[s])))
        new_w = w + delta
        opened = (w <= 1e-9) & (new_w > 1e-9)
        entry[opened] = p_now[opened]
        stop[opened] = p_now[opened] - risk.stop_atr * atr.loc[t, cols][opened]
        take[opened] = p_now[opened] + risk.take_atr * atr.loc[t, cols][opened]
        entry[new_w <= 1e-9] = stop[new_w <= 1e-9] = take[new_w <= 1e-9] = np.nan
        w = new_w
        W.append(w.rename(t))
        cooldown -= 1

        # ---- 3. hold until next fill
        if i + 1 >= len(dates):
            equity.append((t, eq))
            rets.append((t, 0.0))
            break
        t1 = dates[i + 1]
        p_next = px.loc[t1, cols]
        held = w[w > 1e-9].index
        asset_ret = pd.Series(0.0, index=cols)
        for s in held:
            p0, p1 = p_now[s], p_next[s]
            if np.isnan(p1):   # delisted / no data: exit at last traded close, extra 2% haircut
                lc = last_close.loc[:t1, s].iloc[-1]
                asset_ret[s] = lc / p0 * 0.98 - 1
                w_exit = w[s] * (1 + asset_ret[s])
                c = w_exit * eq * costs.fee
                eq -= c
                total_cost += c
                trades.append((t1, s, -float(w[s]), float(lc), float(c), np.nan, 0.0))
                events.append((t1, f"forced_exit_{s}", round(asset_ret[s], 4)))
                cooldown[s] = 10**6
                continue
            if use_stops and not np.isnan(stop[s]):
                lo, hi, op = nxt_low.loc[t, s], nxt_high.loc[t, s], nxt_open.loc[t, s]
                exit_px = None
                if lo <= stop[s]:
                    exit_px = min(stop[s], op) if not np.isnan(op) else stop[s]
                    tag = "stop"
                elif hi >= take[s]:
                    exit_px = max(take[s], op) if not np.isnan(op) else take[s]
                    tag = "take"
                if exit_px is not None:
                    asset_ret[s] = exit_px / p0 - 1
                    w_exit = w[s] * (1 + asset_ret[s])
                    cr_s = cost_rate(adv.loc[t, [s]], vol.loc[t, [s]] * (2 if tag == "stop" else 1),
                                     pd.Series([w_exit * eq], index=[s]), costs).iloc[0]
                    c = w_exit * eq * cr_s
                    eq -= c
                    total_cost += c
                    total_turn += abs(w_exit)
                    trades.append((t1, s, -float(w[s]), float(exit_px), float(c), np.nan, 0.0))
                    cooldown[s] = 3
                    continue
            asset_ret[s] = p1 / p0 - 1
        port_ret = float((w * asset_ret).sum())
        eq_before = eq
        eq *= 1 + port_ret
        # weights drift; stopped/delisted positions go to cash
        stopped = cooldown > 0
        grown = w * (1 + asset_ret)
        w = (grown / (1 + port_ret)).where(~stopped, 0.0)
        prev_ret = eq / (eq_before + 1e-12) - 1
        peak = max(peak, eq)
        equity.append((t1, eq))
        rets.append((t1, eq / (equity[-2][1] if len(equity) > 1 else capital) - 1))

    eq_s = pd.Series(dict(equity)).sort_index()
    ret_s = eq_s.pct_change().fillna(eq_s.iloc[0] / capital - 1)
    tr = pd.DataFrame(trades, columns=["date", "symbol", "dw", "price", "cost", "qty", "w_after"])
    return BTResult(eq_s, ret_s, pd.DataFrame(W), tr, total_cost, total_turn, events)
