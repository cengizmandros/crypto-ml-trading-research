"""Paper-trading engine. Simulated fills against LIVE public order-book quotes
(bid/ask from /api/v3/ticker/bookTicker) plus fee and slippage. No keys, no orders.

State lives in paper/paper.db (SQLite).
"""
from __future__ import annotations

import json
import logging
import sqlite3
from contextlib import closing

import numpy as np
import pandas as pd
import requests

from . import dataset, notify, production
from .backtest import cost_rate
from .config import PAPER, SETTINGS
from .data import base_symbol
from .strategy import PortfolioParams, exposure_series, size

log = logging.getLogger(__name__)
DB = PAPER / "paper.db"
API = "https://api.binance.com/api/v3"

SCHEMA = """
CREATE TABLE IF NOT EXISTS state(key TEXT PRIMARY KEY, value TEXT);
CREATE TABLE IF NOT EXISTS positions(symbol TEXT PRIMARY KEY, qty REAL, entry_px REAL, stop REAL, take REAL, opened_at TEXT);
CREATE TABLE IF NOT EXISTS trades(ts TEXT, symbol TEXT, side TEXT, qty REAL, price REAL, notional REAL, cost REAL, reason TEXT);
CREATE TABLE IF NOT EXISTS signals(date TEXT, symbol TEXT, score REAL, rank INTEGER, target_w REAL, member INTEGER);
CREATE TABLE IF NOT EXISTS equity(ts TEXT PRIMARY KEY, equity REAL, cash REAL, gross REAL, btc_px REAL, p_stress REAL, kind TEXT);
CREATE TABLE IF NOT EXISTS events(ts TEXT, kind TEXT, detail TEXT);
"""


def db() -> sqlite3.Connection:
    con = sqlite3.connect(DB, timeout=30)
    con.executescript(SCHEMA)
    return con


def now() -> str:
    return pd.Timestamp.now(tz="UTC").isoformat(timespec="seconds")


def get_state(con, key, default=None):
    r = con.execute("SELECT value FROM state WHERE key=?", (key,)).fetchone()
    return json.loads(r[0]) if r else default


def set_state(con, key, value):
    con.execute("INSERT OR REPLACE INTO state VALUES(?,?)", (key, json.dumps(value)))


def event(con, kind, detail=""):
    con.execute("INSERT INTO events VALUES(?,?,?)", (now(), kind, str(detail)))
    log.info("event %s %s", kind, detail)


def quotes(names: list[str]) -> pd.DataFrame:
    """Live bid/ask indexed by our internal names ('X@date' segments map to ticker X)."""
    if not names:
        return pd.DataFrame(columns=["bid", "ask"])
    tick = {n: base_symbol(n) for n in names}
    uniq = sorted(set(tick.values()))
    r = requests.get(f"{API}/ticker/bookTicker", params={"symbols": json.dumps(uniq, separators=(",", ":"))}, timeout=20)
    if r.status_code == 400:   # one unknown/delisted symbol fails the batch -> ask one by one
        rows = []
        for u in uniq:
            x = requests.get(f"{API}/ticker/bookTicker", params={"symbol": u}, timeout=20)
            if x.status_code == 200:
                rows.append(x.json())
        q = pd.DataFrame(rows, columns=["symbol", "bidPrice", "askPrice"]).set_index("symbol")
    else:
        r.raise_for_status()
        q = pd.DataFrame(r.json()).set_index("symbol")
    q = q[["bidPrice", "askPrice"]].astype(float).rename(columns={"bidPrice": "bid", "askPrice": "ask"})
    return q.reindex([tick[n] for n in names]).set_axis(names)


def positions(con) -> pd.DataFrame:
    return pd.read_sql("SELECT * FROM positions", con).set_index("symbol")


def equity_now(con, q: pd.DataFrame | None = None) -> tuple[float, float, pd.Series]:
    cash = get_state(con, "cash", SETTINGS.paper_capital)
    pos = positions(con)
    if pos.empty:
        return cash, cash, pd.Series(dtype=float)
    q = q if q is not None else quotes(list(pos.index))
    mv = pos["qty"] * q["bid"].reindex(pos.index).fillna(pos["entry_px"])
    return cash + float(mv.sum()), cash, mv


def init():
    with closing(db()) as con:
        if get_state(con, "cash") is None:
            set_state(con, "cash", SETTINGS.paper_capital)
            set_state(con, "peak", SETTINGS.paper_capital)
            set_state(con, "halted", False)
            set_state(con, "started", now())
            event(con, "init", f"capital {SETTINGS.paper_capital}")
        con.commit()


def _fill(con, symbol: str, qty: float, q: pd.DataFrame, adv: float, vol: float, reason: str):
    """Execute a simulated market order; qty>0 buy at ask, qty<0 sell at bid."""
    side = "BUY" if qty > 0 else "SELL"
    px = q.loc[symbol, "ask"] if qty > 0 else q.loc[symbol, "bid"]
    notional = abs(qty) * px
    c = SETTINGS.costs
    # spread is already paid via bid/ask; add fee + slippage + impact
    rate = c.fee + c.slip_floor + c.impact_k * vol * np.sqrt(notional / max(adv, 1e5))
    cost = notional * rate
    cash = get_state(con, "cash")
    cash += -notional - cost if qty > 0 else notional - cost
    set_state(con, "cash", cash)
    pos = positions(con)
    if symbol in pos.index:
        new_qty = pos.loc[symbol, "qty"] + qty
        if new_qty * px < 1.0:  # dust -> close
            con.execute("DELETE FROM positions WHERE symbol=?", (symbol,))
        else:
            con.execute("UPDATE positions SET qty=? WHERE symbol=?", (new_qty, symbol))
    con.execute("INSERT INTO trades VALUES(?,?,?,?,?,?,?,?)",
                (now(), symbol, side, abs(qty), px, notional, cost, reason))
    return px


def daily_rebalance(params: PortfolioParams | None = None):
    """Run after the daily close: compute signals and rebalance to targets."""
    sel = json.loads((production.REPORTS / "selected.json").read_text())
    params = params or PortfolioParams(exposure_mode=sel["exposure_mode"])
    risk = SETTINGS.risk
    d = dataset.load(research=False)
    score, t, reg = production.predict_latest(d)
    P, M, member = d["P"], d["M"], d["member"]
    with closing(db()) as con:
        last_done = get_state(con, "last_rebalance_date")
        if last_done == str(t.date()):
            log.info("already rebalanced for %s", t.date())
            return
        mem = member.loc[t].reindex(score.index).fillna(False).astype(bool)
        s = score[mem].dropna().sort_values(ascending=False)
        held = list(positions(con).index)
        order = list(s.index)
        keep = [h for h in held if h in order[: params.hold_k]]
        new = [o for o in order if o not in keep][: max(params.top_k - len(keep), 0)]
        chosen = (keep + new)[: params.top_k]
        expo = float(exposure_series(M.loc[[t]], reg, params.exposure_mode).iloc[0])
        rets = np.log(P["close"]).diff()
        w = size(chosen, t, rets, risk, expo)

        halted = get_state(con, "halted", False)
        no_new_risk = get_state(con, "no_new_risk_until", "") > now()
        if halted:
            w = w * 0
            event(con, "halted_skip", "circuit breaker active; manual reset required")
        for i, sym in enumerate(order):
            con.execute("INSERT INTO signals VALUES(?,?,?,?,?,?)",
                        (str(t.date()), sym, float(s[sym]), i + 1, float(w.get(sym, 0.0)), 1))

        syms = sorted(set(held) | set(w.index))
        q = quotes(syms)
        eq, cash, mv = equity_now(con, q)
        adv = P["quote_volume"].iloc[-30:].mean()
        vol = rets.iloc[-30:].std()
        atr = (P["high"] - P["low"]).iloc[-14:].mean()
        cur_w = (mv / eq).reindex(syms).fillna(0.0)
        tgt_w = w.reindex(syms).fillna(0.0)
        if no_new_risk:
            tgt_w = np.minimum(tgt_w, cur_w)
        # sells first to free cash
        for sym in sorted(syms, key=lambda x: tgt_w[x] - cur_w[x]):
            dw = tgt_w[sym] - cur_w[sym]
            if abs(dw) < risk.min_trade and not (tgt_w[sym] == 0 and cur_w[sym] > 0):
                continue
            px = q.loc[sym, "ask"] if dw > 0 else q.loc[sym, "bid"]
            qty = dw * eq / px
            if dw > 0:
                qty = min(qty, get_state(con, "cash") / (px * (1 + 0.003)))
                if qty * px < 10:
                    continue
            was_held = sym in positions(con).index
            fill = _fill(con, sym, qty, q, float(adv.get(sym, 1e6)), float(vol.get(sym, 0.05)), "rebalance")
            if dw > 0 and not was_held:
                a = float(atr.get(sym, fill * 0.05))
                con.execute("INSERT INTO positions VALUES(?,?,?,?,?,?)",
                            (sym, qty, fill, fill - risk.stop_atr * a, fill + risk.take_atr * a, now()))
        eq, cash, mv = equity_now(con)
        btc = quotes(["BTCUSDT"]).loc["BTCUSDT", "bid"]
        con.execute("INSERT OR REPLACE INTO equity VALUES(?,?,?,?,?,?,?)",
                    (now(), eq, cash, float(mv.sum() / eq) if eq else 0, btc, float(reg["p_stress"].iloc[-1]), "daily"))
        set_state(con, "day_start_equity", eq)
        set_state(con, "last_rebalance_date", str(t.date()))
        set_state(con, "peak", max(get_state(con, "peak", eq), eq))
        event(con, "rebalance", f"date={t.date()} chosen={chosen} expo={expo:.2f} equity={eq:.2f}")
        con.commit()


def hourly_check():
    """Stops / take-profits on the last closed hour, daily-loss limit, circuit breaker."""
    risk = SETTINGS.risk
    with closing(db()) as con:
        pos = positions(con)
        if not pos.empty:
            q = quotes(list(pos.index))
            for sym, p in pos.iterrows():
                k = requests.get(f"{API}/klines", params={"symbol": base_symbol(sym), "interval": "1h", "limit": 2}, timeout=20).json()
                o, h, l = float(k[0][1]), float(k[0][2]), float(k[0][3])
                reason = "stop" if l <= p["stop"] else ("take" if h >= p["take"] else None)
                if reason:
                    _fill(con, sym, -p["qty"], q, 1e7, 0.05, reason)
                    con.execute("DELETE FROM positions WHERE symbol=?", (sym,))
                    event(con, reason, f"{sym} low={l} high={h} stop={p['stop']:.6g} take={p['take']:.6g}")
        eq, cash, mv = equity_now(con)
        peak = max(get_state(con, "peak", eq), eq)
        set_state(con, "peak", peak)
        day0 = get_state(con, "day_start_equity", eq)
        if eq / day0 - 1 < -risk.daily_loss_limit and get_state(con, "no_new_risk_until", "") < now():
            until = (pd.Timestamp.now(tz="UTC").normalize() + pd.Timedelta(days=1, hours=1)).isoformat()
            set_state(con, "no_new_risk_until", until)
            event(con, "daily_loss_limit", f"day return {eq / day0 - 1:.2%}; no new risk until {until}")
        if eq / peak - 1 < -risk.breaker_dd and not get_state(con, "halted", False):
            pos = positions(con)
            if not pos.empty:
                q = quotes(list(pos.index))
                for sym, p in pos.iterrows():
                    _fill(con, sym, -p["qty"], q, 1e7, 0.05, "circuit_breaker")
                con.execute("DELETE FROM positions")
            set_state(con, "halted", True)
            event(con, "circuit_breaker", f"drawdown {eq / peak - 1:.2%} -> flat & halted")
            notify.send("Devre kesici")   # content-free push: no balances/positions
        btc = quotes(["BTCUSDT"]).loc["BTCUSDT", "bid"]
        eq, cash, mv = equity_now(con)
        con.execute("INSERT OR REPLACE INTO equity VALUES(?,?,?,?,?,?,?)",
                    (now(), eq, cash, float(mv.sum() / eq) if eq else 0, btc, None, "hourly"))
        con.commit()


def reset_breaker():
    with closing(db()) as con:
        eq = equity_now(con)[0]
        set_state(con, "halted", False)
        set_state(con, "peak", eq)
        event(con, "manual_reset", f"equity {eq:.2f}")
        con.commit()
