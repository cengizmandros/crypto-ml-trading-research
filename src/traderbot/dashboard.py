"""Local dashboard: uvicorn traderbot.dashboard:app --host 127.0.0.1 --port 8050"""
from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd
import plotly
from fastapi import FastAPI
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse

from . import metrics as mt
from .config import REPORTS
from .paper import DB, SCHEMA

app = FastAPI(title="trader-bot")
PLOTLY_JS = Path(plotly.__file__).parent / "package_data" / "plotly.min.js"


def q(sql: str) -> pd.DataFrame:
    con = sqlite3.connect(DB)
    con.executescript(SCHEMA)
    try:
        return pd.read_sql(sql, con)
    finally:
        con.close()


def clean(o):
    return json.loads(json.dumps(o, default=str).replace("NaN", "null").replace("Infinity", "null"))


@app.get("/plotly.min.js")
def plotly_js():
    return FileResponse(PLOTLY_JS, media_type="application/javascript")


@app.get("/api/paper")
def api_paper():
    eq = q("SELECT * FROM equity ORDER BY ts")
    pos = q("SELECT * FROM positions")
    trades = q("SELECT * FROM trades ORDER BY ts DESC LIMIT 100")
    events = q("SELECT * FROM events ORDER BY ts DESC LIMIT 50")
    last_date = q("SELECT MAX(date) d FROM signals")["d"].iloc[0]
    sig = q(f"SELECT * FROM signals WHERE date='{last_date}' ORDER BY rank LIMIT 20") if last_date else pd.DataFrame()
    state = dict(q("SELECT key, value FROM state").values)
    stats = {}
    if len(eq) > 1:
        e = eq.assign(ts=pd.to_datetime(eq["ts"])).set_index("ts")
        daily = e["equity"].resample("D").last().dropna()
        r = daily.pct_change().dropna()
        btc = e["btc_px"].resample("D").last().dropna()
        stats = {"equity": float(e["equity"].iloc[-1]), "return": float(e["equity"].iloc[-1] / e["equity"].iloc[0] - 1),
                 "btc_return": float(btc.iloc[-1] / btc.iloc[0] - 1) if len(btc) else None,
                 "max_dd": mt.max_drawdown(r) if len(r) else 0.0,
                 "sharpe": mt.sharpe(r) if len(r) > 5 else None, "days": len(daily),
                 "n_trades": int(q("SELECT COUNT(*) n FROM trades")["n"].iloc[0]),
                 "costs": float(q("SELECT COALESCE(SUM(cost),0) c FROM trades")["c"].iloc[0])}
    if len(pos):
        from .paper import quotes
        try:
            qt = quotes(list(pos["symbol"]))
            pos["bid"] = pos["symbol"].map(qt["bid"])
            pos["value"] = pos["qty"] * pos["bid"]
            pos["pnl_pct"] = pos["bid"] / pos["entry_px"] - 1
        except Exception:
            pass
    return JSONResponse(clean({"equity": eq.to_dict("list"), "positions": pos.to_dict("records"),
                               "trades": trades.to_dict("records"), "events": events.to_dict("records"),
                               "signals": sig.to_dict("records"), "state": state, "stats": stats}))


@app.get("/api/research")
def api_research():
    out = {}
    f = REPORTS / "wf_results.json"
    if f.exists():
        w = json.loads(f.read_text())
        keep = ["cagr", "sharpe", "sortino", "max_dd", "total_return", "n_trades", "win_rate", "exposure",
                "costs_paid_pct_of_start", "turnover_per_year"]
        out["wf"] = {k: {m: v.get(m) for m in keep} for k, v in w["results"].items()}
        out["selected"] = w["selected"]
        out["period"] = w["period"]
    e = REPORTS / "wf_equity.parquet"
    if e.exists():
        r = pd.read_parquet(e)
        cols = [c for c in ["btc_buy_hold", "ew_universe_monthly", out.get("selected"), "xs_momentum|" + out.get("selected", "|").split("|")[1]] if c in r.columns]
        eq = (1 + r[cols].fillna(0)).cumprod()
        eq = eq.iloc[::3]
        out["curves"] = {"x": [str(i.date()) for i in eq.index], **{c: eq[c].round(4).tolist() for c in cols}}
    h = REPORTS / "holdout_results.json"
    if h.exists():
        hr = json.loads(h.read_text())
        out["holdout"] = {k: {m: v.get(m) for m in ["cagr", "sharpe", "max_dd", "total_return", "n_trades"]}
                          for k, v in hr["results"].items()}
    return JSONResponse(clean(out))


FA_HOME = Path.home() / "funding-arb"   # separate project; read-only here


@app.get("/api/funding_monitor")
def api_funding_monitor():
    """Daily funding-regime monitor written by ~/funding-arb/scripts/monitor.py."""
    out = {"available": False}
    log_csv = FA_HOME / "logs" / "funding_monitor.csv"
    if log_csv.exists():
        df = pd.read_csv(log_csv).drop_duplicates("date", keep="last").sort_values("date").tail(90)
        out.update(available=True, rows=df.to_dict("list"))
    st = FA_HOME / "logs" / "funding_monitor_state.json"
    if st.exists():
        out["state"] = json.loads(st.read_text())
    alarm = FA_HOME / "ALARM.txt"
    out["alarm"] = alarm.read_text() if alarm.exists() else None
    return JSONResponse(clean(out))


@app.get("/", response_class=HTMLResponse)
def index():
    return (Path(__file__).parent / "dashboard.html").read_text()
