"""Point-in-time universe: at each month start pick the top-N USDT pairs by
trailing quote volume, using only data available before that date.
Delisted coins are included as long as they were trading at selection time.
"""
from __future__ import annotations

import pandas as pd

from .config import SETTINGS
from . import data


def daily_quote_volume(symbols: list[str]) -> pd.DataFrame:
    return data.panel(symbols, "1d", "quote_volume")


def monthly_universe(qv: pd.DataFrame, top_n: int | None = None) -> pd.DataFrame:
    """Return long frame [month_start, symbol, rank] of eligible members."""
    u = SETTINGS.universe
    top_n = top_n or u.top_n
    qv = qv.sort_index()
    first_seen = qv.notna().idxmax()
    rows = []
    months = pd.date_range(qv.index.min().normalize() + pd.offsets.MonthBegin(1),
                           qv.index.max() + pd.offsets.MonthBegin(1), freq="MS", tz="UTC")
    for m in months:
        hist = qv.loc[: m - pd.Timedelta(seconds=1)]  # strictly before the month
        if len(hist) < u.lookback_days:
            continue
        win = hist.iloc[-u.lookback_days:]
        alive = hist.iloc[-1].notna()  # traded on the last day before selection
        aged = (m - first_seen) >= pd.Timedelta(days=u.min_history_days)
        score = win.sum(min_count=u.lookback_days // 2)[alive & aged].dropna()
        for rank, (sym, _) in enumerate(score.sort_values(ascending=False).head(top_n).items(), 1):
            rows.append((m, sym, rank))
    return pd.DataFrame(rows, columns=["month", "symbol", "rank"])


def membership_daily(univ: pd.DataFrame, index: pd.DatetimeIndex) -> pd.DataFrame:
    """Boolean frame day x symbol: True if symbol is in the universe that day."""
    syms = sorted(univ["symbol"].unique())
    out = pd.DataFrame(False, index=index, columns=syms)
    for m, g in univ.groupby("month"):
        end = m + pd.offsets.MonthBegin(1)
        sel = (index >= m) & (index < end)
        out.loc[sel, list(g["symbol"])] = True
    return out
