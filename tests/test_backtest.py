import numpy as np
import pandas as pd

from traderbot import backtest as bt
from traderbot.config import Costs

from test_lookahead import synth_panels


def test_buy_and_hold_matches_price_path_minus_entry_cost():
    P = synth_panels(n_days=200)
    P["exec_px"] = P["open"].shift(-1)
    # start after 30 days so the cost model has liquidity history (with none it
    # deliberately assumes the worst-case 1% spread)
    tgt = pd.DataFrame(np.nan, index=P["close"].index[30:150], columns=P["close"].columns)
    tgt.iloc[0] = 0.0
    tgt.iloc[0, 0] = 1.0
    r = bt.run(tgt, P, use_stops=False, use_breaker=False, capital=10_000)
    px = P["exec_px"].iloc[30:150, 0]
    expected = px.iloc[-1] / px.iloc[0]
    got = r.equity.iloc[-1] / 10_000
    assert got < expected  # paid entry costs
    assert abs(got / expected - 1) < 0.005


def test_costs_reduce_returns_monotonically():
    P = synth_panels(n_days=300, seed=3)
    rng = np.random.default_rng(0)
    idx = P["close"].index[:250]
    tgt = pd.DataFrame(rng.dirichlet(np.ones(4), len(idx)) * 0.9, index=idx, columns=P["close"].columns)
    finals = [bt.run(tgt, P, costs=Costs().scaled(m), use_stops=False, use_breaker=False).equity.iloc[-1]
              for m in (0.0, 1.0, 3.0)]
    assert finals[0] > finals[1] > finals[2]


def test_stop_exits_at_stop_price_not_close():
    P = synth_panels(n_days=60, syms=("BTCUSDT", "AAAUSDT"), seed=5)
    idx = P["close"].index
    # crash AAA intraday on day 31 (the day after the fill on day 30)
    P["low"].loc[idx[31], "AAAUSDT"] = P["low"].loc[idx[31], "AAAUSDT"] * 0.3
    tgt = pd.DataFrame(np.nan, index=idx[:40], columns=P["close"].columns)
    tgt.iloc[29] = 0.0
    tgt.loc[idx[29], "AAAUSDT"] = 0.5
    r = bt.run(tgt, P, use_breaker=False)
    sells = r.trades[(r.trades.symbol == "AAAUSDT") & (r.trades.dw < 0)]
    assert len(sells) >= 1
    assert sells.iloc[0]["price"] > P["low"].loc[idx[31], "AAAUSDT"]
