"""Look-ahead tests: perturbing the future must not change any past feature."""
import numpy as np
import pandas as pd

from traderbot import features as fe
from traderbot.regime import RegimeModel


def synth_panels(n_days=400, syms=("BTCUSDT", "AAAUSDT", "BBBUSDT", "CCCUSDT"), seed=0):
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2021-01-01", periods=n_days, freq="D", tz="UTC")
    lr = pd.DataFrame(rng.normal(0, 0.03, (n_days, len(syms))), index=idx, columns=list(syms))
    c = 100 * np.exp(lr.cumsum())
    o = c.shift().fillna(100)
    h = np.maximum(o, c) * (1 + rng.uniform(0, 0.02, c.shape))
    l = np.minimum(o, c) * (1 - rng.uniform(0, 0.02, c.shape))
    qv = pd.DataFrame(rng.lognormal(16, 0.5, c.shape), index=idx, columns=c.columns)
    P = {"close": c, "open": o, "high": h, "low": l, "quote_volume": qv, "volume": qv / c,
         "taker_buy_quote": qv * 0.5, "trades": qv / 100}
    P["exec_px"] = o.shift(-1)
    return P


def perturb_after(P, t):
    Q = {k: v.copy() for k, v in P.items()}
    for k in ("close", "open", "high", "low", "quote_volume", "volume", "taker_buy_quote", "trades"):
        Q[k].loc[Q[k].index > t] *= 3.7
    return Q


def test_asset_and_market_features_have_no_lookahead():
    P = synth_panels()
    t = P["close"].index[250]
    Q = perturb_after(P, t)
    member = pd.DataFrame(True, index=P["close"].index, columns=P["close"].columns)
    for a, b in [(fe.asset_features(P), fe.asset_features(Q)),
                 (fe.market_features(P, member)[1], fe.market_features(Q, member)[1])]:
        for k in a:
            x, y = a[k].loc[:t], b[k].loc[:t]
            assert np.allclose(x.values, y.values, equal_nan=True), f"look-ahead in {k}"
    m1, m2 = fe.market_features(P, member)[0], fe.market_features(Q, member)[0]
    assert np.allclose(m1.loc[:t].values, m2.loc[:t].values, equal_nan=True)


def test_labels_start_after_decision():
    P = synth_panels()
    Y = fe.labels(P, horizons=(1,))
    t = P["close"].index[100]
    # label at t must equal log(open[t+2]/open[t+1]) with exec_px = next open
    o = P["open"]
    exp = np.log(o.shift(-2) / o.shift(-1)).loc[t]
    assert np.allclose(Y["fwd_1"].loc[t].values, exp.values)


def test_regime_filter_is_causal():
    rng = np.random.default_rng(1)
    idx = pd.date_range("2020-01-01", periods=600, freq="D", tz="UTC")
    X = pd.DataFrame(rng.normal(size=(600, 3)), index=idx, columns=["r7", "lv", "br"])
    rm = RegimeModel().fit(X.iloc[:400])
    full = rm.filtered(X)
    X2 = X.copy()
    X2.iloc[500:] += 10
    part = rm.filtered(X2)
    assert np.allclose(full.iloc[:500].values, part.iloc[:500].values)
