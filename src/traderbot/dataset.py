"""Build and cache the full research dataset (features + labels, long format)."""
from __future__ import annotations

import pickle

import numpy as np
import pandas as pd

from . import features as fe
from . import universe
from .config import DATA, HOLDOUT_START

CACHE = DATA / "dataset.pkl"

XS_RANK = ["ret_1", "ret_3", "ret_7", "ret_14", "ret_30", "ret_60", "ret_90", "mom_va_7", "mom_va_14",
           "mom_va_30", "mom_va_60", "mom_va_90", "mom_12_1", "dist_ma_20", "dist_ma_50", "dist_ma_200",
           "vol_7", "vol_30", "vol_ratio", "parkinson_14", "atr_14", "hrv_7", "hrv_30", "hskew_14",
           "skew_60", "max_ret_30", "dd_30", "qv_z_30", "qv_trend", "taker_buy_7", "amihud_30",
           "log_adv_30", "rsi_14", "rsi_2", "macd_hist", "bb_pctb", "bb_width", "beta_mkt_60",
           "corr_btc_60", "idio_vol_30", "rs_btc_7", "rs_btc_30", "rs_mkt_7", "rs_mkt_30"]

MARKET_COLS = ["mkt_ret_1", "mkt_ret_7", "mkt_ret_30", "btc_ret_1", "btc_ret_7", "btc_ret_30",
               "mkt_vol_30", "btc_vol_30", "btc_dist_ma_200", "btc_vol_share", "btc_vol_share_chg_30",
               "breadth_50", "dispersion_7", "avg_corr_60"]


def build(save: bool = True) -> dict:
    univ = pd.read_parquet(DATA / "universe.parquet")
    syms = sorted(univ["symbol"].unique())
    if "BTCUSDT" not in syms:
        syms.append("BTCUSDT")
    P = fe.build_panels(syms)
    member = universe.membership_daily(univ, P["close"].index).reindex(columns=P["close"].columns).fillna(False)
    F = fe.asset_features(P)
    M, X = fe.market_features(P, member)
    F.update(X)
    Y = fe.labels(P)
    df = fe.to_long(F, member, M, Y, rank_cols=XS_RANK)
    out = {"P": P, "member": member, "M": M, "df": df, "univ": univ}
    if save:
        with open(CACHE, "wb") as f:
            pickle.dump(out, f, protocol=5)
    return out


def load(research: bool = True) -> dict:
    """research=True hides the holdout: rows on/after HOLDOUT_START are dropped and any
    label whose horizon reaches into the holdout is set to NaN."""
    with open(CACHE, "rb") as f:
        d = pickle.load(f)
    if research:
        hs = pd.Timestamp(HOLDOUT_START, tz="UTC")
        df = d["df"]
        dates = df.index.get_level_values("date")
        df = df[dates < hs].copy()
        dates = df.index.get_level_values("date")
        for c in [c for c in df.columns if c.startswith("fwd_")]:
            h = int(c.split("_")[1])
            df.loc[dates + pd.Timedelta(days=h + 1) >= hs, c] = np.nan
        d["df"] = df
        d["P"] = {k: v[v.index < hs] for k, v in d["P"].items()}
        d["M"] = d["M"][d["M"].index < hs]
        d["member"] = d["member"][d["member"].index < hs]
    return d


def feature_columns(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c.startswith("xr_")] + [c for c in MARKET_COLS if c in df.columns]
