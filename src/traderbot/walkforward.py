"""Walk-forward training/prediction.

For each test quarter Q:
  train  = [data start, Q.start - purge)        (expanding, >= 2 years)
  inner  = last 6 months of train (for Optuna + early stopping), purged too
  test   = Q (3 months), predictions are strictly out-of-sample
Hyper-parameters are re-tuned every `retune_every` folds and reused in between.
The regime HMM is re-fitted on each fold's training window only.
"""
from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from . import models as mdl
from .regime import RegimeModel, regime_obs

log = logging.getLogger(__name__)


@dataclass
class WFConfig:
    first_test: str = "2021-01-01"
    end: str = "2025-10-01"          # exclusive; = HOLDOUT_START
    test_months: int = 3
    inner_months: int = 6
    horizon: int = 3
    retune_every: int = 4
    trials: dict = field(default_factory=lambda: {"lgbm": 40, "xgb": 30, "transformer": 12})
    models: tuple = ("lgbm", "xgb", "transformer")


def folds(cfg: WFConfig) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
    s = pd.Timestamp(cfg.first_test, tz="UTC")
    end = pd.Timestamp(cfg.end, tz="UTC")
    out = []
    while s < end:
        e = min(s + pd.DateOffset(months=cfg.test_months), end)
        out.append((s, e))
        s = e
    return out


def regime_for_fold(M: pd.DataFrame, train_end: pd.Timestamp, until: pd.Timestamp) -> pd.DataFrame:
    X = regime_obs(M)
    rm = RegimeModel().fit(X[X.index < train_end])
    return rm.filtered(X[X.index < until])


def run(d: dict, feats: list[str], cfg: WFConfig = WFConfig()) -> dict:
    df = d["df"]
    tgt_col = f"fwd_{cfg.horizon}_xs"
    raw_col = f"fwd_{cfg.horizon}"
    dates = df.index.get_level_values("date")
    purge = pd.Timedelta(days=cfg.horizon + 1)
    preds = {m: [] for m in cfg.models}
    regimes, fold_log = [], []
    params = {m: None for m in cfg.models}
    trial_count = {m: 0 for m in cfg.models}
    tensor = None
    if "transformer" in cfg.models:
        # normalisation stats from pre-test data only (no test-period distribution leak)
        pre = df[dates < pd.Timestamp(cfg.first_test, tz="UTC")][feats]
        tensor = mdl.SeqTensor(df, feats, pre.mean(), pre.std().replace(0, 1))

    for k, (ts, te) in enumerate(folds(cfg)):
        t0 = time.time()
        tr_mask = (dates < ts - purge) & df[tgt_col].notna().values
        inner_start = ts - purge - pd.DateOffset(months=cfg.inner_months)
        fit_mask = tr_mask & (dates < inner_start - purge)
        val_mask = tr_mask & (dates >= inner_start)
        te_mask = (dates >= ts) & (dates < te)
        tr, fit, val, te_df = df[tr_mask], df[fit_mask], df[val_mask], df[te_mask]
        y_fit = mdl.gauss_rank_target(fit[tgt_col])
        y_val = mdl.gauss_rank_target(val[tgt_col])
        y_tr = mdl.gauss_rank_target(tr[tgt_col])
        info = {"fold": k, "test_start": str(ts.date()), "test_end": str(te.date()),
                "n_train": len(tr), "n_test": len(te_df), "val_ic": {}}

        # regime model fitted on this fold's training window only
        reg = regime_for_fold(d["M"], ts - purge, te)
        regimes.append(reg[(reg.index >= ts) & (reg.index < te)])

        retune = k % cfg.retune_every == 0
        for name in cfg.models:
            if name in ("lgbm", "xgb"):
                cls = mdl.LGBModel if name == "lgbm" else mdl.XGBModel
                Xf, Xv = fit[feats], val[feats]

                def obj(p, cls=cls, Xf=Xf, Xv=Xv):
                    m = cls(p).fit(Xf, y_fit, Xv, y_val)
                    return mdl.mean_ic(pd.Series(m.predict(Xv), index=Xv.index), val[raw_col])

                if retune or params[name] is None:
                    bp, _, n = mdl.tune(cls, obj, cfg.trials[name], seed=k)
                    params[name] = bp
                    trial_count[name] += n
                m = cls(params[name]).fit(Xf, y_fit, Xv, y_val)
                ic = mdl.mean_ic(pd.Series(m.predict(Xv), index=Xv.index), val[raw_col])
                n_est = int((m.best_iter or 300) * 1.15)
                final = cls({**params[name], "n_estimators": n_est}).fit(tr[feats], y_tr)
                p = pd.Series(final.predict(te_df[feats]), index=te_df.index)
            else:
                def obj(p):
                    m = mdl.SeqModel({**p, "epochs": 15}).fit(tensor, fit.index, y_fit.values, val.index, val[raw_col])
                    return mdl.mean_ic(pd.Series(m.predict_idx(val.index), index=val.index), val[raw_col])

                if retune or params[name] is None:
                    bp, _, n = mdl.tune(mdl.SeqModel, obj, cfg.trials[name], seed=k)
                    params[name] = bp
                    trial_count[name] += n
                m = mdl.SeqModel(params[name]).fit(tensor, fit.index, y_fit.values, val.index, val[raw_col])
                ic = mdl.mean_ic(pd.Series(m.predict_idx(val.index), index=val.index), val[raw_col])
                p = pd.Series(m.predict_idx(te_df.index), index=te_df.index)
            info["val_ic"][name] = round(ic, 4)
            preds[name].append(p)
        info["sec"] = round(time.time() - t0, 1)
        info["params"] = {k2: v for k2, v in params.items()}
        fold_log.append(info)
        log.info("fold %d %s..%s val_ic=%s %.0fs", k, ts.date(), te.date(), info["val_ic"], info["sec"])

    out = {m: pd.concat(v).sort_index() for m, v in preds.items()}
    # ensemble: weights = each model's inner-validation IC in the *same* fold (no peeking at test)
    ens = []
    for k, (ts, te) in enumerate(folds(cfg)):
        w = fold_log[k]["val_ic"]
        part = {m: out[m][(out[m].index.get_level_values("date") >= ts) & (out[m].index.get_level_values("date") < te)] for m in out}
        ens.append(mdl.ensemble(part, w))
    out["ensemble"] = pd.concat(ens).sort_index()
    return {"preds": out, "regimes": pd.concat(regimes).sort_index(), "folds": fold_log,
            "trials": trial_count}
