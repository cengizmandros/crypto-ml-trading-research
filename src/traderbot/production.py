"""Production model: train on all available data, predict the latest day.

Weekly retrain writes a *candidate*; it is promoted only if its inner-validation
IC is not materially worse than the live model's (non-inferiority check).
"""
from __future__ import annotations

import json
import logging
import pickle
import shutil

import numpy as np
import pandas as pd
import torch

from . import dataset, models as mdl
from .config import MODELS, REPORTS
from .regime import RegimeModel, regime_obs

log = logging.getLogger(__name__)
PROD, CAND = MODELS / "prod", MODELS / "candidate"


def _selected() -> dict:
    return json.loads((REPORTS / "selected.json").read_text())


def needed_models(signal: str) -> list[str]:
    if signal == "ensemble":
        return ["lgbm", "xgb", "transformer"]
    if signal == "xs_momentum":
        return []
    return [signal]


def train(d: dict, out_dir=CAND, inner_months: int = 6) -> dict:
    sel = _selected()
    h = sel["horizon"]
    df = d["df"]
    feats = dataset.feature_columns(df)
    tgt, raw = f"fwd_{h}_xs", f"fwd_{h}"
    lab = df[df[tgt].notna()]
    dates = lab.index.get_level_values("date")
    last = dates.max()
    purge = pd.Timedelta(days=h + 1)
    vstart = last - pd.DateOffset(months=inner_months)
    fit, val = lab[dates < vstart - purge], lab[dates >= vstart]
    y_fit, y_val, y_all = (mdl.gauss_rank_target(x[tgt]) for x in (fit, val, lab))
    out_dir.mkdir(parents=True, exist_ok=True)
    meta = {"trained_until": str(last.date()), "signal": sel["signal"], "horizon": h, "val_ic": {}, "feats": feats}

    for name in needed_models(sel["signal"]):
        params = sel["model_params"][name]
        if name in ("lgbm", "xgb"):
            cls = mdl.LGBModel if name == "lgbm" else mdl.XGBModel
            m = cls(params).fit(fit[feats], y_fit, val[feats], y_val)
            meta["val_ic"][name] = mdl.mean_ic(pd.Series(m.predict(val[feats]), index=val.index), val[raw])
            final = cls({**params, "n_estimators": int((m.best_iter or 300) * 1.15)}).fit(lab[feats], y_all)
            pickle.dump(final, open(out_dir / f"{name}.pkl", "wb"))
        else:
            tensor = mdl.SeqTensor(df, feats)
            m = mdl.SeqModel(params).fit(tensor, fit.index, y_fit.values, val.index, val[raw])
            meta["val_ic"][name] = mdl.mean_ic(pd.Series(m.predict_idx(val.index), index=val.index), val[raw])
            torch.save({"state": m.net.state_dict(), "params": m.params, "L": m.L,
                        "mu": tensor.mu, "sd": tensor.sd}, out_dir / "transformer.pt")
    rm = RegimeModel().fit(regime_obs(d["M"]))
    pickle.dump(rm, open(out_dir / "regime.pkl", "wb"))
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=1, default=str))
    return meta


def promote(tolerance: float = 0.01) -> tuple[bool, str]:
    cand = json.loads((CAND / "meta.json").read_text())
    if not (PROD / "meta.json").exists():
        ok, why = True, "no production model yet"
    else:
        prod = json.loads((PROD / "meta.json").read_text())
        c = np.mean(list(cand["val_ic"].values())) if cand["val_ic"] else 0.0
        p = np.mean(list(prod["val_ic"].values())) if prod["val_ic"] else 0.0
        ok = c >= p - tolerance
        why = f"candidate IC {c:.4f} vs prod {p:.4f}"
    if ok:
        if PROD.exists():
            shutil.rmtree(PROD)
        shutil.copytree(CAND, PROD)
    return ok, why


def predict_latest(d: dict) -> tuple[pd.Series, pd.Timestamp, pd.Series]:
    """Scores for the most recent complete day + that day's regime probabilities."""
    meta = json.loads((PROD / "meta.json").read_text())
    df = d["df"]
    t = df.index.get_level_values("date").max()
    today = df.xs(t, level="date", drop_level=False)
    feats = meta["feats"]
    preds = {}
    for name in needed_models(meta["signal"]):
        if name in ("lgbm", "xgb"):
            m = pickle.load(open(PROD / f"{name}.pkl", "rb"))
            preds[name] = pd.Series(m.predict(today[feats]), index=today.index)
        else:
            ck = torch.load(PROD / "transformer.pt", weights_only=False)
            recent = df[df.index.get_level_values("date") > t - pd.Timedelta(days=ck["L"] + 5)]
            tensor = mdl.SeqTensor(recent, feats, ck["mu"], ck["sd"])
            sm = mdl.SeqModel(ck["params"])
            sm.L = ck["L"]
            sm.net = mdl.TinyTransformer(len(feats), int(ck["params"]["d"]), int(ck["params"]["layers"]), 4,
                                         float(ck["params"]["drop"]), sm.L).to(mdl.DEVICE)
            sm.net.load_state_dict(ck["state"])
            preds[name] = pd.Series(sm.predict_idx(today.index, tensor), index=today.index)
    if meta["signal"] == "xs_momentum":
        score = np.log(d["P"]["close"]).diff(30).loc[t]
    elif meta["signal"] == "ensemble":
        score = mdl.ensemble(preds, meta["val_ic"]).droplevel("date")
    else:
        score = preds[meta["signal"]].droplevel("date")
    rm = pickle.load(open(PROD / "regime.pkl", "rb"))
    reg = rm.filtered(regime_obs(d["M"]))
    return score, t, reg
