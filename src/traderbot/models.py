"""Predictive models. All predict a cross-sectional score for the next h days.

Target: per-date gaussianised rank of the forward excess return (robust to the
fat tails of crypto returns). Validation metric: mean daily Spearman rank IC.
"""
from __future__ import annotations

import logging
import math

import lightgbm as lgb
import numpy as np
import optuna
import pandas as pd
import torch
import torch.nn as nn
import xgboost as xgb
from scipy.stats import norm

log = logging.getLogger(__name__)
optuna.logging.set_verbosity(optuna.logging.WARNING)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def gauss_rank_target(y: pd.Series) -> pd.Series:
    r = y.groupby(level="date").rank(pct=True)
    n = y.groupby(level="date").transform("count")
    u = (r * n - 0.5) / n
    return pd.Series(norm.ppf(u.clip(1e-4, 1 - 1e-4)), index=y.index).where(y.notna())


def daily_ic(pred: pd.Series, y: pd.Series) -> pd.Series:
    df = pd.DataFrame({"p": pred, "y": y}).dropna()
    return df.groupby(level="date").apply(
        lambda g: g["p"].rank().corr(g["y"].rank()) if len(g) >= 5 else np.nan).dropna()


def mean_ic(pred, y) -> float:
    ic = daily_ic(pred, y)
    return float(ic.mean()) if len(ic) else -1.0


# ---------------------------------------------------------------- GBMs ----

class LGBModel:
    name = "lgbm"

    def __init__(self, params: dict | None = None):
        self.params = params or {}

    @staticmethod
    def space(trial: optuna.Trial) -> dict:
        return {
            "num_leaves": trial.suggest_int("num_leaves", 7, 63, log=True),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.1, log=True),
            "min_child_samples": trial.suggest_int("min_child_samples", 50, 1500, log=True),
            "feature_fraction": trial.suggest_float("feature_fraction", 0.3, 0.9),
            "bagging_fraction": trial.suggest_float("bagging_fraction", 0.5, 1.0),
            "lambda_l2": trial.suggest_float("lambda_l2", 1e-3, 30, log=True),
        }

    def fit(self, X, y, Xv=None, yv=None):
        p = {"objective": "regression", "verbosity": -1, "bagging_freq": 1, "seed": 7,
             "num_threads": 10, **{k: v for k, v in self.params.items() if k != "n_estimators"}}
        dtr = lgb.Dataset(X, y)
        if Xv is not None:
            dv = lgb.Dataset(Xv, yv)
            self.m = lgb.train(p, dtr, 2000, valid_sets=[dv],
                               callbacks=[lgb.early_stopping(100, verbose=False)])
            self.best_iter = self.m.best_iteration or 2000
        else:
            self.m = lgb.train(p, dtr, int(self.params.get("n_estimators", 300)))
            self.best_iter = None
        return self

    def predict(self, X):
        return self.m.predict(X, num_iteration=self.best_iter)


class XGBModel:
    name = "xgb"

    def __init__(self, params: dict | None = None):
        self.params = params or {}

    @staticmethod
    def space(trial):
        return {
            "max_depth": trial.suggest_int("max_depth", 2, 6),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.1, log=True),
            "min_child_weight": trial.suggest_float("min_child_weight", 5, 500, log=True),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.3, 0.9),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-2, 50, log=True),
        }

    def fit(self, X, y, Xv=None, yv=None):
        p = {"objective": "reg:squarederror", "tree_method": "hist", "device": "cuda",
             "seed": 7, **{k: v for k, v in self.params.items() if k != "n_estimators"}}
        dtr = xgb.DMatrix(X, y)
        if Xv is not None:
            dv = xgb.DMatrix(Xv, yv)
            self.m = xgb.train(p, dtr, 2000, evals=[(dv, "v")], early_stopping_rounds=100, verbose_eval=False)
            self.best_iter = self.m.best_iteration + 1
        else:
            self.m = xgb.train(p, dtr, int(self.params.get("n_estimators", 300)))
            self.best_iter = None
        return self

    def predict(self, X):
        d = xgb.DMatrix(X)
        if self.best_iter:
            return self.m.predict(d, iteration_range=(0, self.best_iter))
        return self.m.predict(d)


# ------------------------------------------------------- sequence model ----

class SeqTensor:
    """Dense (date, symbol, feature) tensor so each sample can look back L days."""

    def __init__(self, df: pd.DataFrame, feats: list[str], mu=None, sd=None):
        X = df[feats]
        self.mu = X.mean() if mu is None else mu
        self.sd = X.std().replace(0, 1) if sd is None else sd
        Z = ((X - self.mu) / self.sd).clip(-5, 5).fillna(0.0).astype("float32")
        self.dates = df.index.get_level_values("date").unique().sort_values()
        self.syms = df.index.get_level_values("symbol").unique().sort_values()
        di = self.dates.get_indexer(Z.index.get_level_values("date"))
        si = self.syms.get_indexer(Z.index.get_level_values("symbol"))
        self.arr = np.zeros((len(self.dates), len(self.syms), len(feats)), dtype="float32")
        self.mask = np.zeros((len(self.dates), len(self.syms)), dtype=bool)
        self.arr[di, si] = Z.values
        self.mask[di, si] = True
        self.pos = pd.Series(list(zip(di, si)), index=Z.index)

    def batch(self, idx: pd.Index, L: int):
        pos = self.pos.loc[idx].values
        out = np.zeros((len(pos), L, self.arr.shape[2]), dtype="float32")
        m = np.zeros((len(pos), L), dtype=bool)
        for j, (d, s) in enumerate(pos):
            lo = max(0, d - L + 1)
            seq = self.arr[lo:d + 1, s]
            out[j, L - len(seq):] = seq
            m[j, L - len(seq):] = self.mask[lo:d + 1, s]
        return out, m


class TinyTransformer(nn.Module):
    def __init__(self, n_feat: int, d: int = 64, layers: int = 2, heads: int = 4, drop: float = 0.1, L: int = 32):
        super().__init__()
        self.inp = nn.Linear(n_feat, d)
        self.pos = nn.Parameter(torch.zeros(1, L, d))
        enc = nn.TransformerEncoderLayer(d, heads, 2 * d, drop, batch_first=True, norm_first=True)
        self.enc = nn.TransformerEncoder(enc, layers)
        self.head = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, d), nn.GELU(), nn.Dropout(drop), nn.Linear(d, 1))

    def forward(self, x, mask):
        h = self.inp(x) + self.pos[:, -x.shape[1]:]
        pad = ~mask
        pad[:, -1] = False  # last step (today) is always valid
        h = self.enc(h, src_key_padding_mask=pad)
        return self.head(h[:, -1]).squeeze(-1)


class SeqModel:
    name = "transformer"

    def __init__(self, params: dict | None = None):
        self.params = {"d": 64, "layers": 2, "drop": 0.15, "lr": 5e-4, "L": 32, "wd": 1e-3,
                       "epochs": 25, **(params or {})}

    @staticmethod
    def space(trial):
        return {
            "d": trial.suggest_categorical("d", [32, 64, 96]),
            "layers": trial.suggest_int("layers", 1, 3),
            "drop": trial.suggest_float("drop", 0.05, 0.4),
            "lr": trial.suggest_float("lr", 1e-4, 2e-3, log=True),
            "L": trial.suggest_categorical("L", [16, 32, 64]),
            "wd": trial.suggest_float("wd", 1e-5, 1e-1, log=True),
        }

    def fit(self, tensor: SeqTensor, idx_tr, y_tr, idx_v=None, y_v=None):
        p = self.params
        torch.manual_seed(7)
        np.random.seed(7)
        self.tensor, self.L = tensor, int(p["L"])
        self.net = TinyTransformer(tensor.arr.shape[2], int(p["d"]), int(p["layers"]), 4, float(p["drop"]), self.L).to(DEVICE)
        opt = torch.optim.AdamW(self.net.parameters(), lr=float(p["lr"]), weight_decay=float(p["wd"]))
        Xtr, Mtr = tensor.batch(idx_tr, self.L)
        Xtr, Mtr = torch.tensor(Xtr, device=DEVICE), torch.tensor(Mtr, device=DEVICE)
        ytr = torch.tensor(np.asarray(y_tr, dtype="float32"), device=DEVICE)
        best, best_state, bad = -1e9, None, 0
        epochs = int(p["epochs"]) if idx_v is not None else int(p.get("best_epochs", p["epochs"]))
        bs = 512
        for ep in range(epochs):
            self.net.train()
            perm = torch.randperm(len(ytr), device=DEVICE)
            for k in range(0, len(perm), bs):
                b = perm[k:k + bs]
                loss = nn.functional.mse_loss(self.net(Xtr[b], Mtr[b]), ytr[b])
                opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.net.parameters(), 1.0)
                opt.step()
            if idx_v is not None:
                ic = mean_ic(pd.Series(self.predict_idx(idx_v), index=idx_v), y_v)
                if ic > best:
                    best, bad, self.best_epochs = ic, 0, ep + 1
                    best_state = {k: v.detach().clone() for k, v in self.net.state_dict().items()}
                else:
                    bad += 1
                    if bad >= 5:
                        break
        if best_state is not None:
            self.net.load_state_dict(best_state)
        return self

    @torch.no_grad()
    def predict_idx(self, idx, tensor: SeqTensor | None = None):
        tensor = tensor or self.tensor
        self.net.eval()
        X, M = tensor.batch(idx, self.L)
        out = []
        for k in range(0, len(X), 4096):
            out.append(self.net(torch.tensor(X[k:k + 4096], device=DEVICE),
                                torch.tensor(M[k:k + 4096], device=DEVICE)).cpu().numpy())
        return np.concatenate(out) if out else np.array([])


def tune(model_cls, objective_fn, n_trials: int, seed: int = 0) -> tuple[dict, float, int]:
    """Run Optuna; returns (best params, best IC, trials run)."""
    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=seed))
    study.optimize(lambda tr: objective_fn(model_cls.space(tr)), n_trials=n_trials, show_progress_bar=False)
    return study.best_params, study.best_value, len(study.trials)


def ensemble(preds: dict[str, pd.Series], weights: dict[str, float]) -> pd.Series:
    """Weighted average of per-date ranks (scale-free)."""
    tot, acc = 0.0, None
    for k, p in preds.items():
        w = max(weights.get(k, 0.0), 0.0)
        if w <= 0:
            continue
        r = p.groupby(level="date").rank(pct=True) - 0.5
        acc = r * w if acc is None else acc.add(r * w, fill_value=0)
        tot += w
    if acc is None:  # no model had positive validation IC -> equal weights
        return ensemble(preds, {k: 1.0 for k in preds})
    return acc / tot
