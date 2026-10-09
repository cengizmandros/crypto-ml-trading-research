"""Market regime detection with a Gaussian HMM.

Fit only on a training window; produce *filtered* state probabilities
P(state_t | obs_1..t) with a forward pass, so no future information leaks
(hmmlearn's predict_proba is the smoothed posterior and would leak).
States are relabelled by volatility/return so labels are stable across refits:
  0 = calm / trend-up, 1 = neutral / range, 2 = high-vol / stress.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from hmmlearn.hmm import GaussianHMM
from scipy.special import logsumexp
from scipy.stats import multivariate_normal

REGIME_NAMES = ["calm_up", "range", "stress"]


def regime_obs(M: pd.DataFrame) -> pd.DataFrame:
    """Observation vector per day: 7d market return, log 30d vol, breadth."""
    X = pd.DataFrame({
        "r7": M["mkt_ret_7"],
        "lv": np.log(M["mkt_vol_30"]),
        "br": M["breadth_50"],
    })
    return X.astype(float).replace([np.inf, -np.inf], np.nan)


class RegimeModel:
    def __init__(self, n_states: int = 3, seed: int = 0):
        self.n = n_states
        self.seed = seed

    def fit(self, X: pd.DataFrame) -> "RegimeModel":
        X = X.dropna()
        self.mu_, self.sd_ = X.mean(), X.std()
        Z = ((X - self.mu_) / self.sd_).values
        best = None
        for s in range(5):  # several restarts, keep best likelihood
            m = GaussianHMM(self.n, covariance_type="full", n_iter=300, random_state=self.seed + s, min_covar=1e-3)
            m.fit(Z)
            ll = m.score(Z)
            if best is None or ll > best[0]:
                best = (ll, m)
        self.hmm_ = best[1]
        # order states: by vol (stress last), then by return
        means = self.hmm_.means_
        vol_rank = np.argsort(means[:, 1])
        hi = vol_rank[-1]
        rest = [k for k in range(self.n) if k != hi]
        rest.sort(key=lambda k: -means[k, 0])   # higher return first
        self.order_ = rest + [hi]
        return self

    def filtered(self, X: pd.DataFrame) -> pd.DataFrame:
        """Forward-filtered probabilities for each row of X (NaN rows carry forward)."""
        m = self.hmm_
        Z = ((X - self.mu_) / self.sd_)
        logA = np.log(m.transmat_ + 1e-300)
        dists = [multivariate_normal(m.means_[k], m.covars_[k], allow_singular=True) for k in range(self.n)]
        la = np.log(m.startprob_ + 1e-300)
        out = np.full((len(Z), self.n), np.nan)
        started = False
        for i, row in enumerate(Z.values):
            if np.isnan(row).any():
                if started:
                    la = logsumexp(la[:, None] + logA, axis=0)  # predict only
                    out[i] = np.exp(la - logsumexp(la))
                continue
            ll = np.array([d.logpdf(row) for d in dists])
            if started:
                la = logsumexp(la[:, None] + logA, axis=0) + ll
            else:
                la = la + ll
                started = True
            la -= logsumexp(la)
            out[i] = np.exp(la)
        df = pd.DataFrame(out[:, self.order_], index=X.index, columns=[f"p_{n}" for n in REGIME_NAMES])
        return df
