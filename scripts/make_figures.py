"""Small PNG figures for the README (docs/img/). Reads only files committed under reports/."""
from __future__ import annotations

import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

from traderbot.config import REPORTS, ROOT  # noqa: E402

OUT = ROOT / "docs" / "img"
OUT.mkdir(parents=True, exist_ok=True)
plt.rcParams.update({"figure.dpi": 110, "font.size": 9, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.alpha": 0.3})
NAMES = {"ensemble|regime": "ML ensemble + regime (selected)", "btc_buy_hold": "BTC buy & hold",
         "btc_vol_target": "BTC, same 35% vol target", "xs_momentum|trend": "Cross-sectional momentum (best)",
         "ew_universe_monthly": "Equal-weight top-20 universe"}
COLORS = {"ensemble|regime": "#2f5fd0", "btc_buy_hold": "#888888", "btc_vol_target": "#b08800",
          "xs_momentum|trend": "#8e44ad", "ew_universe_monthly": "#c0392b"}


def curves(path, title, fname, log=True):
    r = pd.read_parquet(path)
    fig, ax = plt.subplots(figsize=(8, 4))
    for k in NAMES:
        if k in r.columns:
            eq = (1 + r[k].fillna(0)).cumprod()
            ax.plot(eq.index, eq.values, label=NAMES[k], color=COLORS[k], lw=2.2 if k == "ensemble|regime" else 1.2)
    if log:
        ax.set_yscale("log")
    ax.set_ylabel("Growth of 1 (after costs)")
    ax.set_title(title)
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT / fname)
    plt.close(fig)


def monte_carlo():
    w = json.loads((REPORTS / "wf_results.json").read_text())
    rs = w["robustness"]["random_selection"]
    actual = w["results"][w["selected"]]["sharpe"]
    fig, ax = plt.subplots(figsize=(8, 3.4))
    ax.hist(rs["dist"], bins=30, color="#bbbbbb", edgecolor="white", label=f"{rs['n']} random-selection portfolios")
    ax.axvline(actual, color="#2f5fd0", lw=2, label=f"Model portfolio, Sharpe {actual:.2f} (p = {rs['p_value']:.3f})")
    ax.axvline(w["results"]["btc_buy_hold"]["sharpe"], color="#888888", ls="--", lw=1.2,
               label=f"BTC buy & hold, Sharpe {w['results']['btc_buy_hold']['sharpe']:.2f}")
    ax.set_xlabel("Annualised Sharpe ratio, walk-forward 2021-01 .. 2025-09 (same risk overlay, same costs)")
    ax.set_title("Monte Carlo: is coin selection better than chance?")
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT / "monte_carlo.png")
    plt.close(fig)


if __name__ == "__main__":
    curves(REPORTS / "wf_equity.parquet", "Walk-forward out-of-sample, 2021-01 .. 2025-09", "walk_forward.png")
    curves(REPORTS / "holdout_equity.parquet", "Untouched holdout, 2025-10 .. 2026-10 (run once)", "holdout.png", log=False)
    monte_carlo()
    for p in sorted(OUT.glob("*.png")):
        print(p.name, p.stat().st_size // 1024, "KB")
