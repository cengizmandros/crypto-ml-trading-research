"""ONE-SHOT final test on the untouched holdout (HOLDOUT_START .. latest data).

Uses exactly the configuration pre-committed in reports/selected.json and the same
training procedure as the walk-forward (tune on the last 6 months before the
holdout, train on everything before it). Refuses to run twice.
"""
from __future__ import annotations

import json
import logging
import pickle
import sys

import numpy as np
import pandas as pd

from traderbot import dataset, metrics as mt, pipeline as pl
from traderbot.config import HOLDOUT_LOCK, HOLDOUT_START, REPORTS
from traderbot.walkforward import WFConfig, run as wf_run

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
log = logging.getLogger("final_test")


def main() -> int:
    if HOLDOUT_LOCK.exists():
        print(f"Holdout already used ({HOLDOUT_LOCK.read_text().strip()}). Refusing to run again.")
        return 1
    sel = json.loads((REPORTS / "selected.json").read_text())
    HOLDOUT_LOCK.write_text(f"used at {pd.Timestamp.now(tz='UTC').isoformat()} for {sel['selected']}\n")

    d = dataset.load(research=False)
    df, P, M, member = d["df"], d["P"], d["M"], d["member"]
    feats = dataset.feature_columns(df)
    last = df.index.get_level_values("date").max()
    end_excl = (last + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
    cfg = WFConfig(first_test=HOLDOUT_START, end=end_excl, test_months=24, retune_every=1)
    wf = wf_run(d, feats, cfg)
    pickle.dump(wf, open(REPORTS / "holdout_preds.pkl", "wb"))

    start = HOLDOUT_START
    end = (last - pd.Timedelta(days=1)).strftime("%Y-%m-%d")  # last fill needs next-day price
    idx, cols = P["close"].loc[start:end].index, P["close"].columns
    signals = {n: pl.wide(p, idx, cols) for n, p in wf["preds"].items()}
    signals["xs_momentum"] = pl.xs_momentum_scores(P, idx)

    results, rets = {}, {}
    for name, (tgt, kw) in pl.baseline_targets(P, member, start, end).items():
        r = pl.bt.run(tgt, P, **kw)
        rets[name], results[name] = r.returns, mt.summary(r.returns, r)
    bench = rets["btc_buy_hold"]
    for sname, sc in signals.items():
        for mode in ("full", "regime", "trend"):
            key = f"{sname}|{mode}"
            r = pl.run_signal(sc, P, member, M, wf["regimes"], mode, start, end)
            rets[key], results[key] = r.returns, mt.summary(r.returns, r, bench)
            results[key]["events_list"] = [(str(a.date()), b, c) for a, b, c in r.events if b != "daily_loss_limit"][:30]

    best = sel["selected"]
    raw = df["fwd_3"]
    from traderbot.models import daily_ic
    ic = {n: float(daily_ic(p, raw.reindex(p.index)).mean()) for n, p in wf["preds"].items()}
    rb = {
        "bootstrap": mt.block_bootstrap(rets[best]).describe(percentiles=[0.05, 0.5, 0.95]).to_dict(),
        "vs_btc_buy_hold": mt.paired_bootstrap_diff(rets[best], bench),
        "vs_same_overlay_momentum": mt.paired_bootstrap_diff(rets[best], rets["xs_momentum|" + best.split("|")[1]]),
        "cost_x2": mt.summary(pl.run_signal(signals[best.split("|")[0]], P, member, M, wf["regimes"],
                                            best.split("|")[1], start, end,
                                            costs=pl.SETTINGS.costs.scaled(2)).returns),
        "random_selection": pl.random_signal_test(P, member, M, wf["regimes"], best.split("|")[1], start, end,
                                                  results[best]["sharpe"], n=200),
    }
    out = {"period": [start, end], "selected": best, "selected_result": results[best], "ic": ic,
           "results": results, "robustness": rb, "folds": wf["folds"]}
    (REPORTS / "holdout_results.json").write_text(json.dumps(out, indent=1, default=str))
    pd.DataFrame(rets).to_parquet(REPORTS / "holdout_equity.parquet")
    s = results[best]
    log.info("HOLDOUT %s: CAGR %.1f%% Sharpe %.2f MaxDD %.1f%% | BTC B&H %.1f%%", best, 100 * s["cagr"], s["sharpe"],
             100 * s["max_dd"], 100 * results["btc_buy_hold"]["total_return"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
