"""Full walk-forward research run (never touches the holdout).

Outputs (reports/):
  wf_results.json   metrics for every strategy variant + baselines + MC tests
  wf_equity.parquet daily returns of every variant
  wf_preds.pkl      out-of-sample predictions, regimes, fold log
  selected.json     the single configuration pre-committed for the final test
"""
from __future__ import annotations

import argparse
import json
import logging
import pickle
import time

import numpy as np
import pandas as pd

from traderbot import dataset, metrics as mt, pipeline as pl
from traderbot.config import REPORTS, SETTINGS
from traderbot.models import daily_ic
from traderbot.walkforward import WFConfig, run as wf_run

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
log = logging.getLogger("research")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rebuild", action="store_true")
    ap.add_argument("--skip-wf", action="store_true", help="reuse saved predictions")
    ap.add_argument("--mc", type=int, default=200)
    args = ap.parse_args()
    t0 = time.time()

    if args.rebuild or not dataset.CACHE.exists():
        log.info("building dataset ...")
        dataset.build()
    d = dataset.load(research=True)
    df, P, M, member = d["df"], d["P"], d["M"], d["member"]
    feats = dataset.feature_columns(df)
    log.info("dataset: %d rows, %d features, %s..%s", len(df), len(feats),
             df.index.get_level_values("date").min().date(), df.index.get_level_values("date").max().date())

    cfg = WFConfig()
    pf = REPORTS / "wf_preds.pkl"
    if args.skip_wf and pf.exists():
        wf = pickle.load(open(pf, "rb"))
    else:
        wf = wf_run(d, feats, cfg)
        pickle.dump(wf, open(pf, "wb"))
    start, end = cfg.first_test, (pd.Timestamp(cfg.end) - pd.Timedelta(days=1)).strftime("%Y-%m-%d")

    # ---------- out-of-sample signal quality (IC) ----------
    raw = df[f"fwd_{cfg.horizon}"]
    ic = {}
    for name, p in wf["preds"].items():
        dic = daily_ic(p, raw.reindex(p.index))
        ic[name] = {"mean_ic": float(dic.mean()), "ic_tstat": float(dic.mean() / dic.std() * np.sqrt(len(dic) / cfg.horizon)),
                    "pct_days_pos": float((dic > 0).mean())}
    mom = pl.xs_momentum_scores(P, P["close"].loc[start:end].index).stack()
    mom.index.names = ["date", "symbol"]
    mom = mom.reindex(wf["preds"]["lgbm"].index)
    dic = daily_ic(mom, raw.reindex(mom.index))
    ic["xs_momentum"] = {"mean_ic": float(dic.mean()), "ic_tstat": float(dic.mean() / dic.std() * np.sqrt(len(dic) / cfg.horizon)),
                         "pct_days_pos": float((dic > 0).mean())}
    log.info("OOS IC: %s", json.dumps(ic, indent=1))

    # ---------- backtests ----------
    results, rets = {}, {}
    for name, (tgt, kw) in pl.baseline_targets(P, member, start, end).items():
        r = pl.bt.run(tgt, P, **kw)
        rets[name] = r.returns
        results[name] = mt.summary(r.returns, r)
    bench = rets["btc_buy_hold"]

    idx, cols = P["close"].loc[start:end].index, P["close"].columns
    signals = {n: pl.wide(p, idx, cols) for n, p in wf["preds"].items()}
    signals["xs_momentum"] = pl.xs_momentum_scores(P, idx)
    regimes = wf["regimes"]
    bts = {}
    for sname, sc in signals.items():
        for mode in ("full", "regime", "trend"):
            key = f"{sname}|{mode}"
            r = pl.run_signal(sc, P, member, M, regimes, mode, start, end)
            bts[key] = r
            rets[key] = r.returns
            results[key] = mt.summary(r.returns, r, bench)
            results[key]["events_list"] = [(str(a.date()), b, c) for a, b, c in r.events if b != "daily_loss_limit"][:30]
            log.info("%-26s CAGR %6.1f%% Sharpe %5.2f MaxDD %6.1f%% trades %d", key, 100 * results[key]["cagr"],
                     results[key]["sharpe"], 100 * results[key]["max_dd"], results[key]["n_trades"])

    n_variants = len(bts)
    # ---------- pick ONE config for the holdout, by walk-forward Sharpe ----------
    ml_keys = [k for k in bts if not k.startswith("xs_momentum")]
    best = max(ml_keys, key=lambda k: results[k]["sharpe"] if np.isfinite(results[k]["sharpe"]) else -9)
    best_mom = max([k for k in bts if k.startswith("xs_momentum")], key=lambda k: results[k]["sharpe"])
    log.info("selected: %s (best momentum baseline: %s)", best, best_mom)

    # ---------- robustness ----------
    rb = {}
    rb["cost_x2"] = mt.summary(pl.run_signal(signals[best.split("|")[0]], P, member, M, regimes, best.split("|")[1],
                                             start, end, costs=SETTINGS.costs.scaled(2)).returns)
    rb["cost_x3"] = mt.summary(pl.run_signal(signals[best.split("|")[0]], P, member, M, regimes, best.split("|")[1],
                                             start, end, costs=SETTINGS.costs.scaled(3)).returns)
    rb["no_stops_no_breaker"] = mt.summary(pl.run_signal(signals[best.split("|")[0]], P, member, M, regimes,
                                                         best.split("|")[1], start, end, use_stops=False, use_breaker=False).returns)
    bs = mt.block_bootstrap(rets[best])
    rb["bootstrap"] = {c: {"p05": float(bs[c].quantile(0.05)), "p50": float(bs[c].median()), "p95": float(bs[c].quantile(0.95))}
                       for c in bs.columns}
    rb["bootstrap"]["p_sharpe_le_0"] = float((bs["sharpe"] <= 0).mean())
    rb["vs_btc_buy_hold"] = mt.paired_bootstrap_diff(rets[best], bench)
    rb["vs_best_momentum"] = mt.paired_bootstrap_diff(rets[best], rets[best_mom])
    rb["vs_same_overlay_momentum"] = mt.paired_bootstrap_diff(rets[best], rets["xs_momentum|" + best.split("|")[1]])
    total_trials = n_variants + sum(wf["trials"].values())
    rb["deflated_sharpe_variants"] = mt.deflated_sharpe(rets[best], n_variants,
                                                        sr_var_trials=np.var([mt.sharpe(rets[k]) / np.sqrt(365) for k in bts]))
    rb["deflated_sharpe_all_trials"] = mt.deflated_sharpe(rets[best], total_trials,
                                                          sr_var_trials=np.var([mt.sharpe(rets[k]) / np.sqrt(365) for k in bts]))
    log.info("random-selection Monte Carlo (%d runs) ...", args.mc)
    rb["random_selection"] = pl.random_signal_test(P, member, M, regimes, best.split("|")[1], start, end,
                                                   results[best]["sharpe"], n=args.mc)

    out = {"period": [start, end], "horizon": cfg.horizon, "ic": ic, "results": results, "selected": best,
           "best_momentum": best_mom, "robustness": rb, "trials": wf["trials"], "n_variants": n_variants,
           "folds": wf["folds"], "n_features": len(feats), "features": feats,
           "runtime_min": round((time.time() - t0) / 60, 1)}
    (REPORTS / "wf_results.json").write_text(json.dumps(out, indent=1, default=str))
    pd.DataFrame(rets).to_parquet(REPORTS / "wf_equity.parquet")
    last_params = wf["folds"][-1]["params"]
    (REPORTS / "selected.json").write_text(json.dumps({
        "selected": best, "signal": best.split("|")[0], "exposure_mode": best.split("|")[1],
        "horizon": cfg.horizon, "model_params": last_params, "committed_at": pd.Timestamp.now(tz="UTC").isoformat(),
        "wf_sharpe": results[best]["sharpe"]}, indent=1, default=str))
    log.info("done in %.1f min", (time.time() - t0) / 60)


if __name__ == "__main__":
    main()
