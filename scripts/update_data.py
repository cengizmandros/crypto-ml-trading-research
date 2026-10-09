"""Download / incrementally update all market data.

Stage 1: daily bars for every USDT spot pair ever listed (incl. delisted).
Stage 2: point-in-time monthly universe (top candidate_n by trailing volume).
Stage 3: hourly bars for every symbol that was ever a candidate.
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time

from traderbot import data, universe
from traderbot.config import DATA, SETTINGS

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("update_data")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--refresh-symbols", action="store_true", help="re-list all symbols from the archive")
    ap.add_argument("--workers", type=int, default=12)
    args = ap.parse_args()

    t0 = time.time()
    sym_file = DATA / "symbols.json"
    if args.refresh_symbols or not sym_file.exists():
        syms = data.list_symbols()
        sym_file.write_text(json.dumps(syms))
    syms = json.loads(sym_file.read_text())
    log.info("daily update for %d symbols", len(syms))
    res = data.update_many(syms, "1d", workers=args.workers)
    failed = [s for s, n in res.items() if n < 0]
    log.info("daily done: %d ok, %d failed %s", sum(n > 0 for n in res.values()), len(failed), failed[:10])

    have = [s for s, n in res.items() if n > 0]
    qv = universe.daily_quote_volume(have)
    cand = universe.monthly_universe(qv, top_n=SETTINGS.universe.candidate_n)
    cand.to_parquet(DATA / "universe_candidates.parquet", index=False)
    univ = universe.monthly_universe(qv, top_n=SETTINGS.universe.top_n)
    univ.to_parquet(DATA / "universe.parquet", index=False)
    hourly_syms = sorted({data.base_symbol(s) for s in cand["symbol"].unique()})
    log.info("universe: %d months, %d distinct candidates", cand["month"].nunique(), len(hourly_syms))

    res_h = data.update_many(hourly_syms, "1h", workers=args.workers)
    failed_h = [s for s, n in res_h.items() if n < 0]
    log.info("hourly done: %d symbols, %d failed %s", len(res_h), len(failed_h), failed_h)
    log.info("total %.1f min", (time.time() - t0) / 60)
    return 1 if failed_h else 0


if __name__ == "__main__":
    sys.exit(main())
