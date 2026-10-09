"""Paper-trading control.

  paper_ctl.py daily     update data -> rebuild features -> rebalance (run ~00:15 UTC)
  paper_ctl.py hourly    stops / take-profit / loss limits / breaker
  paper_ctl.py retrain   weekly: train candidate, promote if not worse
  paper_ctl.py reset     clear circuit breaker after a manual review
  paper_ctl.py status    print equity, positions, recent events
"""
from __future__ import annotations

import logging
import subprocess
import sys
from contextlib import closing

import pandas as pd

from traderbot import dataset, paper, production
from traderbot.config import LOGS, ROOT

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s",
                    handlers=[logging.FileHandler(LOGS / "paper.log"), logging.StreamHandler()])
log = logging.getLogger("paper_ctl")


def update_data():
    r = subprocess.run([sys.executable, str(ROOT / "scripts" / "update_data.py")], capture_output=True, text=True)
    (LOGS / "update_data_last.log").write_text(r.stdout + r.stderr)
    if r.returncode not in (0, 1):
        raise RuntimeError("data update failed, see logs/update_data_last.log")


def main(cmd: str) -> int:
    paper.init()
    try:
        if cmd == "daily":
            update_data()
            dataset.build()
            if not (production.PROD / "meta.json").exists():
                cmd_retrain()
            paper.daily_rebalance()
        elif cmd == "hourly":
            paper.hourly_check()
        elif cmd == "retrain":
            cmd_retrain()
        elif cmd == "reset":
            paper.reset_breaker()
        elif cmd == "status":
            with closing(paper.db()) as con:
                print(pd.read_sql("SELECT * FROM equity ORDER BY ts DESC LIMIT 5", con).to_string())
                print(pd.read_sql("SELECT * FROM positions", con).to_string())
                print(pd.read_sql("SELECT * FROM events ORDER BY ts DESC LIMIT 10", con).to_string())
        else:
            print(__doc__)
            return 2
    except Exception as e:  # log into the DB so the dashboard shows failures
        log.exception("command %s failed", cmd)
        with closing(paper.db()) as con:
            paper.event(con, f"error_{cmd}", repr(e)[:500])
            con.commit()
        return 1
    return 0


def cmd_retrain():
    d = dataset.load(research=False)
    meta = production.train(d)
    ok, why = production.promote()
    with closing(paper.db()) as con:
        paper.event(con, "retrain", f"promoted={ok} {why} trained_until={meta['trained_until']}")
        con.commit()


if __name__ == "__main__":
    sys.exit(main(sys.argv[1] if len(sys.argv) > 1 else "help"))
