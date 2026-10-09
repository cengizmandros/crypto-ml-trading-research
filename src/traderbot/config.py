"""Central configuration. All paths live under ~/trader-bot."""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(os.environ.get("TB_HOME", Path.home() / "trader-bot"))
DATA = ROOT / "data"
RAW = DATA / "raw"
PARQUET = DATA / "parquet"
MODELS = ROOT / "models"
REPORTS = ROOT / "reports"
PAPER = ROOT / "paper"
LOGS = ROOT / "logs"

QUOTE = "USDT"
START = "2019-01-01"

# The final holdout: never touched by research code. Only scripts/final_test.py
# may read it, and it writes a lock file so it can be run exactly once.
HOLDOUT_START = "2025-10-01"
HOLDOUT_LOCK = REPORTS / "HOLDOUT_USED.lock"

# Assets that are not real risk assets (stablecoins, fiat, wrapped, leveraged).
STABLE_BASES = {
    "USDC", "BUSD", "TUSD", "USDP", "PAX", "DAI", "UST", "USTC", "FDUSD", "USDS",
    "USDSB", "SUSD", "EUR", "GBP", "AUD", "TRY", "BRL", "RUB", "UAH", "NGN",
    "ZAR", "BIDR", "IDRT", "BVND", "AEUR", "EURI", "PYUSD", "USD1", "XUSD",
    "BFUSD", "RLUSD", "USDE", "PAXG", "WBTC", "WBETH", "BETH", "BKRW", "VAI",
}
LEVERAGED_SUFFIXES = ("UP", "DOWN", "BULL", "BEAR")


@dataclass(frozen=True)
class Costs:
    """Realistic Binance spot costs (one side)."""
    fee: float = 0.0010          # 0.10% taker, no BNB discount assumed
    half_spread_min: float = 0.0002  # 2 bp floor for large caps
    impact_k: float = 0.10       # slippage ~ k * daily_vol * sqrt(order / ADV)
    slip_floor: float = 0.0003   # 3 bp minimum execution slippage

    def scaled(self, m: float) -> "Costs":
        return Costs(self.fee * m, self.half_spread_min * m, self.impact_k * m, self.slip_floor * m)


@dataclass(frozen=True)
class Risk:
    target_vol: float = 0.35       # annualised portfolio vol target
    max_weight: float = 0.20       # per-asset cap
    max_gross: float = 1.00        # no leverage
    max_cluster_weight: float = 0.45  # cap for a group of highly correlated assets
    cluster_corr: float = 0.85
    stop_atr: float = 3.0          # stop-loss distance in daily ATRs
    take_atr: float = 6.0          # take-profit distance
    daily_loss_limit: float = 0.05
    breaker_dd: float = 0.25       # halt and go flat beyond this drawdown
    min_trade: float = 0.01        # ignore weight changes below 1% (turnover control)


@dataclass(frozen=True)
class Universe:
    top_n: int = 20                # tradable universe each month
    candidate_n: int = 30          # buffer for hourly downloads
    lookback_days: int = 30
    min_history_days: int = 90     # a coin must have this much history to be eligible


@dataclass
class Settings:
    costs: Costs = field(default_factory=Costs)
    risk: Risk = field(default_factory=Risk)
    universe: Universe = field(default_factory=Universe)
    paper_capital: float = 10_000.0


SETTINGS = Settings()

for _p in (RAW, PARQUET, MODELS, REPORTS, PAPER, LOGS):
    _p.mkdir(parents=True, exist_ok=True)
