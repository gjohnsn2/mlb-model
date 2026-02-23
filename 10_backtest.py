"""
10 -- Profitability Backtest
===============================
Merges walk-forward OOF predictions with historical closing lines from
The Odds API. Simulates betting at actual ML odds across multiple
edge thresholds. Reports W-L record, ROI, p-values, and equity curves.

MLB-specific: Uses actual moneyline odds (not flat -110) for P&L
calculation. This is critical because MLB juice is variable —
a -150 favorite losing costs more than a -110 bet.

Inputs:
  models/oof_margin_predictions.csv
  models/oof_total_predictions.csv
  data/historical/historical_odds.csv

Outputs:
  models/backtest_report.txt
  models/backtest_results.csv
  models/backtest_equity_curve.png

Run: python3 10_backtest.py
"""

import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats
from scipy.stats import norm
from pathlib import Path
from config import (
    MODELS_ROOT, HISTORICAL_DIR,
    ML_EDGE_THRESHOLD, ML_UNIT_TIERS,
    get_logger
)

log = get_logger("10_backtest")


def ml_payout(odds, stake=1.0):
    """Calculate profit for a moneyline bet at given odds."""
    if pd.isna(odds):
        return 0
    if odds > 0:
        return stake * (odds / 100)
    else:
        return stake * (100 / abs(odds))


def ml_to_implied_prob(ml):
    """Convert American ML to implied probability (no-vig adjusted)."""
    if pd.isna(ml):
        return np.nan
    if ml > 0:
        return 100 / (ml + 100)
    else:
        return abs(ml) / (abs(ml) + 100)


def margin_to_win_prob(margin, rmse):
    """Convert predicted margin to win probability via normal CDF."""
    return norm.cdf(margin / rmse)


def run_backtest():
    """Run the full profitability backtest."""
    # Load OOF predictions
    oof_path = MODELS_ROOT / "oof_margin_predictions.csv"
    if not oof_path.exists():
        log.error(f"OOF predictions not found: {oof_path}")
        log.error("Run 06_train_model.py first")
        sys.exit(1)

    oof = pd.read_csv(oof_path)
    oof["date"] = pd.to_datetime(oof["date"])
    log.info(f"Loaded {len(oof):,} OOF predictions")

    # Load historical odds
    odds_path = HISTORICAL_DIR / "historical_odds.csv"
    if not odds_path.exists():
        log.error(f"Historical odds not found: {odds_path}")
        log.error("Fetch historical odds first")
        sys.exit(1)

    odds = pd.read_csv(odds_path)
    odds["date"] = pd.to_datetime(odds["date"])
    log.info(f"Loaded {len(odds):,} historical odds entries")

    # Merge predictions with odds
    # TODO: Implement matching logic once data formats are established
    log.info("\nBacktest infrastructure ready — awaiting historical data population")
    log.info("Run 00_build_historical.py to populate training data, then re-run backtest")


if __name__ == "__main__":
    run_backtest()
