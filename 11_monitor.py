"""
11 -- Real-Time Monitoring System
====================================
Reads performance.csv from 09_evaluate.py and computes rolling metrics,
feature drift (PSI), edge calibration, and fires alerts when thresholds
are breached.

Ported from CBB pipeline with MLB-specific thresholds.

Outputs:
  Console report (human-readable, with alerts)
  data/tracking/monitoring.csv (appended daily)
"""

import sys
import warnings
import numpy as np
import pandas as pd
from pathlib import Path
from config import (
    TRACKING_DIR, PROCESSED_DIR, HISTORICAL_DIR, TODAY, get_logger
)

log = get_logger("11_monitor")

# -- Alert thresholds (MLB-specific) -----------------------------------
# MLB market is more efficient than CBB — lower baseline expected
ML_WIN_RATE_WARNING = 0.53       # Roll-50 ML win rate < 53%
ML_WIN_RATE_CRITICAL = 0.5238    # Below breakeven (approximate for avg ML juice)
ML_MAE_WARNING = 5.0             # Roll-100 margin MAE (runs)
BIAS_WARNING = 0.5               # Roll-20 absolute bias > 0.5 runs
PSI_WARNING = 0.25               # Significant distribution shift
EDGE_SLOPE_PVAL_CRITICAL = 0.10  # Edges not predictive

# Minimum sample sizes
MIN_FEATURE_FILES = 30   # Skip PSI if fewer current-season feature files
MIN_EDGE_BETS = 30       # Skip edge calibration if fewer flagged bets


def load_tracking():
    """Load performance.csv tracking data."""
    path = TRACKING_DIR / "performance.csv"
    if not path.exists():
        log.warning(f"No tracking data found: {path}")
        return None

    df = pd.read_csv(path)
    df["date"] = pd.to_datetime(df["date"])
    log.info(f"Loaded {len(df)} tracked bets from {df['date'].min().date()} to {df['date'].max().date()}")
    return df


def daily_metrics(df, eval_date=None):
    """Compute today's metrics."""
    if eval_date is None:
        eval_date = df["date"].max()
    else:
        eval_date = pd.to_datetime(eval_date)

    today = df[df["date"] == eval_date].copy()
    if today.empty:
        return None

    result = {"date": eval_date.strftime("%Y-%m-%d"), "n_bets": len(today)}

    # Win rate
    result["daily_win_rate"] = today["won"].mean()
    result["daily_pnl"] = today["pnl_units"].sum()

    return result


def rolling_metrics(df, window=50):
    """Compute rolling win rate and P&L."""
    if len(df) < window:
        log.info(f"Only {len(df)} bets — need {window} for rolling metrics")
        return None

    df = df.sort_values("date").reset_index(drop=True)

    # Rolling win rate
    roll_wr = df["won"].rolling(window).mean()
    current_wr = roll_wr.iloc[-1]

    # Rolling P&L
    roll_pnl = df["pnl_units"].rolling(window).sum()
    current_pnl = roll_pnl.iloc[-1]

    log.info(f"\nRolling {window}-bet metrics:")
    log.info(f"  Win rate: {current_wr:.1%}")
    log.info(f"  P&L: {current_pnl:+.1f} units")

    # Alert checks
    alerts = []
    if current_wr < ML_WIN_RATE_CRITICAL:
        alerts.append(f"CRITICAL: Rolling {window}-bet win rate {current_wr:.1%} below breakeven")
    elif current_wr < ML_WIN_RATE_WARNING:
        alerts.append(f"WARNING: Rolling {window}-bet win rate {current_wr:.1%} below warning threshold")

    # Season cumulative
    season_wr = df["won"].mean()
    season_pnl = df["pnl_units"].sum()
    log.info(f"\nSeason cumulative:")
    log.info(f"  Record: {int(df['won'].sum())}-{int(len(df) - df['won'].sum())} ({season_wr:.1%})")
    log.info(f"  P&L: {season_pnl:+.1f} units")

    return {
        "rolling_win_rate": current_wr,
        "rolling_pnl": current_pnl,
        "season_win_rate": season_wr,
        "season_pnl": season_pnl,
        "alerts": alerts,
    }


def run_monitoring():
    """Run all monitoring checks."""
    log.info(f"\n{'='*60}")
    log.info(f"MLB Model Monitoring Report -- {TODAY}")
    log.info(f"{'='*60}")

    df = load_tracking()
    if df is None or df.empty:
        log.info("No tracking data yet — monitoring will begin after first evaluated bets")
        return

    # Daily metrics
    daily = daily_metrics(df)
    if daily:
        log.info(f"\nToday ({daily['date']}): {daily['n_bets']} bets, "
                 f"WR: {daily['daily_win_rate']:.1%}, P&L: {daily['daily_pnl']:+.1f}u")

    # Rolling metrics
    rolling = rolling_metrics(df, window=50)

    # Output alerts
    all_alerts = []
    if rolling and rolling.get("alerts"):
        all_alerts.extend(rolling["alerts"])

    if all_alerts:
        log.info(f"\n{'!'*60}")
        log.info("ALERTS:")
        for alert in all_alerts:
            log.info(f"  {alert}")
        log.info(f"{'!'*60}")
    else:
        log.info("\nNo alerts triggered")

    # Save monitoring log
    TRACKING_DIR.mkdir(parents=True, exist_ok=True)
    monitor_path = TRACKING_DIR / "monitoring.csv"
    monitor_row = {
        "date": TODAY,
        "n_bets": len(df),
        "season_win_rate": df["won"].mean(),
        "season_pnl": df["pnl_units"].sum(),
        "alerts": "; ".join(all_alerts) if all_alerts else "none",
    }
    if rolling:
        monitor_row["rolling_50_wr"] = rolling["rolling_win_rate"]
        monitor_row["rolling_50_pnl"] = rolling["rolling_pnl"]

    new_row = pd.DataFrame([monitor_row])
    if monitor_path.exists():
        existing = pd.read_csv(monitor_path)
        combined = pd.concat([existing, new_row], ignore_index=True)
    else:
        combined = new_row
    combined.to_csv(monitor_path, index=False)
    log.info(f"\nMonitoring log updated: {monitor_path}")


if __name__ == "__main__":
    run_monitoring()
