"""
02b -- Scrape Starting Pitcher Game Logs
==========================================
Pulls game-by-game starting pitcher logs via pybaseball.
These provide the foundation for SP features: ERA, FIP, K%, BB%,
pitch count trends, and recency metrics (last 3 starts).

Starting pitcher is THE most important feature in MLB betting.
This script must run before feature engineering.

Outputs:
  data/raw/pitcher_game_logs_YYYY-MM-DD.csv

Requires: pip install pybaseball
"""

import sys
import pandas as pd
from config import RAW_DIR, TODAY, get_logger

log = get_logger("02b_pitcher_logs")

try:
    from pybaseball import pitching_stats_range, playerid_lookup
except ImportError:
    log.error("pybaseball not installed. Run: pip install pybaseball")
    sys.exit(1)


def scrape_pitcher_logs(start_date=None, end_date=None):
    """Pull pitcher game logs for a date range via pybaseball."""
    if start_date is None:
        # Default: last 14 days
        from datetime import datetime, timedelta
        end_dt = datetime.strptime(TODAY, "%Y-%m-%d")
        start_dt = end_dt - timedelta(days=14)
        start_date = start_dt.strftime("%Y-%m-%d")
        end_date = TODAY

    log.info(f"Fetching pitcher game logs from {start_date} to {end_date}...")
    try:
        df = pitching_stats_range(start_date, end_date)
        if df is not None and len(df) > 0:
            out_path = RAW_DIR / f"pitcher_game_logs_{TODAY}.csv"
            out_path.parent.mkdir(parents=True, exist_ok=True)
            df.to_csv(out_path, index=False)
            log.info(f"Saved {len(df)} pitcher game log rows to {out_path}")
            return df
        else:
            log.warning("No pitcher game logs returned")
            return None
    except Exception as e:
        log.error(f"Failed to fetch pitcher game logs: {e}")
        return None


if __name__ == "__main__":
    scrape_pitcher_logs()
