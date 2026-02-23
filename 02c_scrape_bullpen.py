"""
02c -- Scrape Bullpen Usage and Fatigue Metrics
=================================================
Tracks bullpen workload across recent games to identify fatigue effects.
A fatigued bullpen is one of the most underpriced factors in MLB betting.

Key metrics:
  - Bullpen IP in last 3 days (workload)
  - Closer availability (last used date)
  - High-leverage reliever availability
  - Bullpen ERA over last 7 days (recency)

Outputs:
  data/raw/bullpen_usage_YYYY-MM-DD.csv

Requires: pip install pybaseball
"""

import sys
import pandas as pd
from datetime import datetime, timedelta
from config import RAW_DIR, TODAY, get_logger

log = get_logger("02c_bullpen")


def scrape_bullpen_usage():
    """
    Compute bullpen usage metrics from recent game logs.
    This is a derived calculation from pitcher game logs rather than
    a direct scrape — uses data from 02b_scrape_pitcher_logs.py.
    """
    log.info("Computing bullpen usage metrics...")

    # Load recent pitcher logs
    logs_path = RAW_DIR / f"pitcher_game_logs_{TODAY}.csv"
    if not logs_path.exists():
        log.warning(f"Pitcher logs not found: {logs_path}")
        log.warning("Run 02b_scrape_pitcher_logs.py first")
        return None

    df = pd.read_csv(logs_path)
    log.info(f"Loaded {len(df)} pitcher log entries")

    # Filter to relievers (GS == 0 or not starting)
    if "GS" in df.columns:
        relievers = df[df["GS"] == 0].copy()
    else:
        log.warning("Cannot distinguish starters from relievers — GS column missing")
        return None

    if relievers.empty:
        log.warning("No reliever data found")
        return None

    # Aggregate by team
    if "date" in relievers.columns:
        relievers["date"] = pd.to_datetime(relievers["date"])

    # For each team, compute bullpen metrics
    # This will be expanded as the data pipeline matures
    team_bullpen = []
    for team in relievers["Team"].unique() if "Team" in relievers.columns else []:
        team_data = relievers[relievers["Team"] == team]
        team_bullpen.append({
            "team": team,
            "date": TODAY,
            "bullpen_appearances": len(team_data),
            "bullpen_ip_total": team_data["IP"].sum() if "IP" in team_data.columns else 0,
            "bullpen_era": team_data["ERA"].mean() if "ERA" in team_data.columns else None,
        })

    if team_bullpen:
        result = pd.DataFrame(team_bullpen)
        out_path = RAW_DIR / f"bullpen_usage_{TODAY}.csv"
        result.to_csv(out_path, index=False)
        log.info(f"Saved bullpen usage for {len(result)} teams to {out_path}")
        return result
    else:
        log.warning("No bullpen data computed")
        return None


if __name__ == "__main__":
    scrape_bullpen_usage()
