"""
01 -- Scrape FanGraphs Team + Pitcher Stats
=============================================
Uses pybaseball to pull team-level and pitcher-level season stats from FanGraphs.
This is the primary data source for team batting and pitching metrics.

Outputs:
  data/raw/fangraphs_team_batting_YYYY-MM-DD.csv
  data/raw/fangraphs_team_pitching_YYYY-MM-DD.csv
  data/raw/fangraphs_sp_stats_YYYY-MM-DD.csv

Requires: pip install pybaseball
"""

import sys
import pandas as pd
from config import RAW_DIR, TODAY, get_logger

log = get_logger("01_fangraphs")

try:
    from pybaseball import team_batting, team_pitching, pitching_stats
except ImportError:
    log.error("pybaseball not installed. Run: pip install pybaseball")
    sys.exit(1)


def scrape_team_batting(season=None):
    """Pull FanGraphs team batting stats for the current season."""
    if season is None:
        season = int(TODAY[:4])

    log.info(f"Fetching FanGraphs team batting for {season}...")
    try:
        df = team_batting(season)
        out_path = RAW_DIR / f"fangraphs_team_batting_{TODAY}.csv"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(out_path, index=False)
        log.info(f"Saved {len(df)} team batting rows to {out_path}")
        return df
    except Exception as e:
        log.error(f"Failed to fetch team batting: {e}")
        return None


def scrape_team_pitching(season=None):
    """Pull FanGraphs team pitching stats for the current season."""
    if season is None:
        season = int(TODAY[:4])

    log.info(f"Fetching FanGraphs team pitching for {season}...")
    try:
        df = team_pitching(season)
        out_path = RAW_DIR / f"fangraphs_team_pitching_{TODAY}.csv"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(out_path, index=False)
        log.info(f"Saved {len(df)} team pitching rows to {out_path}")
        return df
    except Exception as e:
        log.error(f"Failed to fetch team pitching: {e}")
        return None


def scrape_sp_stats(season=None):
    """Pull FanGraphs starting pitcher stats for qualified starters."""
    if season is None:
        season = int(TODAY[:4])

    log.info(f"Fetching FanGraphs SP stats for {season}...")
    try:
        df = pitching_stats(season, qual=20)  # 20 IP minimum
        # Filter to starters (GS > 0)
        if "GS" in df.columns:
            df = df[df["GS"] > 0]
        out_path = RAW_DIR / f"fangraphs_sp_stats_{TODAY}.csv"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(out_path, index=False)
        log.info(f"Saved {len(df)} SP stat rows to {out_path}")
        return df
    except Exception as e:
        log.error(f"Failed to fetch SP stats: {e}")
        return None


if __name__ == "__main__":
    scrape_team_batting()
    scrape_team_pitching()
    scrape_sp_stats()
