"""
01b -- Scrape Statcast / Baseball Savant
==========================================
Uses pybaseball to pull Statcast data (pitch-level and aggregated metrics).
Statcast provides advanced metrics like xwOBA, barrel rate, hard-hit%,
and expected stats that are not available through traditional box scores.

Outputs:
  data/raw/statcast_pitcher_YYYY-MM-DD.csv   (pitcher-level xStats)
  data/raw/statcast_team_YYYY-MM-DD.csv      (team-level aggregated)

Requires: pip install pybaseball
"""

import sys
import pandas as pd
from config import RAW_DIR, TODAY, get_logger

log = get_logger("01b_statcast")

try:
    from pybaseball import statcast, statcast_pitcher, pitching_stats
except ImportError:
    log.error("pybaseball not installed. Run: pip install pybaseball")
    sys.exit(1)


def scrape_pitcher_statcast(season=None):
    """Pull Statcast pitcher metrics (xwOBA, barrel%, hard-hit%, etc.)."""
    if season is None:
        season = int(TODAY[:4])

    log.info(f"Fetching Statcast pitcher data for {season}...")
    try:
        # Use pitching_stats with statcast=True for expected metrics
        df = pitching_stats(season, qual=20, ind=0)
        out_path = RAW_DIR / f"statcast_pitcher_{TODAY}.csv"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(out_path, index=False)
        log.info(f"Saved {len(df)} Statcast pitcher rows to {out_path}")
        return df
    except Exception as e:
        log.error(f"Failed to fetch Statcast pitcher data: {e}")
        return None


if __name__ == "__main__":
    scrape_pitcher_statcast()
