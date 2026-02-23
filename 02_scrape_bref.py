"""
02 -- Scrape Baseball Reference
==================================
Pulls standings, team game logs, and schedule data from Baseball Reference
via pybaseball. Provides game results for rolling form computation.

Outputs:
  data/raw/bref_standings_YYYY-MM-DD.csv
  data/raw/bref_game_logs_YYYY-MM-DD.csv

Requires: pip install pybaseball
"""

import sys
import pandas as pd
from config import RAW_DIR, TODAY, get_logger

log = get_logger("02_bref")

try:
    from pybaseball import standings, schedule_and_record
except ImportError:
    log.error("pybaseball not installed. Run: pip install pybaseball")
    sys.exit(1)


# MLB team abbreviations for schedule lookups
MLB_TEAMS = [
    "ARI", "ATL", "BAL", "BOS", "CHC", "CWS", "CIN", "CLE",
    "COL", "DET", "HOU", "KCR", "LAA", "LAD", "MIA", "MIL",
    "MIN", "NYM", "NYY", "OAK", "PHI", "PIT", "SDP", "SFG",
    "SEA", "STL", "TBR", "TEX", "TOR", "WSH",
]


def scrape_standings(season=None):
    """Pull current season standings from Baseball Reference."""
    if season is None:
        season = int(TODAY[:4])

    log.info(f"Fetching Baseball Reference standings for {season}...")
    try:
        dfs = standings(season)
        # standings() returns a list of DataFrames (one per division)
        combined = pd.concat(dfs, ignore_index=True)
        out_path = RAW_DIR / f"bref_standings_{TODAY}.csv"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        combined.to_csv(out_path, index=False)
        log.info(f"Saved {len(combined)} team standings rows to {out_path}")
        return combined
    except Exception as e:
        log.error(f"Failed to fetch standings: {e}")
        return None


def scrape_game_logs(season=None):
    """Pull game-by-game results for all MLB teams."""
    if season is None:
        season = int(TODAY[:4])

    log.info(f"Fetching game logs for {season}...")
    all_logs = []

    for team in MLB_TEAMS:
        try:
            df = schedule_and_record(season, team)
            df["team_abbrev"] = team
            all_logs.append(df)
            log.info(f"  {team}: {len(df)} games")
        except Exception as e:
            log.warning(f"  {team}: Failed ({e})")

    if all_logs:
        combined = pd.concat(all_logs, ignore_index=True)
        out_path = RAW_DIR / f"bref_game_logs_{TODAY}.csv"
        combined.to_csv(out_path, index=False)
        log.info(f"Saved {len(combined)} game log rows to {out_path}")
        return combined
    else:
        log.error("No game logs retrieved")
        return None


if __name__ == "__main__":
    scrape_standings()
    scrape_game_logs()
