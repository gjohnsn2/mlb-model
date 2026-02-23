"""
02e -- Scrape Confirmed Lineups
=================================
Fetches confirmed starting lineups from the MLB Stats API.
Critical for:
  - Confirming starting pitchers (the most important feature)
  - Computing lineup-level batting stats (wRC+ of actual lineup vs. season average)
  - Platoon advantage (lineup handedness vs. opposing SP)

Lineups are typically confirmed 1-3 hours before game time.
The pipeline should run AFTER lineups are posted for maximum accuracy.

Outputs:
  data/raw/lineups_YYYY-MM-DD.csv

No API key required (MLB Stats API is public).
"""

import sys
import requests
import pandas as pd
from config import RAW_DIR, MLB_API_BASE, TODAY, get_logger

log = get_logger("02e_lineups")


def fetch_lineups(game_date=None):
    """Fetch confirmed lineups from MLB Stats API."""
    if game_date is None:
        game_date = TODAY

    log.info(f"Fetching lineups for {game_date}...")

    # MLB Stats API: get schedule with probable pitchers
    url = f"{MLB_API_BASE}/schedule"
    params = {
        "date": game_date,
        "sportId": 1,  # MLB
        "hydrate": "probablePitcher,lineups",
    }

    try:
        resp = requests.get(url, params=params, timeout=30)
        resp.raise_for_status()
        data = resp.json()
    except Exception as e:
        log.error(f"Failed to fetch lineups: {e}")
        return None

    dates = data.get("dates", [])
    if not dates:
        log.warning(f"No games found for {game_date}")
        return None

    games = dates[0].get("games", [])
    log.info(f"Found {len(games)} games")

    lineup_rows = []
    for game in games:
        game_pk = game.get("gamePk")
        status = game.get("status", {}).get("abstractGameState", "")

        home = game.get("teams", {}).get("home", {})
        away = game.get("teams", {}).get("away", {})

        home_team = home.get("team", {}).get("name", "")
        away_team = away.get("team", {}).get("name", "")

        # Probable pitchers
        home_sp = home.get("probablePitcher", {})
        away_sp = away.get("probablePitcher", {})

        lineup_rows.append({
            "game_id": game_pk,
            "date": game_date,
            "status": status,
            "home_team": home_team,
            "away_team": away_team,
            "home_sp_id": home_sp.get("id"),
            "home_sp_name": home_sp.get("fullName", ""),
            "away_sp_id": away_sp.get("id"),
            "away_sp_name": away_sp.get("fullName", ""),
        })

    if lineup_rows:
        df = pd.DataFrame(lineup_rows)
        out_path = RAW_DIR / f"lineups_{TODAY}.csv"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(out_path, index=False)
        log.info(f"Saved {len(df)} lineup entries to {out_path}")

        # Report on confirmed starters
        confirmed = df[df["home_sp_name"].str.len() > 0 & df["away_sp_name"].str.len() > 0]
        log.info(f"  Confirmed both SPs: {len(confirmed)}/{len(df)} games")

        return df

    return None


if __name__ == "__main__":
    fetch_lineups()
