"""
03 -- Fetch Schedule
=====================
Pulls today's MLB schedule from ESPN and/or the MLB Stats API.
Identifies which games are being played and basic matchup info.

Outputs:
  data/raw/schedule_YYYY-MM-DD.csv
"""

import sys
import requests
import pandas as pd
from config import RAW_DIR, ESPN_SCOREBOARD_URL, MLB_API_BASE, TODAY, get_logger

log = get_logger("03_schedule")


def fetch_espn_schedule(game_date=None):
    """Pull today's schedule from ESPN."""
    if game_date is None:
        game_date = TODAY

    api_date = game_date.replace("-", "")
    log.info(f"Fetching ESPN MLB schedule for {game_date}...")

    try:
        resp = requests.get(ESPN_SCOREBOARD_URL, params={
            "dates": api_date,
            "limit": 50,
        }, timeout=30)
        resp.raise_for_status()
        data = resp.json()
    except Exception as e:
        log.error(f"Failed to fetch ESPN schedule: {e}")
        return None

    events = data.get("events", [])
    log.info(f"Found {len(events)} games on ESPN")

    rows = []
    for event in events:
        game_id = event.get("id")
        name = event.get("name", "")
        status = event.get("status", {}).get("type", {}).get("name", "")
        game_time = event.get("date", "")

        comp = event.get("competitions", [{}])[0]
        venue = comp.get("venue", {}).get("fullName", "")

        home_team = away_team = ""
        for team_data in comp.get("competitors", []):
            team_name = team_data.get("team", {}).get("displayName", "")
            if team_data.get("homeAway") == "home":
                home_team = team_name
            else:
                away_team = team_name

        rows.append({
            "game_id": game_id,
            "date": game_date,
            "game_time": game_time,
            "home_team": home_team,
            "away_team": away_team,
            "venue": venue,
            "status": status,
            "name": name,
        })

    if rows:
        df = pd.DataFrame(rows)
        out_path = RAW_DIR / f"schedule_{TODAY}.csv"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(out_path, index=False)
        log.info(f"Saved {len(df)} games to {out_path}")
        return df
    else:
        log.warning("No games found")
        return None


def fetch_mlb_schedule(game_date=None):
    """Pull schedule from official MLB Stats API (backup/supplement)."""
    if game_date is None:
        game_date = TODAY

    log.info(f"Fetching MLB API schedule for {game_date}...")

    try:
        resp = requests.get(
            f"{MLB_API_BASE}/schedule",
            params={"date": game_date, "sportId": 1},
            timeout=30,
        )
        resp.raise_for_status()
        data = resp.json()
    except Exception as e:
        log.error(f"Failed to fetch MLB API schedule: {e}")
        return None

    dates = data.get("dates", [])
    if not dates:
        log.warning(f"No games found for {game_date}")
        return None

    games = dates[0].get("games", [])
    log.info(f"Found {len(games)} games on MLB API")
    return games


if __name__ == "__main__":
    fetch_espn_schedule()
