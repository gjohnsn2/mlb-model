"""
03 -- Fetch Schedule
=====================
Pulls today's MLB schedule from the MLB Stats API.
Returns game_pk, team IDs, probable SPs, venue info — everything
needed by 05_build_features.py.

Falls back to ESPN for display names if needed.

Outputs:
  data/raw/schedule_YYYY-MM-DD.csv
"""

import sys
import requests
import pandas as pd
from config import RAW_DIR, MLB_API_BASE, TODAY, get_logger

log = get_logger("03_schedule")


def fetch_schedule(game_date=None):
    """Pull today's schedule from the MLB Stats API with full game details."""
    if game_date is None:
        game_date = TODAY

    log.info(f"Fetching MLB Stats API schedule for {game_date}...")

    # Use the schedule endpoint with hydrations for probable pitchers and venue
    try:
        resp = requests.get(
            f"{MLB_API_BASE}/schedule",
            params={
                "date": game_date,
                "sportId": 1,
                "hydrate": "probablePitcher,venue,weather,gameInfo,team",
            },
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

    rows = []
    for game in games:
        game_pk = game.get("gamePk")
        game_type = game.get("gameType", "R")
        status = game.get("status", {}).get("detailedState", "")
        game_date_str = game.get("officialDate", game_date)
        game_time = game.get("gameDate", "")  # ISO format

        # Skip non-regular-season/postseason games
        if game_type not in ("R", "F", "D", "L", "W"):
            continue

        # Teams
        teams = game.get("teams", {})
        home = teams.get("home", {})
        away = teams.get("away", {})

        home_team_data = home.get("team", {})
        away_team_data = away.get("team", {})

        home_team = home_team_data.get("name", "")
        away_team = away_team_data.get("name", "")
        home_abbrev = home_team_data.get("abbreviation", "")
        away_abbrev = away_team_data.get("abbreviation", "")
        home_team_id = home_team_data.get("id")
        away_team_id = away_team_data.get("id")

        # Probable pitchers
        home_sp = home.get("probablePitcher", {})
        away_sp = away.get("probablePitcher", {})
        home_sp_id = home_sp.get("id")
        home_sp_name = home_sp.get("fullName", "")
        away_sp_id = away_sp.get("id")
        away_sp_name = away_sp.get("fullName", "")

        # Venue
        venue = game.get("venue", {})
        venue_name = venue.get("name", "")
        venue_id = venue.get("id")

        # Weather (if available)
        weather = game.get("weather", {})
        temp = weather.get("temp")
        wind = weather.get("wind", "")
        condition = weather.get("condition", "")

        # Doubleheader
        dh = game.get("doubleHeader", "N")
        game_num = game.get("gameNumber", 1)

        rows.append({
            "game_pk": game_pk,
            "date": game_date_str,
            "game_time": game_time,
            "home_team": home_team,
            "away_team": away_team,
            "home_abbrev": home_abbrev,
            "away_abbrev": away_abbrev,
            "home_team_id": home_team_id,
            "away_team_id": away_team_id,
            "home_sp_id": home_sp_id,
            "home_sp_name": home_sp_name,
            "away_sp_id": away_sp_id,
            "away_sp_name": away_sp_name,
            "venue_name": venue_name,
            "venue_id": venue_id,
            "temp": temp,
            "wind": wind,
            "condition": condition,
            "doubleheader": dh,
            "game_num": game_num,
            "game_type": game_type,
            "status": status,
        })

    if rows:
        df = pd.DataFrame(rows)
        out_path = RAW_DIR / f"schedule_{TODAY}.csv"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(out_path, index=False)
        log.info(f"Saved {len(df)} games to {out_path}")

        # Summary
        sp_home_pct = df["home_sp_id"].notna().mean() * 100
        sp_away_pct = df["away_sp_id"].notna().mean() * 100
        log.info(f"  Probable SPs: {sp_home_pct:.0f}% home, {sp_away_pct:.0f}% away")

        for _, g in df.iterrows():
            sp_h = g["home_sp_name"] or "TBD"
            sp_a = g["away_sp_name"] or "TBD"
            log.info(f"  {g['away_abbrev']} ({sp_a}) @ {g['home_abbrev']} ({sp_h}) "
                     f"[{g['venue_name']}]")

        return df
    else:
        log.warning("No regular-season/postseason games found")
        return None


if __name__ == "__main__":
    fetch_schedule()
