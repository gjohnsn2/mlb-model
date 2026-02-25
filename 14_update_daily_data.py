"""
14 -- Update Daily Data (Append Yesterday's Results)
======================================================
Fetches yesterday's game results from the MLB Stats API and appends them
to the historical CSV files. This keeps the feature computation up-to-date
for tomorrow's predictions.

Updates:
  - data/historical/game_results_all.csv        (game results)
  - data/historical/pitcher_game_logs_mlbapi.csv (SP box scores)
  - data/historical/team_batting_game_logs.csv   (team batting)
  - data/historical/bullpen_game_logs.csv        (reliever appearances)

Run:
  python3 14_update_daily_data.py                 # Yesterday's games
  MLB_DATE=2025-09-15 python3 14_update_daily_data.py  # Specific date
"""

import sys
import time
import requests
import importlib
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import HISTORICAL_DIR, TODAY, get_logger

log = get_logger("14_update_data")

# Import parsing functions from fetch scripts
_fetch_games = importlib.import_module("scripts.fetch_historical_games")
parse_game_feed = _fetch_games.parse_game_feed
fetch_game_feed = _fetch_games.fetch_game_feed
_get_starting_pitcher = _fetch_games._get_starting_pitcher
_extract_sp_stats = _fetch_games._extract_sp_stats
_extract_team_batting = _fetch_games._extract_team_batting

_fetch_bullpen = importlib.import_module("scripts.fetch_bullpen_data")
extract_bullpen = _fetch_bullpen.extract_bullpen

# File paths
GAME_RESULTS_FILE = HISTORICAL_DIR / "game_results_all.csv"
PITCHER_LOGS_FILE = HISTORICAL_DIR / "pitcher_game_logs_mlbapi.csv"
BATTING_LOGS_FILE = HISTORICAL_DIR / "team_batting_game_logs.csv"
BULLPEN_LOGS_FILE = HISTORICAL_DIR / "bullpen_game_logs.csv"

# MLB Stats API
MLB_API_BASE = "https://statsapi.mlb.com/api/v1"
MAX_RETRIES = 3
TIMEOUT = 30


def get_target_date():
    """Get the date to fetch results for (yesterday by default)."""
    today = datetime.strptime(TODAY, "%Y-%m-%d")
    yesterday = today - timedelta(days=1)
    return yesterday.strftime("%Y-%m-%d")


def fetch_schedule_for_date(date_str):
    """Fetch completed games for a specific date from MLB Stats API."""
    try:
        import statsapi
        # statsapi.schedule expects MM/DD/YYYY format
        dt = datetime.strptime(date_str, "%Y-%m-%d")
        date_fmt = dt.strftime("%m/%d/%Y")
        sched = statsapi.schedule(start_date=date_fmt, end_date=date_fmt)
    except Exception as e:
        log.error(f"Failed to fetch schedule for {date_str}: {e}")
        return []

    completed = []
    for g in sched:
        status = g.get("status", "")
        game_type = g.get("game_type", "R")
        # Only completed regular season + postseason
        if game_type not in ("R", "F", "D", "L", "W"):
            continue
        if "Final" in status or "Game Over" in status:
            completed.append({
                "game_pk": g["game_id"],
                "game_date": g["game_date"],
                "status": status,
                "game_type": game_type,
                "doubleheader": g.get("doubleheader", "N"),
                "game_num": g.get("game_number", 1),
            })

    return completed


def load_existing_pks(filepath):
    """Load existing game_pks from a CSV to avoid duplicates."""
    if filepath.exists():
        df = pd.read_csv(filepath)
        return set(df["game_pk"].unique())
    return set()


def main():
    target_date = get_target_date()
    log.info("=" * 60)
    log.info(f"DAILY DATA UPDATE: fetching results for {target_date}")
    log.info("=" * 60)

    # Get completed games for the target date
    games = fetch_schedule_for_date(target_date)
    if not games:
        log.info(f"No completed games found for {target_date}")
        return

    log.info(f"Found {len(games)} completed games for {target_date}")

    # Load existing game_pks to skip duplicates
    existing_pks = load_existing_pks(GAME_RESULTS_FILE)
    new_games = [g for g in games if g["game_pk"] not in existing_pks]

    if not new_games:
        log.info("All games already in historical data. Nothing to update.")
        return

    log.info(f"New games to fetch: {len(new_games)} "
             f"({len(games) - len(new_games)} already exist)")

    # Fetch and parse each game
    game_rows = []
    pitcher_rows = []
    batting_rows = []
    bullpen_rows = []

    for i, game_meta in enumerate(new_games):
        game_pk = game_meta["game_pk"]
        log.info(f"  Fetching game {game_pk} ({i+1}/{len(new_games)})...")

        feed = fetch_game_feed(game_pk)
        if feed is None:
            log.warning(f"  Failed to fetch game {game_pk}")
            continue

        # Parse game result + SP stats + team batting
        game_row, sp_rows, bat_rows = parse_game_feed(feed, game_meta)
        if game_row:
            game_rows.append(game_row)
        pitcher_rows.extend(sp_rows)
        batting_rows.extend(bat_rows)

        # Parse bullpen (reliever) stats
        boxscore = feed.get("liveData", {}).get("boxscore", {})
        bp_rows = extract_bullpen(boxscore, game_pk, game_meta["game_date"])
        bullpen_rows.extend(bp_rows)

        time.sleep(0.3)  # Be nice to the API

    log.info(f"\nParsed: {len(game_rows)} games, {len(pitcher_rows)} SP logs, "
             f"{len(batting_rows)} batting logs, {len(bullpen_rows)} reliever logs")

    # ── Append to historical files ──
    HISTORICAL_DIR.mkdir(parents=True, exist_ok=True)

    # Game results
    if game_rows:
        new_df = pd.DataFrame(game_rows)
        if GAME_RESULTS_FILE.exists():
            existing = pd.read_csv(GAME_RESULTS_FILE)
            combined = pd.concat([existing, new_df], ignore_index=True)
            combined = combined.drop_duplicates(subset=["game_pk"], keep="first")
        else:
            combined = new_df
        combined.to_csv(GAME_RESULTS_FILE, index=False)
        log.info(f"Updated game_results_all.csv: {len(combined):,} total games")

    # Pitcher logs
    if pitcher_rows:
        new_df = pd.DataFrame(pitcher_rows)
        if PITCHER_LOGS_FILE.exists():
            existing = pd.read_csv(PITCHER_LOGS_FILE)
            combined = pd.concat([existing, new_df], ignore_index=True)
            combined = combined.drop_duplicates(
                subset=["game_pk", "pitcher_id"], keep="first")
        else:
            combined = new_df
        combined.to_csv(PITCHER_LOGS_FILE, index=False)
        log.info(f"Updated pitcher_game_logs_mlbapi.csv: {len(combined):,} total rows")

    # Team batting logs
    if batting_rows:
        new_df = pd.DataFrame(batting_rows)
        if BATTING_LOGS_FILE.exists():
            existing = pd.read_csv(BATTING_LOGS_FILE)
            combined = pd.concat([existing, new_df], ignore_index=True)
            combined = combined.drop_duplicates(
                subset=["game_pk", "team_id"], keep="first")
        else:
            combined = new_df
        combined.to_csv(BATTING_LOGS_FILE, index=False)
        log.info(f"Updated team_batting_game_logs.csv: {len(combined):,} total rows")

    # Bullpen logs
    if bullpen_rows:
        new_df = pd.DataFrame(bullpen_rows)
        if BULLPEN_LOGS_FILE.exists():
            existing = pd.read_csv(BULLPEN_LOGS_FILE)
            combined = pd.concat([existing, new_df], ignore_index=True)
            combined = combined.drop_duplicates(
                subset=["game_pk", "pitcher_id"], keep="first")
        else:
            combined = new_df
        combined.to_csv(BULLPEN_LOGS_FILE, index=False)
        log.info(f"Updated bullpen_game_logs.csv: {len(combined):,} total rows")

    # ── Summary ──
    print(f"\n{'='*60}")
    print(f"DAILY DATA UPDATE COMPLETE: {target_date}")
    print(f"{'='*60}")
    print(f"  Games added:    {len(game_rows)}")
    print(f"  SP logs added:  {len(pitcher_rows)}")
    print(f"  Batting added:  {len(batting_rows)}")
    print(f"  Bullpen added:  {len(bullpen_rows)}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
