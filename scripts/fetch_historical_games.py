"""
Fetch Historical MLB Games (2019-2025) from MLB Stats API
==========================================================
Fetches game feed for every regular season + postseason game.
One API call per game returns everything: boxscore, linescore, weather, umpires.

Uses concurrent requests (4 threads) for ~4x speedup over sequential.

Outputs (data/historical/):
  - game_results_all.csv          — one row/game: scores, F5, NRFI, SPs, venue, weather
  - pitcher_game_logs_mlbapi.csv  — one row/SP/game: counting stats (IP, H, R, ER, K, BB, HR)
  - team_batting_game_logs.csv    — one row/team/game: runs, hits, ABs, HRs, Ks, BBs

Checkpointing: saves progress every 50 games to game_fetch_progress.json.
"""

import sys
import json
import time
import traceback
import statsapi
import requests
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import HISTORICAL_DIR, get_logger

log = get_logger("fetch_hist_games")

# ── Config ──────────────────────────────────────────────────────
SEASONS = list(range(2015, 2026))  # 2015 through 2025
CHECKPOINT_EVERY = 50
MAX_RETRIES = 3
TIMEOUT = 30
NUM_WORKERS = 4  # Concurrent fetch threads

PROGRESS_FILE = HISTORICAL_DIR / "game_fetch_progress.json"
GAME_RESULTS_FILE = HISTORICAL_DIR / "game_results_all.csv"
PITCHER_LOGS_FILE = HISTORICAL_DIR / "pitcher_game_logs_mlbapi.csv"
BATTING_LOGS_FILE = HISTORICAL_DIR / "team_batting_game_logs.csv"


# ── Progress ────────────────────────────────────────────────────
def load_progress():
    if PROGRESS_FILE.exists():
        with open(PROGRESS_FILE) as f:
            return json.load(f)
    return {"fetched_pks": [], "errors": []}


def save_progress(progress):
    HISTORICAL_DIR.mkdir(parents=True, exist_ok=True)
    with open(PROGRESS_FILE, "w") as f:
        json.dump(progress, f)


# ── Schedule fetcher ────────────────────────────────────────────
def _fetch_schedule_with_retry(start_date, end_date, retries=MAX_RETRIES):
    """Fetch schedule with retry logic for flaky MLB Stats API."""
    for attempt in range(retries):
        try:
            return statsapi.schedule(start_date=start_date, end_date=end_date)
        except Exception as e:
            if attempt < retries - 1:
                wait = 2 ** (attempt + 1)
                log.warning(f"Schedule fetch error ({e}), retrying in {wait}s...")
                time.sleep(wait)
            else:
                raise


def get_all_game_pks():
    """Fetch all game PKs for all seasons via statsapi.schedule()."""
    all_pks = []
    for season in SEASONS:
        log.info(f"Fetching schedule for {season}...")
        # Fetch in monthly chunks to avoid API timeouts on large date ranges
        pks = []
        for month in range(3, 12):
            from calendar import monthrange
            _, last_day = monthrange(season, month)
            start = f"{month:02d}/01/{season}"
            end = f"{month:02d}/{last_day:02d}/{season}"
            try:
                sched = _fetch_schedule_with_retry(start, end)
            except Exception as e:
                log.error(f"  Failed to fetch {season}-{month:02d}: {e}")
                continue
            for g in sched:
                status = g.get("status", "")
                game_type = g.get("game_type", "R")
                # Only completed regular season + postseason games
                # R=Regular, F=WildCard, D=Division, L=League, W=WorldSeries
                # Skip: S=Spring, E=Exhibition, A=AllStar
                if game_type not in ("R", "F", "D", "L", "W"):
                    continue
                if "Final" in status or "Game Over" in status:
                    pks.append({
                        "game_pk": g["game_id"],
                        "game_date": g["game_date"],
                        "status": status,
                        "game_type": game_type,
                        "doubleheader": g.get("doubleheader", "N"),
                        "game_num": g.get("game_number", 1),
                    })
            time.sleep(0.3)  # Brief pause between monthly chunks

        log.info(f"  {season}: {len(pks)} completed games")
        all_pks.extend(pks)

    log.info(f"Total game PKs: {len(all_pks)}")
    return all_pks


# ── Game feed fetcher (thread-safe) ─────────────────────────────
def fetch_game_feed(game_pk):
    """Fetch full game feed from MLB Stats API with retries."""
    url = f"https://statsapi.mlb.com/api/v1.1/game/{game_pk}/feed/live"
    for attempt in range(MAX_RETRIES):
        try:
            resp = requests.get(url, timeout=TIMEOUT)
            if resp.status_code == 404:
                return None
            resp.raise_for_status()
            return resp.json()
        except requests.exceptions.RequestException as e:
            if attempt < MAX_RETRIES - 1:
                wait = 2 ** (attempt + 1)
                time.sleep(wait)
            else:
                log.error(f"  Game {game_pk}: failed after {MAX_RETRIES} attempts: {e}")
                return None
    return None


def _fetch_one(game_meta):
    """Fetch and parse a single game. Returns (game_meta, feed) tuple."""
    feed = fetch_game_feed(game_meta["game_pk"])
    return game_meta, feed


# ── Game feed parser ────────────────────────────────────────────
def parse_game_feed(feed, game_meta):
    """
    Parse a game feed into game result, pitcher logs, and batting logs.
    Returns (game_row, pitcher_rows, batting_rows) or (None, [], []) on failure.
    """
    try:
        game_data = feed.get("gameData", {})
        live_data = feed.get("liveData", {})
        game_pk = game_data.get("game", {}).get("pk", game_meta["game_pk"])

        # Teams
        teams = game_data.get("teams", {})
        home_team = teams.get("home", {}).get("name", "")
        away_team = teams.get("away", {}).get("name", "")
        home_abbrev = teams.get("home", {}).get("abbreviation", "")
        away_abbrev = teams.get("away", {}).get("abbreviation", "")
        home_id = teams.get("home", {}).get("id")
        away_id = teams.get("away", {}).get("id")

        # Venue
        venue = game_data.get("venue", {})
        venue_name = venue.get("name", "")
        venue_id = venue.get("id")

        # Weather
        weather = game_data.get("weather", {})
        temp = weather.get("temp")
        wind = weather.get("wind", "")
        condition = weather.get("condition", "")

        # Game info
        game_info = game_data.get("game", {})
        game_date = game_data.get("datetime", {}).get("officialDate", game_meta["game_date"])
        game_type = game_info.get("type", game_meta.get("game_type", "R"))
        is_postseason = game_type in ("F", "D", "L", "W")

        # Doubleheader detection (7-inning games 2020-2022)
        doubleheader = game_meta.get("doubleheader", "N")
        game_num = game_meta.get("game_num", 1)
        season = int(game_date[:4])
        is_7_inning_dh = (
            doubleheader != "N" and
            season >= 2020 and season <= 2022
        )

        # Linescore
        linescore = live_data.get("linescore", {})
        innings = linescore.get("innings", [])

        home_runs = linescore.get("teams", {}).get("home", {}).get("runs")
        away_runs = linescore.get("teams", {}).get("away", {}).get("runs")
        home_hits = linescore.get("teams", {}).get("home", {}).get("hits")
        away_hits = linescore.get("teams", {}).get("away", {}).get("hits")

        # F5 runs (innings 1-5)
        home_f5_runs = 0
        away_f5_runs = 0
        first_inning_home_runs = 0
        first_inning_away_runs = 0

        for inn in innings:
            inn_num = inn.get("num", 0)
            h_r = inn.get("home", {}).get("runs", 0) or 0
            a_r = inn.get("away", {}).get("runs", 0) or 0
            if inn_num <= 5:
                home_f5_runs += h_r
                away_f5_runs += a_r
            if inn_num == 1:
                first_inning_home_runs = h_r
                first_inning_away_runs = a_r

        num_innings = len(innings)

        # HP Umpire
        officials = live_data.get("boxscore", {}).get("officials", [])
        hp_umpire = ""
        hp_umpire_id = None
        for off in officials:
            if off.get("officialType") == "Home Plate":
                official = off.get("official", {})
                hp_umpire = official.get("fullName", "")
                hp_umpire_id = official.get("id")
                break

        # Starting pitchers
        boxscore = live_data.get("boxscore", {})
        home_sp_id, home_sp_name = _get_starting_pitcher(boxscore, "home")
        away_sp_id, away_sp_name = _get_starting_pitcher(boxscore, "away")

        game_row = {
            "game_pk": game_pk,
            "date": game_date,
            "home_team": home_team,
            "away_team": away_team,
            "home_abbrev": home_abbrev,
            "away_abbrev": away_abbrev,
            "home_team_id": home_id,
            "away_team_id": away_id,
            "home_runs": home_runs,
            "away_runs": away_runs,
            "home_hits": home_hits,
            "away_hits": away_hits,
            "home_f5_runs": home_f5_runs,
            "away_f5_runs": away_f5_runs,
            "first_inning_home_runs": first_inning_home_runs,
            "first_inning_away_runs": first_inning_away_runs,
            "num_innings": num_innings,
            "is_7_inning_dh": is_7_inning_dh,
            "game_type": game_type,
            "is_postseason": is_postseason,
            "doubleheader": doubleheader,
            "game_num": game_num,
            "venue_name": venue_name,
            "venue_id": venue_id,
            "temp": temp,
            "wind": wind,
            "condition": condition,
            "hp_umpire": hp_umpire,
            "hp_umpire_id": hp_umpire_id,
            "home_sp_id": home_sp_id,
            "home_sp_name": home_sp_name,
            "away_sp_id": away_sp_id,
            "away_sp_name": away_sp_name,
        }

        # Pitcher game logs (starting pitchers only)
        pitcher_rows = []
        for side in ["home", "away"]:
            sp_row = _extract_sp_stats(boxscore, side, game_pk, game_date)
            if sp_row:
                pitcher_rows.append(sp_row)

        # Team batting logs
        batting_rows = []
        for side, abbrev, team_id in [("home", home_abbrev, home_id),
                                       ("away", away_abbrev, away_id)]:
            bat_row = _extract_team_batting(boxscore, side, game_pk, game_date, abbrev, team_id)
            if bat_row:
                batting_rows.append(bat_row)

        return game_row, pitcher_rows, batting_rows

    except Exception as e:
        log.error(f"  Parse error for game {game_meta['game_pk']}: {e}")
        traceback.print_exc()
        return None, [], []


def _get_starting_pitcher(boxscore, side):
    """Find the starting pitcher for a side (whoever has note='SP' or pitched first)."""
    team_data = boxscore.get("teams", {}).get(side, {})
    players = team_data.get("players", {})
    pitchers = team_data.get("pitchers", [])

    # Method 1: Look for gamesStarted >= 1
    for pid in pitchers:
        player_key = f"ID{pid}"
        pdata = players.get(player_key, {})
        pitching = pdata.get("stats", {}).get("pitching", {})
        if pitching.get("gamesStarted", 0) >= 1:
            person = pdata.get("person", {})
            return person.get("id"), person.get("fullName", "")

    # Method 2: First pitcher in the list is usually the starter
    if pitchers:
        pid = pitchers[0]
        player_key = f"ID{pid}"
        pdata = players.get(player_key, {})
        person = pdata.get("person", {})
        return person.get("id"), person.get("fullName", "")

    return None, ""


def _extract_sp_stats(boxscore, side, game_pk, game_date):
    """Extract starting pitcher counting stats from boxscore."""
    team_data = boxscore.get("teams", {}).get(side, {})
    players = team_data.get("players", {})
    pitchers = team_data.get("pitchers", [])
    team_info = team_data.get("team", {})

    # Find the SP
    sp_id = None
    sp_data = None
    for pid in pitchers:
        player_key = f"ID{pid}"
        pdata = players.get(player_key, {})
        pitching = pdata.get("stats", {}).get("pitching", {})
        if pitching.get("gamesStarted", 0) >= 1:
            sp_id = pid
            sp_data = pdata
            break

    # Fallback: first pitcher
    if sp_data is None and pitchers:
        pid = pitchers[0]
        player_key = f"ID{pid}"
        sp_id = pid
        sp_data = players.get(player_key, {})

    if sp_data is None:
        return None

    person = sp_data.get("person", {})
    pitching = sp_data.get("stats", {}).get("pitching", {})

    # Parse innings pitched (e.g., "6.1" means 6 and 1/3)
    ip_str = pitching.get("inningsPitched", "0")
    try:
        ip = _parse_ip(ip_str)
    except ValueError:
        ip = 0.0

    return {
        "game_pk": game_pk,
        "date": game_date,
        "pitcher_id": person.get("id"),
        "pitcher_name": person.get("fullName", ""),
        "team_id": team_info.get("id"),
        "team_abbrev": team_info.get("abbreviation", ""),
        "side": side,
        "ip": ip,
        "ip_str": ip_str,
        "hits": pitching.get("hits", 0),
        "runs": pitching.get("runs", 0),
        "earned_runs": pitching.get("earnedRuns", 0),
        "strikeouts": pitching.get("strikeOuts", 0),
        "walks": pitching.get("baseOnBalls", 0),
        "home_runs": pitching.get("homeRuns", 0),
        "pitches_thrown": pitching.get("pitchesThrown", 0),
        "batters_faced": pitching.get("battersFaced", 0),
        "strikes": pitching.get("strikes", 0),
    }


def _parse_ip(ip_str):
    """Parse innings pitched string like '6.1' -> 6.333..., '5.2' -> 5.667."""
    ip_str = str(ip_str).strip()
    if "." in ip_str:
        whole, frac = ip_str.split(".")
        whole = int(whole)
        frac = int(frac)
        return whole + frac / 3.0
    return float(ip_str)


def _extract_team_batting(boxscore, side, game_pk, game_date, abbrev, team_id):
    """Extract team batting stats from boxscore."""
    team_data = boxscore.get("teams", {}).get(side, {})
    team_stats = team_data.get("teamStats", {}).get("batting", {})

    if not team_stats:
        return None

    return {
        "game_pk": game_pk,
        "date": game_date,
        "team_id": team_id,
        "team_abbrev": abbrev,
        "side": side,
        "at_bats": team_stats.get("atBats", 0),
        "runs": team_stats.get("runs", 0),
        "hits": team_stats.get("hits", 0),
        "doubles": team_stats.get("doubles", 0),
        "triples": team_stats.get("triples", 0),
        "home_runs": team_stats.get("homeRuns", 0),
        "rbi": team_stats.get("rbi", 0),
        "stolen_bases": team_stats.get("stolenBases", 0),
        "strikeouts": team_stats.get("strikeOuts", 0),
        "walks": team_stats.get("baseOnBalls", 0),
        "left_on_base": team_stats.get("leftOnBase", 0),
        "obp": team_stats.get("obp", ""),
        "slg": team_stats.get("slg", ""),
        "ops": team_stats.get("ops", ""),
        "avg": team_stats.get("avg", ""),
    }


# ── Main ────────────────────────────────────────────────────────
def main():
    HISTORICAL_DIR.mkdir(parents=True, exist_ok=True)

    # Phase 1: Get all game PKs
    log.info("=" * 60)
    log.info("Phase 1: Fetching schedules for all seasons")
    log.info("=" * 60)
    all_games = get_all_game_pks()

    # Phase 2: Fetch game feeds (concurrent)
    log.info("=" * 60)
    log.info(f"Phase 2: Fetching game feeds ({NUM_WORKERS} concurrent threads)")
    log.info("=" * 60)

    progress = load_progress()
    fetched_set = set(progress["fetched_pks"])

    remaining = [g for g in all_games if g["game_pk"] not in fetched_set]
    log.info(f"Total games: {len(all_games)}, already fetched: {len(fetched_set)}, "
             f"remaining: {len(remaining)}")

    # Load existing partial results
    game_rows = []
    pitcher_rows = []
    batting_rows = []

    if GAME_RESULTS_FILE.exists():
        existing = pd.read_csv(GAME_RESULTS_FILE)
        game_rows = existing.to_dict("records")
        log.info(f"Loaded {len(game_rows)} existing game rows")
    if PITCHER_LOGS_FILE.exists():
        existing = pd.read_csv(PITCHER_LOGS_FILE)
        pitcher_rows = existing.to_dict("records")
        log.info(f"Loaded {len(pitcher_rows)} existing pitcher rows")
    if BATTING_LOGS_FILE.exists():
        existing = pd.read_csv(BATTING_LOGS_FILE)
        batting_rows = existing.to_dict("records")
        log.info(f"Loaded {len(batting_rows)} existing batting rows")

    fetch_count = 0
    error_count = 0

    # Process in batches of NUM_WORKERS
    batch_size = NUM_WORKERS
    for batch_start in range(0, len(remaining), batch_size):
        batch = remaining[batch_start:batch_start + batch_size]

        # Fetch concurrently
        with ThreadPoolExecutor(max_workers=NUM_WORKERS) as executor:
            futures = {executor.submit(_fetch_one, gm): gm for gm in batch}
            results = []
            for future in as_completed(futures):
                results.append(future.result())

        # Parse sequentially (fast, keeps lists thread-safe)
        for game_meta, feed in results:
            game_pk = game_meta["game_pk"]

            if feed is None:
                error_count += 1
                progress["errors"].append(game_pk)
                progress["fetched_pks"].append(game_pk)
                if error_count > 100:
                    log.error("Too many errors (>100), stopping.")
                    _save_all(game_rows, pitcher_rows, batting_rows, progress)
                    return
                continue

            game_row, p_rows, b_rows = parse_game_feed(feed, game_meta)
            if game_row is not None:
                game_rows.append(game_row)
                pitcher_rows.extend(p_rows)
                batting_rows.extend(b_rows)

            progress["fetched_pks"].append(game_pk)
            fetch_count += 1

        total_done = batch_start + len(batch)
        if total_done % 100 < batch_size:
            pct = total_done / len(remaining) * 100
            log.info(f"  [{total_done}/{len(remaining)}] ({pct:.1f}%) "
                     f"Games: {len(game_rows)}, Pitchers: {len(pitcher_rows)}, "
                     f"Batting: {len(batting_rows)}, Errors: {error_count}")

        # Checkpoint
        if fetch_count >= CHECKPOINT_EVERY and fetch_count % CHECKPOINT_EVERY < batch_size:
            _save_all(game_rows, pitcher_rows, batting_rows, progress)
            log.info(f"  Checkpoint saved ({len(game_rows)} games)")

    # Final save
    _save_all(game_rows, pitcher_rows, batting_rows, progress)

    log.info("=" * 60)
    log.info("Done!")
    log.info(f"  Game results: {len(game_rows)} rows -> {GAME_RESULTS_FILE}")
    log.info(f"  Pitcher logs: {len(pitcher_rows)} rows -> {PITCHER_LOGS_FILE}")
    log.info(f"  Batting logs: {len(batting_rows)} rows -> {BATTING_LOGS_FILE}")
    log.info(f"  Errors: {error_count}")
    log.info("=" * 60)

    # Quick validation
    if game_rows:
        df = pd.DataFrame(game_rows)
        log.info(f"\nValidation:")
        log.info(f"  Date range: {df['date'].min()} to {df['date'].max()}")
        for gt in df["game_type"].unique():
            n = (df["game_type"] == gt).sum()
            log.info(f"  Game type '{gt}': {n}")
        log.info(f"  F5 runs populated: {df['home_f5_runs'].notna().sum()}/{len(df)}")
        log.info(f"  Home SP populated: {df['home_sp_id'].notna().sum()}/{len(df)}")


def _save_all(game_rows, pitcher_rows, batting_rows, progress):
    """Save all outputs and progress."""
    save_progress(progress)
    if game_rows:
        pd.DataFrame(game_rows).to_csv(GAME_RESULTS_FILE, index=False)
    if pitcher_rows:
        pd.DataFrame(pitcher_rows).to_csv(PITCHER_LOGS_FILE, index=False)
    if batting_rows:
        pd.DataFrame(batting_rows).to_csv(BATTING_LOGS_FILE, index=False)


if __name__ == "__main__":
    main()
