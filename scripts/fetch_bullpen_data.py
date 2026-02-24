"""
Fetch Bullpen Data from MLB Stats API
======================================
Re-fetches game boxscores to extract ALL non-SP pitcher stats per game.
Uses the lighter boxscore-only endpoint (/api/v1/game/{pk}/boxscore).

Uses existing game_pks from game_results_all.csv (no schedule re-fetch).
4 concurrent threads, checkpointing every 200 games.

Output: data/historical/bullpen_game_logs.csv
  - One row per reliever appearance (pitcher_id, game_pk, side, IP, H, R, ER, K, BB, HR)
  - Excludes the starting pitcher (identified by gamesStarted >= 1 or first in list)
"""

import sys
import json
import time
import requests
import pandas as pd
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import HISTORICAL_DIR, get_logger

log = get_logger("fetch_bullpen")

# ── Config ──────────────────────────────────────────────────────
CHECKPOINT_EVERY = 200
MAX_RETRIES = 3
TIMEOUT = 20
NUM_WORKERS = 4

GAME_RESULTS_FILE = HISTORICAL_DIR / "game_results_all.csv"
OUTPUT_FILE = HISTORICAL_DIR / "bullpen_game_logs.csv"
PROGRESS_FILE = HISTORICAL_DIR / "bullpen_fetch_progress.json"


def load_progress():
    if PROGRESS_FILE.exists():
        with open(PROGRESS_FILE) as f:
            return json.load(f)
    return {"fetched_pks": [], "errors": []}


def save_progress(progress):
    with open(PROGRESS_FILE, "w") as f:
        json.dump(progress, f)


def _parse_ip(ip_str):
    """Parse innings pitched string like '6.1' -> 6.333."""
    ip_str = str(ip_str).strip()
    if "." in ip_str:
        whole, frac = ip_str.split(".")
        return int(whole) + int(frac) / 3.0
    return float(ip_str)


def fetch_boxscore(game_pk):
    """Fetch boxscore from MLB Stats API with retries."""
    url = f"https://statsapi.mlb.com/api/v1/game/{game_pk}/boxscore"
    for attempt in range(MAX_RETRIES):
        try:
            resp = requests.get(url, timeout=TIMEOUT)
            if resp.status_code == 404:
                return None
            resp.raise_for_status()
            return resp.json()
        except requests.exceptions.RequestException:
            if attempt < MAX_RETRIES - 1:
                time.sleep(2 ** (attempt + 1))
            else:
                return None
    return None


def _fetch_one(game_pk):
    """Fetch boxscore for a single game. Returns (game_pk, boxscore)."""
    box = fetch_boxscore(game_pk)
    return game_pk, box


def extract_bullpen(boxscore, game_pk, game_date):
    """
    Extract all non-SP pitcher stats from a boxscore.
    Returns list of reliever row dicts.
    """
    rows = []
    teams = boxscore.get("teams", {})

    for side in ["home", "away"]:
        team_data = teams.get(side, {})
        players = team_data.get("players", {})
        pitchers = team_data.get("pitchers", [])
        team_info = team_data.get("team", {})
        team_id = team_info.get("id")

        if not pitchers:
            continue

        # Identify the SP: gamesStarted >= 1, or first pitcher
        sp_id = None
        for pid in pitchers:
            pdata = players.get(f"ID{pid}", {})
            pitching = pdata.get("stats", {}).get("pitching", {})
            if pitching.get("gamesStarted", 0) >= 1:
                sp_id = pid
                break
        if sp_id is None and pitchers:
            sp_id = pitchers[0]

        # Extract all NON-SP pitchers
        for pid in pitchers:
            if pid == sp_id:
                continue

            pdata = players.get(f"ID{pid}", {})
            pitching = pdata.get("stats", {}).get("pitching", {})
            person = pdata.get("person", {})

            # Skip if no pitching stats recorded
            ip_str = pitching.get("inningsPitched", "0")
            try:
                ip = _parse_ip(ip_str)
            except (ValueError, IndexError):
                ip = 0.0

            if ip == 0.0 and pitching.get("battersFaced", 0) == 0:
                continue

            rows.append({
                "game_pk": game_pk,
                "date": game_date,
                "pitcher_id": person.get("id"),
                "pitcher_name": person.get("fullName", ""),
                "team_id": team_id,
                "side": side,
                "ip": ip,
                "hits": pitching.get("hits", 0),
                "runs": pitching.get("runs", 0),
                "earned_runs": pitching.get("earnedRuns", 0),
                "strikeouts": pitching.get("strikeOuts", 0),
                "walks": pitching.get("baseOnBalls", 0),
                "home_runs": pitching.get("homeRuns", 0),
                "pitches_thrown": pitching.get("pitchesThrown", 0),
                "batters_faced": pitching.get("battersFaced", 0),
            })

    return rows


def main():
    HISTORICAL_DIR.mkdir(parents=True, exist_ok=True)

    # Load game PKs from existing results
    if not GAME_RESULTS_FILE.exists():
        log.error(f"Game results not found: {GAME_RESULTS_FILE}")
        log.error("Run scripts/fetch_historical_games.py first")
        sys.exit(1)

    games = pd.read_csv(GAME_RESULTS_FILE)
    all_pks = games[["game_pk", "date"]].to_dict("records")
    log.info(f"Total games from results file: {len(all_pks)}")

    # Load progress
    progress = load_progress()
    fetched_set = set(progress["fetched_pks"])

    remaining = [g for g in all_pks if g["game_pk"] not in fetched_set]
    log.info(f"Already fetched: {len(fetched_set)}, remaining: {len(remaining)}")

    # Load existing partial results
    all_rows = []
    if OUTPUT_FILE.exists():
        existing = pd.read_csv(OUTPUT_FILE)
        all_rows = existing.to_dict("records")
        log.info(f"Loaded {len(all_rows)} existing bullpen rows")

    # Build date lookup for game_pk -> date
    date_lookup = {g["game_pk"]: g["date"] for g in all_pks}

    fetch_count = 0
    error_count = 0
    t_start = time.time()

    for batch_start in range(0, len(remaining), NUM_WORKERS):
        batch = remaining[batch_start:batch_start + NUM_WORKERS]
        batch_pks = [g["game_pk"] for g in batch]

        with ThreadPoolExecutor(max_workers=NUM_WORKERS) as executor:
            futures = {executor.submit(_fetch_one, pk): pk for pk in batch_pks}
            results = []
            for future in as_completed(futures):
                results.append(future.result())

        for game_pk, boxscore in results:
            progress["fetched_pks"].append(game_pk)

            if boxscore is None:
                error_count += 1
                progress["errors"].append(game_pk)
                if error_count > 200:
                    log.error("Too many errors (>200), stopping.")
                    _save(all_rows, progress)
                    return
                continue

            game_date = date_lookup.get(game_pk, "")
            reliever_rows = extract_bullpen(boxscore, game_pk, game_date)
            all_rows.extend(reliever_rows)
            fetch_count += 1

        total_done = batch_start + len(batch)
        if total_done % 500 < NUM_WORKERS:
            elapsed = time.time() - t_start
            rate = total_done / elapsed if elapsed > 0 else 0
            eta_min = (len(remaining) - total_done) / rate / 60 if rate > 0 else 0
            pct = total_done / len(remaining) * 100
            log.info(f"  [{total_done}/{len(remaining)}] ({pct:.1f}%) "
                     f"Rows: {len(all_rows)}, Errors: {error_count}, "
                     f"Rate: {rate:.1f}/s, ETA: {eta_min:.0f}min")

        # Checkpoint
        if fetch_count > 0 and fetch_count % CHECKPOINT_EVERY < NUM_WORKERS:
            _save(all_rows, progress)

    # Final save
    _save(all_rows, progress)

    elapsed = time.time() - t_start
    log.info("=" * 60)
    log.info("Done!")
    log.info(f"  Bullpen rows: {len(all_rows)} -> {OUTPUT_FILE}")
    log.info(f"  Errors: {error_count}")
    log.info(f"  Elapsed: {elapsed / 60:.1f} min")
    log.info("=" * 60)

    # Quick validation
    if all_rows:
        df = pd.DataFrame(all_rows)
        games_covered = df["game_pk"].nunique()
        log.info(f"  Games with bullpen data: {games_covered}/{len(all_pks)}")
        log.info(f"  Avg relievers per game: {len(df) / games_covered:.1f}")
        log.info(f"  Unique relievers: {df['pitcher_id'].nunique()}")


def _save(all_rows, progress):
    """Save output and progress."""
    save_progress(progress)
    if all_rows:
        pd.DataFrame(all_rows).to_csv(OUTPUT_FILE, index=False)


if __name__ == "__main__":
    main()
