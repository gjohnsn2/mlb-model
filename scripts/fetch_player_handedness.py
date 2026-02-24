"""
Fetch Player Handedness from MLB Stats API
===========================================
Collects bat side (L/R/S) and pitch hand (L/R) for all unique player IDs
found in batter_game_logs.csv, pitcher_game_logs_mlbapi.csv, and
bullpen_game_logs.csv.

Uses bulk endpoint: /api/v1/people?personIds=ID1,ID2,...
Batches of 50 IDs per request (~100 API calls total).

Output: data/historical/player_handedness.csv
  - Columns: player_id, player_name, bat_side, pitch_hand
"""

import sys
import time
import requests
import pandas as pd
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import HISTORICAL_DIR, get_logger

log = get_logger("fetch_handedness")

# ── Config ──────────────────────────────────────────────────────
BATCH_SIZE = 50
MAX_RETRIES = 3
TIMEOUT = 20

OUTPUT_FILE = HISTORICAL_DIR / "player_handedness.csv"

# Source files for player IDs
BATTER_LOGS = HISTORICAL_DIR / "batter_game_logs.csv"
PITCHER_LOGS = HISTORICAL_DIR / "pitcher_game_logs_mlbapi.csv"
BULLPEN_LOGS = HISTORICAL_DIR / "bullpen_game_logs.csv"


def fetch_people(person_ids):
    """Fetch player info from MLB Stats API bulk endpoint."""
    ids_str = ",".join(str(pid) for pid in person_ids)
    url = f"https://statsapi.mlb.com/api/v1/people?personIds={ids_str}"
    for attempt in range(MAX_RETRIES):
        try:
            resp = requests.get(url, timeout=TIMEOUT)
            resp.raise_for_status()
            return resp.json()
        except requests.exceptions.RequestException as e:
            if attempt < MAX_RETRIES - 1:
                time.sleep(2 ** (attempt + 1))
            else:
                log.warning(f"Failed to fetch {len(person_ids)} players: {e}")
                return None
    return None


def main():
    HISTORICAL_DIR.mkdir(parents=True, exist_ok=True)

    # Collect all unique player IDs from all sources
    all_ids = set()

    if BATTER_LOGS.exists():
        batters = pd.read_csv(BATTER_LOGS)
        batter_ids = batters["batter_id"].dropna().unique()
        all_ids.update(int(x) for x in batter_ids)
        log.info(f"Batter IDs: {len(batter_ids)}")

    if PITCHER_LOGS.exists():
        pitchers = pd.read_csv(PITCHER_LOGS)
        pitcher_ids = pitchers["pitcher_id"].dropna().unique()
        all_ids.update(int(x) for x in pitcher_ids)
        log.info(f"Pitcher IDs: {len(pitcher_ids)}")

    if BULLPEN_LOGS.exists():
        bullpen = pd.read_csv(BULLPEN_LOGS)
        bp_ids = bullpen["pitcher_id"].dropna().unique()
        all_ids.update(int(x) for x in bp_ids)
        log.info(f"Bullpen IDs: {len(bp_ids)}")

    # Load existing data to skip already-fetched players
    existing_ids = set()
    if OUTPUT_FILE.exists():
        existing = pd.read_csv(OUTPUT_FILE)
        existing_ids = set(existing["player_id"].unique())
        log.info(f"Already fetched: {len(existing_ids)}")

    remaining_ids = sorted(all_ids - existing_ids)
    log.info(f"Total unique player IDs: {len(all_ids)}")
    log.info(f"Remaining to fetch: {len(remaining_ids)}")

    if not remaining_ids:
        log.info("All players already fetched!")
        return

    # Fetch in batches
    rows = []
    if OUTPUT_FILE.exists():
        rows = pd.read_csv(OUTPUT_FILE).to_dict("records")

    t_start = time.time()
    batches = [remaining_ids[i:i + BATCH_SIZE] for i in range(0, len(remaining_ids), BATCH_SIZE)]

    for batch_idx, batch in enumerate(batches):
        data = fetch_people(batch)
        if data is None:
            continue

        people = data.get("people", [])
        for person in people:
            bat_side = person.get("batSide", {}).get("code", "R")
            pitch_hand = person.get("pitchHand", {}).get("code", "R")
            rows.append({
                "player_id": person.get("id"),
                "player_name": person.get("fullName", ""),
                "bat_side": bat_side,
                "pitch_hand": pitch_hand,
            })

        if (batch_idx + 1) % 10 == 0:
            pct = (batch_idx + 1) / len(batches) * 100
            log.info(f"  [{batch_idx+1}/{len(batches)}] ({pct:.0f}%) Players: {len(rows)}")

        # Small delay to be respectful
        time.sleep(0.2)

    # Save
    df = pd.DataFrame(rows)
    df = df.drop_duplicates(subset=["player_id"])
    df.to_csv(OUTPUT_FILE, index=False)

    elapsed = time.time() - t_start
    log.info("=" * 60)
    log.info("Done!")
    log.info(f"  Players: {len(df)} -> {OUTPUT_FILE}")
    log.info(f"  Bat side distribution:")
    for side, count in df["bat_side"].value_counts().items():
        log.info(f"    {side}: {count} ({count/len(df):.0%})")
    log.info(f"  Pitch hand distribution:")
    for hand, count in df["pitch_hand"].value_counts().items():
        log.info(f"    {hand}: {count} ({count/len(df):.0%})")
    log.info(f"  Elapsed: {elapsed:.1f}s")
    log.info("=" * 60)


if __name__ == "__main__":
    main()
