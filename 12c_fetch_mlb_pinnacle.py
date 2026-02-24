"""
12c — Fetch Pinnacle MLB Closing Lines (EU region)
====================================================
Fetches historical Pinnacle H2H (moneyline) and total lines from The Odds API
eu region for MLB games. Pinnacle is the sharpest book — comparing model
performance against Pinnacle vs retail consensus validates edge quality.

Uses dates from historical_mlb_odds.csv (Odds API data, 2020+).
Cost: ~10 credits per date (h2h,totals markets, eu region).
Estimated: 1,284 dates × 10 = ~12,840 credits.

Output: data/historical/pinnacle_mlb_odds.csv

Run: source .env && python3 12c_fetch_mlb_pinnacle.py
"""

import sys
import json
import time
import requests
import pandas as pd
from pathlib import Path
from config import HISTORICAL_DIR, ODDS_API_KEY, ODDS_API_BASE, get_logger

log = get_logger("12c_mlb_pinnacle")

MLB_SPORT = "baseball_mlb"
PROGRESS_FILE = HISTORICAL_DIR / "pinnacle_mlb_progress.json"
OUTPUT_FILE = HISTORICAL_DIR / "pinnacle_mlb_odds.csv"
MLB_ODDS_FILE = HISTORICAL_DIR / "historical_mlb_odds.csv"

REQUEST_DELAY = 0.5
MAX_RETRIES = 3
CHECKPOINT_EVERY = 25


def load_progress():
    if PROGRESS_FILE.exists():
        with open(PROGRESS_FILE) as f:
            return json.load(f)
    return {"dates_fetched": [], "credits_used": 0}


def save_progress(progress):
    HISTORICAL_DIR.mkdir(parents=True, exist_ok=True)
    with open(PROGRESS_FILE, "w") as f:
        json.dump(progress, f, indent=2)


def get_target_dates():
    """Get dates from historical MLB odds file."""
    if not MLB_ODDS_FILE.exists():
        log.error(f"MLB odds file not found: {MLB_ODDS_FILE}")
        sys.exit(1)

    odds = pd.read_csv(MLB_ODDS_FILE)
    all_dates = sorted(odds["fetch_date"].unique())
    log.info(f"Target dates: {len(all_dates)} ({all_dates[0]} to {all_dates[-1]}), "
             f"~{len(all_dates) * 10:,} credits")
    return all_dates


def fetch_pinnacle_for_date(date_str):
    """Fetch Pinnacle H2H + totals from eu region for a given date."""
    iso_date = f"{date_str}T23:00:00Z"

    url = f"{ODDS_API_BASE}/historical/sports/{MLB_SPORT}/odds"
    params = {
        "apiKey": ODDS_API_KEY,
        "regions": "eu",
        "markets": "h2h,totals",
        "oddsFormat": "american",
        "date": iso_date,
    }

    for attempt in range(MAX_RETRIES):
        try:
            resp = requests.get(url, params=params, timeout=30)

            if resp.status_code == 429:
                wait = 2 ** (attempt + 1)
                log.warning(f"Rate limited, waiting {wait}s...")
                time.sleep(wait)
                continue

            if resp.status_code == 422:
                log.warning(f"  {date_str}: 422 — no data")
                return [], 0

            resp.raise_for_status()

            data = resp.json()
            events = data.get("data", [])
            credits = int(resp.headers.get("x-requests-last", 0))

            games = []
            for event in events:
                game = {
                    "home_team": event["home_team"],
                    "away_team": event["away_team"],
                    "commence_time": event["commence_time"],
                    "fetch_date": date_str,
                }

                # Extract Pinnacle lines
                for book in event.get("bookmakers", []):
                    if book["key"] == "pinnacle":
                        for market in book.get("markets", []):
                            if market["key"] == "h2h":
                                for outcome in market["outcomes"]:
                                    if outcome["name"] == event["home_team"]:
                                        game["pinnacle_h2h_home"] = outcome.get("price")
                                    elif outcome["name"] == event["away_team"]:
                                        game["pinnacle_h2h_away"] = outcome.get("price")
                            elif market["key"] == "totals":
                                for outcome in market["outcomes"]:
                                    if outcome["name"] == "Over":
                                        game["pinnacle_total"] = outcome.get("point")
                                        game["pinnacle_total_over_price"] = outcome.get("price")
                                    elif outcome["name"] == "Under":
                                        game["pinnacle_total_under_price"] = outcome.get("price")

                games.append(game)

            return games, credits

        except requests.exceptions.RequestException as e:
            if attempt < MAX_RETRIES - 1:
                wait = 2 ** (attempt + 1)
                log.warning(f"  {date_str}: error ({e}), retry in {wait}s...")
                time.sleep(wait)
            else:
                log.error(f"  {date_str}: failed after {MAX_RETRIES} attempts")
                return None, 0

    return None, 0


def main():
    if "YOUR_ODDS" in ODDS_API_KEY:
        log.error("ODDS_API_KEY not set! Run: source .env")
        sys.exit(1)

    all_dates = get_target_dates()
    progress = load_progress()
    fetched_set = set(progress["dates_fetched"])
    remaining = [d for d in all_dates if d not in fetched_set]

    log.info(f"Total dates: {len(all_dates)}, already fetched: {len(fetched_set)}, "
             f"remaining: {len(remaining)}")

    if not remaining:
        log.info(f"All dates fetched! Output: {OUTPUT_FILE}")
        return

    est_credits = len(remaining) * 10
    log.info(f"Estimated credits: ~{est_credits:,} ({len(remaining)} dates × 10)")

    # Load existing partial results
    all_games = []
    if OUTPUT_FILE.exists():
        existing = pd.read_csv(OUTPUT_FILE)
        all_games = existing.to_dict("records")
        log.info(f"Loaded {len(all_games)} existing rows")

    total_credits = progress["credits_used"]
    fetched_count = 0

    for i, date_str in enumerate(remaining):
        games, credits = fetch_pinnacle_for_date(date_str)

        if games is None:
            continue

        all_games.extend(games)
        total_credits += credits
        fetched_count += 1
        progress["dates_fetched"].append(date_str)
        progress["credits_used"] = total_credits

        pinnacle_h2h = sum(1 for g in games if g.get("pinnacle_h2h_home") is not None)
        pinnacle_total = sum(1 for g in games if g.get("pinnacle_total") is not None)
        pct = (i + 1) / len(remaining) * 100
        log.info(f"  [{i+1}/{len(remaining)}] {date_str}: "
                 f"{len(games)} events, {pinnacle_h2h} H2H / {pinnacle_total} totals "
                 f"({pct:.0f}%, credits: {total_credits})")

        if fetched_count % CHECKPOINT_EVERY == 0:
            save_progress(progress)
            pd.DataFrame(all_games).to_csv(OUTPUT_FILE, index=False)
            log.info(f"  Checkpoint: {len(all_games)} rows saved")

        time.sleep(REQUEST_DELAY)

    # Final save
    save_progress(progress)
    df = pd.DataFrame(all_games)
    if not df.empty:
        df.to_csv(OUTPUT_FILE, index=False)
        with_h2h = df["pinnacle_h2h_home"].notna().sum()
        with_total = df["pinnacle_total"].notna().sum()
        log.info(f"\nDone! {len(df)} rows -> {OUTPUT_FILE}")
        log.info(f"  Pinnacle H2H: {with_h2h}/{len(df)} ({with_h2h/len(df)*100:.0f}%)")
        log.info(f"  Pinnacle total: {with_total}/{len(df)} ({with_total/len(df)*100:.0f}%)")
        log.info(f"  Credits used: {total_credits}")
    else:
        log.warning("No data fetched")


if __name__ == "__main__":
    main()
