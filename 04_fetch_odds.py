"""
04 -- Fetch Odds from The Odds API
=====================================
Pulls current moneyline, run line, and total odds from The Odds API.
MLB's primary market is moneylines (h2h), unlike CBB where spreads dominate.

Markets:
  h2h: Moneyline (primary -- variable juice)
  spreads: Run line (+/- 1.5)
  totals: Over/Under

Outputs:
  data/raw/odds_YYYY-MM-DD.csv
  data/raw/odds_detail_YYYY-MM-DD.json (raw API response)

Requires: ODDS_API_KEY env var
"""

import sys
import json
import requests
import numpy as np
import pandas as pd
from config import (
    RAW_DIR, TODAY,
    ODDS_API_KEY, ODDS_API_BASE, ODDS_SPORT, ODDS_REGIONS, ODDS_MARKETS,
    BOOK_DISPLAY_NAMES, get_logger
)

log = get_logger("04_odds")


def ml_to_implied_prob(ml):
    """Convert American moneyline odds to implied probability."""
    if ml is None or np.isnan(ml):
        return np.nan
    if ml > 0:
        return 100 / (ml + 100)
    else:
        return abs(ml) / (abs(ml) + 100)


def fetch_odds():
    """Fetch odds from The Odds API."""
    if ODDS_API_KEY == "YOUR_ODDS_API_KEY":
        log.error("ODDS_API_KEY not set. Set the env var before running.")
        sys.exit(1)

    log.info(f"Fetching odds for {ODDS_SPORT} on {TODAY}...")

    url = f"{ODDS_API_BASE}/sports/{ODDS_SPORT}/odds/"
    params = {
        "apiKey": ODDS_API_KEY,
        "regions": ODDS_REGIONS,
        "markets": ODDS_MARKETS,
        "oddsFormat": "american",
        "dateFormat": "iso",
    }

    try:
        resp = requests.get(url, params=params, timeout=30)
        resp.raise_for_status()
        data = resp.json()
    except requests.exceptions.HTTPError as e:
        log.error(f"Odds API HTTP error: {e}")
        return None
    except Exception as e:
        log.error(f"Failed to fetch odds: {e}")
        return None

    log.info(f"Received {len(data)} events from Odds API")

    # Save raw JSON
    raw_json_path = RAW_DIR / f"odds_detail_{TODAY}.json"
    raw_json_path.parent.mkdir(parents=True, exist_ok=True)
    with open(raw_json_path, "w") as f:
        json.dump(data, f, indent=2)

    # Parse into structured rows
    rows = []
    for event in data:
        event_id = event.get("id", "")
        sport = event.get("sport_key", "")
        home_team = event.get("home_team", "")
        away_team = event.get("away_team", "")
        commence = event.get("commence_time", "")

        # Collect all bookmaker lines
        ml_home_lines = []
        ml_away_lines = []
        runline_home_lines = []
        total_lines = []
        book_mls = {}

        for book in event.get("bookmakers", []):
            book_key = book.get("key", "")

            for market in book.get("markets", []):
                mkt_key = market.get("key", "")
                outcomes = market.get("outcomes", [])

                if mkt_key == "h2h":
                    # Moneyline
                    for outcome in outcomes:
                        if outcome.get("name") == home_team:
                            ml_home_lines.append(outcome.get("price"))
                            book_mls[f"{book_key}_home_ml"] = outcome.get("price")
                        elif outcome.get("name") == away_team:
                            ml_away_lines.append(outcome.get("price"))
                            book_mls[f"{book_key}_away_ml"] = outcome.get("price")

                elif mkt_key == "spreads":
                    # Run line
                    for outcome in outcomes:
                        if outcome.get("name") == home_team:
                            runline_home_lines.append(outcome.get("point"))

                elif mkt_key == "totals":
                    # Over/under
                    for outcome in outcomes:
                        if outcome.get("name") == "Over":
                            total_lines.append(outcome.get("point"))

        # Compute consensus (median) values
        consensus_ml_home = np.median(ml_home_lines) if ml_home_lines else np.nan
        consensus_ml_away = np.median(ml_away_lines) if ml_away_lines else np.nan
        consensus_runline = np.median(runline_home_lines) if runline_home_lines else np.nan
        consensus_total = np.median(total_lines) if total_lines else np.nan

        rows.append({
            "event_id": event_id,
            "date": TODAY,
            "commence_time": commence,
            "home_team": home_team,
            "away_team": away_team,
            "consensus_ml_home": consensus_ml_home,
            "consensus_ml_away": consensus_ml_away,
            "ml_implied_prob_home": ml_to_implied_prob(consensus_ml_home),
            "ml_implied_prob_away": ml_to_implied_prob(consensus_ml_away),
            "consensus_runline": consensus_runline,
            "consensus_total": consensus_total,
            "num_books": len(event.get("bookmakers", [])),
            "has_odds": len(ml_home_lines) > 0,
            "book_mls_json": json.dumps(book_mls),
        })

    if rows:
        df = pd.DataFrame(rows)
        out_path = RAW_DIR / f"odds_{TODAY}.csv"
        df.to_csv(out_path, index=False)
        log.info(f"Saved odds for {len(df)} games to {out_path}")

        # Report coverage
        with_ml = df[df["has_odds"]].shape[0]
        with_total = df[df["consensus_total"].notna()].shape[0]
        log.info(f"  ML odds: {with_ml}/{len(df)} games")
        log.info(f"  Totals: {with_total}/{len(df)} games")

        return df
    else:
        log.warning("No odds data parsed")
        return None


if __name__ == "__main__":
    fetch_odds()
