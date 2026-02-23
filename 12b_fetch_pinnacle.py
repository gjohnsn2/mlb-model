"""
12b -- Fetch Pinnacle Closing Lines
======================================
Fetches Pinnacle closing moneylines from The Odds API (eu region).
Used for sharp-line validation: does the model's edge hold against
the sharpest book in the market?

Pinnacle is the benchmark for MLB — it takes the highest limits
and has the sharpest lines. If the model beats Pinnacle, the edge
is real.

Outputs:
  data/historical/pinnacle_odds.csv

Requires: ODDS_API_KEY env var
"""

import sys
import json
import requests
import numpy as np
import pandas as pd
from config import (
    HISTORICAL_DIR, ODDS_API_KEY, ODDS_API_BASE, ODDS_SPORT,
    get_logger
)

log = get_logger("12b_pinnacle")


def fetch_pinnacle_odds():
    """Fetch Pinnacle odds from The Odds API (eu region)."""
    if ODDS_API_KEY == "YOUR_ODDS_API_KEY":
        log.error("ODDS_API_KEY not set")
        sys.exit(1)

    log.info("Fetching Pinnacle odds from Odds API (eu region)...")

    url = f"{ODDS_API_BASE}/sports/{ODDS_SPORT}/odds/"
    params = {
        "apiKey": ODDS_API_KEY,
        "regions": "eu",  # Pinnacle is in eu region
        "markets": "h2h",  # Moneylines
        "bookmakers": "pinnacle",
        "oddsFormat": "american",
    }

    try:
        resp = requests.get(url, params=params, timeout=30)
        resp.raise_for_status()
        data = resp.json()
    except Exception as e:
        log.error(f"Failed to fetch Pinnacle odds: {e}")
        return None

    log.info(f"Received {len(data)} events")

    rows = []
    for event in data:
        home_team = event.get("home_team", "")
        away_team = event.get("away_team", "")
        commence = event.get("commence_time", "")

        for book in event.get("bookmakers", []):
            if book.get("key") != "pinnacle":
                continue

            for market in book.get("markets", []):
                if market.get("key") != "h2h":
                    continue

                home_ml = away_ml = None
                for outcome in market.get("outcomes", []):
                    if outcome.get("name") == home_team:
                        home_ml = outcome.get("price")
                    elif outcome.get("name") == away_team:
                        away_ml = outcome.get("price")

                if home_ml is not None and away_ml is not None:
                    rows.append({
                        "date": commence[:10],
                        "home_team": home_team,
                        "away_team": away_team,
                        "pinnacle_ml_home": home_ml,
                        "pinnacle_ml_away": away_ml,
                    })

    if rows:
        df = pd.DataFrame(rows)
        out_path = HISTORICAL_DIR / "pinnacle_odds.csv"
        out_path.parent.mkdir(parents=True, exist_ok=True)

        # Append to existing
        if out_path.exists():
            existing = pd.read_csv(out_path)
            df = pd.concat([existing, df], ignore_index=True)
            df = df.drop_duplicates(subset=["date", "home_team", "away_team"], keep="last")

        df.to_csv(out_path, index=False)
        log.info(f"Saved {len(df)} Pinnacle odds entries to {out_path}")
        return df

    return None


if __name__ == "__main__":
    fetch_pinnacle_odds()
