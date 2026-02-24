"""
05 -- Build Feature Matrix
============================
Merges FanGraphs, Statcast, schedule, odds, weather, and lineup data.
Engineers all features via the shared feature_engine.
Outputs: data/processed/features_YYYY-MM-DD.csv

This is the most important script -- it's where raw data becomes
something a model can learn from.
"""

import sys
import json
import pandas as pd
import numpy as np
from pathlib import Path
from config import RAW_DIR, PROCESSED_DIR, HISTORICAL_DIR, TODAY, get_logger
from feature_engine import (
    compute_all_features, StartingPitcherComputer, BullpenComputer,
    TeamBattingComputer, RollingFormComputer, get_park_factors,
    _load_venue_locations, _sf,
)

log = get_logger("05_features")


# -- Team name matching -----------------------------------------------
# MLB has only 30 teams, but ESPN, Odds API, FanGraphs, and Baseball Reference
# all use slightly different names. 30 MLB teams means a manageable crosswalk.
TEAM_NAME_MAP = {
    # ESPN display names -> canonical
    "Arizona Diamondbacks": "Arizona Diamondbacks",
    "Atlanta Braves": "Atlanta Braves",
    "Baltimore Orioles": "Baltimore Orioles",
    "Boston Red Sox": "Boston Red Sox",
    "Chicago Cubs": "Chicago Cubs",
    "Chicago White Sox": "Chicago White Sox",
    "Cincinnati Reds": "Cincinnati Reds",
    "Cleveland Guardians": "Cleveland Guardians",
    "Colorado Rockies": "Colorado Rockies",
    "Detroit Tigers": "Detroit Tigers",
    "Houston Astros": "Houston Astros",
    "Kansas City Royals": "Kansas City Royals",
    "Los Angeles Angels": "Los Angeles Angels",
    "Los Angeles Dodgers": "Los Angeles Dodgers",
    "Miami Marlins": "Miami Marlins",
    "Milwaukee Brewers": "Milwaukee Brewers",
    "Minnesota Twins": "Minnesota Twins",
    "New York Mets": "New York Mets",
    "New York Yankees": "New York Yankees",
    "Oakland Athletics": "Oakland Athletics",
    "Philadelphia Phillies": "Philadelphia Phillies",
    "Pittsburgh Pirates": "Pittsburgh Pirates",
    "San Diego Padres": "San Diego Padres",
    "San Francisco Giants": "San Francisco Giants",
    "Seattle Mariners": "Seattle Mariners",
    "St. Louis Cardinals": "St. Louis Cardinals",
    "Tampa Bay Rays": "Tampa Bay Rays",
    "Texas Rangers": "Texas Rangers",
    "Toronto Blue Jays": "Toronto Blue Jays",
    "Washington Nationals": "Washington Nationals",

    # Odds API variations
    "Arizona D-backs": "Arizona Diamondbacks",
    "D-backs": "Arizona Diamondbacks",
    "Chi Cubs": "Chicago Cubs",
    "Chi White Sox": "Chicago White Sox",
    "Cleveland Indians": "Cleveland Guardians",  # Pre-2022
    "LA Angels": "Los Angeles Angels",
    "LA Dodgers": "Los Angeles Dodgers",
    "NY Mets": "New York Mets",
    "NY Yankees": "New York Yankees",
    "SD Padres": "San Diego Padres",
    "SF Giants": "San Francisco Giants",
    "TB Rays": "Tampa Bay Rays",
    "St Louis Cardinals": "St. Louis Cardinals",
    "Saint Louis Cardinals": "St. Louis Cardinals",

    # FanGraphs abbreviations
    "ARI": "Arizona Diamondbacks",
    "ATL": "Atlanta Braves",
    "BAL": "Baltimore Orioles",
    "BOS": "Boston Red Sox",
    "CHC": "Chicago Cubs",
    "CHW": "Chicago White Sox",
    "CWS": "Chicago White Sox",
    "CIN": "Cincinnati Reds",
    "CLE": "Cleveland Guardians",
    "COL": "Colorado Rockies",
    "DET": "Detroit Tigers",
    "HOU": "Houston Astros",
    "KCR": "Kansas City Royals",
    "KC":  "Kansas City Royals",
    "LAA": "Los Angeles Angels",
    "LAD": "Los Angeles Dodgers",
    "MIA": "Miami Marlins",
    "MIL": "Milwaukee Brewers",
    "MIN": "Minnesota Twins",
    "NYM": "New York Mets",
    "NYY": "New York Yankees",
    "OAK": "Oakland Athletics",
    "PHI": "Philadelphia Phillies",
    "PIT": "Pittsburgh Pirates",
    "SDP": "San Diego Padres",
    "SD":  "San Diego Padres",
    "SFG": "San Francisco Giants",
    "SF":  "San Francisco Giants",
    "SEA": "Seattle Mariners",
    "STL": "St. Louis Cardinals",
    "TBR": "Tampa Bay Rays",
    "TB":  "Tampa Bay Rays",
    "TEX": "Texas Rangers",
    "TOR": "Toronto Blue Jays",
    "WSH": "Washington Nationals",
    "WSN": "Washington Nationals",
}


def normalize_team_name(name):
    """Normalize team name to canonical form."""
    if not name or pd.isna(name):
        return ""

    name = str(name).strip()

    # Direct lookup
    if name in TEAM_NAME_MAP:
        return TEAM_NAME_MAP[name]

    # Case-insensitive lookup
    for key, val in TEAM_NAME_MAP.items():
        if key.lower() == name.lower():
            return val

    log.warning(f"Unrecognized team name: '{name}'")
    return name


def load_schedule():
    """Load today's schedule."""
    path = RAW_DIR / f"schedule_{TODAY}.csv"
    if not path.exists():
        log.error(f"Schedule not found: {path}")
        log.error("Run 03_fetch_schedule.py first")
        sys.exit(1)

    df = pd.read_csv(path)
    log.info(f"Loaded {len(df)} games from schedule")
    return df


def load_odds():
    """Load today's odds."""
    path = RAW_DIR / f"odds_{TODAY}.csv"
    if not path.exists():
        log.warning(f"Odds not found: {path}")
        return pd.DataFrame()

    df = pd.read_csv(path)
    log.info(f"Loaded odds for {len(df)} games")
    return df


def load_lineups():
    """Load today's confirmed lineups."""
    path = RAW_DIR / f"lineups_{TODAY}.csv"
    if not path.exists():
        log.warning(f"Lineups not found: {path}")
        return pd.DataFrame()

    df = pd.read_csv(path)
    log.info(f"Loaded lineups for {len(df)} games")
    return df


def load_weather():
    """Load today's weather data."""
    path = RAW_DIR / f"weather_{TODAY}.csv"
    if not path.exists():
        log.warning(f"Weather not found: {path}")
        return pd.DataFrame()

    df = pd.read_csv(path)
    log.info(f"Loaded weather for {len(df)} games")
    return df


def build_features():
    """Build the daily feature matrix."""
    schedule = load_schedule()
    odds = load_odds()
    lineups = load_lineups()
    weather = load_weather()
    venue_locations = _load_venue_locations()

    rows = []
    matched_odds = 0

    for _, game in schedule.iterrows():
        home_team = normalize_team_name(game.get("home_team", ""))
        away_team = normalize_team_name(game.get("away_team", ""))
        game_id = game.get("game_id", "")
        game_date = TODAY

        # Match odds
        odds_row = {}
        if not odds.empty:
            # Try matching by team names
            for _, o in odds.iterrows():
                o_home = normalize_team_name(o.get("home_team", ""))
                o_away = normalize_team_name(o.get("away_team", ""))
                if o_home == home_team and o_away == away_team:
                    odds_row = {
                        "consensus_ml_home": _sf(o.get("consensus_ml_home")),
                        "consensus_ml_away": _sf(o.get("consensus_ml_away")),
                        "ml_implied_prob_home": _sf(o.get("ml_implied_prob_home")),
                        "ml_implied_prob_away": _sf(o.get("ml_implied_prob_away")),
                        "consensus_total": _sf(o.get("consensus_total")),
                        "consensus_runline": _sf(o.get("consensus_runline")),
                        "num_books": _sf(o.get("num_books")),
                        "has_odds": bool(o.get("has_odds", False)),
                        "book_mls_json": o.get("book_mls_json", "{}"),
                    }
                    matched_odds += 1
                    break

        # Match lineups (for SP info)
        sp_info = {}
        if not lineups.empty:
            for _, l in lineups.iterrows():
                l_home = normalize_team_name(l.get("home_team", ""))
                if l_home == home_team:
                    sp_info = {
                        "home_sp_id": l.get("home_sp_id"),
                        "home_sp_name": l.get("home_sp_name", ""),
                        "away_sp_id": l.get("away_sp_id"),
                        "away_sp_name": l.get("away_sp_name", ""),
                    }
                    break

        # Match weather
        wx = {}
        if not weather.empty:
            for _, w in weather.iterrows():
                w_home = normalize_team_name(w.get("home_team", ""))
                if w_home == home_team:
                    wx = {
                        "temperature": _sf(w.get("temperature")),
                        "wind_speed": _sf(w.get("wind_speed")),
                        "wind_direction_factor": _sf(w.get("wind_direction_factor")),
                        "humidity": _sf(w.get("humidity")),
                        "precipitation_prob": _sf(w.get("precipitation_prob")),
                        "indoor": bool(w.get("indoor", False)),
                    }
                    break

        # Compute features
        # Note: SP stats, batting stats, form stats would come from their
        # respective Computer classes once data is loaded. For the initial
        # pipeline, we compute what we can from available data.
        park = get_park_factors(home_team)

        features = compute_all_features(
            home_team=home_team,
            away_team=away_team,
            game_date=game_date,
            park_factors=park,
            weather=wx,
            odds=odds_row,
            venue_locations=venue_locations,
        )

        # Add metadata
        features["game_id"] = game_id
        features["date"] = game_date
        features["home_team"] = home_team
        features["away_team"] = away_team
        features["venue"] = game.get("venue", "")
        features["game_time"] = game.get("game_time", "")
        features.update(sp_info)
        features["book_mls_json"] = odds_row.get("book_mls_json", "{}")

        rows.append(features)

    df = pd.DataFrame(rows)
    log.info(f"Built feature matrix: {len(df)} games, {len(df.columns)} columns")
    log.info(f"Matched odds for {matched_odds}/{len(schedule)} games")

    # Save
    out_path = PROCESSED_DIR / f"features_{TODAY}.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    log.info(f"Saved features to {out_path}")

    return df


if __name__ == "__main__":
    build_features()
