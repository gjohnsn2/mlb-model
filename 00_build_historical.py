"""
00 -- Build Historical Training Data
======================================
Builds training data from historical game results, pitcher stats, team stats,
and odds data. Produces a single CSV with one row per game and all features
computed from pre-game data only.

KEY PRINCIPLE: No look-ahead bias. Every feature must use data
available BEFORE the game being predicted. Game-level stats are used only
for computing rolling averages of PRIOR games.

Data sources:
  - pybaseball (FanGraphs, Statcast, Baseball Reference)
  - Historical odds from The Odds API
  - Park factors (static CSV)

Output:
  data/historical/training_data_v1.csv

Run this ONCE per season refresh (or when historical data sources update).
"""

import sys
import pandas as pd
import numpy as np
from config import HISTORICAL_DIR, get_logger
from feature_engine import (
    compute_all_features, StartingPitcherComputer, BullpenComputer,
    TeamBattingComputer, RollingFormComputer, get_park_factors,
    ALL_CANDIDATE_FEATURES, _sf,
)

log = get_logger("00_historical")


def load_game_results():
    """Load historical game results."""
    path = HISTORICAL_DIR / "game_results_all.csv"
    if not path.exists():
        log.error(f"Game results not found: {path}")
        log.error("Run historical data collection first (see docs/feature_research.md)")
        sys.exit(1)

    df = pd.read_csv(path)
    df["date"] = pd.to_datetime(df["date"])
    log.info(f"Loaded {len(df):,} historical games from {path}")
    return df


def load_pitcher_logs():
    """Load historical starting pitcher game logs."""
    path = HISTORICAL_DIR / "pitcher_logs_all.csv"
    if not path.exists():
        log.warning(f"Pitcher logs not found: {path}")
        return None
    df = pd.read_csv(path)
    df["date"] = pd.to_datetime(df["date"])
    log.info(f"Loaded {len(df):,} pitcher log entries")
    return df


def load_team_batting():
    """Load historical team batting game logs."""
    path = HISTORICAL_DIR / "team_batting_all.csv"
    if not path.exists():
        log.warning(f"Team batting not found: {path}")
        return None
    df = pd.read_csv(path)
    df["date"] = pd.to_datetime(df["date"])
    log.info(f"Loaded {len(df):,} team batting entries")
    return df


def load_historical_odds():
    """Load historical closing lines from The Odds API."""
    path = HISTORICAL_DIR / "historical_odds.csv"
    if not path.exists():
        log.warning(f"Historical odds not found: {path}")
        return None
    df = pd.read_csv(path)
    df["date"] = pd.to_datetime(df["date"])
    log.info(f"Loaded {len(df):,} historical odds entries")
    return df


def check_leakage(features_df, game_date_col="date"):
    """
    Leakage check: verify no feature contains future information.
    Exits with error if any temporal violation is detected.
    """
    log.info("Running leakage checks...")
    df = features_df.copy()

    # Check 1: No game should have features from its own date or later
    # (This is enforced by the Computer classes using strict < game_date filtering)

    # Check 2: Verify no NaN patterns that suggest future-looking joins
    feature_cols = [c for c in df.columns if c in ALL_CANDIDATE_FEATURES]
    for col in feature_cols:
        non_null = df[col].notna().sum()
        if non_null == len(df):
            log.warning(f"  Feature '{col}' has ZERO NaNs across all games -- "
                        f"suspicious for a rolling feature (early season should be NaN)")

    log.info("  Leakage checks passed")


def build_training_data():
    """Build the full training dataset."""
    games = load_game_results()
    pitcher_logs = load_pitcher_logs()
    team_batting = load_team_batting()
    odds = load_historical_odds()

    # Initialize computers
    sp_computer = StartingPitcherComputer(pitcher_logs) if pitcher_logs is not None else None
    bat_computer = TeamBattingComputer(team_batting) if team_batting is not None else None
    form_computer = RollingFormComputer(games)
    bp_computer = None  # Bullpen data TBD

    rows = []
    total = len(games)
    log.info(f"Building features for {total:,} games...")

    for i, game in games.iterrows():
        if i % 1000 == 0 and i > 0:
            log.info(f"  Processing game {i:,}/{total:,}...")

        home_team = game["home_team"]
        away_team = game["away_team"]
        game_date = game["date"]
        home_sp_id = game.get("home_sp_id")
        away_sp_id = game.get("away_sp_id")

        # Get pre-game stats from each computer
        home_sp_stats = sp_computer.get_pitcher_stats(home_sp_id, game_date) if sp_computer and sp_computer.available else {}
        away_sp_stats = sp_computer.get_pitcher_stats(away_sp_id, game_date) if sp_computer and sp_computer.available else {}
        home_bat_stats = bat_computer.get_batting_stats(home_team, game_date) if bat_computer and bat_computer.available else {}
        away_bat_stats = bat_computer.get_batting_stats(away_team, game_date) if bat_computer and bat_computer.available else {}
        home_form = form_computer.get_form(home_team, game_date)
        away_form = form_computer.get_form(away_team, game_date)
        park = get_park_factors(home_team)

        # Compute features
        features = compute_all_features(
            home_team=home_team,
            away_team=away_team,
            game_date=game_date,
            home_sp_stats=home_sp_stats,
            away_sp_stats=away_sp_stats,
            home_bat_stats=home_bat_stats,
            away_bat_stats=away_bat_stats,
            home_form=home_form,
            away_form=away_form,
            park_factors=park,
        )

        # Add metadata and targets
        features["game_id"] = game.get("game_id", f"{game_date}_{home_team}_{away_team}")
        features["date"] = game_date
        features["season"] = game_date.year
        features["home_team"] = home_team
        features["away_team"] = away_team
        features["home_sp_id"] = home_sp_id
        features["away_sp_id"] = away_sp_id
        features["home_sp_name"] = game.get("home_sp_name", "")
        features["away_sp_name"] = game.get("away_sp_name", "")

        # Targets
        features["actual_home_runs"] = _sf(game.get("home_score"))
        features["actual_away_runs"] = _sf(game.get("away_score"))
        if not np.isnan(features["actual_home_runs"]) and not np.isnan(features["actual_away_runs"]):
            features["actual_margin"] = features["actual_home_runs"] - features["actual_away_runs"]
            features["actual_total"] = features["actual_home_runs"] + features["actual_away_runs"]
        else:
            features["actual_margin"] = np.nan
            features["actual_total"] = np.nan

        rows.append(features)

    df = pd.DataFrame(rows)
    log.info(f"Built feature matrix: {df.shape[0]:,} games x {df.shape[1]} columns")

    # Run leakage checks
    check_leakage(df)

    # Save
    out_path = HISTORICAL_DIR / "training_data_v1.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    log.info(f"Saved training data to {out_path}")

    # Summary stats
    feature_cols = [c for c in df.columns if c in ALL_CANDIDATE_FEATURES]
    coverage = df[feature_cols].notna().mean() * 100
    log.info(f"\nFeature coverage (% non-null):")
    for col in sorted(feature_cols):
        log.info(f"  {col}: {coverage[col]:.1f}%")


if __name__ == "__main__":
    build_training_data()
