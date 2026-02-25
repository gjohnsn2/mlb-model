"""
05 -- Build Daily Feature Matrix
===================================
Loads historical data files + today's schedule/lineups, computes all 91
no-market features matching the training pipeline (00_build_mlb_historical.py).

Strategy: Append today's games to historical game_results, then run the same
feature computation functions on the combined DataFrame. Extract today's rows
at the end. This guarantees perfect feature parity with training data.

Inputs:
  data/historical/game_results_all.csv
  data/historical/pitcher_game_logs_mlbapi.csv
  data/historical/statcast_pitcher_games.csv
  data/historical/team_batting_game_logs.csv
  data/historical/bullpen_game_logs.csv
  data/historical/batter_game_logs.csv
  data/historical/player_handedness.csv
  data/raw/schedule_{TODAY}.csv
  data/raw/lineups_{TODAY}.csv

Outputs:
  data/processed/features_{TODAY}.csv

Run:
  python3 05_build_features.py
  MLB_DATE=2025-04-15 python3 05_build_features.py  # Historical date
"""

import sys
import importlib
import pandas as pd
import numpy as np
from pathlib import Path
from config import (
    RAW_DIR, PROCESSED_DIR, HISTORICAL_DIR, TODAY,
    MLB_CANDIDATE_FEATURES,
    get_logger
)

log = get_logger("05_features")

# Import compute functions from the historical builder
_mod_00 = importlib.import_module("00_build_mlb_historical")
compute_sp_features = _mod_00.compute_sp_features
compute_team_batting_features = _mod_00.compute_team_batting_features
compute_context_features = _mod_00.compute_context_features
compute_rest_features = _mod_00.compute_rest_features
compute_momentum_features = _mod_00.compute_momentum_features
compute_bullpen_features = _mod_00.compute_bullpen_features
compute_bullpen_availability_features = _mod_00.compute_bullpen_availability_features
compute_travel_features = _mod_00.compute_travel_features
compute_schedule_context_features = _mod_00.compute_schedule_context_features
compute_lineup_features = _mod_00.compute_lineup_features
compute_opponent_adjusted_features = _mod_00.compute_opponent_adjusted_features
compute_handedness_split_features = _mod_00.compute_handedness_split_features
compute_pitch_type_features = _mod_00.compute_pitch_type_features
compute_interaction_features = _mod_00.compute_interaction_features
build_game_opponent_lookup = _mod_00.build_game_opponent_lookup
StartingPitcherComputer = _mod_00.StartingPitcherComputer

# Historical data files
GAMES_FILE = HISTORICAL_DIR / "game_results_all.csv"
PITCHER_LOGS_FILE = HISTORICAL_DIR / "pitcher_game_logs_mlbapi.csv"
STATCAST_FILE = HISTORICAL_DIR / "statcast_pitcher_games.csv"
BATTING_LOGS_FILE = HISTORICAL_DIR / "team_batting_game_logs.csv"
BULLPEN_LOGS_FILE = HISTORICAL_DIR / "bullpen_game_logs.csv"
BATTER_LOGS_FILE = HISTORICAL_DIR / "batter_game_logs.csv"
HANDEDNESS_FILE = HISTORICAL_DIR / "player_handedness.csv"
BATTER_PITCH_TYPE_FILE = HISTORICAL_DIR / "batter_pitch_type_stats.csv"


def load_schedule():
    """Load today's schedule from MLB Stats API output."""
    path = RAW_DIR / f"schedule_{TODAY}.csv"
    if not path.exists():
        log.error(f"Schedule not found: {path}")
        log.error("Run 03_fetch_schedule.py first")
        sys.exit(1)

    df = pd.read_csv(path)
    log.info(f"Loaded {len(df)} games from schedule")
    return df


def load_lineups():
    """Load today's confirmed lineups."""
    path = RAW_DIR / f"lineups_{TODAY}.csv"
    if not path.exists():
        log.warning(f"Lineups not found: {path}. SP info may be from schedule.")
        return pd.DataFrame()

    df = pd.read_csv(path)
    log.info(f"Loaded lineups for {len(df)} games")
    return df


def build_today_games(schedule, lineups):
    """
    Build a games DataFrame for today matching the structure of game_results_all.csv.

    Today's games get placeholder scores (NaN) since they haven't been played.
    SP IDs come from lineups (confirmed) or schedule (probable).
    """
    rows = []
    for _, game in schedule.iterrows():
        game_pk = game.get("game_pk", game.get("game_id"))
        game_date = TODAY

        home_team = game.get("home_team", "")
        away_team = game.get("away_team", "")
        home_abbrev = game.get("home_abbrev", "")
        away_abbrev = game.get("away_abbrev", "")
        home_team_id = game.get("home_team_id")
        away_team_id = game.get("away_team_id")

        # Get SP info from lineups first, fall back to schedule
        home_sp_id = game.get("home_sp_id")
        home_sp_name = game.get("home_sp_name", "")
        away_sp_id = game.get("away_sp_id")
        away_sp_name = game.get("away_sp_name", "")

        if not lineups.empty:
            # Match by game_pk or team names
            for _, l in lineups.iterrows():
                lpk = l.get("game_pk", l.get("game_id"))
                if lpk == game_pk or (
                    str(l.get("home_team", "")) == str(home_team) and
                    str(l.get("away_team", "")) == str(away_team)
                ):
                    if pd.notna(l.get("home_sp_id")):
                        home_sp_id = l["home_sp_id"]
                        home_sp_name = l.get("home_sp_name", home_sp_name)
                    if pd.notna(l.get("away_sp_id")):
                        away_sp_id = l["away_sp_id"]
                        away_sp_name = l.get("away_sp_name", away_sp_name)
                    break

        row = {
            "game_pk": game_pk,
            "date": game_date,
            "home_team": home_team,
            "away_team": away_team,
            "home_abbrev": home_abbrev,
            "away_abbrev": away_abbrev,
            "home_team_id": home_team_id,
            "away_team_id": away_team_id,
            # Placeholder scores — not available pre-game
            "home_runs": np.nan,
            "away_runs": np.nan,
            "home_hits": np.nan,
            "away_hits": np.nan,
            "home_f5_runs": np.nan,
            "away_f5_runs": np.nan,
            "first_inning_home_runs": np.nan,
            "first_inning_away_runs": np.nan,
            "num_innings": np.nan,
            "is_7_inning_dh": False,
            "game_type": game.get("game_type", "R"),
            "is_postseason": game.get("game_type", "R") in ("F", "D", "L", "W"),
            "doubleheader": game.get("doubleheader", "N"),
            "game_num": game.get("game_num", 1),
            "venue_name": game.get("venue_name", game.get("venue", "")),
            "venue_id": game.get("venue_id"),
            "temp": game.get("temp"),
            "wind": game.get("wind", ""),
            "condition": game.get("condition", ""),
            "hp_umpire": game.get("hp_umpire", ""),
            "hp_umpire_id": game.get("hp_umpire_id"),
            "home_sp_id": home_sp_id,
            "home_sp_name": home_sp_name,
            "away_sp_id": away_sp_id,
            "away_sp_name": away_sp_name,
        }
        rows.append(row)

    return pd.DataFrame(rows)


def build_features():
    """Build the daily feature matrix using historical data + today's games."""

    # ── Load today's data ──
    schedule = load_schedule()
    lineups = load_lineups()
    today_games = build_today_games(schedule, lineups)
    n_today = len(today_games)

    if n_today == 0:
        log.warning("No games scheduled for today")
        return pd.DataFrame()

    log.info(f"Today's games: {n_today}")
    for _, g in today_games.iterrows():
        sp_h = g.get("home_sp_name", "TBD") or "TBD"
        sp_a = g.get("away_sp_name", "TBD") or "TBD"
        log.info(f"  {g['away_team']} ({sp_a}) @ {g['home_team']} ({sp_h})")

    # ── Load historical data ──
    log.info("Loading historical data files...")

    if not GAMES_FILE.exists():
        log.error(f"Game results not found: {GAMES_FILE}")
        log.error("Run scripts/fetch_historical_games.py first")
        sys.exit(1)

    hist_games = pd.read_csv(GAMES_FILE)
    hist_games["date"] = pd.to_datetime(hist_games["date"])
    log.info(f"Historical games: {len(hist_games):,}")

    # Combine historical + today, sorted by date
    # IMPORTANT: Sort by date because compute functions (compute_context_features,
    # compute_momentum_features, etc.) sort internally and return features
    # indexed by position. If data isn't pre-sorted, features get misaligned.
    today_games["date"] = pd.to_datetime(today_games["date"])
    today_games = today_games.assign(_is_today=True)
    hist_games = hist_games.assign(_is_today=False)
    all_games = pd.concat([hist_games, today_games], ignore_index=True)
    all_games = all_games.sort_values("date").reset_index(drop=True)
    today_indices = all_games[all_games["_is_today"] == True].index.tolist()
    all_games = all_games.drop(columns=["_is_today"])

    log.info(f"Combined dataset: {len(all_games):,} games "
             f"(today at indices {today_indices[0]}-{today_indices[-1]})")

    # Load pitcher logs + Statcast
    pitcher_logs = None
    if PITCHER_LOGS_FILE.exists():
        pitcher_logs = pd.read_csv(PITCHER_LOGS_FILE)
        if STATCAST_FILE.exists():
            statcast = pd.read_csv(STATCAST_FILE)
            statcast_cols = ["pitcher_id", "game_pk", "statcast_pitches",
                             "xwoba", "hard_hit_pct", "barrel_pct",
                             "groundball_pct", "flyball_pct", "whiff_rate",
                             "xwoba_vs_LHB", "xwoba_vs_RHB",
                             "whiff_rate_vs_LHB", "whiff_rate_vs_RHB",
                             "fastball_pct", "breaking_pct", "offspeed_pct",
                             "primary_pitch_pct", "pitch_mix_entropy",
                             "avg_fastball_velo", "max_fastball_velo",
                             "zone_pct", "csw_pct", "chase_rate"]
            statcast = statcast[[c for c in statcast_cols if c in statcast.columns]]
            pitcher_logs = pitcher_logs.merge(statcast, on=["pitcher_id", "game_pk"], how="left")
            log.info(f"Pitcher logs: {len(pitcher_logs):,} (Statcast merged)")
        else:
            log.info(f"Pitcher logs: {len(pitcher_logs):,} (no Statcast)")
        if "pitches_thrown" in pitcher_logs.columns:
            pitcher_logs = pitcher_logs.rename(columns={"pitches_thrown": "pitches"})
    else:
        log.warning("No pitcher logs found")

    # Load batting logs
    batting_logs = None
    if BATTING_LOGS_FILE.exists():
        batting_logs = pd.read_csv(BATTING_LOGS_FILE)
        log.info(f"Batting logs: {len(batting_logs):,}")

    # Load bullpen logs
    bullpen_logs = None
    if BULLPEN_LOGS_FILE.exists():
        bullpen_logs = pd.read_csv(BULLPEN_LOGS_FILE)
        log.info(f"Bullpen logs: {len(bullpen_logs):,}")

    # Load batter data
    batter_logs = None
    handedness = None
    if BATTER_LOGS_FILE.exists() and HANDEDNESS_FILE.exists():
        batter_logs = pd.read_csv(BATTER_LOGS_FILE)
        handedness = pd.read_csv(HANDEDNESS_FILE)
        log.info(f"Batter logs: {len(batter_logs):,}, Handedness: {len(handedness):,}")

    batter_pitch_stats = None
    if BATTER_PITCH_TYPE_FILE.exists():
        batter_pitch_stats = pd.read_csv(BATTER_PITCH_TYPE_FILE)
        log.info(f"Batter pitch-type stats: {len(batter_pitch_stats):,}")

    # ── Compute all features on combined data ──
    # This replicates 00_build_mlb_historical.py's pipeline exactly.
    # Today's games have NaN scores, but all feature functions use only
    # pre-game data (games before the query date), so today's NaN scores
    # don't affect today's features.

    log.info("=" * 50)
    log.info("Computing features (on combined historical + today)...")
    log.info("=" * 50)

    # SP features
    log.info("  Starting pitcher features...")
    sp_computer = StartingPitcherComputer(pitcher_logs)
    sp_features = compute_sp_features(all_games, sp_computer)

    # Team batting
    log.info("  Team batting features...")
    if batting_logs is not None:
        bat_features = compute_team_batting_features(all_games, batting_logs)
    else:
        bat_features = pd.DataFrame(index=all_games.index)

    # Context (park, umpire, weather, DH)
    log.info("  Context features (park, umpire, weather)...")
    ctx_features = compute_context_features(all_games)

    # Rest (SP rest, SP season IP, team rest)
    log.info("  Rest features...")
    rest_features = compute_rest_features(all_games, pitcher_logs)

    # Momentum (rolling win%, run diff)
    log.info("  Momentum features...")
    mom_features = compute_momentum_features(all_games)

    # Bullpen
    log.info("  Bullpen features...")
    if bullpen_logs is not None:
        bp_features = compute_bullpen_features(all_games, bullpen_logs)
    else:
        bp_features = pd.DataFrame(index=all_games.index)

    # Bullpen availability
    log.info("  Bullpen availability features...")
    if bullpen_logs is not None:
        bp_avail_features = compute_bullpen_availability_features(all_games, bullpen_logs)
    else:
        bp_avail_features = pd.DataFrame(index=all_games.index)

    # Travel
    log.info("  Travel features...")
    travel_features = compute_travel_features(all_games)

    # Schedule context
    log.info("  Schedule context features...")
    sched_features = compute_schedule_context_features(all_games)

    # Lineup features
    log.info("  Lineup features...")
    if batter_logs is not None and handedness is not None:
        lineup_features = compute_lineup_features(all_games, batter_logs, handedness, pitcher_logs)
    else:
        lineup_features = pd.DataFrame(index=all_games.index)

    # Opponent-adjusted
    log.info("  Opponent-adjusted features...")
    opp_lookup = build_game_opponent_lookup(all_games)
    if pitcher_logs is not None and batting_logs is not None:
        opp_adj_features = compute_opponent_adjusted_features(
            all_games, pitcher_logs, batting_logs, opp_lookup)
    else:
        opp_adj_features = pd.DataFrame(index=all_games.index)

    # Handedness splits
    log.info("  Handedness split features...")
    if batter_logs is not None and handedness is not None:
        hand_split_features = compute_handedness_split_features(
            all_games, batter_logs, handedness, pitcher_logs, sp_features)
    else:
        hand_split_features = pd.DataFrame(index=all_games.index)

    # Pitch-type matchups
    log.info("  Pitch-type matchup features...")
    if batter_pitch_stats is not None and batter_logs is not None:
        pitch_type_features = compute_pitch_type_features(
            all_games, batter_pitch_stats, batter_logs, sp_features)
    else:
        pitch_type_features = pd.DataFrame(index=all_games.index)

    # ── Assemble ──
    log.info("  Assembling feature matrix...")

    game_cols = [
        "game_pk", "date", "home_team", "away_team",
        "home_abbrev", "away_abbrev", "home_team_id", "away_team_id",
        "venue_name", "venue_id", "temp", "wind", "condition",
        "hp_umpire", "hp_umpire_id",
        "home_sp_id", "home_sp_name", "away_sp_id", "away_sp_name",
        "doubleheader", "game_type", "is_postseason",
    ]
    game_cols = [c for c in game_cols if c in all_games.columns]

    training = pd.concat([
        all_games[game_cols].reset_index(drop=True),
        sp_features.reset_index(drop=True),
        bat_features.reset_index(drop=True),
        ctx_features.reset_index(drop=True),
        rest_features.reset_index(drop=True),
        mom_features.reset_index(drop=True),
        bp_features.reset_index(drop=True),
        bp_avail_features.reset_index(drop=True),
        travel_features.reset_index(drop=True),
        sched_features.reset_index(drop=True),
        lineup_features.reset_index(drop=True),
        opp_adj_features.reset_index(drop=True),
        hand_split_features.reset_index(drop=True),
        pitch_type_features.reset_index(drop=True),
    ], axis=1)

    # Interaction features
    interaction_features = compute_interaction_features(training)
    training = pd.concat([training, interaction_features], axis=1)

    # Deduplicate columns
    dupes = training.columns[training.columns.duplicated()].tolist()
    if dupes:
        log.warning(f"Removing {len(dupes)} duplicate columns")
        training = training.loc[:, ~training.columns.duplicated()]

    # ── Extract today's games ──
    today_df = training.iloc[today_indices].copy().reset_index(drop=True)

    # Report feature coverage
    feature_cols = [c for c in MLB_CANDIDATE_FEATURES if c in today_df.columns]
    populated = sum(today_df[feature_cols].notna().any() for c in feature_cols
                    if c in today_df.columns)
    log.info(f"\nToday's feature matrix: {len(today_df)} games, "
             f"{len(today_df.columns)} total columns")
    log.info(f"Candidate features available: {len(feature_cols)}/{len(MLB_CANDIDATE_FEATURES)}")

    # Feature coverage per game
    for _, g in today_df.iterrows():
        n_pop = sum(pd.notna(g.get(f)) for f in feature_cols)
        home = g.get("home_team", "?")
        away = g.get("away_team", "?")
        log.info(f"  {away} @ {home}: {n_pop}/{len(feature_cols)} features populated")

    # ── Save ──
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    out_path = PROCESSED_DIR / f"features_{TODAY}.csv"
    today_df.to_csv(out_path, index=False)
    log.info(f"\nSaved features to {out_path}")

    return today_df


if __name__ == "__main__":
    build_features()
