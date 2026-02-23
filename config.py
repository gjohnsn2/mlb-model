"""
MLB Daily Betting Model — Central Configuration
================================================
Fill in your credentials below before running the pipeline.
"""

import os
import json
from datetime import date, datetime
from pathlib import Path

# -- Project paths ----------------------------------------------------
PROJECT_DIR = Path(__file__).parent
DATA_DIR = PROJECT_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
PREDICTIONS_DIR = DATA_DIR / "predictions"
TRACKING_DIR = DATA_DIR / "tracking"
HISTORICAL_DIR = DATA_DIR / "historical"
LINES_DIR = DATA_DIR / "lines"
MODELS_DIR = PROJECT_DIR / "models" / "trained"
MODELS_ROOT = PROJECT_DIR / "models"
REPORTS_DIR = PROJECT_DIR / "reports"

# -- Date --------------------------------------------------------------
# Override with MLB_DATE env var or --date CLI arg
# Usage: MLB_DATE=2026-04-15 ./run_daily.sh full
#    or: ./run_daily.sh --date 2026-04-15 full
TODAY = os.environ.get("MLB_DATE", date.today().isoformat())  # YYYY-MM-DD

# -- API Credentials ---------------------------------------------------
# The Odds API (https://the-odds-api.com) -- sign up for free/pro tier
ODDS_API_KEY = os.environ.get("ODDS_API_KEY", "YOUR_ODDS_API_KEY")

# Weather API (optional -- OpenWeatherMap or Visual Crossing)
WEATHER_API_KEY = os.environ.get("WEATHER_API_KEY", "YOUR_WEATHER_API_KEY")

# -- ESPN API (unofficial, no key needed) ------------------------------
ESPN_SCOREBOARD_URL = (
    "https://site.api.espn.com/apis/site/v2/sports/baseball/mlb/scoreboard"
)
ESPN_TEAMS_URL = (
    "https://site.api.espn.com/apis/site/v2/sports/baseball/mlb/teams"
)

# -- MLB Stats API (official, no key needed) ---------------------------
MLB_API_BASE = "https://statsapi.mlb.com/api/v1"

# -- The Odds API -------------------------------------------------------
ODDS_API_BASE = "https://api.the-odds-api.com/v4"
ODDS_SPORT = "baseball_mlb"
ODDS_REGIONS = "us"
# Full-game + derivative markets. h2h_h1/totals_h1 = first 5 innings.
ODDS_MARKETS = "h2h,spreads,totals,h2h_h1,totals_h1"

# -- Model parameters ---------------------------------------------------
# XGBoost defaults (tuned via Optuna in 06b_tune_hyperparams.py)
XGBOOST_PARAMS_MARGIN = {
    "n_estimators": 400,
    "max_depth": 6,
    "learning_rate": 0.03,
    "subsample": 0.8,
    "colsample_bytree": 0.7,
    "min_child_weight": 5,
    "reg_alpha": 0.1,
    "reg_lambda": 1.0,
    "random_state": 42,
}

XGBOOST_PARAMS_TOTAL = {
    "n_estimators": 400,
    "max_depth": 5,
    "learning_rate": 0.03,
    "subsample": 0.8,
    "colsample_bytree": 0.7,
    "min_child_weight": 5,
    "reg_alpha": 0.1,
    "reg_lambda": 1.0,
    "random_state": 42,
}

# Edge detection thresholds (will be calibrated after initial backtesting)
# Full-game markets
ML_EDGE_THRESHOLD = 0.03       # 3% probability edge for moneyline plays
RUNLINE_EDGE_THRESHOLD = 2.0   # 2.0 point edge for run line plays (mirroring CBB spread)
TOTAL_EDGE_THRESHOLD = 1.5     # 1.5 run edge for totals
# First 5 inning markets
F5_ML_EDGE_THRESHOLD = 0.03    # 3% probability edge for F5 moneyline
F5_TOTAL_EDGE_THRESHOLD = 1.0  # 1.0 run edge for F5 totals (smaller range)
# NRFI/YRFI (binary — edge is probability difference)
NRFI_EDGE_THRESHOLD = 0.05     # 5% probability edge for NRFI/YRFI
# Team totals
TEAM_TOTAL_EDGE_THRESHOLD = 1.0  # 1.0 run edge for team totals

# Moneyline unit tiers (by probability edge)
ML_UNIT_TIERS = [
    (0.15, 3.0, "3u"),   # ml_edge >= 15%: max conviction
    (0.10, 2.0, "2u"),   # ml_edge >= 10%: high conviction
    (0.06, 1.5, "1.5u"), # ml_edge >= 6%: standard+
    (0.03, 1.0, "1u"),   # ml_edge >= 3%: production threshold
]

# Run line unit tiers (by run edge)
RUNLINE_UNIT_TIERS = [
    (5.0, 3.0, "3u"),
    (3.0, 2.0, "2u"),
    (2.0, 1.5, "1.5u"),
    (1.0, 1.0, "1u"),
]

# Total unit tiers (by run edge)
TOTAL_UNIT_TIERS = [
    (3.0, 3.0, "3u"),
    (2.0, 2.0, "2u"),
    (1.5, 1.5, "1.5u"),
    (1.0, 1.0, "1u"),
]

# Staking & bankroll management
BANKROLL_START = 25_000          # Starting bankroll ($)
BANKROLL_UNIT_PCT = 0.01         # 1% of current bankroll per unit
DRAWDOWN_WARNING = 0.15          # Warn at 15% drawdown from peak
DRAWDOWN_PAUSE = 0.25            # Pause betting at 25% drawdown from peak

# ML price filters -- avoid betting huge favorites where juice eats the edge
MAX_ML_PRICE = -250              # Skip moneylines steeper than -250
MIN_IMPLIED_PROB = 0.40          # Skip if market implied prob < 40% (longshots)

# Model RMSEs (set after training -- used for margin-to-prob conversion)
MARGIN_MODEL_RMSE = None         # Full-game margin model
TOTAL_MODEL_RMSE = None          # Full-game total model
F5_MARGIN_MODEL_RMSE = None      # First 5 innings margin model
F5_TOTAL_MODEL_RMSE = None       # First 5 innings total model

# XGBoost params for derivative models (tuned separately via Optuna)
XGBOOST_PARAMS_F5_MARGIN = {
    "n_estimators": 350,
    "max_depth": 5,
    "learning_rate": 0.03,
    "subsample": 0.8,
    "colsample_bytree": 0.7,
    "min_child_weight": 5,
    "reg_alpha": 0.1,
    "reg_lambda": 1.0,
    "random_state": 42,
}

XGBOOST_PARAMS_F5_TOTAL = {
    "n_estimators": 350,
    "max_depth": 5,
    "learning_rate": 0.03,
    "subsample": 0.8,
    "colsample_bytree": 0.7,
    "min_child_weight": 5,
    "reg_alpha": 0.1,
    "reg_lambda": 1.0,
    "random_state": 42,
}

XGBOOST_PARAMS_NRFI = {
    "n_estimators": 300,
    "max_depth": 4,
    "learning_rate": 0.03,
    "subsample": 0.8,
    "colsample_bytree": 0.7,
    "min_child_weight": 10,   # Higher — small target, need regularization
    "reg_alpha": 0.5,
    "reg_lambda": 2.0,
    "objective": "binary:logistic",
    "eval_metric": "logloss",
    "random_state": 42,
}

# Odds API book keys -> display names
BOOK_DISPLAY_NAMES = {
    "draftkings": "DraftKings", "fanduel": "FanDuel", "betmgm": "BetMGM",
    "williamhill_us": "Caesars", "pointsbetus": "PointsBet", "betrivers": "BetRivers",
    "barstool": "ESPN Bet", "superbook": "SuperBook", "pinnacle": "Pinnacle",
    "bovada": "Bovada", "betonlineag": "BetOnline", "mybookieag": "MyBookie",
    "wynnbet": "WynnBET", "twinspires": "TwinSpires", "unibet_us": "Unibet",
    "betfred": "Betfred", "lowvig": "LowVig", "betus": "BetUS",
    "espnbet": "ESPN Bet", "fliff": "Fliff", "hardrock": "Hard Rock",
    "bet365": "Bet365",
}

# Sample weighting -- exponential decay by season
SAMPLE_WEIGHT_HALF_LIFE = 3  # years; weight halves every N seasons back
# MLB changes faster than CBB (rule changes, juiced ball era, etc.)

# -- Feature columns ----------------------------------------------------
# All candidate features (from feature_engine.py)
ALL_CANDIDATE_FEATURES = [
    # Cat 1: Starting Pitcher (20)
    "sp_era_diff", "sp_fip_diff", "sp_xfip_diff", "sp_whip_diff",
    "sp_k_pct_diff", "sp_bb_pct_diff", "sp_k_bb_diff", "sp_hr9_diff",
    "sp_xwoba_diff", "sp_hard_hit_pct_diff",
    "sp_era_last3_diff", "sp_fip_last3_diff", "sp_ip_last3_avg_diff",
    "sp_k_pct_last3_diff", "sp_pitch_count_trend_diff",
    "sp_days_rest_diff", "sp_season_ip_diff",
    "sp_groundball_pct_diff", "sp_flyball_pct_diff", "sp_barrel_pct_diff",
    # Cat 2: Bullpen (10)
    "bullpen_era_diff", "bullpen_fip_diff", "bullpen_whip_diff",
    "bullpen_k_pct_diff", "bullpen_usage_3d_diff",
    "bullpen_high_lev_avail_diff", "closer_available_diff",
    "bullpen_ip_last3d_diff", "bullpen_era_last7d_diff", "bullpen_xfip_diff",
    # Cat 3: Team Batting (15)
    "wrc_plus_diff", "ops_diff", "woba_diff", "iso_diff",
    "babip_diff", "k_pct_bat_diff", "bb_pct_bat_diff",
    "wrc_plus_last14_diff", "ops_last14_diff",
    "wrc_plus_vs_hand_diff", "ops_vs_hand_diff",
    "home_wrc_plus_diff", "away_wrc_plus_diff",
    "runs_scored_last10_avg_diff", "run_diff_last10_diff",
    # Cat 4: Park Factors (5)
    "park_run_factor", "park_hr_factor", "park_hit_factor",
    "park_factor_advantage", "park_total_adjustment",
    # Cat 5: Situational (12)
    "home_flag", "rest_days_diff", "travel_distance",
    "day_night", "series_game_num", "interleague",
    "season_phase", "days_since_allstar",
    "home_record_diff", "away_record_diff",
    "home_win_pct_last20", "away_win_pct_last20",
    # Cat 6: Weather (6)
    "temperature", "wind_speed", "wind_direction_factor",
    "humidity", "precipitation_prob", "indoor_flag",
    # Cat 7: Rolling Form (10)
    "win_pct_10_diff", "win_pct_20_diff",
    "run_diff_10_diff", "run_diff_20_diff",
    "scoring_trend_diff", "pitching_trend_diff",
    "streak_diff", "games_played_diff",
    "pythag_record_diff", "base_running_diff",
    # Cat 8: Market-Derived (8)
    "consensus_ml_home", "consensus_ml_away",
    "ml_implied_prob_home", "ml_implied_prob_away",
    "consensus_total", "consensus_runline",
    "num_books", "has_odds",
    # Cat 9: Matchup-Specific (6)
    "sp_vs_team_ops", "sp_vs_team_woba",
    "team_vs_sp_hand_wrc", "lineup_wrc_plus_diff",
    "batter_pitcher_history_ops", "platoon_advantage",
]

# Dynamic feature loading: use Boruta-selected features if available,
# otherwise fall back to full candidate list
def _load_selected_features():
    """Load Boruta-selected features from models/selected_features.json."""
    path = MODELS_ROOT / "selected_features.json"
    if path.exists():
        with open(path) as f:
            data = json.load(f)
        return data.get("margin_features", ALL_CANDIDATE_FEATURES), \
               data.get("total_features", ALL_CANDIDATE_FEATURES)
    return ALL_CANDIDATE_FEATURES, ALL_CANDIDATE_FEATURES

MARGIN_FEATURES, TOTAL_FEATURES = _load_selected_features()

# -- Logging ------------------------------------------------------------
import logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)

def get_logger(name):
    return logging.getLogger(name)
