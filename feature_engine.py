"""
Feature Engine — Single Source of Truth (MLB)
===============================================
Computes all candidate features for both historical training
(00_build_historical.py) and daily prediction (05_build_features.py).

Feature Categories:
  Cat 1: Starting Pitcher (20)
  Cat 2: Bullpen (10)
  Cat 3: Team Batting (15)
  Cat 4: Park Factors (5)
  Cat 5: Situational/Context (12)
  Cat 6: Weather (6)
  Cat 7: Rolling Form (10)
  Cat 8: Market-Derived (8)
  Cat 9: Matchup-Specific (6)

Architecture note: Each feature must be computable from data available BEFORE
the game being predicted. No in-game or post-game data may be used as features.
This is critical for preventing look-ahead bias.
"""

import math
import json
import numpy as np
import pandas as pd
from pathlib import Path
from config import get_logger

log = get_logger("feature_engine")

# -- Venue locations for travel distance ---------------------------------
VENUE_LOCATIONS = None  # Loaded lazily

def _load_venue_locations():
    global VENUE_LOCATIONS
    if VENUE_LOCATIONS is not None:
        return VENUE_LOCATIONS
    path = Path(__file__).parent / "data" / "venue_locations.csv"
    if path.exists():
        VENUE_LOCATIONS = pd.read_csv(path).set_index("team")
        log.info(f"Loaded venue locations for {len(VENUE_LOCATIONS)} teams")
    else:
        VENUE_LOCATIONS = pd.DataFrame()
        log.warning("Venue locations file not found")
    return VENUE_LOCATIONS


def _haversine(lat1, lon1, lat2, lon2):
    """Haversine distance in miles between two lat/lon points."""
    R = 3959  # Earth radius in miles
    lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = math.sin(dlat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
    return 2 * R * math.asin(math.sqrt(a))


# -- Safe math helpers ---------------------------------------------------

def _sf(val):
    """Safe float conversion."""
    try:
        v = float(val)
        return v if not np.isnan(v) else np.nan
    except (ValueError, TypeError):
        return np.nan


def _diff(a, b):
    a, b = _sf(a), _sf(b)
    if np.isnan(a) or np.isnan(b):
        return np.nan
    return a - b


def _avg(a, b):
    a, b = _sf(a), _sf(b)
    if np.isnan(a) or np.isnan(b):
        return np.nan
    return (a + b) / 2


def _sum(a, b):
    a, b = _sf(a), _sf(b)
    if np.isnan(a) or np.isnan(b):
        return np.nan
    return a + b


# ===================================================================
# Starting Pitcher Stats Computer
# ===================================================================

class StartingPitcherComputer:
    """
    Computes starting pitcher features from game logs.
    Strictly uses only starts BEFORE the query date (no lookahead).
    """

    def __init__(self, pitcher_logs_df):
        """
        pitcher_logs_df: DataFrame with columns like
          date, pitcher_id, pitcher_name, team, era, fip, xfip, whip,
          k_pct, bb_pct, ip, pitches, hits, runs, hr, xwoba,
          hard_hit_pct, groundball_pct, flyball_pct, barrel_pct
        """
        if pitcher_logs_df is None or pitcher_logs_df.empty:
            self.available = False
            return

        self.available = True
        df = pitcher_logs_df.copy()
        df["date"] = pd.to_datetime(df["date"])

        # Parse numeric columns (counting stats from MLB API + Statcast metrics)
        numeric_cols = [
            "ip", "pitches", "hits", "runs", "earned_runs", "home_runs",
            "strikeouts", "walks", "batters_faced",
            # Statcast per-game metrics
            "xwoba", "hard_hit_pct", "groundball_pct", "flyball_pct",
            "barrel_pct", "whiff_rate",
            # Handedness splits (from Statcast)
            "xwoba_vs_LHB", "xwoba_vs_RHB",
            "whiff_rate_vs_LHB", "whiff_rate_vs_RHB",
            # Pitch-type distribution (from Statcast)
            "fastball_pct", "breaking_pct", "offspeed_pct",
            "primary_pitch_pct", "pitch_mix_entropy",
            # Velocity & command (from Statcast)
            "avg_fastball_velo", "max_fastball_velo",
            "zone_pct", "csw_pct", "chase_rate",
        ]
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")

        df = df.sort_values(["pitcher_id", "date"]).reset_index(drop=True)
        self._df = df

        self._pitcher_games = {}
        for pid, group in df.groupby("pitcher_id"):
            self._pitcher_games[pid] = group.reset_index(drop=True)

    def get_pitcher_stats(self, pitcher_id, game_date, season=None):
        """
        Get season-to-date and recent form stats for a pitcher before game_date.
        Returns dict of pitcher feature values, or empty dict if unavailable.
        """
        if not self.available or pitcher_id not in self._pitcher_games:
            return {}

        game_date = pd.to_datetime(game_date)
        pg = self._pitcher_games[pitcher_id]

        # Filter to current season + before game_date
        if season is not None:
            prior = pg[(pg["date"].dt.year == int(season)) & (pg["date"] < game_date)]
        else:
            prior = pg[pg["date"] < game_date]
            # Filter to same calendar year (MLB seasons don't span years)
            prior = prior[prior["date"].dt.year == game_date.year]

        if len(prior) < 2:
            return {}

        result = {}

        # Number of starts this season
        result["sp_starts"] = len(prior)

        # ── Season-to-date rate stats (computed from counting stats) ──
        has_ip = "ip" in prior.columns
        total_ip = prior["ip"].sum() if has_ip else 0
        has_bf = "batters_faced" in prior.columns
        total_bf = prior["batters_faced"].sum() if has_bf else 0

        if has_ip and total_ip > 0:
            result["sp_season_ip"] = total_ip
            result["sp_avg_ip"] = prior["ip"].mean()

            # ERA = (earned_runs / IP) * 9
            if "earned_runs" in prior.columns:
                result["sp_era"] = (prior["earned_runs"].sum() / total_ip) * 9.0

            # WHIP = (hits + walks) / IP
            if "hits" in prior.columns and "walks" in prior.columns:
                result["sp_whip"] = (prior["hits"].sum() + prior["walks"].sum()) / total_ip

            # FIP = ((13*HR + 3*BB - 2*K) / IP) + 3.1
            if all(c in prior.columns for c in ["home_runs", "walks", "strikeouts"]):
                total_hr = prior["home_runs"].sum()
                total_bb = prior["walks"].sum()
                total_k = prior["strikeouts"].sum()
                result["sp_fip"] = ((13 * total_hr + 3 * total_bb - 2 * total_k) / total_ip) + 3.1
                # xFIP: use league-avg HR/FB rate (~10.5%) instead of actual HR
                # Estimate fly balls from Statcast flyball_pct: FB = BIP * FB%
                # BIP ≈ BF - K - BB (batted balls in play)
                if "flyball_pct" in prior.columns:
                    fb_pct_vals = prior["flyball_pct"].dropna()
                    if len(fb_pct_vals) >= 2 and total_bf > 0:
                        avg_fb_pct = fb_pct_vals.mean() / 100.0  # Convert from percentage
                        bip = total_bf - total_k - total_bb
                        if bip > 0:
                            est_fb = bip * avg_fb_pct
                            # League-avg HR/FB rate ≈ 10.5%
                            expected_hr = est_fb * 0.105
                            result["sp_xfip"] = ((13 * expected_hr + 3 * total_bb - 2 * total_k) / total_ip) + 3.1
                        else:
                            result["sp_xfip"] = result["sp_fip"]
                    else:
                        result["sp_xfip"] = result["sp_fip"]
                else:
                    result["sp_xfip"] = result["sp_fip"]

            # Per-9 rates
            if "home_runs" in prior.columns:
                result["sp_hr_per_9"] = (prior["home_runs"].sum() / total_ip) * 9
                result["sp_hr9"] = result["sp_hr_per_9"]
            if "strikeouts" in prior.columns:
                result["sp_k_per_9"] = (prior["strikeouts"].sum() / total_ip) * 9
            if "walks" in prior.columns:
                result["sp_bb_per_9"] = (prior["walks"].sum() / total_ip) * 9

        # Percentage rates (using batters faced)
        if has_bf and total_bf > 0:
            if "strikeouts" in prior.columns:
                result["sp_k_pct"] = prior["strikeouts"].sum() / total_bf
            if "walks" in prior.columns:
                result["sp_bb_pct"] = prior["walks"].sum() / total_bf

        # K-BB%
        if "sp_k_pct" in result and "sp_bb_pct" in result:
            result["sp_k_bb"] = result["sp_k_pct"] - result["sp_bb_pct"]

        # Average pitches per start
        if "pitches" in prior.columns:
            p_vals = prior["pitches"].dropna()
            if len(p_vals) > 0:
                result["sp_avg_pitches"] = p_vals.mean()

        # ── Statcast per-game averages (already rate stats in the data) ──
        for col, key in [
            ("xwoba", "sp_xwoba"), ("hard_hit_pct", "sp_hard_hit_pct"),
            ("groundball_pct", "sp_groundball_pct"),
            ("flyball_pct", "sp_flyball_pct"),
            ("barrel_pct", "sp_barrel_pct"),
            ("whiff_rate", "sp_whiff_rate"),
        ]:
            if col in prior.columns:
                vals = prior[col].dropna()
                if len(vals) >= 2:
                    result[key] = vals.mean()

        # Handedness splits (from Statcast)
        for col, key in [
            ("xwoba_vs_LHB", "sp_xwoba_vs_LHB"),
            ("xwoba_vs_RHB", "sp_xwoba_vs_RHB"),
            ("whiff_rate_vs_LHB", "sp_whiff_rate_vs_LHB"),
            ("whiff_rate_vs_RHB", "sp_whiff_rate_vs_RHB"),
        ]:
            if col in prior.columns:
                vals = prior[col].dropna()
                if len(vals) >= 2:
                    result[key] = vals.mean()

        # Pitch-type distribution (from Statcast)
        for col, key in [
            ("fastball_pct", "sp_fastball_pct"),
            ("breaking_pct", "sp_breaking_pct"),
            ("offspeed_pct", "sp_offspeed_pct"),
            ("primary_pitch_pct", "sp_primary_pitch_pct"),
            ("pitch_mix_entropy", "sp_pitch_mix_entropy"),
        ]:
            if col in prior.columns:
                vals = prior[col].dropna()
                if len(vals) >= 2:
                    result[key] = vals.mean()

        # Velocity & command (from Statcast)
        for col, key in [
            ("avg_fastball_velo", "sp_fastball_velo"),
            ("max_fastball_velo", "sp_max_velo"),
            ("zone_pct", "sp_zone_pct"),
            ("csw_pct", "sp_csw_pct"),
            ("chase_rate", "sp_chase_rate"),
        ]:
            if col in prior.columns:
                vals = prior[col].dropna()
                if len(vals) >= 2:
                    result[key] = vals.mean()

        # Days rest
        if len(prior) > 0:
            last_start = prior["date"].max()
            result["sp_days_rest"] = (game_date - last_start).days

        # ── Last 3 starts recency stats ──
        last3 = prior.tail(3)
        if len(last3) >= 2:
            l3_ip = last3["ip"].sum() if has_ip else 0
            l3_bf = last3["batters_faced"].sum() if has_bf else 0

            # Rate stats from last 3 counting stats
            if l3_ip > 0:
                if "earned_runs" in last3.columns:
                    result["sp_era_last3"] = (last3["earned_runs"].sum() / l3_ip) * 9.0
                if "hits" in last3.columns and "walks" in last3.columns:
                    result["sp_whip_last3"] = (last3["hits"].sum() + last3["walks"].sum()) / l3_ip
                if all(c in last3.columns for c in ["home_runs", "walks", "strikeouts"]):
                    result["sp_fip_last3"] = ((13 * last3["home_runs"].sum() + 3 * last3["walks"].sum() - 2 * last3["strikeouts"].sum()) / l3_ip) + 3.1
            if l3_bf > 0 and "strikeouts" in last3.columns:
                result["sp_k_pct_last3"] = last3["strikeouts"].sum() / l3_bf

            # Statcast averages over last 3
            for col, key in [
                ("xwoba", "sp_xwoba_last3"), ("hard_hit_pct", "sp_hard_hit_pct_last3"),
                ("barrel_pct", "sp_barrel_pct_last3"),
                ("groundball_pct", "sp_groundball_pct_last3"),
                ("flyball_pct", "sp_flyball_pct_last3"),
                ("whiff_rate", "sp_whiff_rate_last3"),
            ]:
                if col in last3.columns:
                    vals = last3[col].dropna()
                    if len(vals) >= 1:
                        result[key] = vals.mean()

            if has_ip:
                result["sp_ip_last3_avg"] = last3["ip"].dropna().mean()
            if "pitches" in last3.columns:
                pitches = last3["pitches"].dropna()
                if len(pitches) >= 2:
                    x = np.arange(len(pitches))
                    if np.std(pitches) > 0:
                        result["sp_pitch_count_trend"] = float(np.polyfit(x, pitches.values, 1)[0])

        return result


# ===================================================================
# Bullpen Stats Computer
# ===================================================================

class BullpenComputer:
    """
    Computes bullpen-level features from reliever game logs.
    Aggregates across all relievers on a team.
    """

    def __init__(self, reliever_logs_df):
        if reliever_logs_df is None or reliever_logs_df.empty:
            self.available = False
            return

        self.available = True
        df = reliever_logs_df.copy()
        df["date"] = pd.to_datetime(df["date"])

        numeric_cols = ["era", "fip", "xfip", "whip", "k_pct", "ip", "pitches"]
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")

        df = df.sort_values(["team", "date"]).reset_index(drop=True)
        self._df = df

        self._team_games = {}
        for team, group in df.groupby("team"):
            self._team_games[team] = group.reset_index(drop=True)

    def get_bullpen_stats(self, team, game_date):
        """Get bullpen aggregate stats for team as of game_date."""
        if not self.available or team not in self._team_games:
            return {}

        game_date = pd.to_datetime(game_date)
        tg = self._team_games[team]
        prior = tg[tg["date"] < game_date]

        # Same season
        prior = prior[prior["date"].dt.year == game_date.year]

        if len(prior) < 10:  # Need enough bullpen appearances
            return {}

        result = {}

        # Season aggregates
        for col, key in [
            ("era", "bullpen_era"), ("fip", "bullpen_fip"),
            ("xfip", "bullpen_xfip"), ("whip", "bullpen_whip"),
            ("k_pct", "bullpen_k_pct"),
        ]:
            if col in prior.columns:
                vals = prior[col].dropna()
                if len(vals) >= 5:
                    result[key] = vals.mean()

        # Recent usage (last 3 days)
        three_days_ago = game_date - pd.Timedelta(days=3)
        recent = prior[prior["date"] >= three_days_ago]
        if "ip" in recent.columns:
            result["bullpen_ip_last3d"] = recent["ip"].sum()

        # Last 7 days ERA
        seven_days_ago = game_date - pd.Timedelta(days=7)
        week = prior[prior["date"] >= seven_days_ago]
        if "era" in week.columns and len(week) >= 3:
            result["bullpen_era_last7d"] = week["era"].mean()

        # Usage intensity last 3 days (higher = more fatigued)
        if "pitches" in recent.columns:
            result["bullpen_usage_3d"] = recent["pitches"].sum()

        return result


# ===================================================================
# Team Batting Computer
# ===================================================================

class TeamBattingComputer:
    """
    Computes team batting features from game-by-game batting data.
    """

    def __init__(self, team_batting_df):
        if team_batting_df is None or team_batting_df.empty:
            self.available = False
            return

        self.available = True
        df = team_batting_df.copy()
        df["date"] = pd.to_datetime(df["date"])

        numeric_cols = [
            "wrc_plus", "ops", "woba", "iso", "babip",
            "k_pct", "bb_pct", "runs_scored",
        ]
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")

        df = df.sort_values(["team", "date"]).reset_index(drop=True)
        self._df = df

        self._team_games = {}
        for team, group in df.groupby("team"):
            self._team_games[team] = group.reset_index(drop=True)

    def get_batting_stats(self, team, game_date):
        """Get team batting stats as of game_date."""
        if not self.available or team not in self._team_games:
            return {}

        game_date = pd.to_datetime(game_date)
        tg = self._team_games[team]
        prior = tg[(tg["date"] < game_date) & (tg["date"].dt.year == game_date.year)]

        if len(prior) < 10:
            return {}

        result = {}

        # Season-to-date
        for col, key in [
            ("wrc_plus", "wrc_plus"), ("ops", "ops"), ("woba", "woba"),
            ("iso", "iso"), ("babip", "babip"),
            ("k_pct", "k_pct_bat"), ("bb_pct", "bb_pct_bat"),
        ]:
            if col in prior.columns:
                vals = prior[col].dropna()
                if len(vals) >= 5:
                    result[key] = vals.mean()

        # Last 14 days
        two_weeks_ago = game_date - pd.Timedelta(days=14)
        recent = prior[prior["date"] >= two_weeks_ago]
        if len(recent) >= 5:
            if "wrc_plus" in recent.columns:
                result["wrc_plus_last14"] = recent["wrc_plus"].dropna().mean()
            if "ops" in recent.columns:
                result["ops_last14"] = recent["ops"].dropna().mean()

        # Scoring stats for rolling form
        if "runs_scored" in prior.columns:
            last10 = prior.tail(10)
            result["runs_scored_last10_avg"] = last10["runs_scored"].mean()

        return result


# ===================================================================
# Rolling Form Computer
# ===================================================================

class RollingFormComputer:
    """
    Precomputes rolling/cumulative stats per team from game-by-game results.
    Strictly uses only games BEFORE the query date (no lookahead).
    """

    def __init__(self, game_results_df):
        if game_results_df is None or game_results_df.empty:
            self.available = False
            return

        self.available = True
        df = game_results_df.copy()
        df["date"] = pd.to_datetime(df["date"])

        df["margin"] = pd.to_numeric(df.get("runs_scored", pd.Series(dtype=float)), errors="coerce") - \
                       pd.to_numeric(df.get("runs_allowed", pd.Series(dtype=float)), errors="coerce")
        df["win"] = (df["margin"] > 0).astype(float)

        df = df.sort_values(["team", "date"]).reset_index(drop=True)
        self._df = df

        self._team_games = {}
        for team, group in df.groupby("team"):
            self._team_games[team] = group.reset_index(drop=True)

    def get_form(self, team, game_date):
        """Get rolling form stats for a team as of game_date (exclusive)."""
        if not self.available or team not in self._team_games:
            return {}

        game_date = pd.to_datetime(game_date)
        tg = self._team_games[team]
        prior = tg[(tg["date"] < game_date) & (tg["date"].dt.year == game_date.year)]

        if len(prior) < 10:
            return {"games_played": len(prior)}

        result = {"games_played": len(prior)}

        # Win pct over last 10 and 20 games
        last10 = prior.tail(10)
        last20 = prior.tail(20)
        result["win_pct_10"] = last10["win"].mean()
        result["win_pct_20"] = last20["win"].mean()

        # Run differential over last 10 and 20
        result["run_diff_10"] = last10["margin"].mean()
        result["run_diff_20"] = last20["margin"].mean()

        # Scoring trend (slope of last 20 margins)
        if len(last20) >= 10:
            margins = last20["margin"].values
            x = np.arange(len(margins))
            if np.std(margins) > 0:
                result["scoring_trend"] = float(np.polyfit(x, margins, 1)[0])
            else:
                result["scoring_trend"] = 0.0

        # Streak
        wins_list = prior["win"].tolist()
        streak = 0
        if wins_list:
            last_val = wins_list[-1]
            for v in reversed(wins_list):
                if v == last_val:
                    streak += 1
                else:
                    break
            if last_val == 0:
                streak = -streak
        result["streak"] = streak

        # Pythagorean record
        if "runs_scored" in prior.columns and "runs_allowed" in prior.columns:
            rs = pd.to_numeric(prior["runs_scored"], errors="coerce").sum()
            ra = pd.to_numeric(prior["runs_allowed"], errors="coerce").sum()
            if rs + ra > 0:
                result["pythag_record"] = rs ** 1.83 / (rs ** 1.83 + ra ** 1.83)

        # Home/away records
        if "location" in prior.columns:
            home = prior[prior["location"] == "H"]
            away = prior[prior["location"] == "A"]
            if len(home) >= 5:
                result["home_win_pct"] = home["win"].mean()
            if len(away) >= 5:
                result["away_win_pct"] = away["win"].mean()

        return result


# ===================================================================
# Park Factors Computer
# ===================================================================

PARK_FACTORS = None  # Loaded lazily

def _load_park_factors():
    global PARK_FACTORS
    if PARK_FACTORS is not None:
        return PARK_FACTORS
    path = Path(__file__).parent / "data" / "park_factors.csv"
    if path.exists():
        PARK_FACTORS = pd.read_csv(path).set_index("team")
        log.info(f"Loaded park factors for {len(PARK_FACTORS)} venues")
    else:
        PARK_FACTORS = pd.DataFrame()
        log.warning("Park factors file not found")
    return PARK_FACTORS


def get_park_factors(home_team):
    """
    Returns park factor dict for the home team's venue.
    Factors > 1.0 = hitter-friendly, < 1.0 = pitcher-friendly.
    """
    pf = _load_park_factors()
    if pf.empty or home_team not in pf.index:
        return {}

    row = pf.loc[home_team]
    return {
        "park_run_factor": _sf(row.get("run_factor", np.nan)),
        "park_hr_factor": _sf(row.get("hr_factor", np.nan)),
        "park_hit_factor": _sf(row.get("hit_factor", np.nan)),
    }


# ===================================================================
# Main Feature Computation
# ===================================================================

def compute_all_features(
    home_team, away_team, game_date,
    home_sp_stats=None, away_sp_stats=None,
    home_bp_stats=None, away_bp_stats=None,
    home_bat_stats=None, away_bat_stats=None,
    home_form=None, away_form=None,
    park_factors=None, weather=None,
    odds=None, matchup_stats=None,
    home_team_record=None, away_team_record=None,
    venue_locations=None,
):
    """
    Compute all features for a single game.
    All inputs are dicts of pre-computed stats (from the Computer classes above).
    Returns dict of {feature_name: value}.

    CRITICAL: Every input must represent data available BEFORE game_date.
    """
    f = {}  # Feature dict
    home_sp = home_sp_stats or {}
    away_sp = away_sp_stats or {}
    home_bp = home_bp_stats or {}
    away_bp = away_bp_stats or {}
    home_bat = home_bat_stats or {}
    away_bat = away_bat_stats or {}
    home_f = home_form or {}
    away_f = away_form or {}
    park = park_factors or {}
    wx = weather or {}
    mkt = odds or {}
    mu = matchup_stats or {}

    # ---- Cat 1: Starting Pitcher (20) ----
    for key in [
        "sp_era", "sp_fip", "sp_xfip", "sp_whip", "sp_k_pct", "sp_bb_pct",
        "sp_k_bb", "sp_hr9", "sp_xwoba", "sp_hard_hit_pct",
        "sp_era_last3", "sp_fip_last3", "sp_ip_last3_avg", "sp_k_pct_last3",
        "sp_pitch_count_trend", "sp_days_rest", "sp_season_ip",
        "sp_groundball_pct", "sp_flyball_pct", "sp_barrel_pct",
    ]:
        # For most pitcher stats, LOWER is better for the pitcher, so:
        # diff = away - home (positive means home pitcher is better)
        # Exception: k_pct, k_bb, groundball_pct where HIGHER is better
        if key in ["sp_k_pct", "sp_k_bb", "sp_k_pct_last3",
                    "sp_groundball_pct", "sp_days_rest", "sp_season_ip",
                    "sp_ip_last3_avg"]:
            f[f"{key}_diff"] = _diff(home_sp.get(key), away_sp.get(key))
        else:
            f[f"{key}_diff"] = _diff(away_sp.get(key), home_sp.get(key))

    # ---- Cat 2: Bullpen (10) ----
    # Lower ERA/FIP/WHIP is better; higher K% is better
    for key in ["bullpen_era", "bullpen_fip", "bullpen_xfip", "bullpen_whip"]:
        f[f"{key}_diff"] = _diff(away_bp.get(key), home_bp.get(key))
    f["bullpen_k_pct_diff"] = _diff(home_bp.get("bullpen_k_pct"), away_bp.get("bullpen_k_pct"))
    # Usage/fatigue: higher = more tired = worse
    f["bullpen_usage_3d_diff"] = _diff(away_bp.get("bullpen_usage_3d"), home_bp.get("bullpen_usage_3d"))
    f["bullpen_ip_last3d_diff"] = _diff(away_bp.get("bullpen_ip_last3d"), home_bp.get("bullpen_ip_last3d"))
    f["bullpen_era_last7d_diff"] = _diff(away_bp.get("bullpen_era_last7d"), home_bp.get("bullpen_era_last7d"))
    # Placeholder for closer/high-leverage availability
    f["bullpen_high_lev_avail_diff"] = np.nan
    f["closer_available_diff"] = np.nan

    # ---- Cat 3: Team Batting (15) ----
    for key in ["wrc_plus", "ops", "woba", "iso", "babip", "bb_pct_bat",
                "wrc_plus_last14", "ops_last14",
                "runs_scored_last10_avg"]:
        f[f"{key}_diff"] = _diff(home_bat.get(key), away_bat.get(key))
    # K% batting: lower is better
    f["k_pct_bat_diff"] = _diff(away_bat.get("k_pct_bat"), home_bat.get("k_pct_bat"))
    # Run differential last 10
    f["run_diff_last10_diff"] = _diff(home_f.get("run_diff_10"), away_f.get("run_diff_10"))
    # Placeholders for split stats (require more complex data)
    f["wrc_plus_vs_hand_diff"] = np.nan
    f["ops_vs_hand_diff"] = np.nan
    f["home_wrc_plus_diff"] = np.nan
    f["away_wrc_plus_diff"] = np.nan

    # ---- Cat 4: Park Factors (5) ----
    f["park_run_factor"] = park.get("park_run_factor", np.nan)
    f["park_hr_factor"] = park.get("park_hr_factor", np.nan)
    f["park_hit_factor"] = park.get("park_hit_factor", np.nan)
    # Park factor advantage: how much the home team's batters benefit relative to visiting team
    # This will be computed in 05_build_features with team-level park adjustment
    f["park_factor_advantage"] = np.nan
    f["park_total_adjustment"] = np.nan

    # ---- Cat 5: Situational (12) ----
    f["home_flag"] = 1  # Always 1 from home team perspective
    f["rest_days_diff"] = np.nan  # Computed in 05_build_features
    f["travel_distance"] = np.nan  # Computed from venue_locations

    # Travel distance
    if venue_locations is not None and not venue_locations.empty:
        if away_team in venue_locations.index and home_team in venue_locations.index:
            away_loc = venue_locations.loc[away_team]
            home_loc = venue_locations.loc[home_team]
            f["travel_distance"] = _haversine(
                away_loc["lat"], away_loc["lon"],
                home_loc["lat"], home_loc["lon"],
            )

    f["day_night"] = wx.get("day_night", np.nan)
    f["series_game_num"] = np.nan  # 1, 2, or 3 within a series
    f["interleague"] = np.nan  # 1 if AL vs NL, 0 otherwise
    f["season_phase"] = np.nan  # Computed from game_date

    # Season phase (1-5: April=1, May-Jun=2, Jul=3, Aug-Sep=4, Oct=5)
    game_dt = pd.to_datetime(game_date)
    month = game_dt.month
    if month <= 4:
        f["season_phase"] = 1
    elif month <= 6:
        f["season_phase"] = 2
    elif month == 7:
        f["season_phase"] = 3
    elif month <= 9:
        f["season_phase"] = 4
    else:
        f["season_phase"] = 5

    # Days since All-Star break (mid-July)
    allstar_date = pd.Timestamp(year=game_dt.year, month=7, day=15)
    if game_dt > allstar_date:
        f["days_since_allstar"] = (game_dt - allstar_date).days
    else:
        f["days_since_allstar"] = 0

    # Records
    f["home_record_diff"] = _diff(
        home_f.get("home_win_pct"), away_f.get("away_win_pct")
    )
    f["away_record_diff"] = _diff(
        home_f.get("away_win_pct"), away_f.get("home_win_pct")
    )
    f["home_win_pct_last20"] = home_f.get("win_pct_20", np.nan)
    f["away_win_pct_last20"] = away_f.get("win_pct_20", np.nan)

    # ---- Cat 6: Weather (6) ----
    f["temperature"] = _sf(wx.get("temperature"))
    f["wind_speed"] = _sf(wx.get("wind_speed"))
    f["wind_direction_factor"] = _sf(wx.get("wind_direction_factor"))
    f["humidity"] = _sf(wx.get("humidity"))
    f["precipitation_prob"] = _sf(wx.get("precipitation_prob"))
    f["indoor_flag"] = 1 if wx.get("indoor", False) else 0

    # ---- Cat 7: Rolling Form (10) ----
    f["win_pct_10_diff"] = _diff(home_f.get("win_pct_10"), away_f.get("win_pct_10"))
    f["win_pct_20_diff"] = _diff(home_f.get("win_pct_20"), away_f.get("win_pct_20"))
    f["run_diff_10_diff"] = _diff(home_f.get("run_diff_10"), away_f.get("run_diff_10"))
    f["run_diff_20_diff"] = _diff(home_f.get("run_diff_20"), away_f.get("run_diff_20"))
    f["scoring_trend_diff"] = _diff(home_f.get("scoring_trend"), away_f.get("scoring_trend"))
    f["pitching_trend_diff"] = np.nan  # Placeholder
    f["streak_diff"] = _diff(home_f.get("streak"), away_f.get("streak"))
    f["games_played_diff"] = _diff(home_f.get("games_played"), away_f.get("games_played"))
    f["pythag_record_diff"] = _diff(home_f.get("pythag_record"), away_f.get("pythag_record"))
    f["base_running_diff"] = np.nan  # Placeholder

    # ---- Cat 8: Market-Derived (8) ----
    f["consensus_ml_home"] = _sf(mkt.get("consensus_ml_home"))
    f["consensus_ml_away"] = _sf(mkt.get("consensus_ml_away"))
    f["ml_implied_prob_home"] = _sf(mkt.get("ml_implied_prob_home"))
    f["ml_implied_prob_away"] = _sf(mkt.get("ml_implied_prob_away"))
    f["consensus_total"] = _sf(mkt.get("consensus_total"))
    f["consensus_runline"] = _sf(mkt.get("consensus_runline"))
    f["num_books"] = _sf(mkt.get("num_books"))
    f["has_odds"] = 1 if mkt.get("has_odds", False) else 0

    # ---- Cat 9: Matchup-Specific (6) ----
    f["sp_vs_team_ops"] = _sf(mu.get("sp_vs_team_ops"))
    f["sp_vs_team_woba"] = _sf(mu.get("sp_vs_team_woba"))
    f["team_vs_sp_hand_wrc"] = _sf(mu.get("team_vs_sp_hand_wrc"))
    f["lineup_wrc_plus_diff"] = _sf(mu.get("lineup_wrc_plus_diff"))
    f["batter_pitcher_history_ops"] = _sf(mu.get("batter_pitcher_history_ops"))
    f["platoon_advantage"] = _sf(mu.get("platoon_advantage"))

    return f


# Re-export all candidate feature names for other modules
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
