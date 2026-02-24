"""
00 — Build MLB Historical Training Data (v2)
=============================================
Assembles training data from:
  - game_results_all.csv              (25,611 games, 2015-2025)
  - pitcher_game_logs_mlbapi.csv      (51,222 SP logs with Statcast)
  - team_batting_game_logs.csv        (51,222 team batting logs)
  - historical_mlb_odds.csv           (Odds API, 2020+)
  - sports_statistics_mlb_odds.csv    (Sports-Statistics.com, 2015-2021)

Computes:
  - Targets: actual_margin, actual_total, actual_f5_margin, actual_f5_total, actual_nrfi
  - SP features: rolling rate stats (ERA, WHIP, FIP, K%, etc.) + Statcast (xwOBA, barrel%, etc.)
  - SP diffs: home_sp - away_sp for each feature
  - Team batting rolling stats (last 10 games)
  - Odds: consensus h2h, spread, total, F5/F1 markets

Output: data/historical/training_data_mlb_v2.csv
"""

import sys
import pandas as pd
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import HISTORICAL_DIR, get_logger
from feature_engine import StartingPitcherComputer

log = get_logger("00_mlb_build")

# ── File paths ───────────────────────────────────────────────────
GAMES_FILE = HISTORICAL_DIR / "game_results_all.csv"
PITCHER_LOGS_FILE = HISTORICAL_DIR / "pitcher_game_logs_mlbapi.csv"
BATTING_LOGS_FILE = HISTORICAL_DIR / "team_batting_game_logs.csv"
STATCAST_FILE = HISTORICAL_DIR / "statcast_pitcher_games.csv"
BULLPEN_LOGS_FILE = HISTORICAL_DIR / "bullpen_game_logs.csv"
ODDS_FILE = HISTORICAL_DIR / "historical_mlb_odds.csv"
SS_ODDS_FILE = HISTORICAL_DIR / "sports_statistics_mlb_odds.csv"
OUTPUT_FILE = HISTORICAL_DIR / "training_data_mlb_v2.csv"

# Rolling window for team batting stats
BATTING_WINDOW = 10


# ── Targets ──────────────────────────────────────────────────────
def compute_targets(df):
    """Compute all MLB targets from game results."""
    df["actual_margin"] = pd.to_numeric(df["home_runs"], errors="coerce") - \
                          pd.to_numeric(df["away_runs"], errors="coerce")
    df["actual_total"] = pd.to_numeric(df["home_runs"], errors="coerce") + \
                         pd.to_numeric(df["away_runs"], errors="coerce")

    h_f5 = pd.to_numeric(df["home_f5_runs"], errors="coerce")
    a_f5 = pd.to_numeric(df["away_f5_runs"], errors="coerce")
    df["actual_f5_margin"] = h_f5 - a_f5
    df["actual_f5_total"] = h_f5 + a_f5

    fi_home = pd.to_numeric(df["first_inning_home_runs"], errors="coerce").fillna(0)
    fi_away = pd.to_numeric(df["first_inning_away_runs"], errors="coerce").fillna(0)
    df["actual_nrfi"] = ((fi_home == 0) & (fi_away == 0)).astype(int)

    log.info(f"Targets computed:")
    log.info(f"  actual_margin mean: {df['actual_margin'].mean():.2f}")
    log.info(f"  actual_total mean: {df['actual_total'].mean():.1f}")
    log.info(f"  actual_f5_total mean: {df['actual_f5_total'].mean():.1f}")
    log.info(f"  NRFI rate: {df['actual_nrfi'].mean():.1%}")
    return df


# ── SP Features ──────────────────────────────────────────────────
def compute_sp_features(games_df, sp_computer):
    """
    For each game, look up both starting pitchers and compute diff features.
    """
    sp_feature_cols = [
        "sp_starts", "sp_era", "sp_whip", "sp_fip", "sp_xfip",
        "sp_k_per_9", "sp_bb_per_9", "sp_hr_per_9", "sp_k_pct", "sp_bb_pct",
        "sp_avg_ip", "sp_avg_pitches",
        "sp_xwoba", "sp_hard_hit_pct", "sp_barrel_pct",
        "sp_groundball_pct", "sp_flyball_pct", "sp_whiff_rate",
        "sp_era_last3", "sp_whip_last3", "sp_fip_last3", "sp_k_pct_last3",
        "sp_xwoba_last3", "sp_hard_hit_pct_last3", "sp_barrel_pct_last3",
        "sp_groundball_pct_last3", "sp_flyball_pct_last3", "sp_whiff_rate_last3",
    ]

    home_rows = []
    away_rows = []
    n_total = len(games_df)

    for i, (_, game) in enumerate(games_df.iterrows()):
        game_date = game["date"]
        home_sp_id = game.get("home_sp_id")
        away_sp_id = game.get("away_sp_id")

        # Get stats for each SP
        home_stats = {}
        away_stats = {}
        if pd.notna(home_sp_id):
            home_stats = sp_computer.get_pitcher_stats(int(home_sp_id), game_date)
        if pd.notna(away_sp_id):
            away_stats = sp_computer.get_pitcher_stats(int(away_sp_id), game_date)

        home_rows.append(home_stats)
        away_rows.append(away_stats)

        if (i + 1) % 2000 == 0:
            pct = (i + 1) / n_total * 100
            log.info(f"  SP features: [{i+1}/{n_total}] ({pct:.0f}%)")

    # Build DataFrames
    home_sp_df = pd.DataFrame(home_rows, index=games_df.index)
    away_sp_df = pd.DataFrame(away_rows, index=games_df.index)

    # Rename with home_/away_ prefix
    home_sp_df = home_sp_df.add_prefix("home_")
    away_sp_df = away_sp_df.add_prefix("away_")

    # Compute diffs (home - away, but for ERA/WHIP/FIP, lower is better for pitcher,
    # so a negative diff means the home SP is better)
    diff_df = pd.DataFrame(index=games_df.index)
    for col in sp_feature_cols:
        h_col = f"home_{col}"
        a_col = f"away_{col}"
        diff_col = f"{col}_diff"
        if h_col in home_sp_df.columns and a_col in away_sp_df.columns:
            diff_df[diff_col] = home_sp_df[h_col] - away_sp_df[a_col]

    result = pd.concat([home_sp_df, away_sp_df, diff_df], axis=1)

    # Coverage report
    for col in sp_feature_cols[:6]:
        h_pct = home_sp_df[f"home_{col}"].notna().mean() * 100
        log.info(f"  home_{col}: {h_pct:.0f}% populated")

    return result


# ── Team Batting Rolling Stats ───────────────────────────────────
def compute_team_batting_features(games_df, batting_df):
    """
    Compute rolling team batting averages (last N games) for home/away.
    """
    batting = batting_df.copy()
    batting["date"] = pd.to_datetime(batting["date"])
    for col in ["at_bats", "runs", "hits", "home_runs", "strikeouts",
                 "walks", "doubles", "triples"]:
        if col in batting.columns:
            batting[col] = pd.to_numeric(batting[col], errors="coerce")

    # Sort and build per-team lookup
    batting = batting.sort_values(["team_id", "date"]).reset_index(drop=True)
    team_games = {}
    for tid, group in batting.groupby("team_id"):
        team_games[tid] = group.reset_index(drop=True)

    rows = []
    n_total = len(games_df)

    for i, (_, game) in enumerate(games_df.iterrows()):
        game_date = pd.to_datetime(game["date"])
        home_tid = game.get("home_team_id")
        away_tid = game.get("away_team_id")

        row = {}
        for side, tid in [("home", home_tid), ("away", away_tid)]:
            if pd.isna(tid) or tid not in team_games:
                continue
            tg = team_games[tid]
            prior = tg[tg["date"] < game_date].tail(BATTING_WINDOW)
            if len(prior) < 3:
                continue

            total_ab = prior["at_bats"].sum()
            if total_ab > 0:
                row[f"{side}_batting_avg"] = prior["hits"].sum() / total_ab
                row[f"{side}_hr_rate"] = prior["home_runs"].sum() / total_ab
                row[f"{side}_k_rate"] = prior["strikeouts"].sum() / total_ab
                row[f"{side}_bb_rate"] = prior["walks"].sum() / total_ab
                row[f"{side}_iso"] = (
                    prior["doubles"].sum() + 2 * prior["triples"].sum() +
                    3 * prior["home_runs"].sum()
                ) / total_ab
            row[f"{side}_runs_per_game"] = prior["runs"].mean()

        rows.append(row)

        if (i + 1) % 2000 == 0:
            pct = (i + 1) / n_total * 100
            log.info(f"  Team batting: [{i+1}/{n_total}] ({pct:.0f}%)")

    bat_features = pd.DataFrame(rows, index=games_df.index)

    # Compute diffs
    for stat in ["batting_avg", "hr_rate", "k_rate", "bb_rate", "iso", "runs_per_game"]:
        h_col = f"home_{stat}"
        a_col = f"away_{stat}"
        if h_col in bat_features.columns and a_col in bat_features.columns:
            bat_features[f"{stat}_diff"] = bat_features[h_col] - bat_features[a_col]

    log.info(f"  Team batting features: {bat_features.shape[1]} columns")
    for col in ["home_batting_avg", "home_runs_per_game"]:
        if col in bat_features.columns:
            pct = bat_features[col].notna().mean() * 100
            log.info(f"  {col}: {pct:.0f}% populated")

    return bat_features


# ── Context Features (Park, Umpire, Weather, DH) ─────────────
def compute_context_features(games_df):
    """
    Compute venue/umpire/weather context features using O(n) accumulation.

    Features:
      park_factor        — Rolling 3yr home/road normalized: venue avg runs / home team's
                           road avg runs. Isolates venue effect from team quality (e.g.,
                           Pirates being bad doesn't deflate PNC Park). Min 50 venue games,
                           30 road games. Falls back to venue/league avg if road data unavailable.
      umpire_runs_factor — Rolling 3yr park-adjusted: divides each ump game's runs by
                           that venue's park factor before averaging. Controls for umpires
                           assigned to hitter/pitcher-friendly parks. Min 30 games.
      is_doubleheader    — 1 if doubleheader (S or Y), else 0.
      is_dome            — 1 if condition = Dome or Roof Closed.
      wind_out_mph       — Signed outbound wind (Out=+, In=-, cross=*0.3, dome=0).
    """
    from collections import defaultdict
    from bisect import bisect_left

    df = games_df.copy()
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)

    # Parse wind string: "12 mph, Out To RF" -> (12, "Out To RF")
    def parse_wind(wind_str):
        if pd.isna(wind_str):
            return 0.0, "None"
        parts = str(wind_str).split(", ", 1)
        try:
            mph = float(parts[0].replace(" mph", ""))
        except (ValueError, IndexError):
            return 0.0, "None"
        direction = parts[1] if len(parts) > 1 else "None"
        return mph, direction

    def wind_out_component(mph, direction, is_dome):
        """Convert wind to signed outbound component."""
        if is_dome:
            return 0.0
        direction_lower = direction.lower()
        if "out to" in direction_lower:
            return mph
        elif "in from" in direction_lower:
            return -mph
        elif direction_lower in ("l to r", "r to l"):
            return mph * 0.3
        else:  # None, Calm, Varies
            return 0.0

    # Accumulators for rolling 3yr stats
    # Park factor: home/road normalization (compare venue runs vs home team's road runs)
    venue_games = defaultdict(list)     # venue_name -> [(date_ord, total_runs)]
    team_road_games = defaultdict(list)  # team_id -> [(date_ord, total_runs)]
    venue_home_team = defaultdict(set)   # venue_name -> {home_team_id, ...}
    # Umpire factor: park-adjusted (controls for venue assignment bias)
    umpire_games = defaultdict(list)    # hp_umpire -> [(date_ord, total_runs, venue_name)]
    # Pre-computed park factors for umpire adjustment (two-pass not needed —
    # we use the park factor from the PREVIOUS iteration, so there's a tiny lag
    # that disappears with 30+ game minimums)
    venue_park_factor_cache = {}        # venue_name -> latest computed PF
    all_game_runs = []  # [(date_ord, total_runs)] for league average

    THREE_YEARS_DAYS = 3 * 365

    park_factors = []
    umpire_factors = []
    is_dh_list = []
    is_dome_list = []
    wind_out_list = []

    n_total = len(df)

    for i, (_, game) in enumerate(df.iterrows()):
        game_date = game["date"]
        date_ord = game_date.toordinal()
        cutoff_ord = date_ord - THREE_YEARS_DAYS

        venue = game.get("venue_name", "")
        umpire = game.get("hp_umpire", "")
        home_tid = game.get("home_team_id", np.nan)
        total_runs = pd.to_numeric(game.get("home_runs", np.nan), errors="coerce") + \
                     pd.to_numeric(game.get("away_runs", np.nan), errors="coerce")

        # ── Park factor: home/road normalization ──
        # Standard method: compare runs at venue vs. runs when the home team
        # plays on the road. This isolates the venue effect from team quality
        # (e.g., Pirates being bad doesn't deflate PNC Park's factor).
        pf = np.nan
        if pd.notna(venue) and venue != "":
            vg = venue_games[venue]
            start_idx = bisect_left([g[0] for g in vg], cutoff_ord)
            recent_venue = vg[start_idx:]
            if len(recent_venue) >= 50:
                venue_avg = np.mean([g[1] for g in recent_venue])
                # Get home team(s) for this venue in the window
                home_teams = venue_home_team.get(venue, set())
                if home_teams:
                    # Collect road games for the home team(s) in same window
                    road_runs = []
                    for tid in home_teams:
                        rg = team_road_games.get(tid, [])
                        r_start = bisect_left([g[0] for g in rg], cutoff_ord)
                        road_runs.extend(g[1] for g in rg[r_start:])
                    if len(road_runs) >= 30:
                        road_avg = np.mean(road_runs)
                        if road_avg > 0:
                            pf = venue_avg / road_avg
                # Fallback: if no road data yet, use league avg (early seasons)
                if np.isnan(pf) if isinstance(pf, float) else pd.isna(pf):
                    lg_start = bisect_left([g[0] for g in all_game_runs], cutoff_ord)
                    lg_recent = all_game_runs[lg_start:]
                    if len(lg_recent) >= 100:
                        league_avg = np.mean([g[1] for g in lg_recent])
                        if league_avg > 0:
                            pf = venue_avg / league_avg
            if pd.notna(pf):
                venue_park_factor_cache[venue] = pf
        park_factors.append(pf)

        # ── Umpire factor: park-adjusted ──
        # Compare umpire's game runs to what's expected given the venues they
        # worked at. This controls for umpires who happen to work more games
        # at hitter-friendly or pitcher-friendly parks.
        uf = np.nan
        if pd.notna(umpire) and umpire != "":
            ug = umpire_games[umpire]
            start_idx = bisect_left([g[0] for g in ug], cutoff_ord)
            recent_ump = ug[start_idx:]
            if len(recent_ump) >= 30:
                # Park-adjusted: for each game, divide runs by that venue's PF
                adjusted_runs = []
                raw_runs = []
                for g_date, g_runs, g_venue in recent_ump:
                    raw_runs.append(g_runs)
                    vpf = venue_park_factor_cache.get(g_venue, 1.0)
                    adjusted_runs.append(g_runs / vpf if vpf > 0 else g_runs)
                ump_adj_avg = np.mean(adjusted_runs)
                # League average (park-adjusted league avg ≈ league avg since
                # PFs are centered around 1.0)
                lg_start = bisect_left([g[0] for g in all_game_runs], cutoff_ord)
                lg_recent = all_game_runs[lg_start:]
                if len(lg_recent) >= 100:
                    league_avg = np.mean([g[1] for g in lg_recent])
                    if league_avg > 0:
                        uf = ump_adj_avg / league_avg
        umpire_factors.append(uf)

        # ── Doubleheader ──
        dh = game.get("doubleheader", "N")
        is_dh_list.append(1 if dh in ("S", "Y") else 0)

        # ── Dome ──
        cond = str(game.get("condition", ""))
        dome = cond.lower() in ("dome", "roof closed")
        is_dome_list.append(1 if dome else 0)

        # ── Wind ──
        mph, direction = parse_wind(game.get("wind"))
        wind_out_list.append(wind_out_component(mph, direction, dome))

        # ── Accumulate for future games (AFTER computing features) ──
        if pd.notna(total_runs):
            if pd.notna(venue) and venue != "":
                venue_games[venue].append((date_ord, total_runs))
                if pd.notna(home_tid):
                    venue_home_team[venue].add(int(home_tid))
            if pd.notna(umpire) and umpire != "":
                umpire_games[umpire].append((date_ord, total_runs, venue))
            # Track road games per team (away team is playing on the road)
            away_tid = game.get("away_team_id", np.nan)
            if pd.notna(away_tid):
                team_road_games[int(away_tid)].append((date_ord, total_runs))
            all_game_runs.append((date_ord, total_runs))

        if (i + 1) % 5000 == 0:
            pct = (i + 1) / n_total * 100
            log.info(f"  Context features: [{i+1}/{n_total}] ({pct:.0f}%)")

    result = pd.DataFrame({
        "park_factor": park_factors,
        "umpire_runs_factor": umpire_factors,
        "is_doubleheader": is_dh_list,
        "is_dome": is_dome_list,
        "wind_out_mph": wind_out_list,
    }, index=games_df.index)

    # Reindex to match original games_df order (we sorted by date internally)
    result = result.reindex(games_df.index)

    for col in ["park_factor", "umpire_runs_factor"]:
        pct = result[col].notna().mean() * 100
        log.info(f"  {col}: {pct:.0f}% populated")
    log.info(f"  is_doubleheader rate: {result['is_doubleheader'].mean():.1%}")
    log.info(f"  is_dome rate: {result['is_dome'].mean():.1%}")
    log.info(f"  wind_out_mph mean: {result['wind_out_mph'].mean():.1f}")

    return result


# ── Rest Features (SP Rest, SP Season IP, Team Rest) ─────────
def compute_rest_features(games_df, pitcher_logs):
    """
    Compute rest/workload features for SPs and teams.

    Features:
      sp_rest_days_diff  — Days since each SP's last start (home - away). Cap at 30.
      sp_season_ip_diff  — Cumulative IP this season (home - away).
      team_rest_days_diff — Days since last game (home - away). 0=DH, 1=normal.
    """
    from bisect import bisect_left

    df = games_df.copy()
    df["date"] = pd.to_datetime(df["date"])

    # ── Build per-pitcher sorted date list for rest/IP lookup ──
    pitcher_dates = {}   # pitcher_id -> sorted list of (date, season, ip)
    if pitcher_logs is not None:
        pl = pitcher_logs.copy()
        pl["date"] = pd.to_datetime(pl["date"])
        pl["ip"] = pd.to_numeric(pl["ip"], errors="coerce").fillna(0)
        pl = pl.sort_values("date")
        for pid, group in pl.groupby("pitcher_id"):
            pitcher_dates[pid] = [
                (row["date"], row["date"].year if row["date"].month >= 3 else row["date"].year - 1, row["ip"])
                for _, row in group.iterrows()
            ]

    # ── Build per-team sorted date list for team rest ──
    team_last_game = {}  # team_id -> sorted list of dates
    games_sorted = df.sort_values("date")
    for _, game in games_sorted.iterrows():
        game_date = game["date"]
        for side in ["home", "away"]:
            tid = game.get(f"{side}_team_id")
            if pd.notna(tid):
                if tid not in team_last_game:
                    team_last_game[tid] = []
                team_last_game[tid].append(game_date)

    rows = []
    n_total = len(df)

    for i, (_, game) in enumerate(df.iterrows()):
        game_date = pd.to_datetime(game["date"])
        # Season: year if month >= 3, else year - 1 (spring training starts in Feb/March)
        game_season = game_date.year if game_date.month >= 3 else game_date.year - 1
        row = {}

        # ── SP rest days and season IP ──
        for side in ["home", "away"]:
            sp_id = game.get(f"{side}_sp_id")
            if pd.notna(sp_id):
                sp_id = int(sp_id)
                entries = pitcher_dates.get(sp_id, [])
                # Find prior starts (before this game date)
                prior = [e for e in entries if e[0] < game_date]
                if prior:
                    last_start_date = prior[-1][0]
                    rest = (game_date - last_start_date).days
                    row[f"{side}_sp_rest_days"] = min(rest, 30)

                    # Season IP: sum IP from all starts in current season before this game
                    season_ip = sum(e[2] for e in prior if e[1] == game_season)
                    row[f"{side}_sp_season_ip"] = season_ip

        # ── Team rest days ──
        for side in ["home", "away"]:
            tid = game.get(f"{side}_team_id")
            if pd.notna(tid) and tid in team_last_game:
                team_dates = team_last_game[tid]
                # Find index of game dates < game_date using bisect
                idx = bisect_left(team_dates, game_date)
                if idx > 0:
                    prev_date = team_dates[idx - 1]
                    rest = (game_date - prev_date).days
                    row[f"{side}_team_rest_days"] = rest

        rows.append(row)

        if (i + 1) % 5000 == 0:
            pct = (i + 1) / n_total * 100
            log.info(f"  Rest features: [{i+1}/{n_total}] ({pct:.0f}%)")

    rest_df = pd.DataFrame(rows, index=df.index)

    # Compute diffs (home - away)
    for feat in ["sp_rest_days", "sp_season_ip", "team_rest_days"]:
        h_col = f"home_{feat}"
        a_col = f"away_{feat}"
        if h_col in rest_df.columns and a_col in rest_df.columns:
            rest_df[f"{feat}_diff"] = rest_df[h_col] - rest_df[a_col]

    # Coverage report
    for col in ["sp_rest_days_diff", "sp_season_ip_diff", "team_rest_days_diff"]:
        if col in rest_df.columns:
            pct = rest_df[col].notna().mean() * 100
            log.info(f"  {col}: {pct:.0f}% populated")
        else:
            log.warning(f"  {col}: not computed")

    return rest_df


# ── Momentum Features (Rolling Win%, Run Diff) ───────────────
def compute_momentum_features(games_df):
    """
    Compute rolling 10-game momentum features per team.

    Features:
      team_win_pct_10_diff  — Rolling 10-game win % (home - away). Min 3 games.
      team_run_diff_10_diff — Rolling 10-game run differential/game (home - away). Min 3 games.
    """
    WINDOW = 10
    MIN_GAMES = 3

    df = games_df.copy()
    df["date"] = pd.to_datetime(df["date"])
    df["home_runs_n"] = pd.to_numeric(df["home_runs"], errors="coerce")
    df["away_runs_n"] = pd.to_numeric(df["away_runs"], errors="coerce")

    # Build per-team game history: list of (date, won, run_diff) sorted by date
    team_history = {}  # team_id -> [(date, won: bool, run_diff: float)]
    for _, game in df.sort_values("date").iterrows():
        hr = game["home_runs_n"]
        ar = game["away_runs_n"]
        if pd.isna(hr) or pd.isna(ar):
            continue

        h_tid = game.get("home_team_id")
        a_tid = game.get("away_team_id")

        if pd.notna(h_tid):
            if h_tid not in team_history:
                team_history[h_tid] = []
            team_history[h_tid].append((game["date"], hr > ar, hr - ar))

        if pd.notna(a_tid):
            if a_tid not in team_history:
                team_history[a_tid] = []
            team_history[a_tid].append((game["date"], ar > hr, ar - hr))

    rows = []
    n_total = len(df)

    for i, (_, game) in enumerate(df.iterrows()):
        game_date = game["date"]
        row = {}

        for side in ["home", "away"]:
            tid = game.get(f"{side}_team_id")
            if pd.isna(tid) or tid not in team_history:
                continue
            hist = team_history[tid]
            # Get prior games (before this date)
            prior = [h for h in hist if h[0] < game_date]
            recent = prior[-WINDOW:]  # last N games
            if len(recent) >= MIN_GAMES:
                row[f"{side}_win_pct_10"] = np.mean([h[1] for h in recent])
                row[f"{side}_run_diff_10"] = np.mean([h[2] for h in recent])

        rows.append(row)

        if (i + 1) % 5000 == 0:
            pct = (i + 1) / n_total * 100
            log.info(f"  Momentum features: [{i+1}/{n_total}] ({pct:.0f}%)")

    mom_df = pd.DataFrame(rows, index=df.index)

    # Compute diffs
    for feat in ["win_pct_10", "run_diff_10"]:
        h_col = f"home_{feat}"
        a_col = f"away_{feat}"
        if h_col in mom_df.columns and a_col in mom_df.columns:
            mom_df[f"team_{feat}_diff"] = mom_df[h_col] - mom_df[a_col]

    # Coverage
    for col in ["team_win_pct_10_diff", "team_run_diff_10_diff"]:
        if col in mom_df.columns:
            pct = mom_df[col].notna().mean() * 100
            log.info(f"  {col}: {pct:.0f}% populated")

    return mom_df


# ── Bullpen Features (Rolling ERA, WHIP, Usage) ──────────────
def compute_bullpen_features(games_df, bullpen_df):
    """
    Compute rolling 10-game team bullpen features.

    Features:
      bullpen_era_diff   — Rolling 10-game team bullpen ERA (home - away).
      bullpen_whip_diff  — Rolling 10-game team bullpen WHIP (home - away).
      bullpen_usage_diff — Rolling 10-game bullpen IP per team game (home - away).
    """
    WINDOW = 10
    MIN_GAMES = 3

    df = games_df.copy()
    df["date"] = pd.to_datetime(df["date"])

    bp = bullpen_df.copy()
    bp["date"] = pd.to_datetime(bp["date"])
    for col in ["ip", "hits", "earned_runs", "walks"]:
        bp[col] = pd.to_numeric(bp[col], errors="coerce").fillna(0)

    # Aggregate bullpen stats per team per game
    bp_game = bp.groupby(["game_pk", "team_id"]).agg(
        bp_ip=("ip", "sum"),
        bp_hits=("hits", "sum"),
        bp_er=("earned_runs", "sum"),
        bp_walks=("walks", "sum"),
        bp_date=("date", "first"),
    ).reset_index()
    bp_game = bp_game.sort_values(["team_id", "bp_date"]).reset_index(drop=True)

    # Build per-team sorted game lists
    team_bp = {}
    for tid, group in bp_game.groupby("team_id"):
        team_bp[tid] = group.reset_index(drop=True)

    rows = []
    n_total = len(df)

    for i, (_, game) in enumerate(df.iterrows()):
        game_date = game["date"]
        row = {}

        for side in ["home", "away"]:
            tid = game.get(f"{side}_team_id")
            if pd.isna(tid) or tid not in team_bp:
                continue
            tg = team_bp[tid]
            prior = tg[tg["bp_date"] < game_date].tail(WINDOW)
            if len(prior) < MIN_GAMES:
                continue

            total_ip = prior["bp_ip"].sum()
            total_hits = prior["bp_hits"].sum()
            total_er = prior["bp_er"].sum()
            total_walks = prior["bp_walks"].sum()
            n_games = len(prior)

            # Bullpen ERA: (ER / IP) * 9
            if total_ip > 0:
                row[f"{side}_bullpen_era"] = (total_er / total_ip) * 9.0
                row[f"{side}_bullpen_whip"] = (total_walks + total_hits) / total_ip
            row[f"{side}_bullpen_usage"] = total_ip / n_games  # IP per game

        rows.append(row)

        if (i + 1) % 5000 == 0:
            pct = (i + 1) / n_total * 100
            log.info(f"  Bullpen features: [{i+1}/{n_total}] ({pct:.0f}%)")

    bp_features = pd.DataFrame(rows, index=df.index)

    # Compute diffs
    for feat in ["bullpen_era", "bullpen_whip", "bullpen_usage"]:
        h_col = f"home_{feat}"
        a_col = f"away_{feat}"
        if h_col in bp_features.columns and a_col in bp_features.columns:
            bp_features[f"{feat}_diff"] = bp_features[h_col] - bp_features[a_col]

    for col in ["bullpen_era_diff", "bullpen_whip_diff", "bullpen_usage_diff"]:
        if col in bp_features.columns:
            pct = bp_features[col].notna().mean() * 100
            log.info(f"  {col}: {pct:.0f}% populated")
        else:
            log.warning(f"  {col}: not computed (no bullpen data?)")

    return bp_features


# ── Bullpen Availability Features (Game-Day State) ────────────
def compute_bullpen_availability_features(games_df, bullpen_df):
    """
    Compute daily bullpen availability features from per-reliever appearance data.

    Unlike rolling aggregate bullpen_era_diff (which the market already prices),
    these features capture WHICH specific arms are available TODAY.

    Features:
      bp_availability_score_diff  — Weighted rest score across active bullpen.
      bp_high_leverage_rested_diff — Same but only top 3 relievers by season IP.
      bp_pitches_3d_diff          — Total reliever pitches in last 3 calendar days.
      bp_arms_unavailable_diff    — Count of relievers who threw 25+ pitches in last 2 days.
    """
    from collections import defaultdict

    df = games_df.copy()
    df["date"] = pd.to_datetime(df["date"])

    bp = bullpen_df.copy()
    bp["date"] = pd.to_datetime(bp["date"])
    bp["pitches_thrown"] = pd.to_numeric(bp["pitches_thrown"], errors="coerce").fillna(0)
    bp["ip"] = pd.to_numeric(bp["ip"], errors="coerce").fillna(0)

    # Build per-reliever appearance history: {(team_id, pitcher_id): [(date, pitches, ip)]}
    # Sorted by date for efficient lookback
    reliever_history = defaultdict(list)
    for _, row in bp.sort_values("date").iterrows():
        key = (row["team_id"], row["pitcher_id"])
        reliever_history[key].append((row["date"], row["pitches_thrown"], row["ip"]))

    # Build per-team active reliever set per season: {(team_id, season): set of pitcher_ids}
    # And cumulative IP for high-leverage identification
    team_season_relievers = defaultdict(lambda: defaultdict(float))
    for _, row in bp.iterrows():
        season = row["date"].year if row["date"].month >= 3 else row["date"].year - 1
        team_season_relievers[(row["team_id"], season)][row["pitcher_id"]] += row["ip"]

    rows = []
    n_total = len(df)

    for i, (_, game) in enumerate(df.iterrows()):
        game_date = game["date"]
        game_season = game_date.year if game_date.month >= 3 else game_date.year - 1
        row = {}

        for side in ["home", "away"]:
            tid = game.get(f"{side}_team_id")
            if pd.isna(tid):
                continue

            season_key = (tid, game_season)
            if season_key not in team_season_relievers:
                continue

            # Get all relievers active for this team this season (before this game)
            reliever_ip = {}
            for pid, total_ip in team_season_relievers[season_key].items():
                # Only count relievers with appearances before this game
                hist = reliever_history.get((tid, pid), [])
                prior = [h for h in hist if h[0] < game_date]
                if prior:
                    reliever_ip[pid] = sum(h[2] for h in prior)

            if not reliever_ip:
                continue

            # Top 3 relievers by season IP (high-leverage)
            top3_pids = set(sorted(reliever_ip, key=reliever_ip.get, reverse=True)[:3])

            # Compute availability metrics for each reliever
            availability_scores = []
            hl_scores = []
            total_pitches_3d = 0
            arms_unavailable = 0

            cutoff_14d = game_date - pd.Timedelta(days=14)
            cutoff_3d = game_date - pd.Timedelta(days=3)
            cutoff_2d = game_date - pd.Timedelta(days=2)

            for pid in reliever_ip:
                hist = reliever_history.get((tid, pid), [])
                recent_14d = [h for h in hist if cutoff_14d <= h[0] < game_date]

                if not recent_14d:
                    # No appearance in 14 days = fully rested
                    availability_scores.append(1.0)
                    if pid in top3_pids:
                        hl_scores.append(1.0)
                    continue

                # Days since last appearance
                last_date = max(h[0] for h in recent_14d)
                days_since = (game_date - last_date).days
                rest_score = min(days_since / 3.0, 1.0)

                # Pitches in last 3 days
                pitches_3d = sum(h[1] for h in recent_14d if h[0] >= cutoff_3d)
                fatigue = min(pitches_3d / 100.0, 1.0)

                avail = rest_score * (1.0 - fatigue)
                availability_scores.append(avail)

                if pid in top3_pids:
                    hl_scores.append(avail)

                total_pitches_3d += pitches_3d

                # Arms unavailable: 25+ pitches in any of last 2 days
                pitches_2d = sum(h[1] for h in recent_14d if h[0] >= cutoff_2d)
                if pitches_2d >= 25:
                    arms_unavailable += 1

            if availability_scores:
                row[f"{side}_bp_availability_score"] = np.mean(availability_scores)
            if hl_scores:
                row[f"{side}_bp_high_leverage_rested"] = np.mean(hl_scores)
            row[f"{side}_bp_pitches_3d"] = total_pitches_3d
            row[f"{side}_bp_arms_unavailable"] = arms_unavailable

        rows.append(row)

        if (i + 1) % 5000 == 0:
            pct = (i + 1) / n_total * 100
            log.info(f"  Bullpen availability: [{i+1}/{n_total}] ({pct:.0f}%)")

    avail_df = pd.DataFrame(rows, index=df.index)

    # Compute diffs
    for feat in ["bp_availability_score", "bp_high_leverage_rested",
                  "bp_pitches_3d", "bp_arms_unavailable"]:
        h_col = f"home_{feat}"
        a_col = f"away_{feat}"
        if h_col in avail_df.columns and a_col in avail_df.columns:
            avail_df[f"{feat}_diff"] = avail_df[h_col] - avail_df[a_col]

    for col in ["bp_availability_score_diff", "bp_high_leverage_rested_diff",
                "bp_pitches_3d_diff", "bp_arms_unavailable_diff"]:
        if col in avail_df.columns:
            pct = avail_df[col].notna().mean() * 100
            log.info(f"  {col}: {pct:.0f}% populated")
        else:
            log.warning(f"  {col}: not computed")

    return avail_df


# ── Travel & Fatigue Features ────────────────────────────────
def compute_travel_features(games_df):
    """
    Compute travel/fatigue features per team using stadium lat/lon.

    Features:
      travel_distance_diff    — Haversine miles from previous venue to this venue.
      road_trip_length_diff   — Consecutive away games for each team.
      total_distance_7d_diff  — Sum of travel distances in last 7 calendar days.
      tz_changes_7d_diff      — Count of timezone boundary crossings in last 7 days.
      games_in_7d_diff        — Games played in last 7 calendar days.
    """
    from math import radians, sin, cos, sqrt, atan2
    from collections import defaultdict

    # Load stadium locations
    stadium_file = Path(__file__).parent / "data" / "mlb_stadium_locations.csv"
    if not stadium_file.exists():
        log.warning(f"  Stadium locations not found: {stadium_file}")
        return pd.DataFrame(index=games_df.index)

    stadiums = pd.read_csv(stadium_file)
    venue_loc = {}   # venue_id -> (lat, lon)
    venue_tz = {}    # venue_id -> timezone offset hours from ET

    # Map timezone names to approximate UTC offsets for distance calculation
    tz_offsets = {
        "America/New_York": -5, "America/Chicago": -6, "America/Denver": -7,
        "America/Los_Angeles": -8, "America/Phoenix": -7,
        "America/Puerto_Rico": -4, "America/Mexico_City": -6,
        "Asia/Tokyo": 9, "Asia/Seoul": 9, "Europe/London": 0,
    }

    for _, row in stadiums.iterrows():
        vid = row["venue_id"]
        venue_loc[vid] = (row["lat"], row["lon"])
        venue_tz[vid] = tz_offsets.get(row["timezone"], -5)

    def haversine_miles(lat1, lon1, lat2, lon2):
        R = 3959  # Earth radius in miles
        dlat = radians(lat2 - lat1)
        dlon = radians(lon2 - lon1)
        a = sin(dlat / 2) ** 2 + cos(radians(lat1)) * cos(radians(lat2)) * sin(dlon / 2) ** 2
        return R * 2 * atan2(sqrt(a), sqrt(1 - a))

    df = games_df.copy()
    df["date"] = pd.to_datetime(df["date"])
    df["venue_id"] = pd.to_numeric(df["venue_id"], errors="coerce")

    # Build per-team sorted game history: [(date, venue_id, is_home)]
    team_games = defaultdict(list)
    for _, game in df.sort_values("date").iterrows():
        game_date = game["date"]
        vid = game["venue_id"]
        h_tid = game.get("home_team_id")
        a_tid = game.get("away_team_id")
        if pd.notna(h_tid):
            team_games[h_tid].append((game_date, vid, True))
        if pd.notna(a_tid):
            team_games[a_tid].append((game_date, vid, False))

    rows = []
    n_total = len(df)

    for i, (_, game) in enumerate(df.iterrows()):
        game_date = game["date"]
        vid = game["venue_id"]
        row = {}

        for side in ["home", "away"]:
            tid = game.get(f"{side}_team_id")
            if pd.isna(tid) or tid not in team_games:
                continue

            hist = team_games[tid]
            # Get prior games (before this date)
            prior = [h for h in hist if h[0] < game_date]
            if not prior:
                continue

            # Travel distance: from last game's venue to this venue
            last_vid = prior[-1][1]
            if pd.notna(vid) and vid in venue_loc and pd.notna(last_vid) and last_vid in venue_loc:
                lat1, lon1 = venue_loc[last_vid]
                lat2, lon2 = venue_loc[vid]
                dist = haversine_miles(lat1, lon1, lat2, lon2)
                row[f"{side}_travel_distance"] = dist

            # Road trip length: consecutive away games ending now
            road_count = 0
            for pg in reversed(prior):
                if not pg[2]:  # is_home = False means away
                    road_count += 1
                else:
                    break
            # If current game is away, add 1 for this game
            if side == "away":
                road_count += 1
            row[f"{side}_road_trip_length"] = road_count

            # Last 7 days stats
            cutoff_7d = game_date - pd.Timedelta(days=7)
            recent_7d = [h for h in prior if h[0] >= cutoff_7d]

            # Games in 7d
            row[f"{side}_games_in_7d"] = len(recent_7d)

            # Total distance in 7d
            total_dist_7d = 0.0
            # Build venue sequence for last 7 days
            recent_venues = [h[1] for h in recent_7d]
            if recent_venues:
                # Include travel from last pre-7d game to first 7d game
                pre_7d = [h for h in prior if h[0] < cutoff_7d]
                if pre_7d:
                    recent_venues = [pre_7d[-1][1]] + recent_venues
                for j in range(1, len(recent_venues)):
                    v1, v2 = recent_venues[j - 1], recent_venues[j]
                    if pd.notna(v1) and v1 in venue_loc and pd.notna(v2) and v2 in venue_loc:
                        total_dist_7d += haversine_miles(*venue_loc[v1], *venue_loc[v2])
                # Also add travel to current game venue
                if pd.notna(vid) and vid in venue_loc and pd.notna(recent_venues[-1]) and recent_venues[-1] in venue_loc:
                    total_dist_7d += haversine_miles(*venue_loc[recent_venues[-1]], *venue_loc[vid])
            row[f"{side}_total_distance_7d"] = total_dist_7d

            # Timezone changes in 7d
            tz_changes = 0
            venue_seq = recent_venues if recent_venues else []
            if vid in venue_tz:
                venue_seq = venue_seq + [vid]
            for j in range(1, len(venue_seq)):
                v1, v2 = venue_seq[j - 1], venue_seq[j]
                if pd.notna(v1) and v1 in venue_tz and pd.notna(v2) and v2 in venue_tz:
                    if venue_tz[v1] != venue_tz[v2]:
                        tz_changes += 1
            row[f"{side}_tz_changes_7d"] = tz_changes

        rows.append(row)

        if (i + 1) % 5000 == 0:
            pct = (i + 1) / n_total * 100
            log.info(f"  Travel features: [{i+1}/{n_total}] ({pct:.0f}%)")

    travel_df = pd.DataFrame(rows, index=df.index)

    # Compute diffs
    for feat in ["travel_distance", "road_trip_length", "total_distance_7d",
                  "tz_changes_7d", "games_in_7d"]:
        h_col = f"home_{feat}"
        a_col = f"away_{feat}"
        if h_col in travel_df.columns and a_col in travel_df.columns:
            travel_df[f"{feat}_diff"] = travel_df[h_col] - travel_df[a_col]

    for col in ["travel_distance_diff", "road_trip_length_diff",
                "total_distance_7d_diff", "tz_changes_7d_diff", "games_in_7d_diff"]:
        if col in travel_df.columns:
            pct = travel_df[col].notna().mean() * 100
            log.info(f"  {col}: {pct:.0f}% populated")
        else:
            log.warning(f"  {col}: not computed")

    return travel_df


# ── Schedule Context Features ────────────────────────────────
def compute_schedule_context_features(games_df):
    """
    Compute schedule context features.

    Features:
      is_interleague       — Teams from different leagues (AL vs NL).
      series_game_num      — Which game in a consecutive series at same venue (1-4).
      post_allstar_diff    — Binary: first 5 games after All-Star break per team.
    """
    from collections import defaultdict

    # AL/NL team mapping (2024+ realignment: Oakland→Sacramento, same league)
    AL_TEAMS = {108, 110, 111, 114, 116, 117, 118, 133, 136, 139, 140, 141, 142, 145, 147}
    NL_TEAMS = {109, 112, 113, 115, 119, 120, 121, 134, 135, 137, 138, 143, 144, 146, 158}

    df = games_df.copy()
    df["date"] = pd.to_datetime(df["date"])

    # ── Is interleague ──
    def is_interleague(h_tid, a_tid):
        if pd.isna(h_tid) or pd.isna(a_tid):
            return np.nan
        h_al = h_tid in AL_TEAMS
        a_al = a_tid in AL_TEAMS
        if h_tid not in AL_TEAMS and h_tid not in NL_TEAMS:
            return np.nan
        if a_tid not in AL_TEAMS and a_tid not in NL_TEAMS:
            return np.nan
        return int(h_al != a_al)

    il_list = []
    for _, game in df.iterrows():
        il_list.append(is_interleague(game.get("home_team_id"), game.get("away_team_id")))

    # ── Series game number ──
    # Group by (home_team_id, away_team_id, venue_id) and sort by date within each group
    # to detect consecutive games in the same series
    df_with_idx = df.copy()
    df_with_idx["_orig_idx"] = df.index
    df_with_idx = df_with_idx.sort_values(["home_team_id", "away_team_id", "venue_id", "date"])

    series_num_map = {}
    prev_row = None
    current_series = 1

    for _, row in df_with_idx.iterrows():
        if prev_row is not None:
            same_home = row["home_team_id"] == prev_row["home_team_id"]
            same_away = row["away_team_id"] == prev_row["away_team_id"]
            same_venue = row["venue_id"] == prev_row["venue_id"]
            date_gap = (row["date"] - prev_row["date"]).days
            if same_home and same_away and same_venue and 0 <= date_gap <= 2:
                current_series += 1
            else:
                current_series = 1
        else:
            current_series = 1

        series_num_map[row["_orig_idx"]] = current_series
        prev_row = row

    series_map = series_num_map

    # ── Post All-Star break (first 5 games after 3+ day mid-July gap) ──
    team_games = defaultdict(list)
    for _, game in df.sort_values("date").iterrows():
        game_date = game["date"]
        for side in ["home", "away"]:
            tid = game.get(f"{side}_team_id")
            if pd.notna(tid):
                team_games[tid].append(game_date)

    # Find All-Star break per team per season: longest gap >= 3 days in July
    team_post_asb = defaultdict(set)  # (team_id, season) -> set of post-ASB game dates
    for tid, dates in team_games.items():
        dates_sorted = sorted(set(dates))
        for j in range(1, len(dates_sorted)):
            gap = (dates_sorted[j] - dates_sorted[j - 1]).days
            month = dates_sorted[j - 1].month
            if gap >= 3 and month in (6, 7):
                # Found ASB — mark next 5 games
                season = dates_sorted[j].year
                post_games = [d for d in dates_sorted[j:] if d >= dates_sorted[j]][:5]
                for d in post_games:
                    team_post_asb[(tid, season)].add(d)
                break  # One ASB per season

    # Build feature columns
    post_asb_home = []
    post_asb_away = []
    series_nums = []

    for idx, (_, game) in enumerate(df.iterrows()):
        game_date = game["date"]
        season = game_date.year if game_date.month >= 3 else game_date.year - 1

        h_tid = game.get("home_team_id")
        a_tid = game.get("away_team_id")

        h_post = 0
        a_post = 0
        if pd.notna(h_tid):
            h_post = int(game_date in team_post_asb.get((h_tid, season), set()))
        if pd.notna(a_tid):
            a_post = int(game_date in team_post_asb.get((a_tid, season), set()))

        post_asb_home.append(h_post)
        post_asb_away.append(a_post)

        # Series game number from sorted mapping
        orig_idx = df.index[idx]
        series_nums.append(series_map.get(orig_idx, 1))

    result = pd.DataFrame({
        "is_interleague": il_list,
        "series_game_num": series_nums,
        "home_post_allstar": post_asb_home,
        "away_post_allstar": post_asb_away,
    }, index=df.index)

    # Compute diff for post_allstar
    result["post_allstar_diff"] = result["home_post_allstar"] - result["away_post_allstar"]

    log.info(f"  is_interleague rate: {result['is_interleague'].mean():.1%}")
    log.info(f"  series_game_num mean: {result['series_game_num'].mean():.1f}")
    log.info(f"  post_allstar games: {(result['home_post_allstar'] + result['away_post_allstar']).gt(0).sum()}")

    return result


# ── Lineup Features (Phase 3: Player-Level) ──────────────────
def compute_lineup_features(games_df, batter_df, handedness_df, pitcher_logs):
    """
    Compute player-level lineup features from individual batter data.

    Requires:
      - batter_game_logs.csv (per-batter per-game stats)
      - player_handedness.csv (bat/pitch side for all players)
      - pitcher_game_logs_mlbapi.csv (for SP handedness cross-reference)

    Features:
      lineup_ops_diff           — Avg rolling 20-game OPS of actual starters.
      lineup_power_diff         — Avg rolling 20-game ISO of actual starters.
      lineup_k_rate_diff        — Weighted rolling 20-game K-rate of starters.
      platoon_advantage_pct_diff — % of lineup with platoon advantage vs opposing SP.
      star_missing_ops_diff     — Sum of missing top-3 batters' OPS (not in lineup).
      lineup_continuity_diff    — Fraction of starters who also started previous game.
      lineup_hot_streak_diff    — Avg (last-5-OPS minus season-OPS) across starters.
      lineup_obp_diff           — Avg rolling 20-game OBP of actual starters.
    """
    from collections import defaultdict

    ROLLING_WINDOW = 20
    STREAK_WINDOW = 5
    MIN_GAMES = 5

    df = games_df.copy()
    df["date"] = pd.to_datetime(df["date"])

    bat = batter_df.copy()
    bat["date"] = pd.to_datetime(bat["date"])
    for col in ["at_bats", "hits", "doubles", "home_runs", "walks",
                 "strikeouts", "stolen_bases"]:
        if col in bat.columns:
            bat[col] = pd.to_numeric(bat[col], errors="coerce").fillna(0)

    # Build handedness lookup: player_id -> bat_side ('L', 'R', 'S')
    hand_lookup = {}
    if handedness_df is not None:
        for _, row in handedness_df.iterrows():
            hand_lookup[row["player_id"]] = {
                "bat_side": row.get("bat_side", "R"),
                "pitch_hand": row.get("pitch_hand", "R"),
            }

    # Build SP handedness from pitcher_logs + handedness
    sp_hand = {}  # pitcher_id -> 'L' or 'R'
    if pitcher_logs is not None:
        for pid in pitcher_logs["pitcher_id"].unique():
            h = hand_lookup.get(pid, {})
            sp_hand[pid] = h.get("pitch_hand", "R")

    # Build per-batter sorted game history: batter_id -> [(date, ab, h, 2b, hr, bb, k)]
    batter_history = defaultdict(list)
    for _, row in bat.sort_values("date").iterrows():
        bid = row["batter_id"]
        batter_history[bid].append((
            row["date"],
            row["at_bats"],
            row["hits"],
            row.get("doubles", 0),
            row["home_runs"],
            row["walks"],
            row["strikeouts"],
        ))

    # Build per-game lineup: game_pk -> {side: [batter_ids in order]}
    game_lineups = defaultdict(lambda: defaultdict(list))
    for _, row in bat.sort_values(["game_pk", "side", "batting_order"]).iterrows():
        bo = pd.to_numeric(row.get("batting_order", 0), errors="coerce")
        if pd.notna(bo) and bo > 0:
            game_lineups[row["game_pk"]][row["side"]].append(row["batter_id"])

    # Build per-team season top-3 batters by cumulative OPS
    # We'll track this incrementally
    team_season_batter_stats = defaultdict(lambda: defaultdict(lambda: {"ab": 0, "h": 0, "bb": 0, "hr": 0, "2b": 0, "sf": 0}))

    # Pre-compute per-team season stats from batter data
    for _, row in bat.iterrows():
        season = row["date"].year if row["date"].month >= 3 else row["date"].year - 1
        tid = row["team_id"]
        bid = row["batter_id"]
        key = (tid, season)
        stats = team_season_batter_stats[key][bid]
        stats["ab"] += row["at_bats"]
        stats["h"] += row["hits"]
        stats["bb"] += row["walks"]
        stats["hr"] += row["home_runs"]
        stats["2b"] += row.get("doubles", 0)

    # Track previous game lineup per team for continuity
    prev_lineup = {}  # team_id -> set of batter_ids

    rows = []
    n_total = len(df)

    for i, (_, game) in enumerate(df.iterrows()):
        game_date = game["date"]
        game_pk = game["game_pk"]
        game_season = game_date.year if game_date.month >= 3 else game_date.year - 1
        row = {}

        for side in ["home", "away"]:
            tid = game.get(f"{side}_team_id")
            sp_id = game.get(f"{'away' if side == 'home' else 'home'}_sp_id")  # opposing SP

            if pd.isna(tid):
                continue

            # Get lineup for this game
            lineup_side = "home" if side == "home" else "away"
            lineup = game_lineups.get(game_pk, {}).get(lineup_side, [])
            if len(lineup) < 5:
                continue

            # Compute rolling stats for each batter in lineup
            ops_list = []
            iso_list = []
            k_rate_list = []
            obp_list = []
            streak_list = []
            platoon_count = 0

            # Get opposing SP handedness
            opp_sp_hand = "R"
            if pd.notna(sp_id):
                opp_sp_hand = sp_hand.get(int(sp_id), "R")

            for order_idx, bid in enumerate(lineup[:9]):
                hist = batter_history.get(bid, [])
                prior = [h for h in hist if h[0] < game_date]

                if len(prior) >= MIN_GAMES:
                    # Rolling window stats
                    recent = prior[-ROLLING_WINDOW:]
                    total_ab = sum(h[1] for h in recent)
                    total_h = sum(h[2] for h in recent)
                    total_2b = sum(h[3] for h in recent)
                    total_hr = sum(h[4] for h in recent)
                    total_bb = sum(h[5] for h in recent)
                    total_k = sum(h[6] for h in recent)
                    total_pa = total_ab + total_bb

                    if total_ab > 0:
                        avg = total_h / total_ab
                        slg = (total_h - total_2b - total_hr + 2 * total_2b + 4 * total_hr) / total_ab
                        iso_val = slg - avg
                        iso_list.append(iso_val)

                    if total_pa > 0:
                        obp_val = (total_h + total_bb) / total_pa
                        ops_val = obp_val + (slg if total_ab > 0 else 0)
                        ops_list.append(ops_val)
                        obp_list.append(obp_val)

                    if total_pa > 0:
                        k_rate = total_k / total_pa
                        # Weight top 4 batters 1.5x
                        weight = 1.5 if order_idx < 4 else 1.0
                        k_rate_list.append((k_rate, weight))

                    # Streak: last 5 games OPS minus season OPS
                    if len(prior) >= ROLLING_WINDOW:
                        last5 = prior[-STREAK_WINDOW:]
                        l5_ab = sum(h[1] for h in last5)
                        l5_h = sum(h[2] for h in last5)
                        l5_2b = sum(h[3] for h in last5)
                        l5_hr = sum(h[4] for h in last5)
                        l5_bb = sum(h[5] for h in last5)
                        l5_pa = l5_ab + l5_bb

                        if l5_pa > 0 and l5_ab > 0:
                            l5_avg = l5_h / l5_ab
                            l5_slg = (l5_h - l5_2b - l5_hr + 2 * l5_2b + 4 * l5_hr) / l5_ab
                            l5_obp = (l5_h + l5_bb) / l5_pa
                            l5_ops = l5_obp + l5_slg

                            # Season OPS from all prior games
                            s_ab = sum(h[1] for h in prior)
                            s_h = sum(h[2] for h in prior)
                            s_2b = sum(h[3] for h in prior)
                            s_hr = sum(h[4] for h in prior)
                            s_bb = sum(h[5] for h in prior)
                            s_pa = s_ab + s_bb
                            if s_pa > 0 and s_ab > 0:
                                s_avg = s_h / s_ab
                                s_slg = (s_h - s_2b - s_hr + 2 * s_2b + 4 * s_hr) / s_ab
                                s_obp = (s_h + s_bb) / s_pa
                                s_ops = s_obp + s_slg
                                streak_list.append(l5_ops - s_ops)

                # Platoon advantage
                bat_side = hand_lookup.get(bid, {}).get("bat_side", "R")
                if bat_side == "S":
                    platoon_count += 1  # Switch hitters always have advantage
                elif bat_side == "L" and opp_sp_hand == "R":
                    platoon_count += 1
                elif bat_side == "R" and opp_sp_hand == "L":
                    platoon_count += 1

            # Store features
            n_batters = min(len(lineup), 9)
            if ops_list:
                row[f"{side}_lineup_ops"] = np.mean(ops_list)
            if iso_list:
                row[f"{side}_lineup_power"] = np.mean(iso_list)
            if k_rate_list:
                total_weight = sum(w for _, w in k_rate_list)
                row[f"{side}_lineup_k_rate"] = sum(k * w for k, w in k_rate_list) / total_weight
            if obp_list:
                row[f"{side}_lineup_obp"] = np.mean(obp_list)
            if streak_list:
                row[f"{side}_lineup_hot_streak"] = np.mean(streak_list)

            row[f"{side}_platoon_advantage_pct"] = platoon_count / n_batters if n_batters > 0 else 0

            # Star missing: identify top 3 by season OPS, check if in today's lineup
            season_key = (tid, game_season)
            season_stats = team_season_batter_stats.get(season_key, {})
            # Compute OPS for each batter with enough ABs
            batter_ops_season = {}
            for bid2, stats in season_stats.items():
                if stats["ab"] >= 50:
                    avg2 = stats["h"] / stats["ab"]
                    slg2 = (stats["h"] - stats["2b"] - stats["hr"] + 2 * stats["2b"] + 4 * stats["hr"]) / stats["ab"]
                    pa2 = stats["ab"] + stats["bb"]
                    obp2 = (stats["h"] + stats["bb"]) / pa2 if pa2 > 0 else 0
                    batter_ops_season[bid2] = obp2 + slg2

            top3_pids = sorted(batter_ops_season, key=batter_ops_season.get, reverse=True)[:3]
            lineup_set = set(lineup[:9])
            missing_ops = sum(batter_ops_season.get(p, 0) for p in top3_pids if p not in lineup_set)
            row[f"{side}_star_missing_ops"] = missing_ops

            # Lineup continuity
            prev = prev_lineup.get(tid, set())
            if prev:
                overlap = len(lineup_set & prev) / max(len(lineup_set), 1)
                row[f"{side}_lineup_continuity"] = overlap

            # Update previous lineup tracker
            prev_lineup[tid] = lineup_set

        rows.append(row)

        if (i + 1) % 5000 == 0:
            pct = (i + 1) / n_total * 100
            log.info(f"  Lineup features: [{i+1}/{n_total}] ({pct:.0f}%)")

    lineup_df = pd.DataFrame(rows, index=df.index)

    # Compute diffs
    for feat in ["lineup_ops", "lineup_power", "lineup_k_rate", "lineup_obp",
                  "lineup_hot_streak", "platoon_advantage_pct",
                  "star_missing_ops", "lineup_continuity"]:
        h_col = f"home_{feat}"
        a_col = f"away_{feat}"
        if h_col in lineup_df.columns and a_col in lineup_df.columns:
            lineup_df[f"{feat}_diff"] = lineup_df[h_col] - lineup_df[a_col]

    for col in ["lineup_ops_diff", "lineup_power_diff", "lineup_k_rate_diff",
                "platoon_advantage_pct_diff", "star_missing_ops_diff",
                "lineup_continuity_diff", "lineup_hot_streak_diff", "lineup_obp_diff"]:
        if col in lineup_df.columns:
            pct = lineup_df[col].notna().mean() * 100
            log.info(f"  {col}: {pct:.0f}% populated")
        else:
            log.warning(f"  {col}: not computed (no data)")

    return lineup_df


# ── Odds Merge ───────────────────────────────────────────────────
def merge_odds(games_df, odds_df, ss_odds_df=None):
    """
    Merge historical odds onto game results from two sources:
    1. Odds API (2020+, multi-book consensus, but has extreme outliers)
    2. Sports-Statistics.com (2010-2021, single-book closing lines, clean)

    Strategy: Use Odds API as primary for 2022+. For 2019-2021, use SS odds
    as primary (cleaner). Backfill any gaps with SS data. Clean extreme values.
    """
    odds = odds_df.copy()

    # Clean extreme Odds API values: clip H2H to [-1000, +1000] range
    # Normal MLB moneylines are rarely beyond -500/+500
    MAX_ML = 1000
    for col in ["consensus_h2h_home", "consensus_h2h_away"]:
        if col in odds.columns:
            extreme = odds[col].abs() > MAX_ML
            if extreme.any():
                log.info(f"  Clipping {extreme.sum()} extreme {col} values (|val| > {MAX_ML})")
                odds.loc[extreme, col] = np.nan  # Set to NaN rather than clip

    # Extract date from commence_time for matching
    odds["odds_date"] = pd.to_datetime(odds["commence_time"]).dt.strftime("%Y-%m-%d")

    # Build Odds API lookup by (date, home_team)
    oa_lookup = {}
    for _, row in odds.iterrows():
        key = (row["odds_date"], row["home_team"])
        oa_lookup[key] = row

    # Build SS odds lookup by (date, home_abbrev) if available
    ss_lookup = {}
    if ss_odds_df is not None:
        for _, row in ss_odds_df.iterrows():
            key = (row["date"], row["home_abbrev"])
            ss_lookup[key] = row
        log.info(f"  SS odds lookup: {len(ss_lookup)} games")

    odds_cols = [
        "consensus_h2h_home", "consensus_h2h_away",
        "consensus_spread", "consensus_total",
        "consensus_f5_h2h_home", "consensus_f5_h2h_away",
        "consensus_f5_total", "consensus_f1_total",
        "num_books",
    ]

    matched_oa = 0
    matched_ss = 0
    rows = []
    for _, game in games_df.iterrows():
        game_date = pd.to_datetime(game["date"]).strftime("%Y-%m-%d")
        home_team = game["home_team"]
        home_abbrev = game.get("home_abbrev", "")

        result = {col: None for col in odds_cols}

        # Try Odds API first
        oa_row = oa_lookup.get((game_date, home_team))
        if oa_row is not None:
            for col in odds_cols:
                val = oa_row.get(col)
                if pd.notna(val):
                    result[col] = val
            if pd.notna(result.get("consensus_h2h_home")):
                matched_oa += 1

        # Backfill H2H and total from SS odds if OA is missing
        ss_row = ss_lookup.get((game_date, home_abbrev))
        if ss_row is not None:
            if pd.isna(result.get("consensus_h2h_home")) and pd.notna(ss_row.get("ss_h2h_home")):
                result["consensus_h2h_home"] = ss_row["ss_h2h_home"]
                result["consensus_h2h_away"] = ss_row["ss_h2h_away"]
                if pd.isna(result.get("num_books")):
                    result["num_books"] = 1  # Single source
                matched_ss += 1
            if pd.isna(result.get("consensus_total")) and pd.notna(ss_row.get("ss_total")):
                result["consensus_total"] = ss_row["ss_total"]

        rows.append(result)

    odds_features = pd.DataFrame(rows, index=games_df.index)
    total_h2h = odds_features["consensus_h2h_home"].notna().sum()
    match_pct = total_h2h / len(games_df) * 100
    log.info(f"  Odds API matched: {matched_oa}")
    log.info(f"  SS backfilled: {matched_ss}")
    log.info(f"  Total H2H coverage: {total_h2h}/{len(games_df)} ({match_pct:.0f}%)")

    for col in ["consensus_spread", "consensus_total", "consensus_f5_total"]:
        if col in odds_features.columns:
            pct = odds_features[col].notna().mean() * 100
            log.info(f"  {col}: {pct:.0f}% populated")

    return odds_features


# ── Main ─────────────────────────────────────────────────────────
def main():
    HISTORICAL_DIR.mkdir(parents=True, exist_ok=True)

    # ── Load data ────────────────────────────────────────────────
    log.info("=" * 60)
    log.info("Loading data files")
    log.info("=" * 60)

    if not GAMES_FILE.exists():
        log.error(f"Game results not found: {GAMES_FILE}")
        log.error("Run scripts/fetch_historical_games.py first")
        sys.exit(1)

    games = pd.read_csv(GAMES_FILE)
    games["date"] = pd.to_datetime(games["date"])
    log.info(f"Games: {len(games)} rows ({games['date'].min()} to {games['date'].max()})")

    pitcher_logs = None
    if PITCHER_LOGS_FILE.exists():
        pitcher_logs = pd.read_csv(PITCHER_LOGS_FILE)
        log.info(f"Pitcher logs: {len(pitcher_logs)} rows")
        # Merge Statcast data (xwoba, barrel_pct, hard_hit_pct, etc.)
        if STATCAST_FILE.exists():
            statcast = pd.read_csv(STATCAST_FILE)
            statcast_cols = ["pitcher_id", "game_pk", "statcast_pitches",
                             "xwoba", "hard_hit_pct", "barrel_pct",
                             "groundball_pct", "flyball_pct", "whiff_rate"]
            statcast = statcast[[c for c in statcast_cols if c in statcast.columns]]
            pitcher_logs = pitcher_logs.merge(statcast, on=["pitcher_id", "game_pk"], how="left")
            sc_pct = pitcher_logs["xwoba"].notna().mean()
            log.info(f"Statcast merged: {sc_pct:.0%} coverage ({len(statcast)} rows)")
        else:
            log.warning(f"Statcast file not found: {STATCAST_FILE}")
    else:
        log.warning(f"Pitcher logs not found: {PITCHER_LOGS_FILE}")

    batting_logs = None
    if BATTING_LOGS_FILE.exists():
        batting_logs = pd.read_csv(BATTING_LOGS_FILE)
        log.info(f"Batting logs: {len(batting_logs)} rows")
    else:
        log.warning(f"Batting logs not found: {BATTING_LOGS_FILE}")

    odds = None
    if ODDS_FILE.exists():
        odds = pd.read_csv(ODDS_FILE)
        log.info(f"Odds API: {len(odds)} rows")
    else:
        log.warning(f"Odds API not found: {ODDS_FILE}")

    ss_odds = None
    if SS_ODDS_FILE.exists():
        ss_odds = pd.read_csv(SS_ODDS_FILE)
        log.info(f"SS odds: {len(ss_odds)} rows ({ss_odds['source_year'].min()}-{ss_odds['source_year'].max()})")
    else:
        log.warning(f"SS odds not found: {SS_ODDS_FILE}")

    bullpen_logs = None
    if BULLPEN_LOGS_FILE.exists():
        bullpen_logs = pd.read_csv(BULLPEN_LOGS_FILE)
        log.info(f"Bullpen logs: {len(bullpen_logs)} rows ({bullpen_logs['game_pk'].nunique()} games)")
    else:
        log.warning(f"Bullpen logs not found: {BULLPEN_LOGS_FILE}")
        log.warning("Run scripts/fetch_bullpen_data.py to fetch bullpen data")

    # ── Compute targets ──────────────────────────────────────────
    log.info("=" * 60)
    log.info("Computing targets")
    log.info("=" * 60)
    games = compute_targets(games)

    # ── SP features ──────────────────────────────────────────────
    log.info("=" * 60)
    log.info("Computing starting pitcher features")
    log.info("=" * 60)
    sp_computer = StartingPitcherComputer(pitcher_logs)
    sp_features = compute_sp_features(games, sp_computer)

    # ── Team batting features ────────────────────────────────────
    log.info("=" * 60)
    log.info("Computing team batting features")
    log.info("=" * 60)
    if batting_logs is not None:
        bat_features = compute_team_batting_features(games, batting_logs)
    else:
        bat_features = pd.DataFrame(index=games.index)

    # ── Context features (park, umpire, weather, DH) ─────────────
    log.info("=" * 60)
    log.info("Computing context features (park, umpire, weather)")
    log.info("=" * 60)
    ctx_features = compute_context_features(games)

    # ── Rest features (SP rest, SP season IP, team rest) ──────────
    log.info("=" * 60)
    log.info("Computing rest features (SP rest, SP season IP, team rest)")
    log.info("=" * 60)
    rest_features = compute_rest_features(games, pitcher_logs)

    # ── Momentum features (rolling win%, run diff) ────────────────
    log.info("=" * 60)
    log.info("Computing momentum features (rolling win%, run diff)")
    log.info("=" * 60)
    mom_features = compute_momentum_features(games)

    # ── Bullpen features (rolling ERA, WHIP, usage) ───────────────
    log.info("=" * 60)
    log.info("Computing bullpen features (rolling ERA, WHIP, usage)")
    log.info("=" * 60)
    if bullpen_logs is not None:
        bp_features = compute_bullpen_features(games, bullpen_logs)
    else:
        bp_features = pd.DataFrame(index=games.index)
        log.warning("  Skipping bullpen features (no data)")

    # ── Bullpen availability features (game-day state) ────────────
    log.info("=" * 60)
    log.info("Computing bullpen availability features (game-day state)")
    log.info("=" * 60)
    if bullpen_logs is not None:
        bp_avail_features = compute_bullpen_availability_features(games, bullpen_logs)
    else:
        bp_avail_features = pd.DataFrame(index=games.index)
        log.warning("  Skipping bullpen availability features (no data)")

    # ── Travel features ─────────────────────────────────────────
    log.info("=" * 60)
    log.info("Computing travel features (distance, road trip, timezone)")
    log.info("=" * 60)
    travel_features = compute_travel_features(games)

    # ── Schedule context features ───────────────────────────────
    log.info("=" * 60)
    log.info("Computing schedule context features (interleague, series, ASB)")
    log.info("=" * 60)
    sched_features = compute_schedule_context_features(games)

    # ── Load batter data for lineup features (Phase 3) ──────────
    batter_logs = None
    BATTER_LOGS_FILE = HISTORICAL_DIR / "batter_game_logs.csv"
    HANDEDNESS_FILE = HISTORICAL_DIR / "player_handedness.csv"
    if BATTER_LOGS_FILE.exists() and HANDEDNESS_FILE.exists():
        batter_logs = pd.read_csv(BATTER_LOGS_FILE)
        handedness = pd.read_csv(HANDEDNESS_FILE)
        log.info(f"Batter logs: {len(batter_logs)} rows ({batter_logs['batter_id'].nunique()} batters)")
        log.info(f"Handedness: {len(handedness)} rows")
    else:
        handedness = None
        if not BATTER_LOGS_FILE.exists():
            log.warning(f"Batter logs not found: {BATTER_LOGS_FILE}")
            log.warning("Run scripts/fetch_batter_data.py to fetch batter data")
        if not HANDEDNESS_FILE.exists():
            log.warning(f"Handedness not found: {HANDEDNESS_FILE}")
            log.warning("Run scripts/fetch_player_handedness.py to fetch handedness data")

    # ── Lineup features (Phase 3) ───────────────────────────────
    log.info("=" * 60)
    log.info("Computing lineup features (composition, platoon, star impact)")
    log.info("=" * 60)
    if batter_logs is not None and handedness is not None:
        lineup_features = compute_lineup_features(games, batter_logs, handedness, pitcher_logs)
    else:
        lineup_features = pd.DataFrame(index=games.index)
        log.warning("  Skipping lineup features (no batter/handedness data)")

    # ── Merge odds ───────────────────────────────────────────────
    log.info("=" * 60)
    log.info("Merging odds")
    log.info("=" * 60)
    if odds is not None or ss_odds is not None:
        odds_features = merge_odds(games, odds if odds is not None else pd.DataFrame(),
                                    ss_odds)
    else:
        odds_features = pd.DataFrame(index=games.index)

    # ── Assemble training data ───────────────────────────────────
    log.info("=" * 60)
    log.info("Assembling training data")
    log.info("=" * 60)

    # Select game-level columns to keep
    game_cols = [
        "game_pk", "date", "home_team", "away_team",
        "home_abbrev", "away_abbrev", "home_team_id", "away_team_id",
        "home_runs", "away_runs", "home_f5_runs", "away_f5_runs",
        "first_inning_home_runs", "first_inning_away_runs",
        "num_innings", "is_7_inning_dh", "game_type", "is_postseason",
        "venue_name", "venue_id", "temp", "wind", "condition", "doubleheader",
        "hp_umpire", "hp_umpire_id",
        "home_sp_id", "home_sp_name", "away_sp_id", "away_sp_name",
        # Targets
        "actual_margin", "actual_total",
        "actual_f5_margin", "actual_f5_total", "actual_nrfi",
    ]
    game_cols = [c for c in game_cols if c in games.columns]

    training = pd.concat([
        games[game_cols].reset_index(drop=True),
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
        odds_features.reset_index(drop=True),
    ], axis=1)

    # ── Leakage check ────────────────────────────────────────────
    log.info("Running leakage check...")
    target_cols = ["actual_margin", "actual_total", "actual_f5_margin",
                   "actual_f5_total", "actual_nrfi"]
    feature_cols = [c for c in training.columns if c not in game_cols]

    # Check that no feature has unreasonably high correlation with target
    for target in target_cols:
        if target not in training.columns:
            continue
        for feat in feature_cols:
            if feat not in training.columns:
                continue
            valid = training[[target, feat]].dropna()
            if len(valid) < 100:
                continue
            corr = valid[target].corr(valid[feat])
            if abs(corr) > 0.90:
                log.error(f"LEAKAGE ALERT: {feat} corr with {target} = {corr:.3f}")
                sys.exit(1)
    log.info("  Leakage check passed")

    # ── Save ─────────────────────────────────────────────────────
    training.to_csv(OUTPUT_FILE, index=False)
    log.info(f"\nSaved {len(training)} rows x {training.shape[1]} columns -> {OUTPUT_FILE}")

    # ── Summary ──────────────────────────────────────────────────
    log.info("=" * 60)
    log.info("Summary")
    log.info("=" * 60)
    log.info(f"  Games: {len(training)}")
    log.info(f"  Columns: {training.shape[1]}")
    log.info(f"  Date range: {training['date'].min()} to {training['date'].max()}")

    # Feature coverage
    log.info(f"\n  Feature coverage:")
    for col in sorted(training.columns):
        if col in game_cols:
            continue
        pct = training[col].notna().mean() * 100
        if pct < 100:
            log.info(f"    {col}: {pct:.0f}%")

    # Target stats
    log.info(f"\n  Target stats:")
    for col in target_cols:
        if col in training.columns:
            log.info(f"    {col}: mean={training[col].mean():.2f}, "
                     f"std={training[col].std():.2f}")

    log.info(f"\n  Game types:")
    for gt in training["game_type"].unique():
        n = (training["game_type"] == gt).sum()
        log.info(f"    {gt}: {n}")


if __name__ == "__main__":
    main()
