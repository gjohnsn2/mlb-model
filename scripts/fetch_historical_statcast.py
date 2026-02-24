"""
Fetch Historical Statcast Data (2019-2025) via pybaseball
==========================================================
Fetches pitch-level data one month at a time, then aggregates to
per-pitcher-per-game Statcast metrics.

Output (data/historical/):
  - statcast_pitcher_games.csv — one row/pitcher/game with 7 Statcast columns:
    xwOBA-against, barrel%, hard-hit%, GB%, FB%, whiff rate, pitches

Checkpointing: tracks completed months in statcast_fetch_progress.json.
Uses pybaseball cache for idempotent re-runs.
Runtime: ~45-60 minutes.
"""

import sys
import json
import time
import traceback
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
from calendar import monthrange

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import HISTORICAL_DIR, get_logger

log = get_logger("fetch_statcast")

# ── Config ──────────────────────────────────────────────────────
SEASONS = list(range(2015, 2026))  # Statcast data starts 2015
OUTPUT_FILE = HISTORICAL_DIR / "statcast_pitcher_games.csv"
PROGRESS_FILE = HISTORICAL_DIR / "statcast_fetch_progress.json"

# Enable pybaseball cache
try:
    from pybaseball import cache
    cache.enable()
    log.info("pybaseball cache enabled")
except Exception:
    pass

from pybaseball import statcast


# ── Progress ────────────────────────────────────────────────────
def load_progress():
    if PROGRESS_FILE.exists():
        with open(PROGRESS_FILE) as f:
            return json.load(f)
    return {"completed_months": []}


def save_progress(progress):
    HISTORICAL_DIR.mkdir(parents=True, exist_ok=True)
    with open(PROGRESS_FILE, "w") as f:
        json.dump(progress, f)


# ── Month chunks ────────────────────────────────────────────────
def get_month_chunks():
    """Generate (start_date, end_date, label) for each month with MLB games."""
    chunks = []
    for season in SEASONS:
        # MLB regular season: late March through September
        # Postseason: October (sometimes early November)
        for month in range(3, 12):
            _, last_day = monthrange(season, month)
            start = f"{season}-{month:02d}-01"
            end = f"{season}-{month:02d}-{last_day:02d}"
            label = f"{season}-{month:02d}"
            chunks.append((start, end, label))
    return chunks


# ── Statcast aggregation ───────────────────────────────────────
def aggregate_pitcher_games(df):
    """
    Aggregate pitch-level Statcast data to per-pitcher-per-game metrics.

    Input: pitch-level DataFrame from pybaseball.statcast()
    Output: DataFrame with one row per (pitcher, game_pk)
    """
    if df is None or df.empty:
        return pd.DataFrame()

    # Filter to pitcher-related data (each row is a pitch)
    # Key columns: pitcher, game_pk, game_date, estimated_woba_using_speedangle,
    #   launch_speed, launch_angle, bb_type, description
    required_cols = ["pitcher", "game_pk", "game_date"]
    for col in required_cols:
        if col not in df.columns:
            log.warning(f"Missing required column: {col}")
            return pd.DataFrame()

    # Group by pitcher + game
    groups = df.groupby(["pitcher", "game_pk", "game_date"])
    rows = []

    for (pitcher_id, game_pk, game_date), pitches in groups:
        n_pitches = len(pitches)
        if n_pitches < 5:
            continue  # Skip very short appearances (not meaningful)

        row = {
            "pitcher_id": int(pitcher_id),
            "game_pk": int(game_pk),
            "date": str(game_date)[:10],
            "statcast_pitches": n_pitches,
        }

        # xwOBA against (estimated_woba_using_speedangle on batted balls)
        if "estimated_woba_using_speedangle" in pitches.columns:
            xwoba_vals = pd.to_numeric(
                pitches["estimated_woba_using_speedangle"], errors="coerce"
            ).dropna()
            row["xwoba"] = round(xwoba_vals.mean(), 3) if len(xwoba_vals) > 0 else np.nan
        else:
            row["xwoba"] = np.nan

        # Hard hit % (launch_speed >= 95 mph on batted balls)
        if "launch_speed" in pitches.columns:
            ls = pd.to_numeric(pitches["launch_speed"], errors="coerce").dropna()
            if len(ls) > 0:
                row["hard_hit_pct"] = round((ls >= 95).mean() * 100, 1)
            else:
                row["hard_hit_pct"] = np.nan
        else:
            row["hard_hit_pct"] = np.nan

        # Barrel % (barrel classification from Statcast)
        if "launch_speed_angle" in pitches.columns:
            lsa = pitches["launch_speed_angle"].dropna()
            if len(lsa) > 0:
                row["barrel_pct"] = round((lsa == 6).mean() * 100, 1)  # 6 = barrel
            else:
                row["barrel_pct"] = np.nan
        else:
            row["barrel_pct"] = np.nan

        # GB% and FB% from bb_type (ground_ball, fly_ball, line_drive, popup)
        if "bb_type" in pitches.columns:
            batted = pitches["bb_type"].dropna()
            n_batted = len(batted)
            if n_batted > 0:
                row["groundball_pct"] = round(
                    (batted == "ground_ball").sum() / n_batted * 100, 1
                )
                row["flyball_pct"] = round(
                    (batted == "fly_ball").sum() / n_batted * 100, 1
                )
            else:
                row["groundball_pct"] = np.nan
                row["flyball_pct"] = np.nan
        else:
            row["groundball_pct"] = np.nan
            row["flyball_pct"] = np.nan

        # Whiff rate (swinging strikes / total swings)
        if "description" in pitches.columns:
            desc = pitches["description"].fillna("")
            swings = desc.isin([
                "swinging_strike", "swinging_strike_blocked",
                "foul", "foul_tip", "foul_bunt",
                "hit_into_play", "hit_into_play_no_out", "hit_into_play_score",
                "missed_bunt",
            ])
            whiffs = desc.isin(["swinging_strike", "swinging_strike_blocked"])
            n_swings = swings.sum()
            if n_swings > 0:
                row["whiff_rate"] = round(whiffs.sum() / n_swings * 100, 1)
            else:
                row["whiff_rate"] = np.nan
        else:
            row["whiff_rate"] = np.nan

        rows.append(row)

    return pd.DataFrame(rows)


# ── Main ────────────────────────────────────────────────────────
def main():
    HISTORICAL_DIR.mkdir(parents=True, exist_ok=True)

    month_chunks = get_month_chunks()
    progress = load_progress()
    completed = set(progress["completed_months"])

    remaining = [(s, e, lbl) for s, e, lbl in month_chunks if lbl not in completed]
    log.info(f"Total months: {len(month_chunks)}, completed: {len(completed)}, "
             f"remaining: {len(remaining)}")

    # Load existing partial results
    all_rows = []
    if OUTPUT_FILE.exists():
        existing = pd.read_csv(OUTPUT_FILE)
        all_rows = existing.to_dict("records")
        log.info(f"Loaded {len(all_rows)} existing Statcast rows")

    for i, (start_dt, end_dt, label) in enumerate(remaining):
        log.info(f"[{i+1}/{len(remaining)}] Fetching Statcast: {label} "
                 f"({start_dt} to {end_dt})...")
        try:
            df = statcast(start_dt=start_dt, end_dt=end_dt)
            if df is None or df.empty:
                log.info(f"  {label}: no data (off-season?)")
                progress["completed_months"].append(label)
                save_progress(progress)
                continue

            log.info(f"  {label}: {len(df)} pitches fetched")

            # Aggregate to pitcher-game level
            agg = aggregate_pitcher_games(df)
            if not agg.empty:
                all_rows.extend(agg.to_dict("records"))
                log.info(f"  {label}: {len(agg)} pitcher-game rows aggregated "
                         f"(total: {len(all_rows)})")

            progress["completed_months"].append(label)
            save_progress(progress)

            # Save checkpoint
            if all_rows:
                pd.DataFrame(all_rows).to_csv(OUTPUT_FILE, index=False)

        except Exception as e:
            log.error(f"  {label}: error — {e}")
            traceback.print_exc()
            # Don't mark as completed so we retry next run
            continue

    # Final save
    if all_rows:
        df_out = pd.DataFrame(all_rows)
        df_out.to_csv(OUTPUT_FILE, index=False)
        log.info(f"\nDone! {len(df_out)} pitcher-game Statcast rows -> {OUTPUT_FILE}")

        # Validation
        log.info(f"  Unique pitchers: {df_out['pitcher_id'].nunique()}")
        log.info(f"  Unique games: {df_out['game_pk'].nunique()}")
        log.info(f"  Date range: {df_out['date'].min()} to {df_out['date'].max()}")
        for col in ["xwoba", "hard_hit_pct", "barrel_pct", "groundball_pct",
                     "flyball_pct", "whiff_rate"]:
            pct = df_out[col].notna().mean() * 100
            log.info(f"  {col}: {pct:.0f}% populated")
    else:
        log.warning("No Statcast data fetched")


if __name__ == "__main__":
    main()
