"""
Merge Pitcher Logs: MLB API + Statcast
=======================================
Left joins MLB API pitcher counting stats with Statcast advanced metrics
on (pitcher_id, game_pk).

Inputs (data/historical/):
  - pitcher_game_logs_mlbapi.csv   — counting stats (IP, H, R, ER, K, BB, HR, etc.)
  - statcast_pitcher_games.csv     — advanced metrics (xwOBA, barrel%, hard-hit%, etc.)

Output:
  - pitcher_logs_all.csv           — unified pitcher log with all columns

Runtime: ~10 seconds.
"""

import sys
import pandas as pd
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import HISTORICAL_DIR, get_logger

log = get_logger("merge_pitcher_logs")

MLBAPI_FILE = HISTORICAL_DIR / "pitcher_game_logs_mlbapi.csv"
STATCAST_FILE = HISTORICAL_DIR / "statcast_pitcher_games.csv"
OUTPUT_FILE = HISTORICAL_DIR / "pitcher_logs_all.csv"


def main():
    # Load MLB API pitcher logs
    if not MLBAPI_FILE.exists():
        log.error(f"MLB API pitcher logs not found: {MLBAPI_FILE}")
        log.error("Run scripts/fetch_historical_games.py first")
        sys.exit(1)

    mlbapi = pd.read_csv(MLBAPI_FILE)
    log.info(f"MLB API pitcher logs: {len(mlbapi)} rows, {mlbapi['pitcher_id'].nunique()} pitchers")

    # Load Statcast data (optional — merge is a left join)
    statcast = None
    if STATCAST_FILE.exists():
        statcast = pd.read_csv(STATCAST_FILE)
        log.info(f"Statcast pitcher games: {len(statcast)} rows, "
                 f"{statcast['pitcher_id'].nunique()} pitchers")
    else:
        log.warning(f"Statcast file not found: {STATCAST_FILE}")
        log.warning("Output will have MLB API stats only (no Statcast metrics)")

    # Merge on (pitcher_id, game_pk)
    if statcast is not None and not statcast.empty:
        # Select only Statcast-specific columns for merge
        statcast_cols = ["pitcher_id", "game_pk", "statcast_pitches",
                         "xwoba", "hard_hit_pct", "barrel_pct",
                         "groundball_pct", "flyball_pct", "whiff_rate"]
        statcast_merge = statcast[[c for c in statcast_cols if c in statcast.columns]].copy()

        # Ensure matching types
        mlbapi["pitcher_id"] = mlbapi["pitcher_id"].astype("Int64")
        mlbapi["game_pk"] = mlbapi["game_pk"].astype("Int64")
        statcast_merge["pitcher_id"] = statcast_merge["pitcher_id"].astype("Int64")
        statcast_merge["game_pk"] = statcast_merge["game_pk"].astype("Int64")

        merged = mlbapi.merge(
            statcast_merge,
            on=["pitcher_id", "game_pk"],
            how="left",
            suffixes=("", "_sc"),
        )

        # Report match rate
        matched = merged["xwoba"].notna().sum() if "xwoba" in merged.columns else 0
        match_pct = matched / len(merged) * 100 if len(merged) > 0 else 0
        log.info(f"Statcast match rate: {matched}/{len(merged)} ({match_pct:.1f}%)")
    else:
        merged = mlbapi.copy()
        # Add empty Statcast columns
        for col in ["statcast_pitches", "xwoba", "hard_hit_pct", "barrel_pct",
                     "groundball_pct", "flyball_pct", "whiff_rate"]:
            merged[col] = pd.NA

    # Save
    HISTORICAL_DIR.mkdir(parents=True, exist_ok=True)
    merged.to_csv(OUTPUT_FILE, index=False)
    log.info(f"\nSaved {len(merged)} rows -> {OUTPUT_FILE}")
    log.info(f"  Columns: {list(merged.columns)}")

    # Validation
    log.info(f"\nValidation:")
    log.info(f"  Unique pitchers: {merged['pitcher_id'].nunique()}")
    log.info(f"  Unique games: {merged['game_pk'].nunique()}")
    log.info(f"  Date range: {merged['date'].min()} to {merged['date'].max()}")
    for col in merged.columns:
        pct = merged[col].notna().mean() * 100
        if pct < 100:
            log.info(f"  {col}: {pct:.0f}% populated")


if __name__ == "__main__":
    main()
