"""
Fetch Batter Pitch-Type Arsenal Stats (2019-2025) via pybaseball
================================================================
Annual batter performance broken down by pitch type.
Uses Statcast-derived data from Baseball Savant.

Output (data/historical/):
  - batter_pitch_type_stats.csv — one row per (batter_id, season, pitch_type)
    with pa, ba, slg, woba, whiff_pct, est_woba

Checkpointed per season. ~4 minutes total (web API).
"""

import sys
import json
import pandas as pd
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import HISTORICAL_DIR, get_logger

log = get_logger("fetch_batter_pitch_type")

# ── Config ──────────────────────────────────────────────────────
SEASONS = list(range(2019, 2026))  # Statcast pitch arsenal starts ~2019
MIN_PA = 25
OUTPUT_FILE = HISTORICAL_DIR / "batter_pitch_type_stats.csv"
PROGRESS_FILE = HISTORICAL_DIR / "batter_pitch_type_progress.json"

# Enable pybaseball cache
try:
    from pybaseball import cache
    cache.enable()
    log.info("pybaseball cache enabled")
except Exception:
    pass

from pybaseball import statcast_batter_pitch_arsenal


# ── Progress ────────────────────────────────────────────────────
def load_progress():
    if PROGRESS_FILE.exists():
        with open(PROGRESS_FILE) as f:
            return json.load(f)
    return {"completed_seasons": []}


def save_progress(progress):
    HISTORICAL_DIR.mkdir(parents=True, exist_ok=True)
    with open(PROGRESS_FILE, "w") as f:
        json.dump(progress, f)


# ── Main ────────────────────────────────────────────────────────
def main():
    HISTORICAL_DIR.mkdir(parents=True, exist_ok=True)

    progress = load_progress()
    completed = set(progress["completed_seasons"])

    remaining = [s for s in SEASONS if s not in completed]
    log.info(f"Total seasons: {len(SEASONS)}, completed: {len(completed)}, "
             f"remaining: {len(remaining)}")

    # Load existing partial results
    all_rows = []
    if OUTPUT_FILE.exists():
        existing = pd.read_csv(OUTPUT_FILE)
        all_rows = existing.to_dict("records")
        log.info(f"Loaded {len(all_rows)} existing rows")

    for season in remaining:
        log.info(f"Fetching batter pitch arsenal stats: {season} (minPA={MIN_PA})...")
        try:
            df = statcast_batter_pitch_arsenal(season, minPA=MIN_PA)
            if df is None or df.empty:
                log.info(f"  {season}: no data")
                progress["completed_seasons"].append(season)
                save_progress(progress)
                continue

            # Select and rename columns
            keep_cols = {
                "player_id": "batter_id",
                "pitch_type": "pitch_type",
                "pa": "pa",
                "ba": "ba",
                "slg": "slg",
                "woba": "woba",
                "whiff_percent": "whiff_pct",
                "est_woba": "est_woba",
            }
            df_out = df[[c for c in keep_cols if c in df.columns]].copy()
            df_out = df_out.rename(columns=keep_cols)
            df_out["season"] = season

            all_rows.extend(df_out.to_dict("records"))
            log.info(f"  {season}: {len(df_out)} rows "
                     f"({df_out['batter_id'].nunique()} batters, "
                     f"{df_out['pitch_type'].nunique()} pitch types)")

            progress["completed_seasons"].append(season)
            save_progress(progress)

            # Checkpoint save
            if all_rows:
                pd.DataFrame(all_rows).to_csv(OUTPUT_FILE, index=False)

        except Exception as e:
            log.error(f"  {season}: error — {e}")
            import traceback
            traceback.print_exc()
            continue

    # Final save
    if all_rows:
        df_final = pd.DataFrame(all_rows)
        df_final.to_csv(OUTPUT_FILE, index=False)
        log.info(f"\nDone! {len(df_final)} rows -> {OUTPUT_FILE}")
        log.info(f"  Unique batters: {df_final['batter_id'].nunique()}")
        log.info(f"  Seasons: {sorted(df_final['season'].unique())}")
        log.info(f"  Pitch types: {sorted(df_final['pitch_type'].unique())}")
        log.info(f"  Rows per season:")
        for s in sorted(df_final["season"].unique()):
            n = (df_final["season"] == s).sum()
            log.info(f"    {s}: {n}")
    else:
        log.warning("No data fetched")


if __name__ == "__main__":
    main()
