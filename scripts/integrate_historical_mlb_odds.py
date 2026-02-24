"""
Integrate Sports-Statistics.com historical MLB odds (2010-2021)
==============================================================
Parses Excel files, pairs V/H rows into games, maps team abbreviations,
and merges closing moneyline + total odds into the existing training data.

The existing training data (2019-2025) has Odds API H2H odds for 2020+.
This script backfills 2019 and adds 2010-2018 odds from Sports-Statistics.com.
For the 2020-2021 overlap period, we validate consistency then prefer
Odds API data (more books, true consensus) where available.

Run: python3 scripts/integrate_historical_mlb_odds.py
"""

import sys
import numpy as np
import pandas as pd
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import HISTORICAL_DIR, get_logger

log = get_logger("integrate_mlb_odds")

RAW_DIR = HISTORICAL_DIR / "mlb_odds_raw"

# Map Sports-Statistics team abbreviations → our MLB Stats API abbreviations
TEAM_MAP = {
    "ARI": "AZ",
    "CUB": "CHC",
    "KAN": "KC",
    "SDG": "SD",
    "SFO": "SF",
    "TAM": "TB",
    "WAS": "WSH",
    # These match already
    "ATL": "ATL", "BAL": "BAL", "BOS": "BOS", "CIN": "CIN",
    "CLE": "CLE", "COL": "COL", "CWS": "CWS", "DET": "DET",
    "HOU": "HOU", "LAA": "LAA", "LAD": "LAD", "MIA": "MIA",
    "MIL": "MIL", "MIN": "MIN", "NYM": "NYM", "NYY": "NYY",
    "OAK": "OAK", "PHI": "PHI", "PIT": "PIT", "SEA": "SEA",
    "STL": "STL", "TEX": "TEX", "TOR": "TOR",
    # Oakland renamed to Athletics in 2025, mapped in training data as ATH
    # but for 2010-2021 data they're OAK
}


def parse_odds_file(year):
    """Parse a single year's Excel file into game-level rows."""
    path = RAW_DIR / f"mlb-odds-{year}.xlsx"
    if not path.exists():
        log.warning(f"File not found: {path}")
        return pd.DataFrame()

    df = pd.read_excel(path)
    log.info(f"  {year}: {len(df)} rows ({len(df)//2} games)")

    games = []
    i = 0
    while i < len(df) - 1:
        row_v = df.iloc[i]
        row_h = df.iloc[i + 1]

        # Validate V/H pairing
        if row_v["VH"] != "V" or row_h["VH"] != "H":
            # Try to recover — sometimes there are extra rows
            if row_v["VH"] == "V":
                i += 1
                continue
            i += 1
            continue

        # Parse date: format is MMDD (e.g., 404 = April 4)
        date_val = row_v["Date"]
        try:
            date_str = str(int(date_val)).zfill(4)
            month = int(date_str[:-2])
            day = int(date_str[-2:])
            game_date = pd.Timestamp(year=year, month=month, day=day)
        except (ValueError, TypeError):
            i += 2
            continue

        # Map teams
        away_abbrev = TEAM_MAP.get(row_v["Team"], row_v["Team"])
        home_abbrev = TEAM_MAP.get(row_h["Team"], row_h["Team"])

        # Closing moneylines
        away_ml = row_v.get("Close")
        home_ml = row_h.get("Close")

        # Closing total — column name varies by year
        total_line = None
        for col_name in ["Close OU", "CloseOU"]:
            if col_name in df.columns:
                total_line = row_v.get(col_name)  # Same for both rows
                if pd.isna(total_line):
                    total_line = row_h.get(col_name)
                break

        # Final scores
        away_runs = row_v.get("Final")
        home_runs = row_h.get("Final")

        # Clean numeric values
        try:
            away_ml = float(away_ml) if pd.notna(away_ml) else np.nan
            home_ml = float(home_ml) if pd.notna(home_ml) else np.nan
            total_line = float(total_line) if pd.notna(total_line) else np.nan
            away_runs = int(away_runs) if pd.notna(away_runs) else np.nan
            home_runs = int(home_runs) if pd.notna(home_runs) else np.nan
        except (ValueError, TypeError):
            i += 2
            continue

        games.append({
            "date": game_date.strftime("%Y-%m-%d"),
            "home_abbrev": home_abbrev,
            "away_abbrev": away_abbrev,
            "ss_h2h_home": home_ml,
            "ss_h2h_away": away_ml,
            "ss_total": total_line,
            "ss_home_runs": home_runs,
            "ss_away_runs": away_runs,
            "source_year": year,
        })

        i += 2

    return pd.DataFrame(games)


def parse_all_years():
    """Parse all available years."""
    all_games = []
    for year in range(2010, 2022):
        games = parse_odds_file(year)
        if not games.empty:
            all_games.append(games)

    df = pd.concat(all_games, ignore_index=True)
    log.info(f"\nTotal parsed: {len(df)} games across {df['source_year'].nunique()} years")
    return df


def validate_overlap(ss_odds, training_df):
    """Validate Sports-Statistics odds against existing Odds API data for 2020-2021."""
    training_df = training_df.copy()
    training_df["date_str"] = pd.to_datetime(training_df["date"]).dt.strftime("%Y-%m-%d")

    overlap = ss_odds[ss_odds["source_year"].isin([2020, 2021])].copy()

    # Merge on date + teams
    merged = overlap.merge(
        training_df[["date_str", "home_abbrev", "away_abbrev",
                      "consensus_h2h_home", "consensus_h2h_away",
                      "home_runs", "away_runs"]],
        left_on=["date", "home_abbrev", "away_abbrev"],
        right_on=["date_str", "home_abbrev", "away_abbrev"],
        how="inner"
    )

    log.info(f"\nOverlap validation (2020-2021):")
    log.info(f"  SS games: {len(overlap)}, matched to training: {len(merged)}")

    # Check score consistency
    if len(merged) > 0:
        score_match = (
            (merged["ss_home_runs"] == merged["home_runs"]) &
            (merged["ss_away_runs"] == merged["away_runs"])
        ).mean()
        log.info(f"  Score match rate: {score_match:.1%}")

        # Compare odds where both have them
        both_odds = merged[
            merged["consensus_h2h_home"].notna() & merged["ss_h2h_home"].notna()
        ]
        if len(both_odds) > 0:
            home_corr = both_odds["ss_h2h_home"].corr(both_odds["consensus_h2h_home"])
            away_corr = both_odds["ss_h2h_away"].corr(both_odds["consensus_h2h_away"])
            home_diff = (both_odds["ss_h2h_home"] - both_odds["consensus_h2h_home"]).abs().median()
            log.info(f"  Games with both SS and Odds API H2H: {len(both_odds)}")
            log.info(f"  Home ML correlation: {home_corr:.3f}")
            log.info(f"  Away ML correlation: {away_corr:.3f}")
            log.info(f"  Median |home ML difference|: {home_diff:.0f}")

    return merged


def merge_odds_into_training(ss_odds, training_path):
    """Merge Sports-Statistics odds into the training data."""
    training_df = pd.read_csv(training_path)
    training_df["date_str"] = pd.to_datetime(training_df["date"]).dt.strftime("%Y-%m-%d")

    log.info(f"\nMerging odds into training data ({len(training_df)} games)...")

    # Merge SS odds by date + teams
    merged = training_df.merge(
        ss_odds[["date", "home_abbrev", "away_abbrev", "ss_h2h_home", "ss_h2h_away", "ss_total"]],
        left_on=["date_str", "home_abbrev", "away_abbrev"],
        right_on=["date", "home_abbrev", "away_abbrev"],
        how="left",
        suffixes=("", "_ss")
    )

    matched = merged["ss_h2h_home"].notna().sum()
    log.info(f"  Matched SS odds to training games: {matched}/{len(training_df)}")

    # Backfill: use SS odds where Odds API odds are missing
    before_h2h = training_df["consensus_h2h_home"].notna().sum()
    backfilled_h2h = 0
    backfilled_total = 0

    mask_no_h2h = merged["consensus_h2h_home"].isna() & merged["ss_h2h_home"].notna()
    merged.loc[mask_no_h2h, "consensus_h2h_home"] = merged.loc[mask_no_h2h, "ss_h2h_home"]
    merged.loc[mask_no_h2h, "consensus_h2h_away"] = merged.loc[mask_no_h2h, "ss_h2h_away"]
    backfilled_h2h = mask_no_h2h.sum()

    mask_no_total = merged["consensus_total"].isna() & merged["ss_total"].notna()
    merged.loc[mask_no_total, "consensus_total"] = merged.loc[mask_no_total, "ss_total"]
    backfilled_total = mask_no_total.sum()

    # Set num_books = 1 for SS-only games (single source)
    mask_ss_only = mask_no_h2h & merged["num_books"].isna()
    merged.loc[mask_ss_only, "num_books"] = 1

    after_h2h = merged["consensus_h2h_home"].notna().sum()
    log.info(f"  H2H odds: {before_h2h} → {after_h2h} (+{backfilled_h2h} backfilled)")
    log.info(f"  Totals backfilled: {backfilled_total}")

    # Drop temp columns
    merged.drop(columns=["date_str", "date_ss", "ss_h2h_home", "ss_h2h_away", "ss_total"],
                errors="ignore", inplace=True)

    # Per-year summary
    merged["_year"] = pd.to_datetime(merged["date"]).dt.year
    log.info(f"\n  H2H coverage by year:")
    for y in sorted(merged["_year"].unique()):
        sub = merged[merged["_year"] == y]
        has = sub["consensus_h2h_home"].notna().sum()
        log.info(f"    {y}: {has}/{len(sub)} ({has/len(sub)*100:.1f}%)")
    merged.drop(columns=["_year"], inplace=True)

    return merged


def extend_training_data(ss_odds, training_path):
    """
    For years 2010-2018 that aren't in the training data at all,
    we need game results + SP stats from MLB Stats API.
    This function only adds odds — the full pipeline rebuild is needed
    for the SP/batting features.

    For now, just report what we'd gain and handle the 2019 backfill
    (which IS in the training data but has 0% odds coverage).
    """
    training_df = pd.read_csv(training_path)
    training_years = set(pd.to_datetime(training_df["date"]).dt.year.unique())

    ss_years = set(ss_odds["source_year"].unique())
    new_years = ss_years - training_years

    if new_years:
        log.info(f"\nYears in SS data but NOT in training data: {sorted(new_years)}")
        for y in sorted(new_years):
            n = len(ss_odds[ss_odds["source_year"] == y])
            log.info(f"  {y}: {n} games (would need SP/batting features from MLB Stats API)")
    else:
        log.info(f"\nAll SS years already in training data — backfill only")


def main():
    log.info("=" * 60)
    log.info("INTEGRATE SPORTS-STATISTICS MLB HISTORICAL ODDS")
    log.info("=" * 60)

    # Parse all Excel files
    ss_odds = parse_all_years()

    # Quick stats
    for y in sorted(ss_odds["source_year"].unique()):
        sub = ss_odds[ss_odds["source_year"] == y]
        has_ml = sub["ss_h2h_home"].notna().sum()
        has_total = sub["ss_total"].notna().sum()
        log.info(f"  {y}: {len(sub)} games, {has_ml} with ML ({has_ml/len(sub)*100:.1f}%), "
                 f"{has_total} with total ({has_total/len(sub)*100:.1f}%)")

    # Validate overlap with existing data
    training_path = HISTORICAL_DIR / "training_data_mlb_v1.csv"
    training_df = pd.read_csv(training_path)
    validate_overlap(ss_odds, training_df)

    # Check what years need full rebuilds vs backfill
    extend_training_data(ss_odds, training_path)

    # Merge odds into training data (backfill 2019 + supplement 2020-2021)
    updated = merge_odds_into_training(ss_odds, training_path)

    # Save updated training data
    output_path = HISTORICAL_DIR / "training_data_mlb_v2.csv"
    updated.to_csv(output_path, index=False)
    log.info(f"\nSaved updated training data → {output_path}")
    log.info(f"  {len(updated)} games, {updated['consensus_h2h_home'].notna().sum()} with H2H odds")

    # Save parsed SS odds as standalone CSV for reference
    ss_path = HISTORICAL_DIR / "sports_statistics_mlb_odds.csv"
    ss_odds.to_csv(ss_path, index=False)
    log.info(f"Saved parsed SS odds → {ss_path}")


if __name__ == "__main__":
    main()
