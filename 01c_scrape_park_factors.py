"""
01c -- Scrape Park Factors
============================
Fetches park factor data from FanGraphs (via pybaseball) or uses static
reference CSV. Park factors quantify how much a venue inflates or deflates
run scoring, home runs, hits, etc. relative to league average.

Critical for:
  - Total (O/U) predictions: Coors Field adds 1.5+ runs
  - Run line adjustments: hitter-friendly parks have more variance
  - ML adjustments: pitcher-friendly parks favor favorites slightly

Outputs:
  data/park_factors.csv (static reference, updated seasonally)
  data/raw/park_factors_YYYY-MM-DD.csv (daily snapshot)

Note: Park factors are relatively stable year-to-year. The static CSV
is the primary reference; this script refreshes it periodically.
"""

import sys
import pandas as pd
from pathlib import Path
from config import RAW_DIR, DATA_DIR, TODAY, get_logger

log = get_logger("01c_parks")


# Default park factors (2024 ESPN park factors, runs)
# Values > 100 = hitter-friendly, < 100 = pitcher-friendly
# Will be replaced with live data once scraper is connected
DEFAULT_PARK_FACTORS = {
    "COL": {"team": "Colorado Rockies", "run_factor": 1.38, "hr_factor": 1.36, "hit_factor": 1.15},
    "BOS": {"team": "Boston Red Sox", "run_factor": 1.12, "hr_factor": 1.06, "hit_factor": 1.07},
    "CIN": {"team": "Cincinnati Reds", "run_factor": 1.10, "hr_factor": 1.22, "hit_factor": 1.03},
    "TEX": {"team": "Texas Rangers", "run_factor": 1.08, "hr_factor": 1.15, "hit_factor": 1.02},
    "ARI": {"team": "Arizona Diamondbacks", "run_factor": 1.06, "hr_factor": 1.08, "hit_factor": 1.04},
    "CHC": {"team": "Chicago Cubs", "run_factor": 1.05, "hr_factor": 1.12, "hit_factor": 1.01},
    "PHI": {"team": "Philadelphia Phillies", "run_factor": 1.04, "hr_factor": 1.10, "hit_factor": 1.00},
    "TOR": {"team": "Toronto Blue Jays", "run_factor": 1.04, "hr_factor": 1.08, "hit_factor": 1.01},
    "MIN": {"team": "Minnesota Twins", "run_factor": 1.03, "hr_factor": 1.06, "hit_factor": 1.00},
    "MIL": {"team": "Milwaukee Brewers", "run_factor": 1.02, "hr_factor": 1.08, "hit_factor": 0.99},
    "NYY": {"team": "New York Yankees", "run_factor": 1.02, "hr_factor": 1.12, "hit_factor": 0.98},
    "ATL": {"team": "Atlanta Braves", "run_factor": 1.01, "hr_factor": 1.05, "hit_factor": 1.00},
    "BAL": {"team": "Baltimore Orioles", "run_factor": 1.00, "hr_factor": 1.08, "hit_factor": 0.98},
    "LAA": {"team": "Los Angeles Angels", "run_factor": 1.00, "hr_factor": 0.98, "hit_factor": 1.01},
    "HOU": {"team": "Houston Astros", "run_factor": 0.99, "hr_factor": 1.02, "hit_factor": 0.99},
    "DET": {"team": "Detroit Tigers", "run_factor": 0.99, "hr_factor": 0.92, "hit_factor": 1.02},
    "WSH": {"team": "Washington Nationals", "run_factor": 0.98, "hr_factor": 1.02, "hit_factor": 0.98},
    "CLE": {"team": "Cleveland Guardians", "run_factor": 0.98, "hr_factor": 0.94, "hit_factor": 1.00},
    "KCR": {"team": "Kansas City Royals", "run_factor": 0.97, "hr_factor": 0.90, "hit_factor": 1.02},
    "PIT": {"team": "Pittsburgh Pirates", "run_factor": 0.97, "hr_factor": 0.85, "hit_factor": 1.00},
    "SEA": {"team": "Seattle Mariners", "run_factor": 0.96, "hr_factor": 0.92, "hit_factor": 0.99},
    "CWS": {"team": "Chicago White Sox", "run_factor": 0.96, "hr_factor": 1.04, "hit_factor": 0.95},
    "STL": {"team": "St. Louis Cardinals", "run_factor": 0.95, "hr_factor": 0.88, "hit_factor": 0.99},
    "LAD": {"team": "Los Angeles Dodgers", "run_factor": 0.95, "hr_factor": 0.92, "hit_factor": 0.98},
    "SDP": {"team": "San Diego Padres", "run_factor": 0.94, "hr_factor": 0.86, "hit_factor": 0.98},
    "NYM": {"team": "New York Mets", "run_factor": 0.93, "hr_factor": 0.90, "hit_factor": 0.97},
    "TBR": {"team": "Tampa Bay Rays", "run_factor": 0.93, "hr_factor": 0.88, "hit_factor": 0.97},
    "SFG": {"team": "San Francisco Giants", "run_factor": 0.92, "hr_factor": 0.80, "hit_factor": 0.96},
    "MIA": {"team": "Miami Marlins", "run_factor": 0.91, "hr_factor": 0.82, "hit_factor": 0.97},
    "OAK": {"team": "Oakland Athletics", "run_factor": 0.90, "hr_factor": 0.78, "hit_factor": 0.96},
}


def build_default_park_factors():
    """Write default park factors to static CSV."""
    rows = []
    for abbrev, data in DEFAULT_PARK_FACTORS.items():
        rows.append({
            "team": data["team"],
            "abbrev": abbrev,
            "run_factor": data["run_factor"],
            "hr_factor": data["hr_factor"],
            "hit_factor": data["hit_factor"],
        })
    df = pd.DataFrame(rows)
    out_path = DATA_DIR / "park_factors.csv"
    df.to_csv(out_path, index=False)
    log.info(f"Saved default park factors for {len(df)} teams to {out_path}")

    # Also save dated snapshot
    raw_path = RAW_DIR / f"park_factors_{TODAY}.csv"
    raw_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(raw_path, index=False)
    return df


if __name__ == "__main__":
    build_default_park_factors()
