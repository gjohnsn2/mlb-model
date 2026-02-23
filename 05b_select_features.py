"""
05b -- Boruta Feature Selection
=================================
Runs BorutaPy with XGBoost estimator on training data to identify
genuinely predictive features. Separate runs for margin and total targets.

Ported directly from CBB pipeline -- same methodology, different features.

Outputs:
  models/selected_features.json -- {margin_features: [...], total_features: [...]}
  models/boruta_report.txt -- Human-readable selection report

Usage:
  python 05b_select_features.py
"""

import sys
import json
import numpy as np
import pandas as pd
from pathlib import Path
from config import HISTORICAL_DIR, MODELS_ROOT, get_logger
from feature_engine import ALL_CANDIDATE_FEATURES

log = get_logger("05b_boruta")

try:
    from boruta import BorutaPy
    import xgboost as xgb
except ImportError as e:
    log.error(f"Missing package: {e}")
    log.error("Run: pip install boruta xgboost")
    sys.exit(1)


def load_training_data():
    """Load training_data_v1.csv."""
    path = HISTORICAL_DIR / "training_data_v1.csv"
    if not path.exists():
        log.error(f"Training data not found: {path}")
        log.error("Run 00_build_historical.py first")
        sys.exit(1)

    df = pd.read_csv(path)
    df["date"] = pd.to_datetime(df["date"])
    log.info(f"Loaded {len(df):,} games from {path}")
    return df


def run_boruta(X, y, name="model", n_iter=100, alpha=0.05):
    """
    Run Boruta feature selection.
    Returns lists of confirmed and tentative features.
    """
    log.info(f"\n{'='*60}")
    log.info(f"Running Boruta for {name}")
    log.info(f"  Samples: {len(X)}, Features: {X.shape[1]}")
    log.info(f"  Iterations: {n_iter}, Alpha: {alpha}")

    # XGBoost estimator for Boruta
    estimator = xgb.XGBRegressor(
        n_estimators=100,
        max_depth=5,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=3,
        random_state=42,
        n_jobs=-1,
        verbosity=0,
    )

    selector = BorutaPy(
        estimator=estimator,
        n_estimators="auto",
        max_iter=n_iter,
        alpha=alpha,
        random_state=42,
        verbose=0,
    )

    log.info("  Fitting Boruta (this may take a few minutes)...")

    # sklearn needs no NaN; fill with 0 for Boruta
    X_clean = X.fillna(0)
    y_clean = y.fillna(y.median())

    selector.fit(X_clean.values, y_clean.values)

    confirmed = X.columns[selector.support_].tolist()
    tentative = X.columns[selector.support_weak_].tolist()
    rejected = X.columns[~selector.support_ & ~selector.support_weak_].tolist()

    log.info(f"  Confirmed: {len(confirmed)}")
    log.info(f"  Tentative: {len(tentative)}")
    log.info(f"  Rejected: {len(rejected)}")

    for f in confirmed:
        log.info(f"    + {f}")
    for f in tentative:
        log.info(f"    ~ {f}")

    return confirmed, tentative, rejected


def main():
    """Run Boruta for both margin and total models."""
    df = load_training_data()

    # Identify available features
    available = [f for f in ALL_CANDIDATE_FEATURES if f in df.columns]
    log.info(f"Available features: {len(available)}/{len(ALL_CANDIDATE_FEATURES)}")

    # Margin model
    margin_mask = df["actual_margin"].notna()
    X_margin = df.loc[margin_mask, available]
    y_margin = df.loc[margin_mask, "actual_margin"]

    confirmed_m, tentative_m, rejected_m = run_boruta(X_margin, y_margin, "margin")
    margin_features = confirmed_m + tentative_m

    # Total model
    total_mask = df["actual_total"].notna()
    X_total = df.loc[total_mask, available]
    y_total = df.loc[total_mask, "actual_total"]

    confirmed_t, tentative_t, rejected_t = run_boruta(X_total, y_total, "total")
    total_features = confirmed_t + tentative_t

    # Save selected features
    output = {
        "margin_features": margin_features,
        "total_features": total_features,
        "margin_confirmed": confirmed_m,
        "margin_tentative": tentative_m,
        "total_confirmed": confirmed_t,
        "total_tentative": tentative_t,
    }

    out_path = MODELS_ROOT / "selected_features.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    log.info(f"\nSaved selected features to {out_path}")

    # Save report
    report_path = MODELS_ROOT / "boruta_report.txt"
    with open(report_path, "w") as f:
        f.write("Boruta Feature Selection Report\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Training data: {len(df):,} games\n\n")
        f.write(f"MARGIN MODEL ({len(margin_features)} features)\n")
        f.write(f"  Confirmed ({len(confirmed_m)}): {', '.join(confirmed_m)}\n")
        f.write(f"  Tentative ({len(tentative_m)}): {', '.join(tentative_m)}\n")
        f.write(f"  Rejected ({len(rejected_m)}): {', '.join(rejected_m)}\n\n")
        f.write(f"TOTAL MODEL ({len(total_features)} features)\n")
        f.write(f"  Confirmed ({len(confirmed_t)}): {', '.join(confirmed_t)}\n")
        f.write(f"  Tentative ({len(tentative_t)}): {', '.join(tentative_t)}\n")
        f.write(f"  Rejected ({len(rejected_t)}): {', '.join(rejected_t)}\n")

    log.info(f"Saved report to {report_path}")


if __name__ == "__main__":
    main()
