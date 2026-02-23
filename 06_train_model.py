"""
06 -- Train XGBoost Models (Walk-Forward Validation)
======================================================
Trains two models (margin + total) using honest walk-forward validation
with per-fold Boruta feature selection.

Walk-forward design (folds TBD -- example with 5 seasons):
  Fold 1: Train 2019-2022, Test 2023
  Fold 2: Train 2019-2023, Test 2024
  Fold 3: Train 2019-2024, Test 2025
  Fold 4: Train 2019-2025, Test 2026

Each fold runs Boruta on train data ONLY, then trains XGBoost on the
selected features and predicts the held-out test season. This eliminates
feature selection leakage.

Production models are trained on ALL data using features from
models/selected_features.json (output of 05b_select_features.py).

Isotonic calibrators are fitted on walk-forward OOF predictions.

Ported from CBB pipeline -- same architecture, MLB-specific parameters.

Outputs: models/trained/margin_model.pkl, models/trained/total_model.pkl,
         models/trained/shap_margin.png, models/trained/shap_total.png,
         models/walkforward_report.txt, models/training_metrics.json
"""

import sys
import json
import pickle
import importlib
import time
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend
import matplotlib.pyplot as plt
from pathlib import Path
from collections import Counter
from config import (
    HISTORICAL_DIR, PROCESSED_DIR, MODELS_DIR, MODELS_ROOT,
    XGBOOST_PARAMS_MARGIN, XGBOOST_PARAMS_TOTAL,
    MARGIN_FEATURES, TOTAL_FEATURES,
    SAMPLE_WEIGHT_HALF_LIFE, get_logger
)
from feature_engine import ALL_CANDIDATE_FEATURES

log = get_logger("06_train")

try:
    from sklearn.metrics import root_mean_squared_error, mean_absolute_error
    from sklearn.isotonic import IsotonicRegression
    import xgboost as xgb
    import shap
except ImportError as e:
    log.error(f"Missing package: {e}")
    log.error("Run: pip install scikit-learn xgboost shap --break-system-packages")
    sys.exit(1)

# Import run_boruta from 05b_select_features.py
_mod_05b = importlib.import_module("05b_select_features")
run_boruta = _mod_05b.run_boruta

# Walk-forward configuration
# MLB seasons are calendar years, so folds are by year
TEST_SEASONS = [2023, 2024, 2025, 2026]


def compute_sample_weights(seasons, half_life=SAMPLE_WEIGHT_HALF_LIFE):
    """Exponential decay weighting by season recency."""
    max_season = seasons.max()
    age = max_season - seasons
    return np.power(2.0, -age / half_life)


def fit_tail_aware_calibrator(predictions, actuals, name="model"):
    """
    Tail-aware calibration: isotonic in the core + linear regression at tails.
    Ported directly from CBB pipeline.
    """
    predictions = np.array(predictions, dtype=float)
    actuals = np.array(actuals, dtype=float)

    lo_thresh = np.percentile(predictions, 5)
    hi_thresh = np.percentile(predictions, 95)

    # Core isotonic (middle 90%)
    core_mask = (predictions >= lo_thresh) & (predictions <= hi_thresh)
    iso = IsotonicRegression(out_of_bounds="clip")
    iso.fit(predictions[core_mask], actuals[core_mask])

    # Low tail: linear regression on bottom 10%
    lo_mask = predictions < lo_thresh
    lo_slope, lo_intercept = 1.0, 0.0
    if lo_mask.sum() > 20:
        from numpy.polynomial.polynomial import polyfit
        lo_coeffs = polyfit(predictions[lo_mask], actuals[lo_mask], 1)
        lo_intercept, lo_slope = lo_coeffs[0], lo_coeffs[1]

    # High tail: linear regression on top 10%
    hi_mask = predictions > hi_thresh
    hi_slope, hi_intercept = 1.0, 0.0
    if hi_mask.sum() > 20:
        from numpy.polynomial.polynomial import polyfit
        hi_coeffs = polyfit(predictions[hi_mask], actuals[hi_mask], 1)
        hi_intercept, hi_slope = hi_coeffs[0], hi_coeffs[1]

    log.info(f"  {name} tail-aware calibrator: core [{lo_thresh:.1f}, {hi_thresh:.1f}], "
             f"lo_slope={lo_slope:.3f}, hi_slope={hi_slope:.3f}")

    return {
        "iso": iso,
        "lo_thresh": lo_thresh,
        "hi_thresh": hi_thresh,
        "lo_slope": lo_slope,
        "lo_intercept": lo_intercept,
        "hi_slope": hi_slope,
        "hi_intercept": hi_intercept,
    }


def apply_tail_aware_calibrator(cal, predictions):
    """Apply tail-aware calibrator to predictions."""
    predictions = np.array(predictions, dtype=float)
    result = np.empty_like(predictions)

    lo = cal["lo_thresh"]
    hi = cal["hi_thresh"]

    core_mask = (predictions >= lo) & (predictions <= hi)
    lo_mask = predictions < lo
    hi_mask = predictions > hi

    if core_mask.any():
        result[core_mask] = cal["iso"].predict(predictions[core_mask])
    if lo_mask.any():
        result[lo_mask] = cal["lo_intercept"] + cal["lo_slope"] * predictions[lo_mask]
    if hi_mask.any():
        result[hi_mask] = cal["hi_intercept"] + cal["hi_slope"] * predictions[hi_mask]

    return result


def load_training_data():
    """Load historical training data."""
    path = HISTORICAL_DIR / "training_data_v1.csv"
    if not path.exists():
        log.error(f"Training data not found: {path}")
        log.error("Run 00_build_historical.py first")
        sys.exit(1)

    df = pd.read_csv(path)
    df["date"] = pd.to_datetime(df["date"])
    if "season" not in df.columns:
        df["season"] = df["date"].dt.year
    log.info(f"Loaded {len(df):,} training games")
    return df


def walk_forward_validate(df, target_col, params, features=None, model_name="model"):
    """
    Walk-forward validation with per-fold Boruta.
    Returns OOF predictions and per-fold metrics.
    """
    log.info(f"\n{'='*60}")
    log.info(f"Walk-forward validation for {model_name}")
    log.info(f"  Target: {target_col}")
    log.info(f"  Test seasons: {TEST_SEASONS}")

    oof_preds = []
    oof_actuals = []
    oof_meta = []
    fold_metrics = []
    feature_counts = Counter()

    for fold_idx, test_season in enumerate(TEST_SEASONS):
        log.info(f"\n--- Fold {fold_idx+1}: Test {test_season} ---")

        train = df[df["season"] < test_season].copy()
        test = df[df["season"] == test_season].copy()

        if len(test) == 0:
            log.warning(f"  No test data for {test_season}, skipping")
            continue

        # Filter to games with target
        train = train[train[target_col].notna()]
        test = test[test[target_col].notna()]

        log.info(f"  Train: {len(train):,} games, Test: {len(test):,} games")

        # Per-fold Boruta feature selection
        available = [f for f in ALL_CANDIDATE_FEATURES if f in df.columns]
        X_boruta = train[available].copy()
        y_boruta = train[target_col]

        confirmed, tentative, _ = run_boruta(X_boruta, y_boruta, f"{model_name}_fold{fold_idx+1}")
        fold_features = confirmed + tentative

        if len(fold_features) < 3:
            log.warning(f"  Only {len(fold_features)} features selected, using top 10 from full list")
            fold_features = available[:10]

        for feat in fold_features:
            feature_counts[feat] += 1

        # Prepare data
        X_train = train[fold_features].fillna(0)
        y_train = train[target_col]
        X_test = test[fold_features].fillna(0)
        y_test = test[target_col]

        # Sample weights
        weights = compute_sample_weights(train["season"])

        # Train XGBoost
        model = xgb.XGBRegressor(**params, verbosity=0, n_jobs=-1)
        model.fit(X_train, y_train, sample_weight=weights)

        # Predict
        preds = model.predict(X_test)

        # Metrics
        rmse = root_mean_squared_error(y_test, preds)
        mae = mean_absolute_error(y_test, preds)

        fold_metrics.append({
            "fold": fold_idx + 1,
            "test_season": test_season,
            "n_train": len(train),
            "n_test": len(test),
            "n_features": len(fold_features),
            "rmse": rmse,
            "mae": mae,
        })

        log.info(f"  RMSE: {rmse:.3f}, MAE: {mae:.3f}")

        oof_preds.extend(preds)
        oof_actuals.extend(y_test.values)
        oof_meta.extend(test[["game_id", "date", "home_team", "away_team", "season"]].to_dict("records"))

    # Aggregate metrics
    all_preds = np.array(oof_preds)
    all_actuals = np.array(oof_actuals)
    overall_rmse = root_mean_squared_error(all_actuals, all_preds)
    overall_mae = mean_absolute_error(all_actuals, all_preds)

    log.info(f"\n{'='*60}")
    log.info(f"Walk-Forward Results ({model_name}):")
    log.info(f"  Overall RMSE: {overall_rmse:.3f}")
    log.info(f"  Overall MAE: {overall_mae:.3f}")
    log.info(f"  Total OOF samples: {len(all_preds):,}")
    log.info(f"\nFeature selection frequency:")
    for feat, count in feature_counts.most_common():
        log.info(f"  {feat}: {count}/{len(TEST_SEASONS)} folds")

    return {
        "oof_preds": all_preds,
        "oof_actuals": all_actuals,
        "oof_meta": oof_meta,
        "fold_metrics": fold_metrics,
        "feature_counts": dict(feature_counts),
        "overall_rmse": overall_rmse,
        "overall_mae": overall_mae,
    }


def train_production_model(df, target_col, features, params, model_name="model"):
    """Train production model on ALL data."""
    log.info(f"\nTraining production {model_name} model on ALL data...")

    mask = df[target_col].notna()
    X = df.loc[mask, features].fillna(0)
    y = df.loc[mask, target_col]
    weights = compute_sample_weights(df.loc[mask, "season"])

    model = xgb.XGBRegressor(**params, verbosity=0, n_jobs=-1)
    model.fit(X, y, sample_weight=weights)

    log.info(f"  Trained on {len(X):,} games with {len(features)} features")

    return model


def main():
    """Run full training pipeline."""
    df = load_training_data()

    # Margin model
    margin_results = walk_forward_validate(
        df, "actual_margin", XGBOOST_PARAMS_MARGIN,
        model_name="margin"
    )

    # Total model
    total_results = walk_forward_validate(
        df, "actual_total", XGBOOST_PARAMS_TOTAL,
        model_name="total"
    )

    # Fit calibrators on OOF predictions
    log.info("\nFitting calibrators on OOF predictions...")
    margin_cal = fit_tail_aware_calibrator(
        margin_results["oof_preds"], margin_results["oof_actuals"], "margin"
    )
    total_cal = fit_tail_aware_calibrator(
        total_results["oof_preds"], total_results["oof_actuals"], "total"
    )

    # Train production models
    # Use features that appeared in majority of folds
    n_folds = len(TEST_SEASONS)
    margin_features = [f for f, c in margin_results["feature_counts"].items() if c >= n_folds // 2]
    total_features = [f for f, c in total_results["feature_counts"].items() if c >= n_folds // 2]

    margin_model = train_production_model(df, "actual_margin", margin_features, XGBOOST_PARAMS_MARGIN, "margin")
    total_model = train_production_model(df, "actual_total", total_features, XGBOOST_PARAMS_TOTAL, "total")

    # Save model bundles
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    for name, model, features, cal, results in [
        ("margin", margin_model, margin_features, margin_cal, margin_results),
        ("total", total_model, total_features, total_cal, total_results),
    ]:
        bundle = {
            "model": model,
            "features": features,
            "calibrator": cal,
            "metrics": {
                "n_samples": len(df),
                "cv_rmse_mean": results["overall_rmse"],
                "cv_mae_mean": results["overall_mae"],
                "feature_counts": results["feature_counts"],
            },
        }
        out_path = MODELS_DIR / f"{name}_model.pkl"
        with open(out_path, "wb") as f:
            pickle.dump(bundle, f)
        log.info(f"Saved {name} model to {out_path}")

    # Save OOF predictions for backtest
    for name, results in [("margin", margin_results), ("total", total_results)]:
        oof_df = pd.DataFrame(results["oof_meta"])
        oof_df[f"predicted_{name}"] = results["oof_preds"]
        oof_df[f"actual_{name}"] = results["oof_actuals"]
        out_path = MODELS_ROOT / f"oof_{name}_predictions.csv"
        oof_df.to_csv(out_path, index=False)
        log.info(f"Saved OOF {name} predictions to {out_path}")

    # Save training metrics
    metrics = {
        "margin_rmse": margin_results["overall_rmse"],
        "margin_mae": margin_results["overall_mae"],
        "total_rmse": total_results["overall_rmse"],
        "total_mae": total_results["overall_mae"],
        "margin_features": margin_features,
        "total_features": total_features,
    }
    metrics_path = MODELS_ROOT / "training_metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    log.info(f"Saved metrics to {metrics_path}")

    log.info("\nTraining complete!")
    log.info(f"  Margin RMSE: {margin_results['overall_rmse']:.3f}")
    log.info(f"  Total RMSE: {total_results['overall_rmse']:.3f}")


if __name__ == "__main__":
    main()
