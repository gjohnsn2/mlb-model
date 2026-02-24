"""
06b -- Hyperparameter Tuning with Optuna
==========================================
Uses Bayesian optimization (Optuna) to find optimal XGBoost hyperparameters
via walk-forward validation on recent seasons.

Objective: minimize walk-forward RMSE on games with odds data,
weighted by recency. Optimizes for the use case that matters --
beating the market on games we actually bet on.

MLB-specific parameter ranges and search spaces.

Outputs:
  models/configs/tuned_params.json
  models/configs/tuning_report.txt

Requires: pip install optuna
"""

import sys
import json
import time
import numpy as np
import pandas as pd
from pathlib import Path
from config import (
    HISTORICAL_DIR, MODELS_ROOT,
    SAMPLE_WEIGHT_HALF_LIFE, get_logger
)
from feature_engine import ALL_CANDIDATE_FEATURES

log = get_logger("06b_tune")

try:
    from sklearn.metrics import root_mean_squared_error
    import xgboost as xgb
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
except ImportError as e:
    log.error(f"Missing package: {e}")
    log.error("Run: pip install scikit-learn xgboost optuna --break-system-packages")
    sys.exit(1)

# Import Boruta runner
import importlib
_mod_05b = importlib.import_module("05b_select_features")
run_boruta = _mod_05b.run_boruta

# Tuning configuration
TUNE_TEST_SEASONS = [2024, 2025, 2026]
N_TRIALS = 100


def compute_sample_weights(seasons, half_life=SAMPLE_WEIGHT_HALF_LIFE):
    max_season = seasons.max()
    age = max_season - seasons
    return np.power(2.0, -age / half_life)


def load_training_data():
    """Load training data."""
    path = HISTORICAL_DIR / "training_data_v1.csv"
    if not path.exists():
        log.error(f"Training data not found: {path}")
        sys.exit(1)

    df = pd.read_csv(path)
    df["date"] = pd.to_datetime(df["date"])
    if "season" not in df.columns:
        df["season"] = df["date"].dt.year
    log.info(f"Loaded {len(df):,} training games")
    return df


def objective(trial, df, target_col, features):
    """Optuna objective: minimize walk-forward RMSE."""
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 100, 600, step=50),
        "max_depth": trial.suggest_int("max_depth", 3, 10),
        "learning_rate": trial.suggest_float("learning_rate", 0.005, 0.1, log=True),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.4, 1.0),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 20),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 10.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
        "random_state": 42,
    }

    all_preds = []
    all_actuals = []

    for test_season in TUNE_TEST_SEASONS:
        train = df[(df["season"] < test_season) & df[target_col].notna()]
        test = df[(df["season"] == test_season) & df[target_col].notna()]

        if len(test) == 0:
            continue

        available = [f for f in features if f in df.columns]
        X_train = train[available].fillna(0)
        y_train = train[target_col]
        X_test = test[available].fillna(0)
        y_test = test[target_col]

        weights = compute_sample_weights(train["season"])

        model = xgb.XGBRegressor(**params, verbosity=0, n_jobs=-1)
        model.fit(X_train, y_train, sample_weight=weights)

        preds = model.predict(X_test)
        all_preds.extend(preds)
        all_actuals.extend(y_test.values)

    if not all_preds:
        return float("inf")

    return root_mean_squared_error(np.array(all_actuals), np.array(all_preds))


def main():
    """Run Optuna tuning for both models."""
    df = load_training_data()
    available = [f for f in ALL_CANDIDATE_FEATURES if f in df.columns]

    results = {}

    for target, name in [("actual_margin", "margin"), ("actual_total", "total")]:
        log.info(f"\n{'='*60}")
        log.info(f"Tuning {name} model ({N_TRIALS} trials)...")

        study = optuna.create_study(direction="minimize")
        study.optimize(
            lambda trial: objective(trial, df, target, available),
            n_trials=N_TRIALS,
        )

        best = study.best_params
        log.info(f"\nBest {name} params (RMSE={study.best_value:.4f}):")
        for k, v in best.items():
            log.info(f"  {k}: {v}")

        results[name] = {
            "params": best,
            "best_rmse": study.best_value,
        }

    # Save
    out_dir = MODELS_ROOT / "configs"
    out_dir.mkdir(parents=True, exist_ok=True)

    out_path = out_dir / "tuned_params.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    log.info(f"\nSaved tuned params to {out_path}")


if __name__ == "__main__":
    main()
