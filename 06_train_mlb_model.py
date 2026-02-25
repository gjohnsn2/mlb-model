"""
06 — Train MLB XGBoost Models (Walk-Forward Validation)
=========================================================
Trains five models (margin, total, F5 margin, F5 total, NRFI) using
honest walk-forward validation with per-fold Boruta feature selection.

Flags:
  --no-market  Exclude all market-derived features. Outputs get _nomarket
               suffix to avoid overwriting market-based models.
  --ensemble   Merge no-market OOF predictions as features alongside market
               features. Requires --no-market to have been run first.
               Outputs get _ensemble suffix.

Run: python3 06_train_mlb_model.py
     python3 06_train_mlb_model.py --no-market
     python3 06_train_mlb_model.py --ensemble
"""

import sys
import json
import pickle
import importlib
import time
import argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from collections import Counter
from config import (
    HISTORICAL_DIR, MODELS_DIR,
    MLB_XGBOOST_PARAMS, MLB_CANDIDATE_FEATURES,
    MLB_TEST_SEASONS, MLB_SAMPLE_WEIGHT_HALF_LIFE,
    XGBOOST_PARAMS_F5_MARGIN, XGBOOST_PARAMS_F5_TOTAL, XGBOOST_PARAMS_NRFI,
    F5_EXTRA_CANDIDATE_FEATURES, NRFI_EXTRA_CANDIDATE_FEATURES,
    get_logger
)

log = get_logger("06_train_mlb")

try:
    from sklearn.metrics import root_mean_squared_error, mean_absolute_error
    from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score
    from sklearn.isotonic import IsotonicRegression
    from sklearn.calibration import CalibratedClassifierCV
    import xgboost as xgb
    import shap
except ImportError as e:
    log.error(f"Missing package: {e}")
    log.error("Run: pip install scikit-learn xgboost shap --break-system-packages")
    sys.exit(1)

# Import run_boruta from 05b_select_features.py
_mod_05b = importlib.import_module("05b_select_features")
run_boruta = _mod_05b.run_boruta
run_boruta_classifier = _mod_05b.run_boruta_classifier

# All market-derived feature names (excluded by --no-market)
MARKET_FEATURE_NAMES = {
    "market_implied_prob", "market_logit", "consensus_total", "num_books",
    "f5_market_implied_prob", "f5_market_logit", "consensus_f5_total",
    "consensus_f1_total",
}

# Import calibration utilities from 06_train_model.py
_mod_06 = importlib.import_module("06_train_model")
fit_tail_aware_calibrator = _mod_06.fit_tail_aware_calibrator
apply_tail_aware_calibrator = _mod_06.apply_tail_aware_calibrator


# ── Market feature engineering ─────────────────────────────────

def american_to_implied_prob(odds):
    """Convert American odds to implied probability (no-vig)."""
    if pd.isna(odds):
        return np.nan
    if odds < 0:
        return abs(odds) / (abs(odds) + 100)
    elif odds > 0:
        return 100 / (odds + 100)
    else:
        return np.nan


def engineer_market_features(df):
    """
    Convert consensus_h2h_home American odds to market_implied_prob + market_logit.

    market_implied_prob: home win probability implied by H2H moneyline
    market_logit: log(p / (1-p)) — unbounded, better for tree models
    """
    df = df.copy()

    # Convert home H2H odds to implied probability
    df["market_implied_prob"] = df["consensus_h2h_home"].apply(american_to_implied_prob)

    # Also get away implied prob for de-vigging
    df["_away_prob"] = df["consensus_h2h_away"].apply(american_to_implied_prob)

    # De-vig: normalize so probs sum to 1
    total_prob = df["market_implied_prob"] + df["_away_prob"]
    mask = total_prob.notna() & (total_prob > 0)
    df.loc[mask, "market_implied_prob"] = df.loc[mask, "market_implied_prob"] / total_prob[mask]

    # Market logit: log(p / (1-p))
    p = df["market_implied_prob"].clip(0.01, 0.99)
    df["market_logit"] = np.log(p / (1 - p))

    df = df.drop(columns=["_away_prob"])

    n_valid = df["market_implied_prob"].notna().sum()
    log.info(f"Engineered market features: {n_valid}/{len(df)} games with H2H odds")

    return df


def engineer_f5_market_features(df):
    """
    Compute F5 market features from sparse consensus_f5_h2h_home/away odds.
    ~19% coverage (mostly 2023+). XGBoost handles NaN natively.

    f5_market_implied_prob: home F5 win probability from F5 H2H moneyline
    f5_market_logit: log(p / (1-p)) — unbounded, better for tree models
    """
    df = df.copy()

    if "consensus_f5_h2h_home" not in df.columns:
        df["f5_market_implied_prob"] = np.nan
        df["f5_market_logit"] = np.nan
        log.info("No F5 H2H odds columns found — F5 market features set to NaN")
        return df

    df["f5_market_implied_prob"] = df["consensus_f5_h2h_home"].apply(american_to_implied_prob)
    away_prob = df["consensus_f5_h2h_away"].apply(american_to_implied_prob)

    # De-vig: normalize so probs sum to 1
    total_prob = df["f5_market_implied_prob"] + away_prob
    mask = total_prob.notna() & (total_prob > 0)
    df.loc[mask, "f5_market_implied_prob"] = (
        df.loc[mask, "f5_market_implied_prob"] / total_prob[mask]
    )

    # F5 market logit
    p = df["f5_market_implied_prob"].clip(0.01, 0.99)
    df["f5_market_logit"] = np.where(p.notna(), np.log(p / (1 - p)), np.nan)

    n_valid = df["f5_market_implied_prob"].notna().sum()
    log.info(f"Engineered F5 market features: {n_valid}/{len(df)} games with F5 H2H odds "
             f"({n_valid/len(df)*100:.1f}%)")

    return df


# ── Data loading ───────────────────────────────────────────────

def load_mlb_training_data():
    """Load MLB training data CSV and derive season."""
    path = HISTORICAL_DIR / "training_data_mlb_v2.csv"
    if not path.exists():
        log.error(f"MLB training data not found: {path}")
        sys.exit(1)

    df = pd.read_csv(path)
    df["date"] = pd.to_datetime(df["date"])
    df["season"] = df["date"].dt.year

    log.info(f"Loaded {len(df):,} MLB games from {path}")
    log.info(f"  Seasons: {sorted(df['season'].unique())}")
    log.info(f"  Date range: {df['date'].min()} to {df['date'].max()}")

    return df


def prepare_mlb_features(df, feature_cols):
    """
    Select and clean feature columns for modeling.

    Market features keep NaN — XGBoost handles natively.
    SP/batting features fill NaN with 0.
    """
    KEEP_NAN_FEATURES = {"market_implied_prob", "market_logit",
                         "consensus_total", "num_books",
                         "f5_market_implied_prob", "f5_market_logit",
                         "consensus_f5_total", "consensus_f1_total",
                         "nomarket_pred_margin", "nomarket_pred_f5_margin",
                         "nomarket_pred_nrfi"}

    X = df.reindex(columns=feature_cols).copy()
    for col in X.columns:
        if col not in KEEP_NAN_FEATURES:
            X[col] = X[col].fillna(0)

    return X


# ── Sample weighting ───────────────────────────────────────────

def compute_sample_weights(seasons, half_life=MLB_SAMPLE_WEIGHT_HALF_LIFE):
    """Exponential decay weights: recent seasons contribute more."""
    max_season = seasons.max()
    age = max_season - seasons
    return np.power(2.0, -age / half_life)


# ── Walk-forward validation ───────────────────────────────────

def walk_forward_validate_mlb(df, candidate_features, target_col, model_name,
                               test_seasons=None, xgb_params=None):
    """
    Expanding-window walk-forward validation with per-fold Boruta.

    For each fold:
      1. Run Boruta on train data ONLY to select features
      2. Train XGBoost on train data with selected features
      3. Predict held-out test season
    """
    if test_seasons is None:
        test_seasons = MLB_TEST_SEASONS
    if xgb_params is None:
        xgb_params = MLB_XGBOOST_PARAMS

    log.info(f"\n{'='*60}")
    log.info(f"Walk-forward validation for MLB {model_name}")
    log.info(f"  Test seasons: {test_seasons}")
    log.info(f"  Candidate features: {len(candidate_features)}")

    season = df["season"]

    # Build candidate feature matrix
    KEEP_NAN = {"market_implied_prob", "market_logit",
                "consensus_total", "num_books",
                "f5_market_implied_prob", "f5_market_logit",
                "consensus_f5_total", "consensus_f1_total",
                "nomarket_pred_margin", "nomarket_pred_f5_margin",
                "nomarket_pred_nrfi"}
    X_all = df[candidate_features].copy()
    for col in X_all.columns:
        if col not in KEEP_NAN:
            X_all[col] = X_all[col].fillna(0)

    # Drop zero-variance columns
    zero_var = [c for c in X_all.columns
                if X_all[c].dropna().std() == 0 and X_all[c].notna().any()]
    if zero_var:
        log.info(f"  Dropping {len(zero_var)} zero-variance features: {zero_var}")
        X_all = X_all.drop(columns=zero_var)
        candidate_features = [f for f in candidate_features if f not in zero_var]

    y_all = df[target_col].astype(float)

    # Accumulators
    oof_predictions = np.full(len(df), np.nan)
    oof_actuals = np.full(len(df), np.nan)
    oof_folds = np.full(len(df), -1, dtype=int)
    fold_results = []
    all_selected_features = []

    total_start = time.time()

    for fold_num, test_season in enumerate(test_seasons):
        fold_start = time.time()
        train_mask = season < test_season
        test_mask = season == test_season

        n_train = train_mask.sum()
        n_test = test_mask.sum()

        if n_test == 0:
            log.warning(f"  Fold {fold_num+1} (test={test_season}): no test games, skipping")
            continue

        X_train = X_all[train_mask]
        y_train = y_all[train_mask]
        X_test = X_all[test_mask]
        y_test = y_all[test_mask]

        log.info(f"\n  Fold {fold_num+1}/{len(test_seasons)}: "
                 f"train=2019-{test_season-1} ({n_train:,}), "
                 f"test={test_season} ({n_test:,})")

        # 1. Run Boruta on train data ONLY
        boruta_start = time.time()
        X_train_boruta = X_train.fillna(0)
        confirmed, tentative, rejected, rankings = run_boruta(
            X_train_boruta, y_train,
            name=f"mlb_{model_name}_fold{fold_num+1}",
            n_iter=100,
            alpha=0.05,
        )
        fold_features = sorted(confirmed + tentative)
        boruta_time = time.time() - boruta_start

        # Fallback: if Boruta selects 0, take top 10 by ranking
        if not fold_features:
            log.warning(f"    Boruta selected 0 features — using top 10 by ranking")
            fold_features = sorted(rankings, key=rankings.get)[:10]

        log.info(f"    Boruta: {len(confirmed)}C + {len(tentative)}T = "
                 f"{len(fold_features)} features ({boruta_time:.0f}s)")

        # 2. Train XGBoost with sample weights
        train_seasons = season[train_mask]
        train_weights = compute_sample_weights(train_seasons)

        model = xgb.XGBRegressor(**xgb_params)
        model.fit(X_train[fold_features], y_train,
                  sample_weight=train_weights, verbose=False)

        # 3. Predict test fold
        preds = model.predict(X_test[fold_features])

        # 4. Compute metrics
        rmse = root_mean_squared_error(y_test, preds)
        mae = mean_absolute_error(y_test, preds)
        fold_time = time.time() - fold_start

        log.info(f"    RMSE: {rmse:.2f}, MAE: {mae:.2f} ({fold_time:.0f}s)")

        # Store OOF predictions
        test_indices = np.where(test_mask)[0]
        oof_predictions[test_indices] = preds
        oof_actuals[test_indices] = y_test.values
        oof_folds[test_indices] = fold_num + 1

        fold_results.append({
            "fold": fold_num + 1,
            "test_season": int(test_season),
            "n_train": int(n_train),
            "n_test": int(n_test),
            "n_features": len(fold_features),
            "features": fold_features,
            "confirmed": sorted(confirmed),
            "tentative": sorted(tentative),
            "rmse": float(rmse),
            "mae": float(mae),
            "time_seconds": float(fold_time),
        })
        all_selected_features.extend(fold_features)

    total_time = time.time() - total_start

    # Overall metrics on all OOF predictions
    valid_mask = ~np.isnan(oof_predictions)
    overall_rmse = root_mean_squared_error(oof_actuals[valid_mask], oof_predictions[valid_mask])
    overall_mae = mean_absolute_error(oof_actuals[valid_mask], oof_predictions[valid_mask])

    # Feature stability
    feature_counts = Counter(all_selected_features)
    n_folds = len(fold_results)
    feature_stability = {
        feat: {"count": count, "pct": count / n_folds}
        for feat, count in feature_counts.most_common()
    }

    log.info(f"\n  Walk-forward complete ({total_time:.0f}s total)")
    log.info(f"  Overall RMSE: {overall_rmse:.2f}, MAE: {overall_mae:.2f}")
    log.info(f"  OOF samples: {valid_mask.sum():,}")
    log.info(f"  Features in all {n_folds} folds: "
             f"{sum(1 for v in feature_stability.values() if v['count'] == n_folds)}")

    # Build OOF DataFrame
    oof_df = pd.DataFrame({
        "game_pk": df["game_pk"].values[valid_mask],
        "date": df["date"].values[valid_mask],
        "home_team": df["home_team"].values[valid_mask],
        "away_team": df["away_team"].values[valid_mask],
        "fold": oof_folds[valid_mask],
        "predicted": oof_predictions[valid_mask],
        "actual": oof_actuals[valid_mask],
        "season": df["season"].values[valid_mask],
    })

    return {
        "oof_predictions": oof_predictions[valid_mask],
        "oof_actuals": oof_actuals[valid_mask],
        "oof_df": oof_df,
        "fold_results": fold_results,
        "feature_stability": feature_stability,
        "overall_rmse": float(overall_rmse),
        "overall_mae": float(overall_mae),
        "n_oof_samples": int(valid_mask.sum()),
        "total_time_seconds": float(total_time),
    }


# ── Classifier walk-forward validation ─────────────────────────

def walk_forward_validate_mlb_classifier(df, candidate_features, target_col, model_name,
                                          test_seasons=None, xgb_params=None):
    """
    Expanding-window walk-forward validation for binary classification (NRFI).
    Same structure as regressor version but uses XGBClassifier and reports
    Brier score, log loss, and AUC instead of RMSE/MAE.
    """
    if test_seasons is None:
        test_seasons = MLB_TEST_SEASONS
    if xgb_params is None:
        xgb_params = XGBOOST_PARAMS_NRFI

    log.info(f"\n{'='*60}")
    log.info(f"Walk-forward validation for MLB {model_name} (classifier)")
    log.info(f"  Test seasons: {test_seasons}")
    log.info(f"  Candidate features: {len(candidate_features)}")

    season = df["season"]

    # Build candidate feature matrix
    KEEP_NAN = {"market_implied_prob", "market_logit", "consensus_total",
                "num_books", "f5_market_implied_prob", "f5_market_logit",
                "consensus_f5_total", "consensus_f1_total",
                "nomarket_pred_margin", "nomarket_pred_f5_margin",
                "nomarket_pred_nrfi"}
    X_all = df[candidate_features].copy()
    for col in X_all.columns:
        if col not in KEEP_NAN:
            X_all[col] = X_all[col].fillna(0)

    # Drop zero-variance columns
    zero_var = [c for c in X_all.columns
                if X_all[c].dropna().std() == 0 and X_all[c].notna().any()]
    if zero_var:
        log.info(f"  Dropping {len(zero_var)} zero-variance features: {zero_var}")
        X_all = X_all.drop(columns=zero_var)
        candidate_features = [f for f in candidate_features if f not in zero_var]

    y_all = df[target_col].astype(int)

    # Accumulators
    oof_probs = np.full(len(df), np.nan)
    oof_actuals = np.full(len(df), np.nan)
    oof_folds = np.full(len(df), -1, dtype=int)
    fold_results = []
    all_selected_features = []

    total_start = time.time()

    for fold_num, test_season in enumerate(test_seasons):
        fold_start = time.time()
        train_mask = season < test_season
        test_mask = season == test_season

        n_train = train_mask.sum()
        n_test = test_mask.sum()

        if n_test == 0:
            log.warning(f"  Fold {fold_num+1} (test={test_season}): no test games, skipping")
            continue

        X_train = X_all[train_mask]
        y_train = y_all[train_mask]
        X_test = X_all[test_mask]
        y_test = y_all[test_mask]

        log.info(f"\n  Fold {fold_num+1}/{len(test_seasons)}: "
                 f"train=2015-{test_season-1} ({n_train:,}), "
                 f"test={test_season} ({n_test:,}), "
                 f"base rate={y_train.mean():.3f}")

        # 1. Run Boruta (classifier) on train data ONLY
        boruta_start = time.time()
        X_train_boruta = X_train.fillna(0)
        confirmed, tentative, rejected, rankings = run_boruta_classifier(
            X_train_boruta, y_train,
            name=f"mlb_{model_name}_fold{fold_num+1}",
            n_iter=100,
            alpha=0.05,
        )
        fold_features = sorted(confirmed + tentative)
        boruta_time = time.time() - boruta_start

        # Fallback: if Boruta selects 0, take top 10 by ranking
        if not fold_features:
            log.warning(f"    Boruta selected 0 features — using top 10 by ranking")
            fold_features = sorted(rankings, key=rankings.get)[:10]

        log.info(f"    Boruta: {len(confirmed)}C + {len(tentative)}T = "
                 f"{len(fold_features)} features ({boruta_time:.0f}s)")

        # 2. Train XGBClassifier with sample weights
        train_seasons = season[train_mask]
        train_weights = compute_sample_weights(train_seasons)

        # Strip non-classifier params
        clf_params = {k: v for k, v in xgb_params.items()}
        model = xgb.XGBClassifier(**clf_params)
        model.fit(X_train[fold_features], y_train,
                  sample_weight=train_weights, verbose=False)

        # 3. Predict test fold (probabilities)
        probs = model.predict_proba(X_test[fold_features])[:, 1]

        # 4. Compute metrics
        brier = brier_score_loss(y_test, probs)
        logloss = log_loss(y_test, probs)
        auc = roc_auc_score(y_test, probs) if y_test.nunique() > 1 else 0.5
        fold_time = time.time() - fold_start

        log.info(f"    Brier: {brier:.4f}, LogLoss: {logloss:.4f}, AUC: {auc:.3f} ({fold_time:.0f}s)")

        # Store OOF predictions
        test_indices = np.where(test_mask)[0]
        oof_probs[test_indices] = probs
        oof_actuals[test_indices] = y_test.values
        oof_folds[test_indices] = fold_num + 1

        fold_results.append({
            "fold": fold_num + 1,
            "test_season": int(test_season),
            "n_train": int(n_train),
            "n_test": int(n_test),
            "n_features": len(fold_features),
            "features": fold_features,
            "confirmed": sorted(confirmed),
            "tentative": sorted(tentative),
            "brier": float(brier),
            "logloss": float(logloss),
            "auc": float(auc),
            "time_seconds": float(fold_time),
        })
        all_selected_features.extend(fold_features)

    total_time = time.time() - total_start

    # Overall metrics
    valid_mask = ~np.isnan(oof_probs)
    overall_brier = brier_score_loss(oof_actuals[valid_mask].astype(int), oof_probs[valid_mask])
    overall_logloss = log_loss(oof_actuals[valid_mask].astype(int), oof_probs[valid_mask])
    overall_auc = roc_auc_score(oof_actuals[valid_mask].astype(int), oof_probs[valid_mask])

    # Feature stability
    feature_counts = Counter(all_selected_features)
    n_folds = len(fold_results)
    feature_stability = {
        feat: {"count": count, "pct": count / n_folds}
        for feat, count in feature_counts.most_common()
    }

    log.info(f"\n  Walk-forward complete ({total_time:.0f}s total)")
    log.info(f"  Overall Brier: {overall_brier:.4f}, LogLoss: {overall_logloss:.4f}, AUC: {overall_auc:.3f}")
    log.info(f"  OOF samples: {valid_mask.sum():,}")
    base_rate = oof_actuals[valid_mask].mean()
    naive_brier = base_rate * (1 - base_rate) ** 2 + (1 - base_rate) * base_rate ** 2
    log.info(f"  Base rate: {base_rate:.3f}, Naive Brier: {naive_brier:.4f}")

    # Build OOF DataFrame
    oof_df = pd.DataFrame({
        "game_pk": df["game_pk"].values[valid_mask],
        "date": df["date"].values[valid_mask],
        "home_team": df["home_team"].values[valid_mask],
        "away_team": df["away_team"].values[valid_mask],
        "fold": oof_folds[valid_mask],
        "predicted": oof_probs[valid_mask],
        "actual": oof_actuals[valid_mask].astype(int),
        "season": df["season"].values[valid_mask],
    })

    return {
        "oof_predictions": oof_probs[valid_mask],
        "oof_actuals": oof_actuals[valid_mask].astype(int),
        "oof_df": oof_df,
        "fold_results": fold_results,
        "feature_stability": feature_stability,
        "overall_brier": float(overall_brier),
        "overall_logloss": float(overall_logloss),
        "overall_auc": float(overall_auc),
        "base_rate": float(base_rate),
        "naive_brier": float(naive_brier),
        "n_oof_samples": int(valid_mask.sum()),
        "total_time_seconds": float(total_time),
    }


# ── Production model training ─────────────────────────────────

def train_production_model_mlb(X, y, name, seasons=None, xgb_params=None):
    """Train final production model on ALL data."""
    if xgb_params is None:
        xgb_params = MLB_XGBOOST_PARAMS

    log.info(f"\n{'='*50}")
    log.info(f"Training production MLB {name} model")
    log.info(f"  Samples: {len(X)}, Features: {X.shape[1]}")

    model = xgb.XGBRegressor(**xgb_params)
    if seasons is not None:
        weights = compute_sample_weights(seasons)
        model.fit(X, y, sample_weight=weights, verbose=False)
        log.info(f"  Weights: min={weights.min():.3f}, max={weights.max():.3f}, "
                 f"mean={weights.mean():.3f}")
    else:
        model.fit(X, y, verbose=False)

    log.info(f"  Production MLB {name} model trained")
    return model


def train_production_classifier_mlb(X, y, name, seasons=None, xgb_params=None):
    """Train final production XGBClassifier + IsotonicRegression calibrator."""
    if xgb_params is None:
        xgb_params = XGBOOST_PARAMS_NRFI

    log.info(f"\n{'='*50}")
    log.info(f"Training production MLB {name} classifier")
    log.info(f"  Samples: {len(X)}, Features: {X.shape[1]}")
    log.info(f"  Base rate: {y.mean():.3f}")

    clf_params = {k: v for k, v in xgb_params.items()}
    model = xgb.XGBClassifier(**clf_params)

    if seasons is not None:
        weights = compute_sample_weights(seasons)
        model.fit(X, y, sample_weight=weights, verbose=False)
        log.info(f"  Weights: min={weights.min():.3f}, max={weights.max():.3f}")
    else:
        model.fit(X, y, verbose=False)

    # Fit isotonic calibrator on training data (in production, use OOF for better calibration)
    raw_probs = model.predict_proba(X)[:, 1]
    iso_cal = IsotonicRegression(out_of_bounds="clip")
    iso_cal.fit(raw_probs, y)

    log.info(f"  Production MLB {name} classifier trained + isotonic calibrator fit")
    return model, iso_cal


# ── Report generation ──────────────────────────────────────────

def build_walkforward_report(margin_wf, total_wf,
                              f5_margin_wf=None, f5_total_wf=None, nrfi_wf=None):
    """Build human-readable walk-forward report for all models."""
    lines = []
    lines.append("MLB WALK-FORWARD VALIDATION REPORT")
    lines.append("=" * 60)
    lines.append(f"Generated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')}")
    lines.append("")

    # Regressor models
    regressor_models = [
        ("MARGIN (full-game)", margin_wf),
        ("TOTAL (full-game)", total_wf),
        ("F5 MARGIN (first 5 innings)", f5_margin_wf),
        ("F5 TOTAL (first 5 innings)", f5_total_wf),
    ]

    for model_name, wf in regressor_models:
        if wf is None:
            continue
        lines.append(f"\n{'='*60}")
        lines.append(f"{model_name} MODEL")
        lines.append(f"{'='*60}")
        lines.append(f"Overall RMSE: {wf['overall_rmse']:.2f}")
        lines.append(f"Overall MAE:  {wf['overall_mae']:.2f}")
        lines.append(f"OOF samples:  {wf['n_oof_samples']:,}")
        lines.append(f"Total time:   {wf['total_time_seconds']:.0f}s")
        lines.append("")

        # Per-fold table
        lines.append("Per-Fold Results:")
        lines.append(f"{'Fold':>4}  {'Test':>6}  {'Train':>7}  {'Test':>6}  "
                      f"{'Feats':>5}  {'RMSE':>6}  {'MAE':>6}")
        lines.append("-" * 52)
        for fr in wf["fold_results"]:
            lines.append(
                f"  {fr['fold']:>2}  {fr['test_season']:>6}  "
                f"{fr['n_train']:>7,}  {fr['n_test']:>6,}  "
                f"{fr['n_features']:>5}  {fr['rmse']:>6.2f}  {fr['mae']:>6.2f}"
            )

        fold_rmses = [fr["rmse"] for fr in wf["fold_results"]]
        lines.append(f"       {'Mean':>6}  {'':>7}  {'':>6}  {'':>5}  "
                      f"{np.mean(fold_rmses):>6.2f}  "
                      f"{np.mean([fr['mae'] for fr in wf['fold_results']]):>6.2f}")

        # Feature stability
        lines.append(f"\nFeature Stability (selected in N of {len(wf['fold_results'])} folds):")
        lines.append(f"{'Feature':<35} {'Count':>5}  {'Pct':>5}")
        lines.append("-" * 48)
        n_folds = len(wf["fold_results"])
        for feat, info in sorted(wf["feature_stability"].items(),
                                  key=lambda x: (-x[1]["count"], x[0])):
            bar = "#" * info["count"] + "." * (n_folds - info["count"])
            lines.append(f"  {feat:<33} {info['count']:>5}  {info['pct']:>5.0%}  {bar}")

    # Classifier models (NRFI)
    if nrfi_wf is not None:
        wf = nrfi_wf
        lines.append(f"\n{'='*60}")
        lines.append(f"NRFI MODEL (classifier)")
        lines.append(f"{'='*60}")
        lines.append(f"Overall Brier:   {wf['overall_brier']:.4f}")
        lines.append(f"Overall LogLoss: {wf['overall_logloss']:.4f}")
        lines.append(f"Overall AUC:     {wf['overall_auc']:.3f}")
        lines.append(f"Base rate:       {wf['base_rate']:.3f}")
        lines.append(f"Naive Brier:     {wf['naive_brier']:.4f}")
        lines.append(f"Brier skill:     {1 - wf['overall_brier']/wf['naive_brier']:.3f}")
        lines.append(f"OOF samples:     {wf['n_oof_samples']:,}")
        lines.append(f"Total time:      {wf['total_time_seconds']:.0f}s")
        lines.append("")

        lines.append("Per-Fold Results:")
        lines.append(f"{'Fold':>4}  {'Test':>6}  {'Train':>7}  {'Test':>6}  "
                      f"{'Feats':>5}  {'Brier':>7}  {'LogLoss':>8}  {'AUC':>5}")
        lines.append("-" * 60)
        for fr in wf["fold_results"]:
            lines.append(
                f"  {fr['fold']:>2}  {fr['test_season']:>6}  "
                f"{fr['n_train']:>7,}  {fr['n_test']:>6,}  "
                f"{fr['n_features']:>5}  {fr['brier']:>7.4f}  "
                f"{fr['logloss']:>8.4f}  {fr['auc']:>5.3f}"
            )

        fold_briers = [fr["brier"] for fr in wf["fold_results"]]
        fold_aucs = [fr["auc"] for fr in wf["fold_results"]]
        lines.append(f"       {'Mean':>6}  {'':>7}  {'':>6}  {'':>5}  "
                      f"{np.mean(fold_briers):>7.4f}  "
                      f"{np.mean([fr['logloss'] for fr in wf['fold_results']]):>8.4f}  "
                      f"{np.mean(fold_aucs):>5.3f}")

        # Feature stability
        lines.append(f"\nFeature Stability (selected in N of {len(wf['fold_results'])} folds):")
        lines.append(f"{'Feature':<35} {'Count':>5}  {'Pct':>5}")
        lines.append("-" * 48)
        n_folds = len(wf["fold_results"])
        for feat, info in sorted(wf["feature_stability"].items(),
                                  key=lambda x: (-x[1]["count"], x[0])):
            bar = "#" * info["count"] + "." * (n_folds - info["count"])
            lines.append(f"  {feat:<33} {info['count']:>5}  {info['pct']:>5.0%}  {bar}")

    return "\n".join(lines)


def generate_shap_plots(model, X, name="model"):
    """Generate SHAP summary plot."""
    log.info(f"Generating SHAP plot for MLB {name}...")
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)

    fig, ax = plt.subplots(figsize=(10, 8))
    shap.summary_plot(shap_values, X, show=False, max_display=15)
    plt.title(f"MLB {name.title()} - SHAP Feature Importance", fontsize=14)
    plt.tight_layout()

    out_path = MODELS_DIR / f"mlb_shap_{name}.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info(f"  Saved SHAP plot -> {out_path}")


# ── Main ───────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Train MLB XGBoost models")
    mode_group = parser.add_mutually_exclusive_group()
    mode_group.add_argument("--no-market", action="store_true",
                            help="Exclude all market-derived features")
    mode_group.add_argument("--ensemble", action="store_true",
                            help="Ensemble mode: merge no-market OOF preds as features alongside market features")
    args = parser.parse_args()

    no_market = args.no_market
    ensemble = args.ensemble
    suffix = "_nomarket" if no_market else ("_ensemble" if ensemble else "")
    mode_label = ("NO-MARKET (baseball-only)" if no_market
                  else ("ENSEMBLE (market + baseball stacking)" if ensemble else "standard"))

    log.info("=" * 60)
    log.info(f"MLB MODEL TRAINING [{mode_label}]")
    log.info(f"  5 models: margin, total, F5 margin, F5 total, NRFI")
    if no_market:
        log.info(f"  EXCLUDING {len(MARKET_FEATURE_NAMES)} market features: "
                 f"{sorted(MARKET_FEATURE_NAMES)}")
        log.info(f"  Output suffix: {suffix}")
    if ensemble:
        log.info(f"  ENSEMBLE: merging no-market OOF predictions as features")
        log.info(f"  Output suffix: {suffix}")
    log.info("=" * 60)

    # Load data
    df = load_mlb_training_data()

    # Engineer market features (always — needed for data columns even if excluded from candidates)
    df = engineer_market_features(df)
    df = engineer_f5_market_features(df)

    # ── Ensemble: merge no-market OOF predictions as features ──
    if ensemble:
        oof_map = {
            "nomarket_pred_margin": "mlb_oof_margin_nomarket_predictions.csv",
            "nomarket_pred_f5_margin": "mlb_oof_f5_margin_nomarket_predictions.csv",
            "nomarket_pred_nrfi": "mlb_oof_nrfi_nomarket_predictions.csv",
        }
        for col_name, filename in oof_map.items():
            oof_path = MODELS_DIR / filename
            if not oof_path.exists():
                log.error(f"Ensemble requires {oof_path} — run: python3 06_train_mlb_model.py --no-market")
                sys.exit(1)
            oof_df = pd.read_csv(oof_path)
            # Rename 'predicted' to the ensemble feature name, keep only game_pk + prediction
            oof_df = oof_df[["game_pk", "predicted"]].rename(columns={"predicted": col_name})
            before_len = len(df)
            df = df.merge(oof_df, on="game_pk", how="left")
            matched = df[col_name].notna().sum()
            log.info(f"  Merged {col_name}: {matched}/{before_len} games matched "
                     f"({matched/before_len*100:.1f}%)")

    # Identify available candidate features (base 69)
    available = [f for f in MLB_CANDIDATE_FEATURES if f in df.columns]
    missing = [f for f in MLB_CANDIDATE_FEATURES if f not in df.columns]
    if missing:
        log.warning(f"Missing candidate features: {missing}")

    # F5 candidates = base + F5 market features
    f5_extra = [f for f in F5_EXTRA_CANDIDATE_FEATURES if f in df.columns]
    f5_available = available + f5_extra

    # NRFI candidates = base + consensus_f1_total
    nrfi_extra = [f for f in NRFI_EXTRA_CANDIDATE_FEATURES if f in df.columns]
    nrfi_available = available + nrfi_extra

    # Apply --no-market filter
    if no_market:
        before = len(available)
        available = [f for f in available if f not in MARKET_FEATURE_NAMES]
        f5_available = [f for f in f5_available if f not in MARKET_FEATURE_NAMES]
        nrfi_available = [f for f in nrfi_available if f not in MARKET_FEATURE_NAMES]
        log.info(f"Filtered market features: {before} -> {len(available)} base candidates")

    log.info(f"Available base candidate features: {len(available)}")
    log.info(f"F5 candidate features: {len(f5_available)}")
    log.info(f"NRFI candidate features: {len(nrfi_available)}")

    # ══════════════════════════════════════════════════════════════
    # MODEL 1: FULL-GAME MARGIN
    # ══════════════════════════════════════════════════════════════
    margin_wf = None
    margin_df = df[df["actual_margin"].notna()].copy()
    log.info(f"Margin model: {len(margin_df):,} games with actual_margin")

    margin_wf = walk_forward_validate_mlb(
        margin_df, available, "actual_margin", "margin"
    )

    # ══════════════════════════════════════════════════════════════
    # MODEL 2: FULL-GAME TOTAL
    # ══════════════════════════════════════════════════════════════
    total_wf = None
    # Exclude 7-inning doubleheader games for full-game total model
    total_df = df[(df["actual_total"].notna()) &
                  (df["is_7_inning_dh"] != True)].copy()
    log.info(f"Total model: {len(total_df):,} games "
             f"(excluded {df['is_7_inning_dh'].sum()} 7-inning DH games)")

    total_wf = walk_forward_validate_mlb(
        total_df, available, "actual_total", "total"
    )

    # ══════════════════════════════════════════════════════════════
    # MODEL 3: F5 MARGIN (first 5 innings)
    # ══════════════════════════════════════════════════════════════
    f5_margin_wf = None
    f5_margin_df = df[df["actual_f5_margin"].notna()].copy()
    log.info(f"F5 Margin model: {len(f5_margin_df):,} games with actual_f5_margin")

    f5_margin_wf = walk_forward_validate_mlb(
        f5_margin_df, f5_available, "actual_f5_margin", "f5_margin",
        xgb_params=XGBOOST_PARAMS_F5_MARGIN,
    )

    # ══════════════════════════════════════════════════════════════
    # MODEL 4: F5 TOTAL (first 5 innings)
    # ══════════════════════════════════════════════════════════════
    # Include 7-inning DH games — F5 is valid for those
    f5_total_wf = None
    f5_total_df = df[df["actual_f5_total"].notna()].copy()
    log.info(f"F5 Total model: {len(f5_total_df):,} games with actual_f5_total "
             f"(7-inning DH games INCLUDED — F5 is valid)")

    f5_total_wf = walk_forward_validate_mlb(
        f5_total_df, f5_available, "actual_f5_total", "f5_total",
        xgb_params=XGBOOST_PARAMS_F5_TOTAL,
    )

    # ══════════════════════════════════════════════════════════════
    # MODEL 5: NRFI (binary classifier)
    # ══════════════════════════════════════════════════════════════
    nrfi_wf = None
    nrfi_df = df[df["actual_nrfi"].notna()].copy()
    nrfi_df["actual_nrfi"] = nrfi_df["actual_nrfi"].astype(int)
    log.info(f"NRFI model: {len(nrfi_df):,} games with actual_nrfi "
             f"(base rate: {nrfi_df['actual_nrfi'].mean():.3f})")

    nrfi_wf = walk_forward_validate_mlb_classifier(
        nrfi_df, nrfi_available, "actual_nrfi", "nrfi",
        xgb_params=XGBOOST_PARAMS_NRFI,
    )

    # ══════════════════════════════════════════════════════════════
    # SAVE OOF PREDICTIONS
    # ══════════════════════════════════════════════════════════════
    for name, wf in [("margin", margin_wf), ("total", total_wf),
                     ("f5_margin", f5_margin_wf), ("f5_total", f5_total_wf),
                     ("nrfi", nrfi_wf)]:
        if wf is not None:
            oof_path = MODELS_DIR / f"mlb_oof_{name}{suffix}_predictions.csv"
            wf["oof_df"].to_csv(oof_path, index=False)
            log.info(f"Saved {name} OOF ({len(wf['oof_df'])} games) -> {oof_path}")

    # ══════════════════════════════════════════════════════════════
    # DETERMINE SELECTED FEATURES (>= 50% of folds)
    # ══════════════════════════════════════════════════════════════
    selected = {
        "margin_features": [], "total_features": [],
        "f5_margin_features": [], "f5_total_features": [],
        "nrfi_features": [],
    }

    for key, wf in [("margin_features", margin_wf), ("total_features", total_wf),
                     ("f5_margin_features", f5_margin_wf), ("f5_total_features", f5_total_wf),
                     ("nrfi_features", nrfi_wf)]:
        if wf is not None:
            n_folds = len(wf["fold_results"])
            threshold = max(1, n_folds // 2)
            sel = [f for f, info in wf["feature_stability"].items()
                   if info["count"] >= threshold]
            selected[key] = sorted(sel)
            log.info(f"{key}: {len(sel)} features selected (>= {threshold}/{n_folds} folds)")

    sel_path = MODELS_DIR / f"mlb_selected_features{suffix}.json"
    with open(sel_path, "w") as f:
        json.dump(selected, f, indent=2)
    log.info(f"Saved selected features -> {sel_path}")

    # ══════════════════════════════════════════════════════════════
    # TRAIN PRODUCTION MODELS ON ALL DATA
    # ══════════════════════════════════════════════════════════════

    # --- Margin production model ---
    if margin_wf is not None:
        prod_features = selected["margin_features"] or available
        X_margin_prod = prepare_mlb_features(margin_df, prod_features)

        margin_model = train_production_model_mlb(
            X_margin_prod, margin_df["actual_margin"].astype(float),
            "margin", seasons=margin_df["season"]
        )

        margin_cal = None
        if len(margin_wf["oof_predictions"]) > 100:
            margin_cal = fit_tail_aware_calibrator(
                margin_wf["oof_predictions"], margin_wf["oof_actuals"], "margin"
            )
            cal_preds = apply_tail_aware_calibrator(
                margin_cal, margin_wf["oof_predictions"]
            )
            cal_rmse = root_mean_squared_error(margin_wf["oof_actuals"], cal_preds)
            log.info(f"  Margin calibrated RMSE: {cal_rmse:.2f}")
        else:
            cal_rmse = margin_wf["overall_rmse"]

        margin_metrics = {
            "walkforward_overall_rmse": margin_wf["overall_rmse"],
            "walkforward_overall_mae": margin_wf["overall_mae"],
            "walkforward_calibrated_rmse": float(cal_rmse),
            "walkforward_n_oof": margin_wf["n_oof_samples"],
            "n_samples": len(X_margin_prod),
            "n_features": X_margin_prod.shape[1],
            "features": list(X_margin_prod.columns),
            "has_calibrator": margin_cal is not None,
            "sample_weight_half_life": MLB_SAMPLE_WEIGHT_HALF_LIFE,
        }

        margin_path = MODELS_DIR / f"mlb_margin{suffix}_model.pkl"
        with open(margin_path, "wb") as f:
            pickle.dump({
                "model": margin_model,
                "features": list(X_margin_prod.columns),
                "metrics": margin_metrics,
                "calibrator": margin_cal,
            }, f)
        log.info(f"Saved margin model -> {margin_path}")
        generate_shap_plots(margin_model, X_margin_prod, f"margin{suffix}")

    # --- Total production model ---
    if total_wf is not None:
        prod_features_t = selected["total_features"] or available
        X_total_prod = prepare_mlb_features(total_df, prod_features_t)

        total_model = train_production_model_mlb(
            X_total_prod, total_df["actual_total"].astype(float),
            "total", seasons=total_df["season"]
        )

        total_cal = None
        if len(total_wf["oof_predictions"]) > 100:
            total_cal = fit_tail_aware_calibrator(
                total_wf["oof_predictions"], total_wf["oof_actuals"], "total"
            )
            cal_preds = apply_tail_aware_calibrator(
                total_cal, total_wf["oof_predictions"]
            )
            cal_rmse_t = root_mean_squared_error(total_wf["oof_actuals"], cal_preds)
            log.info(f"  Total calibrated RMSE: {cal_rmse_t:.2f}")
        else:
            cal_rmse_t = total_wf["overall_rmse"]

        total_metrics = {
            "walkforward_overall_rmse": total_wf["overall_rmse"],
            "walkforward_overall_mae": total_wf["overall_mae"],
            "walkforward_calibrated_rmse": float(cal_rmse_t),
            "walkforward_n_oof": total_wf["n_oof_samples"],
            "n_samples": len(X_total_prod),
            "n_features": X_total_prod.shape[1],
            "features": list(X_total_prod.columns),
            "has_calibrator": total_cal is not None,
            "sample_weight_half_life": MLB_SAMPLE_WEIGHT_HALF_LIFE,
        }

        total_path = MODELS_DIR / f"mlb_total{suffix}_model.pkl"
        with open(total_path, "wb") as f:
            pickle.dump({
                "model": total_model,
                "features": list(X_total_prod.columns),
                "metrics": total_metrics,
                "calibrator": total_cal,
            }, f)
        log.info(f"Saved total model -> {total_path}")
        generate_shap_plots(total_model, X_total_prod, f"total{suffix}")

    # --- F5 Margin production model ---
    if f5_margin_wf is not None:
        prod_f5m = selected["f5_margin_features"] or f5_available
        X_f5m_prod = prepare_mlb_features(f5_margin_df, prod_f5m)

        f5_margin_model = train_production_model_mlb(
            X_f5m_prod, f5_margin_df["actual_f5_margin"].astype(float),
            "f5_margin", seasons=f5_margin_df["season"],
            xgb_params=XGBOOST_PARAMS_F5_MARGIN,
        )

        f5m_cal = None
        if len(f5_margin_wf["oof_predictions"]) > 100:
            f5m_cal = fit_tail_aware_calibrator(
                f5_margin_wf["oof_predictions"], f5_margin_wf["oof_actuals"], "f5_margin"
            )
            cal_preds = apply_tail_aware_calibrator(f5m_cal, f5_margin_wf["oof_predictions"])
            cal_rmse_f5m = root_mean_squared_error(f5_margin_wf["oof_actuals"], cal_preds)
            log.info(f"  F5 Margin calibrated RMSE: {cal_rmse_f5m:.2f}")
        else:
            cal_rmse_f5m = f5_margin_wf["overall_rmse"]

        f5m_metrics = {
            "walkforward_overall_rmse": f5_margin_wf["overall_rmse"],
            "walkforward_overall_mae": f5_margin_wf["overall_mae"],
            "walkforward_calibrated_rmse": float(cal_rmse_f5m),
            "walkforward_n_oof": f5_margin_wf["n_oof_samples"],
            "n_samples": len(X_f5m_prod),
            "n_features": X_f5m_prod.shape[1],
            "features": list(X_f5m_prod.columns),
            "has_calibrator": f5m_cal is not None,
        }

        f5m_path = MODELS_DIR / f"mlb_f5_margin{suffix}_model.pkl"
        with open(f5m_path, "wb") as f:
            pickle.dump({
                "model": f5_margin_model,
                "features": list(X_f5m_prod.columns),
                "metrics": f5m_metrics,
                "calibrator": f5m_cal,
            }, f)
        log.info(f"Saved F5 margin model -> {f5m_path}")
        generate_shap_plots(f5_margin_model, X_f5m_prod, f"f5_margin{suffix}")

    # --- F5 Total production model ---
    if f5_total_wf is not None:
        prod_f5t = selected["f5_total_features"] or f5_available
        X_f5t_prod = prepare_mlb_features(f5_total_df, prod_f5t)

        f5_total_model = train_production_model_mlb(
            X_f5t_prod, f5_total_df["actual_f5_total"].astype(float),
            "f5_total", seasons=f5_total_df["season"],
            xgb_params=XGBOOST_PARAMS_F5_TOTAL,
        )

        f5t_cal = None
        if len(f5_total_wf["oof_predictions"]) > 100:
            f5t_cal = fit_tail_aware_calibrator(
                f5_total_wf["oof_predictions"], f5_total_wf["oof_actuals"], "f5_total"
            )
            cal_preds = apply_tail_aware_calibrator(f5t_cal, f5_total_wf["oof_predictions"])
            cal_rmse_f5t = root_mean_squared_error(f5_total_wf["oof_actuals"], cal_preds)
            log.info(f"  F5 Total calibrated RMSE: {cal_rmse_f5t:.2f}")
        else:
            cal_rmse_f5t = f5_total_wf["overall_rmse"]

        f5t_metrics = {
            "walkforward_overall_rmse": f5_total_wf["overall_rmse"],
            "walkforward_overall_mae": f5_total_wf["overall_mae"],
            "walkforward_calibrated_rmse": float(cal_rmse_f5t),
            "walkforward_n_oof": f5_total_wf["n_oof_samples"],
            "n_samples": len(X_f5t_prod),
            "n_features": X_f5t_prod.shape[1],
            "features": list(X_f5t_prod.columns),
            "has_calibrator": f5t_cal is not None,
        }

        f5t_path = MODELS_DIR / f"mlb_f5_total{suffix}_model.pkl"
        with open(f5t_path, "wb") as f:
            pickle.dump({
                "model": f5_total_model,
                "features": list(X_f5t_prod.columns),
                "metrics": f5t_metrics,
                "calibrator": f5t_cal,
            }, f)
        log.info(f"Saved F5 total model -> {f5t_path}")
        generate_shap_plots(f5_total_model, X_f5t_prod, f"f5_total{suffix}")

    # --- NRFI production classifier ---
    if nrfi_wf is not None:
        prod_nrfi = selected["nrfi_features"] or nrfi_available
        X_nrfi_prod = prepare_mlb_features(nrfi_df, prod_nrfi)

        nrfi_model, nrfi_iso_cal = train_production_classifier_mlb(
            X_nrfi_prod, nrfi_df["actual_nrfi"].astype(int),
            "nrfi", seasons=nrfi_df["season"],
            xgb_params=XGBOOST_PARAMS_NRFI,
        )

        nrfi_metrics = {
            "walkforward_overall_brier": nrfi_wf["overall_brier"],
            "walkforward_overall_logloss": nrfi_wf["overall_logloss"],
            "walkforward_overall_auc": nrfi_wf["overall_auc"],
            "walkforward_base_rate": nrfi_wf["base_rate"],
            "walkforward_naive_brier": nrfi_wf["naive_brier"],
            "walkforward_n_oof": nrfi_wf["n_oof_samples"],
            "n_samples": len(X_nrfi_prod),
            "n_features": X_nrfi_prod.shape[1],
            "features": list(X_nrfi_prod.columns),
        }

        nrfi_path = MODELS_DIR / f"mlb_nrfi{suffix}_model.pkl"
        with open(nrfi_path, "wb") as f:
            pickle.dump({
                "model": nrfi_model,
                "features": list(X_nrfi_prod.columns),
                "metrics": nrfi_metrics,
                "calibrator": nrfi_iso_cal,
            }, f)
        log.info(f"Saved NRFI model -> {nrfi_path}")

    # ══════════════════════════════════════════════════════════════
    # SAVE COMBINED METRICS
    # ══════════════════════════════════════════════════════════════
    metrics = {}
    if margin_wf is not None:
        metrics["margin"] = margin_metrics
        metrics["margin_walkforward"] = {
            "fold_results": margin_wf["fold_results"],
            "feature_stability": margin_wf["feature_stability"],
        }
    if total_wf is not None:
        metrics["total"] = total_metrics
        metrics["total_walkforward"] = {
            "fold_results": total_wf["fold_results"],
            "feature_stability": total_wf["feature_stability"],
        }
    if f5_margin_wf is not None:
        metrics["f5_margin"] = f5m_metrics
        metrics["f5_margin_walkforward"] = {
            "fold_results": f5_margin_wf["fold_results"],
            "feature_stability": f5_margin_wf["feature_stability"],
        }
    if f5_total_wf is not None:
        metrics["f5_total"] = f5t_metrics
        metrics["f5_total_walkforward"] = {
            "fold_results": f5_total_wf["fold_results"],
            "feature_stability": f5_total_wf["feature_stability"],
        }
    if nrfi_wf is not None:
        metrics["nrfi"] = nrfi_metrics
        metrics["nrfi_walkforward"] = {
            "fold_results": nrfi_wf["fold_results"],
            "feature_stability": nrfi_wf["feature_stability"],
        }

    metrics_path = MODELS_DIR / f"mlb_training_metrics{suffix}.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    log.info(f"Saved training metrics -> {metrics_path}")

    # ══════════════════════════════════════════════════════════════
    # WALK-FORWARD REPORT
    # ══════════════════════════════════════════════════════════════
    report = build_walkforward_report(margin_wf, total_wf,
                                       f5_margin_wf, f5_total_wf, nrfi_wf)
    report_path = MODELS_DIR / f"mlb_walkforward_report{suffix}.txt"
    with open(report_path, "w") as f:
        f.write(report)
    log.info(f"Saved walk-forward report -> {report_path}")

    # ══════════════════════════════════════════════════════════════
    # FINAL SUMMARY
    # ══════════════════════════════════════════════════════════════
    print(f"\n{'='*60}")
    print(f"MLB TRAINING COMPLETE — {mode_label} (5 models)")
    print(f"{'='*60}")
    if margin_wf is not None:
        fold_rmses = [fr["rmse"] for fr in margin_wf["fold_results"]]
        print(f"  Margin     RMSE: {margin_wf['overall_rmse']:.2f} "
              f"(mean: {np.mean(fold_rmses):.2f} +/- {np.std(fold_rmses):.2f}), "
              f"{len(selected['margin_features'])} features")
    if total_wf is not None:
        fold_rmses = [fr["rmse"] for fr in total_wf["fold_results"]]
        print(f"  Total      RMSE: {total_wf['overall_rmse']:.2f} "
              f"(mean: {np.mean(fold_rmses):.2f} +/- {np.std(fold_rmses):.2f}), "
              f"{len(selected['total_features'])} features")
    if f5_margin_wf is not None:
        fold_rmses = [fr["rmse"] for fr in f5_margin_wf["fold_results"]]
        print(f"  F5 Margin  RMSE: {f5_margin_wf['overall_rmse']:.2f} "
              f"(mean: {np.mean(fold_rmses):.2f} +/- {np.std(fold_rmses):.2f}), "
              f"{len(selected['f5_margin_features'])} features")
    if f5_total_wf is not None:
        fold_rmses = [fr["rmse"] for fr in f5_total_wf["fold_results"]]
        print(f"  F5 Total   RMSE: {f5_total_wf['overall_rmse']:.2f} "
              f"(mean: {np.mean(fold_rmses):.2f} +/- {np.std(fold_rmses):.2f}), "
              f"{len(selected['f5_total_features'])} features")
    if nrfi_wf is not None:
        fold_briers = [fr["brier"] for fr in nrfi_wf["fold_results"]]
        fold_aucs = [fr["auc"] for fr in nrfi_wf["fold_results"]]
        print(f"  NRFI       Brier: {nrfi_wf['overall_brier']:.4f} "
              f"(naive: {nrfi_wf['naive_brier']:.4f}), "
              f"AUC: {nrfi_wf['overall_auc']:.3f}, "
              f"{len(selected['nrfi_features'])} features")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
