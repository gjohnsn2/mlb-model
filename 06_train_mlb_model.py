"""
06 — Train MLB XGBoost Models (Walk-Forward Validation)
=========================================================
Trains two models (margin + total) using honest walk-forward validation
with per-fold Boruta feature selection.

Walk-forward design (4 folds, expanding window):
  Fold 1: Train 2019-2021, Test 2022
  Fold 2: Train 2019-2022, Test 2023
  Fold 3: Train 2019-2023, Test 2024
  Fold 4: Train 2019-2024, Test 2025

Season = calendar year (no Nov crossover like CBB).

Critical data insight: MLB consensus_spread is the runline (+/-1.5 in 88%
of games) — NOT a true spread. The market signal comes from H2H moneyline
odds, converted to market_implied_prob as the primary market feature.

Inputs:
  data/historical/training_data_mlb_v1.csv

Outputs:
  models/mlb_margin_model.pkl, models/mlb_total_model.pkl
  models/mlb_oof_margin_predictions.csv, models/mlb_oof_total_predictions.csv
  models/mlb_selected_features.json, models/mlb_walkforward_report.txt,
  models/mlb_training_metrics.json

Run: python3 06_train_mlb_model.py
"""

import sys
import json
import pickle
import importlib
import time
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
    get_logger
)

log = get_logger("06_train_mlb")

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

    Market features (market_implied_prob, market_logit, consensus_total, num_books)
    keep NaN — XGBoost handles natively. SP/batting features fill NaN with 0.
    """
    KEEP_NAN_FEATURES = {"market_implied_prob", "market_logit",
                         "consensus_total", "num_books"}

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
    MARKET_FEATURES = {"market_implied_prob", "market_logit",
                       "consensus_total", "num_books"}
    X_all = df[candidate_features].copy()
    for col in X_all.columns:
        if col not in MARKET_FEATURES:
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


# ── Report generation ──────────────────────────────────────────

def build_walkforward_report(margin_wf, total_wf):
    """Build human-readable walk-forward report."""
    lines = []
    lines.append("MLB WALK-FORWARD VALIDATION REPORT")
    lines.append("=" * 60)
    lines.append(f"Generated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')}")
    lines.append("")

    for model_name, wf in [("MARGIN", margin_wf), ("TOTAL", total_wf)]:
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
    log.info("=" * 60)
    log.info("MLB MODEL TRAINING")
    log.info("=" * 60)

    # Load data
    df = load_mlb_training_data()

    # Engineer market features from H2H odds
    df = engineer_market_features(df)

    # Identify available candidate features
    available = [f for f in MLB_CANDIDATE_FEATURES if f in df.columns]
    missing = [f for f in MLB_CANDIDATE_FEATURES if f not in df.columns]
    if missing:
        log.warning(f"Missing candidate features: {missing}")
    log.info(f"Available candidate features: {len(available)}")

    # ── Walk-forward: MARGIN model ─────────────────────────────
    margin_wf = None
    margin_df = df[df["actual_margin"].notna()].copy()
    log.info(f"Margin model: {len(margin_df):,} games with actual_margin")

    margin_wf = walk_forward_validate_mlb(
        margin_df, available, "actual_margin", "margin"
    )

    # ── Walk-forward: TOTAL model ──────────────────────────────
    total_wf = None
    # Exclude 7-inning doubleheader games for total model
    total_df = df[(df["actual_total"].notna()) &
                  (df["is_7_inning_dh"] != True)].copy()
    log.info(f"Total model: {len(total_df):,} games "
             f"(excluded {df['is_7_inning_dh'].sum()} 7-inning DH games)")

    total_wf = walk_forward_validate_mlb(
        total_df, available, "actual_total", "total"
    )

    # ── Save OOF predictions ───────────────────────────────────
    if margin_wf is not None:
        oof_path = MODELS_DIR / "mlb_oof_margin_predictions.csv"
        margin_wf["oof_df"].to_csv(oof_path, index=False)
        log.info(f"Saved margin OOF ({len(margin_wf['oof_df'])} games) -> {oof_path}")

    if total_wf is not None:
        oof_path = MODELS_DIR / "mlb_oof_total_predictions.csv"
        total_wf["oof_df"].to_csv(oof_path, index=False)
        log.info(f"Saved total OOF ({len(total_wf['oof_df'])} games) -> {oof_path}")

    # ── Determine selected features (all-folds union) ──────────
    selected = {"margin_features": [], "total_features": []}

    if margin_wf is not None:
        # Features selected in >= 50% of folds
        n_folds = len(margin_wf["fold_results"])
        margin_selected = [f for f, info in margin_wf["feature_stability"].items()
                           if info["count"] >= max(1, n_folds // 2)]
        selected["margin_features"] = sorted(margin_selected)
        log.info(f"Margin: {len(margin_selected)} features selected (>= {max(1, n_folds//2)}/{n_folds} folds)")

    if total_wf is not None:
        n_folds = len(total_wf["fold_results"])
        total_selected = [f for f, info in total_wf["feature_stability"].items()
                          if info["count"] >= max(1, n_folds // 2)]
        selected["total_features"] = sorted(total_selected)
        log.info(f"Total: {len(total_selected)} features selected (>= {max(1, n_folds//2)}/{n_folds} folds)")

    sel_path = MODELS_DIR / "mlb_selected_features.json"
    with open(sel_path, "w") as f:
        json.dump(selected, f, indent=2)
    log.info(f"Saved selected features -> {sel_path}")

    # ── Train production models on ALL data ────────────────────
    if margin_wf is not None:
        prod_features = selected["margin_features"] or available
        X_margin_prod = prepare_mlb_features(margin_df, prod_features)

        margin_model = train_production_model_mlb(
            X_margin_prod, margin_df["actual_margin"].astype(float),
            "margin", seasons=margin_df["season"]
        )

        # Fit calibrator on OOF predictions
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

        margin_path = MODELS_DIR / "mlb_margin_model.pkl"
        with open(margin_path, "wb") as f:
            pickle.dump({
                "model": margin_model,
                "features": list(X_margin_prod.columns),
                "metrics": margin_metrics,
                "calibrator": margin_cal,
            }, f)
        log.info(f"Saved margin model -> {margin_path}")

        generate_shap_plots(margin_model, X_margin_prod, "margin")

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

        total_path = MODELS_DIR / "mlb_total_model.pkl"
        with open(total_path, "wb") as f:
            pickle.dump({
                "model": total_model,
                "features": list(X_total_prod.columns),
                "metrics": total_metrics,
                "calibrator": total_cal,
            }, f)
        log.info(f"Saved total model -> {total_path}")

        generate_shap_plots(total_model, X_total_prod, "total")

    # ── Save combined metrics ──────────────────────────────────
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

    metrics_path = MODELS_DIR / "mlb_training_metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    log.info(f"Saved training metrics -> {metrics_path}")

    # ── Walk-forward report ────────────────────────────────────
    report = build_walkforward_report(margin_wf, total_wf)
    report_path = MODELS_DIR / "mlb_walkforward_report.txt"
    with open(report_path, "w") as f:
        f.write(report)
    log.info(f"Saved walk-forward report -> {report_path}")

    # ── Final summary ──────────────────────────────────────────
    print(f"\n{'='*60}")
    print("MLB TRAINING COMPLETE")
    print(f"{'='*60}")
    if margin_wf is not None:
        fold_rmses = [fr["rmse"] for fr in margin_wf["fold_results"]]
        print(f"  Margin RMSE: {margin_wf['overall_rmse']:.2f} "
              f"(mean fold: {np.mean(fold_rmses):.2f} +/- {np.std(fold_rmses):.2f})")
        print(f"  Margin features: {len(selected['margin_features'])}")
    if total_wf is not None:
        fold_rmses = [fr["rmse"] for fr in total_wf["fold_results"]]
        print(f"  Total  RMSE: {total_wf['overall_rmse']:.2f} "
              f"(mean fold: {np.mean(fold_rmses):.2f} +/- {np.std(fold_rmses):.2f})")
        print(f"  Total  features: {len(selected['total_features'])}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
