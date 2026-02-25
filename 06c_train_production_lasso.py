"""
06c_train_production_lasso — Train Production Lasso Model
==========================================================
Trains Lasso on ALL historical data (2015-2025) and saves a .pkl bundle
for daily prediction. Uses the same methodology as 06c_ridge_lasso_experiment.py
but trains on the full dataset instead of walk-forward folds.

Also computes calibration parameters from OOF predictions (from the walk-forward
experiment) so daily predictions can be calibrated to market scale.

Bundle format (models/trained/lasso_margin_nomarket.pkl):
  {
      "model": LassoCV,
      "scaler": StandardScaler,
      "features": [feature_names],
      "alpha": float,
      "n_nonzero": int,
      "rmse": float,              # OOF RMSE from walk-forward
      "calibration": {
          "model_mean": float,
          "model_std": float,
          "market_mean": float,
          "market_std": float,
      },
      "trained_on": "2015-2025",
      "n_samples": int,
  }

Run:
  python3 06c_train_production_lasso.py
"""

import sys
import pickle
import importlib
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.stats import norm
from config import (
    HISTORICAL_DIR, MODELS_DIR,
    MLB_CANDIDATE_FEATURES, MLB_SAMPLE_WEIGHT_HALF_LIFE,
    get_logger
)

log = get_logger("06c_prod_lasso")

try:
    from sklearn.linear_model import LassoCV
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import root_mean_squared_error
except ImportError as e:
    log.error(f"Missing package: {e}")
    sys.exit(1)

# Import shared functions from training scripts
_mod_06m = importlib.import_module("06_train_mlb_model")
load_mlb_training_data = _mod_06m.load_mlb_training_data
engineer_market_features = _mod_06m.engineer_market_features
engineer_f5_market_features = _mod_06m.engineer_f5_market_features
compute_sample_weights = _mod_06m.compute_sample_weights
MARKET_FEATURE_NAMES = _mod_06m.MARKET_FEATURE_NAMES

# Same alpha grid as walk-forward experiment
LASSO_ALPHAS = np.logspace(-4, 2, 50)

# For calibration: de-vig odds
def american_to_implied_prob(odds):
    if pd.isna(odds) or odds == 0:
        return np.nan
    if odds < 0:
        return abs(odds) / (abs(odds) + 100)
    else:
        return 100 / (odds + 100)


def compute_calibration(oof_path, training_df, rmse):
    """
    Compute calibration parameters from OOF predictions + training odds.
    Matches the methodology in 10_backtest_mlb.py:calibrate_predictions().
    """
    if not oof_path.exists():
        log.warning(f"OOF predictions not found: {oof_path}")
        log.warning("Run 06c_ridge_lasso_experiment.py --no-market first")
        return None

    oof = pd.read_csv(oof_path)
    oof["date"] = oof["date"].astype(str).str[:10]

    # Merge with training data for odds
    odds_cols = ["game_pk", "consensus_h2h_home", "consensus_h2h_away"]
    available = [c for c in odds_cols if c in training_df.columns]
    merged = oof.merge(training_df[available], on="game_pk", how="left")

    # Filter corrupt H2H
    MIN_ML = 100
    for col in ["consensus_h2h_home", "consensus_h2h_away"]:
        if col in merged.columns:
            corrupt = merged[col].notna() & (merged[col].abs() < MIN_ML)
            if corrupt.any():
                merged.loc[corrupt, col] = np.nan

    # De-vig market probabilities
    h2h_mask = merged["consensus_h2h_home"].notna() & merged["consensus_h2h_away"].notna()
    has_odds = merged[h2h_mask].copy()

    raw_home = has_odds["consensus_h2h_home"].apply(american_to_implied_prob)
    raw_away = has_odds["consensus_h2h_away"].apply(american_to_implied_prob)
    total_vig = raw_home + raw_away
    valid = total_vig.notna() & (total_vig > 0)
    has_odds.loc[valid, "market_home_prob"] = raw_home[valid] / total_vig[valid]

    # Convert market prob to implied margin
    has_odds["market_implied_margin"] = rmse * norm.ppf(
        has_odds["market_home_prob"].clip(0.001, 0.999)
    )

    # Calibration parameters
    model_mean = float(has_odds["predicted"].mean())
    model_std = float(has_odds["predicted"].std())
    market_mean = float(has_odds["market_implied_margin"].mean())
    market_std = float(has_odds["market_implied_margin"].std())

    log.info(f"  Calibration: model mean={model_mean:.4f}, std={model_std:.4f}")
    log.info(f"               market mean={market_mean:.4f}, std={market_std:.4f}")

    return {
        "model_mean": model_mean,
        "model_std": model_std,
        "market_mean": market_mean,
        "market_std": market_std,
    }


def train_production_model(df, candidate_features, target_col, model_name,
                           oof_path, training_df):
    """Train a production Lasso model on all data and save bundle."""

    # Fill NaN with 0 for linear models
    X = df[candidate_features].copy().fillna(0)

    # Drop zero-variance columns
    zero_var = [c for c in X.columns if X[c].std() == 0]
    if zero_var:
        log.info(f"  Dropping {len(zero_var)} zero-variance features")
        X = X.drop(columns=zero_var)
        candidate_features = [f for f in candidate_features if f not in zero_var]

    y = df[target_col].astype(float)
    feature_names = list(X.columns)

    log.info(f"\nTraining production {model_name} Lasso:")
    log.info(f"  Samples: {len(df):,}")
    log.info(f"  Features: {len(feature_names)}")

    # Sample weights (exponential decay)
    weights = compute_sample_weights(df["season"])

    # Standardize
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X.values)

    # Train LassoCV
    lasso = LassoCV(alphas=LASSO_ALPHAS, cv=5, max_iter=10000, random_state=42)
    lasso.fit(X_scaled, y.values, sample_weight=weights)

    preds = lasso.predict(X_scaled)
    train_rmse = root_mean_squared_error(y.values, preds)
    n_nonzero = int(np.sum(np.abs(lasso.coef_) > 1e-6))

    log.info(f"  Alpha: {lasso.alpha_:.6f}")
    log.info(f"  Train RMSE: {train_rmse:.3f}")
    log.info(f"  Nonzero features: {n_nonzero}/{len(feature_names)}")

    # Load OOF RMSE from walk-forward experiment
    oof_rmse = None
    if oof_path.exists():
        oof_df = pd.read_csv(oof_path)
        oof_rmse = float(root_mean_squared_error(oof_df["actual"], oof_df["predicted"]))
        log.info(f"  OOF RMSE (walk-forward): {oof_rmse:.3f}")
    else:
        oof_rmse = train_rmse
        log.warning(f"  No OOF predictions found, using train RMSE as fallback")

    # Compute calibration
    calibration = compute_calibration(oof_path, training_df, oof_rmse)

    # Report nonzero features
    coef_pairs = sorted(zip(feature_names, lasso.coef_),
                        key=lambda x: -abs(x[1]))
    log.info(f"\n  Top features ({model_name}):")
    for fname, coef in coef_pairs[:15]:
        if abs(coef) > 1e-6:
            log.info(f"    {fname:<40} {coef:+.4f}")

    # Build bundle
    bundle = {
        "model": lasso,
        "scaler": scaler,
        "features": feature_names,
        "alpha": float(lasso.alpha_),
        "n_nonzero": n_nonzero,
        "rmse": oof_rmse,
        "calibration": calibration,
        "trained_on": f"{int(df['season'].min())}-{int(df['season'].max())}",
        "n_samples": len(df),
    }

    return bundle


def main():
    log.info("=" * 60)
    log.info("PRODUCTION LASSO TRAINING (no-market)")
    log.info("=" * 60)

    # Load data
    df = load_mlb_training_data()
    df = engineer_market_features(df)
    df = engineer_f5_market_features(df)

    # Load raw training data for calibration (needs odds columns)
    training_df = pd.read_csv(HISTORICAL_DIR / "training_data_mlb_v2.csv")

    # No-market features only
    available = [f for f in MLB_CANDIDATE_FEATURES if f in df.columns]
    available = [f for f in available if f not in MARKET_FEATURE_NAMES]
    log.info(f"Candidate features (no-market): {len(available)}")

    # Ensure output directory
    trained_dir = MODELS_DIR / "trained"
    trained_dir.mkdir(parents=True, exist_ok=True)

    # ── MARGIN MODEL ──
    margin_df = df[df["actual_margin"].notna()].copy()
    margin_oof_path = MODELS_DIR / "mlb_oof_margin_lasso_nomarket_predictions.csv"

    margin_bundle = train_production_model(
        margin_df, available, "actual_margin", "margin",
        margin_oof_path, training_df
    )

    margin_path = trained_dir / "lasso_margin_nomarket.pkl"
    with open(margin_path, "wb") as f:
        pickle.dump(margin_bundle, f)
    log.info(f"\nSaved margin model -> {margin_path}")

    # ── TOTAL MODEL ──
    total_df = df[(df["actual_total"].notna()) &
                  (df["is_7_inning_dh"] != True)].copy()
    total_oof_path = MODELS_DIR / "mlb_oof_total_lasso_nomarket_predictions.csv"

    total_bundle = train_production_model(
        total_df, available, "actual_total", "total",
        total_oof_path, training_df
    )

    total_path = trained_dir / "lasso_total_nomarket.pkl"
    with open(total_path, "wb") as f:
        pickle.dump(total_bundle, f)
    log.info(f"Saved total model -> {total_path}")

    # ── Summary ──
    print(f"\n{'='*60}")
    print("PRODUCTION LASSO TRAINING COMPLETE")
    print(f"{'='*60}")
    print(f"  Margin model: {margin_path}")
    print(f"    Alpha: {margin_bundle['alpha']:.6f}")
    print(f"    Nonzero: {margin_bundle['n_nonzero']}/{len(margin_bundle['features'])}")
    print(f"    OOF RMSE: {margin_bundle['rmse']:.3f}")
    print(f"    Calibration: {'OK' if margin_bundle['calibration'] else 'MISSING'}")
    print(f"  Total model: {total_path}")
    print(f"    Alpha: {total_bundle['alpha']:.6f}")
    print(f"    Nonzero: {total_bundle['n_nonzero']}/{len(total_bundle['features'])}")
    print(f"    OOF RMSE: {total_bundle['rmse']:.3f}")
    print(f"    Calibration: {'OK' if total_bundle['calibration'] else 'MISSING'}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
