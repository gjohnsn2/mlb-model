"""
07 -- Generate Daily Predictions (Lasso)
==========================================
Loads today's feature matrix + trained Lasso model bundle.
Outputs raw predictions for every game.

Each row includes:
  - Raw model predicted margin (home perspective, positive = home favored)
  - Raw model predicted total
  - Top Lasso coefficient drivers

Outputs: data/predictions/picks_{TODAY}.csv

Run: python3 07_predict.py
"""

import sys
import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from config import (
    PROCESSED_DIR, PREDICTIONS_DIR, MODELS_DIR,
    TODAY, get_logger
)

log = get_logger("07_predict")


def load_lasso_bundle(name):
    """Load a trained Lasso model bundle."""
    path = MODELS_DIR / "trained" / f"lasso_{name}_nomarket.pkl"
    if not path.exists():
        log.error(f"Model not found: {path}")
        log.error("Run 06c_train_production_lasso.py first")
        sys.exit(1)

    with open(path, "rb") as f:
        bundle = pickle.load(f)

    log.info(f"Loaded {name} Lasso model:")
    log.info(f"  Trained on: {bundle['trained_on']} ({bundle['n_samples']:,} samples)")
    log.info(f"  Alpha: {bundle['alpha']:.6f}")
    log.info(f"  Features: {bundle['n_nonzero']}/{len(bundle['features'])} nonzero")
    log.info(f"  OOF RMSE: {bundle['rmse']:.3f}")
    return bundle


def get_lasso_drivers(model, feature_names, X_row, top_n=5):
    """
    Return top N coefficient * feature value drivers for a single prediction.
    Shows which features are pushing the prediction and by how much.
    """
    contributions = model.coef_ * X_row
    pairs = list(zip(feature_names, contributions))
    pairs.sort(key=lambda x: abs(x[1]), reverse=True)
    parts = []
    for fname, contrib in pairs[:top_n]:
        if abs(contrib) < 1e-6:
            continue
        sign = "+" if contrib >= 0 else ""
        parts.append(f"{fname}: {sign}{contrib:.3f}")
    return " | ".join(parts) if parts else "no significant drivers"


def predict():
    """Generate predictions for today's games."""
    # Load features
    features_path = PROCESSED_DIR / f"features_{TODAY}.csv"
    if not features_path.exists():
        log.error(f"Features not found: {features_path}")
        log.error("Run 05_build_features.py first")
        sys.exit(1)

    df = pd.read_csv(features_path)
    log.info(f"Loaded {len(df)} games from features")

    # Load Lasso bundles
    margin_bundle = load_lasso_bundle("margin")
    total_bundle = load_lasso_bundle("total")

    # ── Margin predictions ──
    margin_features = margin_bundle["features"]
    margin_scaler = margin_bundle["scaler"]
    margin_model = margin_bundle["model"]

    # Select features (fill missing with 0 — matching training pipeline)
    X_margin = df.reindex(columns=margin_features).fillna(0)
    missing_margin = [f for f in margin_features if f not in df.columns]
    if missing_margin:
        log.warning(f"  Missing {len(missing_margin)} margin features (filled with 0): "
                    f"{missing_margin[:5]}...")

    X_margin_scaled = margin_scaler.transform(X_margin.values)
    raw_margin = margin_model.predict(X_margin_scaled)

    # ── Total predictions ──
    total_features = total_bundle["features"]
    total_scaler = total_bundle["scaler"]
    total_model = total_bundle["model"]

    X_total = df.reindex(columns=total_features).fillna(0)
    missing_total = [f for f in total_features if f not in df.columns]
    if missing_total:
        log.warning(f"  Missing {len(missing_total)} total features (filled with 0): "
                    f"{missing_total[:5]}...")

    X_total_scaled = total_scaler.transform(X_total.values)
    raw_total = total_model.predict(X_total_scaled)

    # ── Lasso driver explanations ──
    margin_drivers = []
    total_drivers = []
    for i in range(len(df)):
        margin_drivers.append(get_lasso_drivers(
            margin_model, margin_features, X_margin_scaled[i]))
        total_drivers.append(get_lasso_drivers(
            total_model, total_features, X_total_scaled[i]))

    # ── Assemble output ──
    new_cols = {
        "raw_margin_pred": np.round(raw_margin, 4),
        "raw_total_pred": np.round(raw_total, 4),
        "margin_rmse": margin_bundle["rmse"],
        "total_rmse": total_bundle["rmse"],
        "margin_drivers": margin_drivers,
        "total_drivers": total_drivers,
    }
    if margin_bundle["calibration"]:
        cal = margin_bundle["calibration"]
        new_cols["cal_model_mean"] = cal["model_mean"]
        new_cols["cal_model_std"] = cal["model_std"]
        new_cols["cal_market_mean"] = cal["market_mean"]
        new_cols["cal_market_std"] = cal["market_std"]
    df = pd.concat([df, pd.DataFrame(new_cols, index=df.index)], axis=1)

    # ── Save ──
    PREDICTIONS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = PREDICTIONS_DIR / f"picks_{TODAY}.csv"
    df.to_csv(out_path, index=False)
    log.info(f"Saved predictions to {out_path}")

    # ── Summary ──
    log.info(f"\nPrediction Summary:")
    for _, row in df.iterrows():
        margin = row["raw_margin_pred"]
        total = row["raw_total_pred"]
        home = row.get("home_team", "?")
        away = row.get("away_team", "?")
        sp_h = row.get("home_sp_name", "TBD") or "TBD"
        sp_a = row.get("away_sp_name", "TBD") or "TBD"
        log.info(f"  {away} ({sp_a}) @ {home} ({sp_h}): "
                 f"margin={margin:+.2f}, total={total:.1f}")
        log.info(f"    Drivers: {row['margin_drivers']}")

    return df


if __name__ == "__main__":
    predict()
