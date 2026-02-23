"""
07 -- Generate Daily Predictions
===================================
Loads today's feature matrix + trained models.
Outputs predictions with SHAP explanations for every game.

Each row includes:
  - Model predicted margin (home perspective, positive = home favored)
  - Model predicted total (combined runs)
  - Derived win probability (from margin via normal CDF)
  - Top 3 SHAP drivers for each prediction

Outputs: data/predictions/picks_YYYY-MM-DD.csv
"""

import sys
import pickle
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import norm
from config import (
    PROCESSED_DIR, PREDICTIONS_DIR, MODELS_DIR, REPORTS_DIR,
    TODAY, MARGIN_FEATURES, TOTAL_FEATURES, MARGIN_MODEL_RMSE,
    get_logger
)

log = get_logger("07_predict")

try:
    import xgboost as xgb
    import shap
except ImportError as e:
    log.error(f"Missing package: {e}")
    log.error("Run: pip install xgboost shap --break-system-packages")
    sys.exit(1)


def _apply_calibrator(calibrator, predictions):
    """Apply calibrator -- handles both legacy and tail-aware format."""
    if isinstance(calibrator, dict) and "iso" in calibrator:
        result = np.empty_like(predictions)
        lo = calibrator["lo_thresh"]
        hi = calibrator["hi_thresh"]
        core_mask = (predictions >= lo) & (predictions <= hi)
        lo_mask = predictions < lo
        hi_mask = predictions > hi
        if core_mask.any():
            result[core_mask] = calibrator["iso"].predict(predictions[core_mask])
        if lo_mask.any():
            result[lo_mask] = calibrator["lo_intercept"] + calibrator["lo_slope"] * predictions[lo_mask]
        if hi_mask.any():
            result[hi_mask] = calibrator["hi_intercept"] + calibrator["hi_slope"] * predictions[hi_mask]
        return result
    else:
        return calibrator.predict(predictions)


def load_model(name):
    """Load a pickled model bundle."""
    path = MODELS_DIR / f"{name}_model.pkl"
    if not path.exists():
        log.error(f"Model not found: {path}")
        log.error("Run 06_train_model.py first")
        sys.exit(1)

    with open(path, "rb") as f:
        bundle = pickle.load(f)

    log.info(f"Loaded {name} model ({bundle['metrics']['n_samples']} training samples, "
             f"CV RMSE: {bundle['metrics']['cv_rmse_mean']:.2f})")
    return bundle


def get_shap_explanations(model, X, feature_names, top_n=3):
    """Return top N SHAP drivers as human-readable strings per prediction."""
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)

    explanations = []
    for i in range(len(X)):
        sv = shap_values[i]
        pairs = list(zip(feature_names, sv))
        pairs.sort(key=lambda x: abs(x[1]), reverse=True)
        parts = []
        for fname, fval in pairs[:top_n]:
            sign = "+" if fval >= 0 else ""
            parts.append(f"{fname}: {sign}{fval:.2f}")
        explanations.append(" | ".join(parts))

    return explanations


def margin_to_win_prob(margin, rmse):
    """Convert predicted margin to win probability via normal CDF."""
    if rmse is None or rmse <= 0:
        return 0.5
    return norm.cdf(margin / rmse)


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

    # Load models
    margin_bundle = load_model("margin")
    total_bundle = load_model("total")

    margin_model = margin_bundle["model"]
    margin_features = margin_bundle["features"]
    margin_cal = margin_bundle.get("calibrator")

    total_model = total_bundle["model"]
    total_features = total_bundle["features"]
    total_cal = total_bundle.get("calibrator")

    # Prepare feature matrices
    available_margin = [f for f in margin_features if f in df.columns]
    available_total = [f for f in total_features if f in df.columns]

    X_margin = df[available_margin].fillna(0)
    X_total = df[available_total].fillna(0)

    # Predict
    raw_margin = margin_model.predict(X_margin)
    raw_total = total_model.predict(X_total)

    # Calibrate
    if margin_cal:
        cal_margin = _apply_calibrator(margin_cal, raw_margin)
    else:
        cal_margin = raw_margin

    if total_cal:
        cal_total = _apply_calibrator(total_cal, raw_total)
    else:
        cal_total = raw_total

    # Win probabilities (from margin model)
    rmse = margin_bundle["metrics"]["cv_rmse_mean"]
    win_probs = [margin_to_win_prob(m, rmse) for m in cal_margin]

    # SHAP explanations
    margin_shap = get_shap_explanations(margin_model, X_margin, available_margin)
    total_shap = get_shap_explanations(total_model, X_total, available_total)

    # Assemble output
    df["model_margin"] = np.round(cal_margin, 2)
    df["model_total"] = np.round(cal_total, 1)
    df["model_win_prob_home"] = np.round(win_probs, 4)
    df["model_win_prob_away"] = np.round(1 - np.array(win_probs), 4)
    df["raw_margin"] = np.round(raw_margin, 2)
    df["raw_total"] = np.round(raw_total, 1)
    df["margin_shap"] = margin_shap
    df["total_shap"] = total_shap

    # Save
    PREDICTIONS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = PREDICTIONS_DIR / f"picks_{TODAY}.csv"
    df.to_csv(out_path, index=False)
    log.info(f"Saved predictions to {out_path}")

    # Summary
    log.info(f"\nPrediction Summary:")
    for _, row in df.iterrows():
        margin = row["model_margin"]
        wp = row["model_win_prob_home"]
        total = row["model_total"]
        home = row.get("home_team", "?")
        away = row.get("away_team", "?")
        log.info(f"  {away} @ {home}: margin={margin:+.1f}, "
                 f"home_prob={wp:.1%}, total={total:.1f}")

    # Generate daily SHAP chart
    try:
        REPORTS_DIR.mkdir(parents=True, exist_ok=True)
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        shap.summary_plot(
            shap.TreeExplainer(margin_model).shap_values(X_margin),
            X_margin, feature_names=available_margin,
            show=False, plot_type="bar", ax=axes[0]
        )
        axes[0].set_title("Margin Model SHAP")
        shap.summary_plot(
            shap.TreeExplainer(total_model).shap_values(X_total),
            X_total, feature_names=available_total,
            show=False, plot_type="bar", ax=axes[1]
        )
        axes[1].set_title("Total Model SHAP")
        plt.tight_layout()
        plt.savefig(REPORTS_DIR / f"shap_daily_{TODAY}.png", dpi=150)
        plt.close()
        log.info(f"Saved SHAP chart to reports/shap_daily_{TODAY}.png")
    except Exception as e:
        log.warning(f"SHAP chart failed: {e}")

    return df


if __name__ == "__main__":
    predict()
