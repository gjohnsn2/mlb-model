"""
06c — Ridge/Lasso Walk-Forward Experiment
==========================================
Tests whether regularized linear models find baseball signal that XGBoost
misses. The hypothesis: XGBoost needs strong splits to use a feature, but
weak linear trends (e.g., SP K% adds 0.02 runs of edge) get buried in noise.
Ridge/Lasso can detect these because they aggregate many small linear effects.

Approach:
  - Same walk-forward folds as 06_train_mlb_model.py (test 2017-2025)
  - StandardScaler fit on train only (Ridge/Lasso need scaled features)
  - RidgeCV / LassoCV / ElasticNetCV with built-in alpha cross-validation
  - No Boruta — Lasso does its own selection, Ridge uses all features
  - Same sample weighting (exponential decay, half_life=3yr)
  - Saves OOF predictions compatible with 10_backtest_mlb.py

Run:
  python3 06c_ridge_lasso_experiment.py               # with market features
  python3 06c_ridge_lasso_experiment.py --no-market    # baseball-only
"""

import sys
import json
import time
import importlib
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from collections import Counter
from config import (
    HISTORICAL_DIR, MODELS_DIR,
    MLB_CANDIDATE_FEATURES, MLB_TEST_SEASONS,
    MLB_SAMPLE_WEIGHT_HALF_LIFE,
    F5_EXTRA_CANDIDATE_FEATURES,
    get_logger
)

log = get_logger("06c_ridge_lasso")

try:
    from sklearn.linear_model import RidgeCV, LassoCV, ElasticNetCV
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import root_mean_squared_error, mean_absolute_error
except ImportError as e:
    log.error(f"Missing package: {e}")
    sys.exit(1)

# Import data loading from 06_train_mlb_model.py
_mod_06m = importlib.import_module("06_train_mlb_model")
load_mlb_training_data = _mod_06m.load_mlb_training_data
engineer_market_features = _mod_06m.engineer_market_features
engineer_f5_market_features = _mod_06m.engineer_f5_market_features
compute_sample_weights = _mod_06m.compute_sample_weights
MARKET_FEATURE_NAMES = _mod_06m.MARKET_FEATURE_NAMES

# Alpha grids for CV
RIDGE_ALPHAS = np.logspace(-2, 5, 50)   # 0.01 to 100,000
LASSO_ALPHAS = np.logspace(-4, 2, 50)   # 0.0001 to 100
ELASTICNET_L1_RATIOS = [0.1, 0.3, 0.5, 0.7, 0.9]


def walk_forward_linear(df, candidate_features, target_col, model_name,
                         test_seasons=None):
    """
    Walk-forward validation with Ridge, Lasso, and ElasticNet.

    For each fold:
      1. StandardScaler fit on train only
      2. RidgeCV / LassoCV / ElasticNetCV with internal 5-fold CV for alpha
      3. Predict held-out test season
      4. Record coefficients and selected alpha

    Returns results dict for each model type.
    """
    if test_seasons is None:
        test_seasons = MLB_TEST_SEASONS

    season = df["season"]

    # Build feature matrix — fill NaN with 0 for linear models
    X_all = df[candidate_features].copy().fillna(0)

    # Drop zero-variance columns
    zero_var = [c for c in X_all.columns if X_all[c].std() == 0]
    if zero_var:
        log.info(f"  Dropping {len(zero_var)} zero-variance features")
        X_all = X_all.drop(columns=zero_var)
        candidate_features = [f for f in candidate_features if f not in zero_var]

    y_all = df[target_col].astype(float)
    feature_names = list(X_all.columns)

    log.info(f"\n{'='*60}")
    log.info(f"Walk-forward linear models for {model_name}")
    log.info(f"  Test seasons: {test_seasons}")
    log.info(f"  Features: {len(feature_names)}")

    # Per-model accumulators
    model_types = ["ridge", "lasso", "elasticnet"]
    accum = {mt: {
        "oof_predictions": np.full(len(df), np.nan),
        "oof_actuals": np.full(len(df), np.nan),
        "oof_folds": np.full(len(df), -1, dtype=int),
        "fold_results": [],
        "all_coefs": [],
    } for mt in model_types}

    total_start = time.time()

    for fold_num, test_season in enumerate(test_seasons):
        fold_start = time.time()
        train_mask = season < test_season
        test_mask = season == test_season

        n_train = train_mask.sum()
        n_test = test_mask.sum()

        if n_test == 0:
            continue

        X_train = X_all[train_mask].values
        y_train = y_all[train_mask].values
        X_test = X_all[test_mask].values
        y_test = y_all[test_mask].values

        # Sample weights
        train_seasons = season[train_mask].values
        weights = compute_sample_weights(pd.Series(train_seasons))

        # Standardize (fit on train only)
        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_train)
        X_test_s = scaler.transform(X_test)

        log.info(f"\n  Fold {fold_num+1}/{len(test_seasons)}: "
                 f"train<{test_season} ({n_train:,}), test={test_season} ({n_test:,})")

        # ── Ridge ──
        ridge = RidgeCV(alphas=RIDGE_ALPHAS, cv=5)
        ridge.fit(X_train_s, y_train, sample_weight=weights)
        ridge_preds = ridge.predict(X_test_s)
        ridge_rmse = root_mean_squared_error(y_test, ridge_preds)
        ridge_mae = mean_absolute_error(y_test, ridge_preds)

        test_idx = np.where(test_mask)[0]
        accum["ridge"]["oof_predictions"][test_idx] = ridge_preds
        accum["ridge"]["oof_actuals"][test_idx] = y_test
        accum["ridge"]["oof_folds"][test_idx] = fold_num + 1
        accum["ridge"]["fold_results"].append({
            "fold": fold_num + 1, "test_season": int(test_season),
            "n_train": int(n_train), "n_test": int(n_test),
            "rmse": float(ridge_rmse), "mae": float(ridge_mae),
            "alpha": float(ridge.alpha_),
            "n_nonzero": int(np.sum(np.abs(ridge.coef_) > 1e-6)),
        })
        accum["ridge"]["all_coefs"].append(ridge.coef_.copy())

        # ── Lasso ──
        lasso = LassoCV(alphas=LASSO_ALPHAS, cv=5, max_iter=10000, random_state=42)
        lasso.fit(X_train_s, y_train, sample_weight=weights)
        lasso_preds = lasso.predict(X_test_s)
        lasso_rmse = root_mean_squared_error(y_test, lasso_preds)
        lasso_mae = mean_absolute_error(y_test, lasso_preds)
        lasso_nonzero = int(np.sum(np.abs(lasso.coef_) > 1e-6))

        accum["lasso"]["oof_predictions"][test_idx] = lasso_preds
        accum["lasso"]["oof_actuals"][test_idx] = y_test
        accum["lasso"]["oof_folds"][test_idx] = fold_num + 1
        accum["lasso"]["fold_results"].append({
            "fold": fold_num + 1, "test_season": int(test_season),
            "n_train": int(n_train), "n_test": int(n_test),
            "rmse": float(lasso_rmse), "mae": float(lasso_mae),
            "alpha": float(lasso.alpha_),
            "n_nonzero": lasso_nonzero,
        })
        accum["lasso"]["all_coefs"].append(lasso.coef_.copy())

        # ── ElasticNet ──
        enet = ElasticNetCV(
            l1_ratio=ELASTICNET_L1_RATIOS,
            alphas=LASSO_ALPHAS, cv=5, max_iter=10000, random_state=42
        )
        enet.fit(X_train_s, y_train, sample_weight=weights)
        enet_preds = enet.predict(X_test_s)
        enet_rmse = root_mean_squared_error(y_test, enet_preds)
        enet_mae = mean_absolute_error(y_test, enet_preds)
        enet_nonzero = int(np.sum(np.abs(enet.coef_) > 1e-6))

        accum["elasticnet"]["oof_predictions"][test_idx] = enet_preds
        accum["elasticnet"]["oof_actuals"][test_idx] = y_test
        accum["elasticnet"]["oof_folds"][test_idx] = fold_num + 1
        accum["elasticnet"]["fold_results"].append({
            "fold": fold_num + 1, "test_season": int(test_season),
            "n_train": int(n_train), "n_test": int(n_test),
            "rmse": float(enet_rmse), "mae": float(enet_mae),
            "alpha": float(enet.alpha_),
            "l1_ratio": float(enet.l1_ratio_),
            "n_nonzero": enet_nonzero,
        })
        accum["elasticnet"]["all_coefs"].append(enet.coef_.copy())

        fold_time = time.time() - fold_start
        log.info(f"    Ridge:  RMSE {ridge_rmse:.3f}, MAE {ridge_mae:.3f}, "
                 f"alpha={ridge.alpha_:.1f}")
        log.info(f"    Lasso:  RMSE {lasso_rmse:.3f}, MAE {lasso_mae:.3f}, "
                 f"alpha={lasso.alpha_:.4f}, {lasso_nonzero}/{len(feature_names)} features")
        log.info(f"    ENet:   RMSE {enet_rmse:.3f}, MAE {enet_mae:.3f}, "
                 f"alpha={enet.alpha_:.4f}, L1={enet.l1_ratio_:.1f}, "
                 f"{enet_nonzero}/{len(feature_names)} features")
        log.info(f"    ({fold_time:.1f}s)")

    total_time = time.time() - total_start

    # Compute overall metrics and build OOF DataFrames
    results = {}
    for mt in model_types:
        valid = ~np.isnan(accum[mt]["oof_predictions"])
        preds = accum[mt]["oof_predictions"][valid]
        actuals = accum[mt]["oof_actuals"][valid]

        overall_rmse = root_mean_squared_error(actuals, preds)
        overall_mae = mean_absolute_error(actuals, preds)
        corr = np.corrcoef(preds, actuals)[0, 1]

        # Mean absolute coefficient across folds
        mean_coefs = np.mean(np.abs(np.array(accum[mt]["all_coefs"])), axis=0)
        coef_ranking = sorted(zip(feature_names, mean_coefs),
                              key=lambda x: -x[1])

        # For Lasso/ElasticNet: count how often each feature is nonzero
        nonzero_counts = {}
        for coefs in accum[mt]["all_coefs"]:
            for i, fname in enumerate(feature_names):
                if abs(coefs[i]) > 1e-6:
                    nonzero_counts[fname] = nonzero_counts.get(fname, 0) + 1

        oof_df = pd.DataFrame({
            "game_pk": df["game_pk"].values[valid],
            "date": df["date"].values[valid],
            "home_team": df["home_team"].values[valid],
            "away_team": df["away_team"].values[valid],
            "fold": accum[mt]["oof_folds"][valid],
            "predicted": preds,
            "actual": actuals,
            "season": df["season"].values[valid],
        })

        results[mt] = {
            "overall_rmse": float(overall_rmse),
            "overall_mae": float(overall_mae),
            "corr_pred_actual": float(corr),
            "n_oof_samples": int(valid.sum()),
            "fold_results": accum[mt]["fold_results"],
            "coef_ranking": coef_ranking,
            "nonzero_counts": nonzero_counts,
            "oof_df": oof_df,
        }

        log.info(f"\n  {mt.upper()} overall: RMSE {overall_rmse:.3f}, "
                 f"MAE {overall_mae:.3f}, corr {corr:.4f}")

    log.info(f"\n  Total time: {total_time:.0f}s")
    return results, feature_names


def build_report(results, feature_names, model_name, xgb_rmse=None, suffix=""):
    """Build human-readable comparison report."""
    lines = []
    lines.append("RIDGE/LASSO WALK-FORWARD EXPERIMENT")
    lines.append("=" * 80)
    lines.append(f"Generated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')}")
    lines.append(f"Model: {model_name}")
    lines.append(f"Mode: {'no-market (baseball-only)' if 'nomarket' in suffix else 'standard'}")
    lines.append(f"Candidate features: {len(feature_names)}")
    lines.append(f"Walk-forward folds: {len(results['ridge']['fold_results'])}")
    lines.append(f"OOF samples: {results['ridge']['n_oof_samples']:,}")
    lines.append("")

    # ── Overall comparison ──
    lines.append("=" * 80)
    lines.append("OVERALL COMPARISON")
    lines.append("=" * 80)
    lines.append(f"  {'Model':<15} | {'RMSE':>7} | {'MAE':>7} | {'Corr':>7} | {'vs XGBoost':>10}")
    lines.append("  " + "-" * 55)

    if xgb_rmse is not None:
        lines.append(f"  {'XGBoost':.<15} | {xgb_rmse:>7.3f} | {'':>7} | {'':>7} | {'baseline':>10}")

    for mt in ["ridge", "lasso", "elasticnet"]:
        r = results[mt]
        xgb_diff = ""
        if xgb_rmse is not None:
            diff = r["overall_rmse"] - xgb_rmse
            xgb_diff = f"{diff:>+.3f}"
        lines.append(
            f"  {mt.capitalize():.<15} | {r['overall_rmse']:>7.3f} | "
            f"{r['overall_mae']:>7.3f} | {r['corr_pred_actual']:>7.4f} | {xgb_diff:>10}"
        )
    lines.append("")

    # ── Per-fold comparison ──
    lines.append("=" * 80)
    lines.append("PER-FOLD RMSE COMPARISON")
    lines.append("=" * 80)
    lines.append(f"  {'Season':>6} | {'Ridge':>7} | {'Lasso':>7} | {'ENet':>7} | "
                 f"{'Ridge α':>10} | {'Lasso α':>10} | {'Lasso NZ':>8}")
    lines.append("  " + "-" * 70)

    for i, fr_r in enumerate(results["ridge"]["fold_results"]):
        fr_l = results["lasso"]["fold_results"][i]
        fr_e = results["elasticnet"]["fold_results"][i]
        lines.append(
            f"  {fr_r['test_season']:>6} | {fr_r['rmse']:>7.3f} | "
            f"{fr_l['rmse']:>7.3f} | {fr_e['rmse']:>7.3f} | "
            f"{fr_r['alpha']:>10.1f} | {fr_l['alpha']:>10.4f} | "
            f"{fr_l['n_nonzero']:>4}/{len(feature_names)}"
        )
    lines.append("")

    # ── Lasso feature selection (the key result) ──
    lines.append("=" * 80)
    lines.append("LASSO FEATURE SELECTION (features with nonzero coefficient)")
    lines.append("=" * 80)
    lines.append(f"  {'Feature':<40} | {'Folds':>5} | {'Mean |coef|':>11} | {'Direction':>9}")
    lines.append("  " + "-" * 72)

    n_folds = len(results["lasso"]["fold_results"])
    lasso_coefs = results["lasso"]["coef_ranking"]
    nonzero = results["lasso"]["nonzero_counts"]

    sorted_feats = sorted(
        [(fname, nonzero.get(fname, 0), mean_abs)
         for fname, mean_abs in lasso_coefs if nonzero.get(fname, 0) > 0],
        key=lambda x: (-x[1], -x[2])
    )

    for fname, folds_nz, mean_abs in sorted_feats:
        bar = "#" * folds_nz + "." * (n_folds - folds_nz)
        lines.append(
            f"  {fname:<40} | {folds_nz:>3}/{n_folds} | {mean_abs:>11.4f} | {bar}")

    total_nz = len(sorted_feats)
    lines.append(f"\n  Total features selected by Lasso: {total_nz}/{len(feature_names)}")
    lines.append(f"  Features in all {n_folds} folds: "
                 f"{sum(1 for _, nz, _ in sorted_feats if nz == n_folds)}")
    lines.append(f"  Features in >= {n_folds//2} folds: "
                 f"{sum(1 for _, nz, _ in sorted_feats if nz >= n_folds//2)}")
    lines.append("")

    # ── Ridge top coefficients ──
    lines.append("=" * 80)
    lines.append("RIDGE TOP 20 FEATURES (by mean |coefficient| across folds)")
    lines.append("=" * 80)
    lines.append(f"  {'Feature':<40} | {'Mean |coef|':>11}")
    lines.append("  " + "-" * 55)

    for fname, mean_abs in results["ridge"]["coef_ranking"][:20]:
        lines.append(f"  {fname:<40} | {mean_abs:>11.4f}")
    lines.append("")

    # ── ElasticNet summary ──
    lines.append("=" * 80)
    lines.append("ELASTICNET SUMMARY")
    lines.append("=" * 80)
    enet_nz = results["elasticnet"]["nonzero_counts"]
    enet_sorted = sorted(
        [(fname, enet_nz.get(fname, 0), mean_abs)
         for fname, mean_abs in results["elasticnet"]["coef_ranking"]
         if enet_nz.get(fname, 0) > 0],
        key=lambda x: (-x[1], -x[2])
    )
    lines.append(f"  L1 ratios chosen: {[fr['l1_ratio'] for fr in results['elasticnet']['fold_results']]}")
    lines.append(f"  Features selected: {len(enet_sorted)}/{len(feature_names)}")
    lines.append(f"  {'Feature':<40} | {'Folds':>5} | {'Mean |coef|':>11}")
    lines.append("  " + "-" * 62)
    for fname, folds_nz, mean_abs in enet_sorted[:20]:
        lines.append(f"  {fname:<40} | {folds_nz:>3}/{n_folds} | {mean_abs:>11.4f}")
    lines.append("")

    # ── Key findings ──
    lines.append("=" * 80)
    lines.append("KEY FINDINGS")
    lines.append("=" * 80)

    # Compare RMSE
    best_linear = min(results, key=lambda mt: results[mt]["overall_rmse"])
    best_rmse = results[best_linear]["overall_rmse"]

    if xgb_rmse is not None:
        if best_rmse < xgb_rmse:
            lines.append(f"  {best_linear.capitalize()} BEATS XGBoost: "
                         f"RMSE {best_rmse:.3f} vs {xgb_rmse:.3f} "
                         f"({xgb_rmse - best_rmse:.3f} improvement)")
        else:
            lines.append(f"  XGBoost wins: RMSE {xgb_rmse:.3f} vs best linear "
                         f"{best_rmse:.3f} ({best_linear})")

    # Lasso selection vs Boruta
    lasso_stable = [f for f, nz, _ in sorted_feats if nz >= n_folds // 2]
    lines.append(f"  Lasso stable features (>= {n_folds//2}/{n_folds} folds): "
                 f"{len(lasso_stable)}")
    if lasso_stable:
        lines.append(f"    {lasso_stable}")

    lines.append("")
    lines.append("NOTES")
    lines.append("=" * 80)
    lines.append("- All predictions are truly out-of-sample (walk-forward)")
    lines.append("- Features are StandardScaled per fold (fit on train only)")
    lines.append("- Alpha selected by internal 5-fold CV within each training set")
    lines.append("- Sample weighting: exponential decay (half_life=3yr)")
    lines.append("- No Boruta — Lasso does its own L1 feature selection")
    lines.append("- ElasticNet combines L1 (sparsity) and L2 (grouping) penalties")
    lines.append("- Coefficients are on standardized features — compare magnitudes, not raw values")

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Ridge/Lasso walk-forward experiment")
    parser.add_argument("--no-market", action="store_true",
                        help="Exclude market-derived features")
    args = parser.parse_args()

    no_market = args.no_market
    suffix = "_nomarket" if no_market else ""
    mode_label = "NO-MARKET (baseball-only)" if no_market else "standard"

    log.info("=" * 60)
    log.info(f"RIDGE/LASSO EXPERIMENT [{mode_label}]")
    log.info("=" * 60)

    # Load data (same as 06_train_mlb_model.py)
    df = load_mlb_training_data()
    df = engineer_market_features(df)
    df = engineer_f5_market_features(df)

    # Candidate features
    available = [f for f in MLB_CANDIDATE_FEATURES if f in df.columns]
    if no_market:
        available = [f for f in available if f not in MARKET_FEATURE_NAMES]
    log.info(f"Candidate features: {len(available)}")

    # ── MARGIN MODEL ──
    margin_df = df[df["actual_margin"].notna()].copy()
    log.info(f"\nMargin model: {len(margin_df):,} games")

    margin_results, margin_features = walk_forward_linear(
        margin_df, available, "actual_margin", "margin"
    )

    # ── TOTAL MODEL ──
    total_df = df[(df["actual_total"].notna()) &
                  (df["is_7_inning_dh"] != True)].copy()
    log.info(f"\nTotal model: {len(total_df):,} games")

    total_results, total_features = walk_forward_linear(
        total_df, available, "actual_total", "total"
    )

    # ── Load XGBoost baselines for comparison ──
    xgb_margin_rmse = None
    xgb_total_rmse = None
    metrics_path = MODELS_DIR / f"mlb_training_metrics{suffix}.json"
    if metrics_path.exists():
        with open(metrics_path) as f:
            metrics = json.load(f)
        xgb_margin_rmse = metrics.get("margin", {}).get("walkforward_overall_rmse")
        xgb_total_rmse = metrics.get("total", {}).get("walkforward_overall_rmse")
        log.info(f"XGBoost baselines: margin RMSE={xgb_margin_rmse}, total RMSE={xgb_total_rmse}")

    # ── Save OOF predictions (Lasso — best sparse model for backtest) ──
    for model_type in ["ridge", "lasso", "elasticnet"]:
        for name, results in [("margin", margin_results), ("total", total_results)]:
            oof_path = MODELS_DIR / f"mlb_oof_{name}_{model_type}{suffix}_predictions.csv"
            results[model_type]["oof_df"].to_csv(oof_path, index=False)
            log.info(f"Saved {name} {model_type} OOF -> {oof_path}")

    # ── Build reports ──
    margin_report = build_report(
        margin_results, margin_features, "MARGIN", xgb_margin_rmse, suffix
    )
    total_report = build_report(
        total_results, total_features, "TOTAL", xgb_total_rmse, suffix
    )

    full_report = margin_report + "\n\n" + "=" * 80 + "\n" * 3 + total_report

    report_path = MODELS_DIR / f"mlb_ridge_lasso_report{suffix}.txt"
    with open(report_path, "w") as f:
        f.write(full_report)
    log.info(f"\nSaved report -> {report_path}")

    # ── Print summary ──
    print(f"\n{'='*70}")
    print(f"RIDGE/LASSO EXPERIMENT SUMMARY [{mode_label}]")
    print(f"{'='*70}")

    for name, results, xgb_rmse in [("MARGIN", margin_results, xgb_margin_rmse),
                                      ("TOTAL", total_results, xgb_total_rmse)]:
        print(f"\n  {name}:")
        if xgb_rmse:
            print(f"    XGBoost:    RMSE {xgb_rmse:.3f}")
        for mt in ["ridge", "lasso", "elasticnet"]:
            r = results[mt]
            nz = sum(1 for v in r["nonzero_counts"].values() if v > 0) if mt != "ridge" else "all"
            print(f"    {mt.capitalize():<12}  RMSE {r['overall_rmse']:.3f}, "
                  f"corr {r['corr_pred_actual']:.4f}, features: {nz}")

    # Lasso feature summary
    print(f"\n  LASSO selected features (margin, >= {len(margin_results['lasso']['fold_results'])//2} folds):")
    n_folds = len(margin_results["lasso"]["fold_results"])
    nz = margin_results["lasso"]["nonzero_counts"]
    stable = sorted([(f, c) for f, c in nz.items() if c >= n_folds // 2],
                    key=lambda x: -x[1])
    for fname, count in stable:
        print(f"    {fname}: {count}/{n_folds} folds")

    print(f"\n  Full report: {report_path}")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()
