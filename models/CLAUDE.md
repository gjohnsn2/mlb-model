# MLB Model — Models Directory

## Structure
```
models/
├── CLAUDE.md
├── selected_features.json    # Boruta output (margin_features + total_features)
├── training_metrics.json     # Walk-forward RMSE/MAE per model
├── boruta_report.txt         # Human-readable feature selection report
├── oof_margin_predictions.csv  # Walk-forward OOF predictions (margin)
├── oof_total_predictions.csv   # Walk-forward OOF predictions (total)
├── walkforward_report.txt    # Fold-by-fold results
├── configs/
│   └── tuned_params.json     # Optuna-tuned hyperparameters
└── trained/
    ├── margin_model.pkl      # Production margin model + calibrator
    └── total_model.pkl       # Production total model + calibrator
```

## Model Architecture
- **Margin model**: XGBoost regressor predicting home team run margin
  - Target: actual_margin (home_score - away_score)
  - Used for: moneyline bets (margin -> win probability via normal CDF)
  - Also used for: run line bets (margin vs. 1.5 run line)
- **Total model**: XGBoost regressor predicting combined runs
  - Target: actual_total (home_score + away_score)
  - Used for: over/under bets

## Walk-Forward Validation
- Expanding window, per-fold Boruta feature selection
- Test seasons: TBD (targeting 2023-2026 initially)
- Per-fold: Boruta on train only -> train XGBoost -> predict test
- Production model: trained on ALL data with consensus features

## Calibration
- Tail-aware isotonic calibration
- Core 5th-95th percentile: isotonic regression
- Tails: linear regression for extremes
- Fitted on OOF walk-forward predictions (truly out-of-sample)

## Model Bundle Format (.pkl)
Each pickle contains:
```python
{
    "model": xgb.XGBRegressor,       # Trained model
    "features": ["feat1", "feat2"],   # Feature names in order
    "calibrator": {                    # Tail-aware calibrator dict
        "iso": IsotonicRegression,
        "lo_thresh": float,
        "hi_thresh": float,
        "lo_slope": float, "lo_intercept": float,
        "hi_slope": float, "hi_intercept": float,
    },
    "metrics": {
        "n_samples": int,
        "cv_rmse_mean": float,
        "cv_mae_mean": float,
        "feature_counts": dict,
    },
}
```
