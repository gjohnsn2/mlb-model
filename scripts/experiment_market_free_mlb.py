"""
Experiment: MLB model WITHOUT market features
=============================================
Strips market_implied_prob, market_logit, consensus_total, num_books
and retrains with only SP + batting features to measure the model's
standalone predictive power.

If the SP/batting features have real signal, this model should:
  1. Have corr(pred, actual) > 0 (predicts margin direction)
  2. Show some betting edge when combined with market odds at backtest time

If the signal is mostly in market features, this model will be ~useless.

Run: python3 scripts/experiment_market_free_mlb.py
"""

import sys
import numpy as np
import pandas as pd
from scipy.stats import norm
from pathlib import Path
from sklearn.metrics import root_mean_squared_error, mean_absolute_error

# Add parent dir to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import (
    MODELS_DIR, HISTORICAL_DIR,
    MLB_XGBOOST_PARAMS, MLB_TEST_SEASONS, MLB_SAMPLE_WEIGHT_HALF_LIFE,
    MLB_CANDIDATE_FEATURES, get_logger
)

log = get_logger("experiment_market_free")

MARKET_FEATURES = {"market_implied_prob", "market_logit", "consensus_total", "num_books"}
UNIT_SIZE = 100


def american_to_implied_prob(odds):
    if pd.isna(odds) or odds == 0:
        return np.nan
    if odds < 0:
        return abs(odds) / (abs(odds) + 100)
    return 100 / (odds + 100)


def american_to_decimal(odds):
    if pd.isna(odds) or odds == 0:
        return np.nan
    if odds < 0:
        return 1 + 100 / abs(odds)
    return 1 + odds / 100


def load_data():
    path = HISTORICAL_DIR / "training_data_mlb_v1.csv"
    df = pd.read_csv(path)
    df["date"] = pd.to_datetime(df["date"])
    df["season"] = df["date"].dt.year
    log.info(f"Loaded {len(df)} games, seasons {df['season'].min()}-{df['season'].max()}")
    return df


def compute_sample_weights(seasons, test_season, half_life):
    max_season = test_season - 1
    age = max_season - seasons
    return np.exp(-np.log(2) * age / half_life)


def walk_forward_no_market(df):
    """Walk-forward validation without market features."""
    from xgboost import XGBRegressor

    # Non-market candidate features
    candidates = [f for f in MLB_CANDIDATE_FEATURES if f not in MARKET_FEATURES]
    log.info(f"Non-market candidate features: {len(candidates)}")
    log.info(f"  {candidates}")

    all_oof = []
    fold_results = []

    for fold_idx, test_season in enumerate(MLB_TEST_SEASONS):
        train_mask = df["season"] < test_season
        test_mask = df["season"] == test_season
        train_df = df[train_mask].copy()
        test_df = df[test_mask].copy()

        log.info(f"\nFold {fold_idx+1}: train {len(train_df)}, test {len(test_df)} "
                 f"(test season {test_season})")

        # Use all non-market candidates (skip Boruta for simplicity)
        available = [f for f in candidates if f in train_df.columns]
        # Drop features with zero variance in train
        for f in available[:]:
            if train_df[f].std() == 0 or train_df[f].isna().all():
                available.remove(f)

        X_train = train_df[available].fillna(0)
        y_train = train_df["actual_margin"]
        X_test = test_df[available].fillna(0)
        y_test = test_df["actual_margin"]

        weights = compute_sample_weights(
            train_df["season"].values, test_season, MLB_SAMPLE_WEIGHT_HALF_LIFE
        )

        model = XGBRegressor(**MLB_XGBOOST_PARAMS)
        model.fit(X_train, y_train, sample_weight=weights)
        preds = model.predict(X_test)

        rmse = root_mean_squared_error(y_test, preds)
        mae = mean_absolute_error(y_test, preds)
        corr = np.corrcoef(preds, y_test)[0, 1]

        log.info(f"  RMSE={rmse:.3f}, MAE={mae:.3f}, corr={corr:.3f}")

        # Feature importance
        importances = dict(zip(available, model.feature_importances_))
        top5 = sorted(importances.items(), key=lambda x: -x[1])[:5]
        log.info(f"  Top 5: {[(f, round(v, 3)) for f, v in top5]}")

        fold_results.append({
            "fold": fold_idx + 1, "test_season": test_season,
            "rmse": rmse, "mae": mae, "corr": corr,
            "n_features": len(available),
        })

        oof = test_df[["game_pk", "date", "season", "actual_margin",
                        "consensus_h2h_home", "consensus_h2h_away"]].copy()
        oof["predicted"] = preds
        all_oof.append(oof)

    oof_df = pd.concat(all_oof, ignore_index=True)
    return oof_df, fold_results


def backtest_no_market(oof_df):
    """Backtest the market-free model using calibrated margin-space edges."""
    # Need odds for backtesting
    h2h_mask = oof_df["consensus_h2h_home"].notna() & oof_df["consensus_h2h_away"].notna()
    has_odds = oof_df[h2h_mask].copy()

    if len(has_odds) == 0:
        log.warning("No games with H2H odds for backtesting")
        return

    # OOF RMSE
    margin_rmse = root_mean_squared_error(has_odds["actual_margin"], has_odds["predicted"])

    # De-vig market probabilities
    raw_home = has_odds["consensus_h2h_home"].apply(american_to_implied_prob)
    raw_away = has_odds["consensus_h2h_away"].apply(american_to_implied_prob)
    total_vig = raw_home + raw_away
    valid = total_vig.notna() & (total_vig > 0)
    has_odds.loc[valid, "market_home_prob"] = raw_home[valid] / total_vig[valid]

    # Market implied margin
    has_odds["market_implied_margin"] = margin_rmse * norm.ppf(
        has_odds["market_home_prob"].clip(0.001, 0.999)
    )

    # Calibrate: rescale model to market distribution
    model_mean = has_odds["predicted"].mean()
    model_std = has_odds["predicted"].std()
    market_mean = has_odds["market_implied_margin"].mean()
    market_std = has_odds["market_implied_margin"].std()

    has_odds["calibrated_pred"] = (
        (has_odds["predicted"] - model_mean) / model_std * market_std + market_mean
    )
    has_odds["margin_edge"] = has_odds["calibrated_pred"] - has_odds["market_implied_margin"]

    # Diagnostics
    r_model = has_odds["predicted"].corr(has_odds["actual_margin"])
    r_market = has_odds["market_implied_margin"].corr(has_odds["actual_margin"])
    resid = has_odds["actual_margin"] - has_odds["market_implied_margin"]
    r_resid = has_odds["margin_edge"].corr(resid)

    log.info(f"\n{'='*60}")
    log.info(f"MARKET-FREE MODEL DIAGNOSTICS")
    log.info(f"{'='*60}")
    log.info(f"  Games with odds:      {len(has_odds)}")
    log.info(f"  OOF RMSE:             {margin_rmse:.3f}")
    log.info(f"  Model pred std:       {has_odds['predicted'].std():.3f}")
    log.info(f"  Market margin std:    {market_std:.3f}")
    log.info(f"  corr(model, actual):  {r_model:.3f}")
    log.info(f"  corr(market, actual): {r_market:.3f}")
    log.info(f"  corr(edge, market_residual): {r_resid:.3f}")

    # Simulate ML bets at thresholds
    thresholds = [0.0, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0]
    log.info(f"\n{'='*60}")
    log.info(f"ML BACKTEST (market-free model, calibrated edges)")
    log.info(f"{'='*60}")
    log.info(f"{'Thresh':>7} | {'Bets':>5} | {'W-L':>10} | {'Win%':>6} | "
             f"{'Profit':>9} | {'ROI':>7} | {'Dog%':>5} | {'p-val':>7}")
    log.info("-" * 70)

    for thresh in thresholds:
        bets = []
        for _, row in has_odds.iterrows():
            edge = row["margin_edge"]
            if pd.isna(edge) or abs(edge) < thresh:
                continue

            if edge > 0:
                odds_used = row["consensus_h2h_home"]
                won = row["actual_margin"] > 0
                is_dog = row["market_home_prob"] < 0.5
            else:
                odds_used = row["consensus_h2h_away"]
                won = row["actual_margin"] < 0
                is_dog = row["market_home_prob"] >= 0.5

            dec = american_to_decimal(odds_used)
            if pd.isna(dec):
                continue

            push = (row["actual_margin"] == 0)
            if push:
                profit = 0
            elif won:
                profit = round(UNIT_SIZE * (dec - 1))
            else:
                profit = -UNIT_SIZE

            bets.append({"won": won, "push": push, "profit": profit, "is_dog": is_dog,
                          "season": row["season"]})

        if not bets:
            continue

        bets_df = pd.DataFrame(bets)
        non_push = bets_df[~bets_df["push"]]
        wins = int(non_push["won"].sum())
        losses = len(non_push) - wins
        n = wins + losses
        win_pct = wins / n * 100 if n > 0 else 0
        profit = bets_df["profit"].sum()
        roi = profit / (n * UNIT_SIZE) * 100 if n > 0 else 0
        dog_pct = bets_df["is_dog"].mean() * 100

        # Profit z-test
        profits = bets_df["profit"].values
        p_mean = profits.mean()
        p_std = profits.std(ddof=1)
        p_val = 1 - norm.cdf(p_mean * np.sqrt(len(profits)) / p_std) if p_std > 0 else 1.0

        log.info(f"  >= {thresh:>4.2f} | {n:>5} | {wins:>4}-{losses:<4} | "
                 f"{win_pct:>5.1f}% | ${profit:>+8.0f} | {roi:>+6.1f}% | "
                 f"{dog_pct:>4.0f}% | {p_val:>7.4f}")

    # Per-season at production threshold (0.5)
    log.info(f"\n{'='*60}")
    log.info(f"BY SEASON (>= 0.5 runs, market-free model)")
    log.info(f"{'='*60}")
    for season in sorted(has_odds["season"].unique()):
        sdf = has_odds[has_odds["season"] == season]
        bets = []
        for _, row in sdf.iterrows():
            edge = row["margin_edge"]
            if pd.isna(edge) or abs(edge) < 0.5:
                continue
            if edge > 0:
                odds_used = row["consensus_h2h_home"]
                won = row["actual_margin"] > 0
                is_dog = row["market_home_prob"] < 0.5
            else:
                odds_used = row["consensus_h2h_away"]
                won = row["actual_margin"] < 0
                is_dog = row["market_home_prob"] >= 0.5
            dec = american_to_decimal(odds_used)
            if pd.isna(dec):
                continue
            push = (row["actual_margin"] == 0)
            if push:
                profit = 0
            elif won:
                profit = round(UNIT_SIZE * (dec - 1))
            else:
                profit = -UNIT_SIZE
            bets.append({"won": won, "push": push, "profit": profit, "is_dog": is_dog})

        if not bets:
            log.info(f"  {int(season)}: No bets")
            continue

        bets_df = pd.DataFrame(bets)
        non_push = bets_df[~bets_df["push"]]
        wins = int(non_push["won"].sum())
        losses = len(non_push) - wins
        n = wins + losses
        win_pct = wins / n * 100 if n > 0 else 0
        profit = bets_df["profit"].sum()
        roi = profit / (n * UNIT_SIZE) * 100 if n > 0 else 0
        dog_pct = bets_df["is_dog"].mean() * 100
        log.info(f"  {int(season)}: {wins}-{losses} ({win_pct:.1f}%), "
                 f"ROI {roi:+.1f}%, dog%={dog_pct:.0f}%")


def main():
    log.info("=" * 60)
    log.info("EXPERIMENT: Market-free MLB model")
    log.info("=" * 60)

    df = load_data()

    # Walk-forward without market features
    oof_df, fold_results = walk_forward_no_market(df)

    # Summary
    log.info(f"\n{'='*60}")
    log.info("WALK-FORWARD SUMMARY (no market features)")
    log.info(f"{'='*60}")
    rmses = [f["rmse"] for f in fold_results]
    corrs = [f["corr"] for f in fold_results]
    overall_rmse = root_mean_squared_error(oof_df["actual_margin"], oof_df["predicted"])
    overall_corr = np.corrcoef(oof_df["predicted"], oof_df["actual_margin"])[0, 1]
    log.info(f"  Overall RMSE: {overall_rmse:.3f}")
    log.info(f"  Overall corr: {overall_corr:.3f}")
    for fr in fold_results:
        log.info(f"  Fold {fr['fold']} ({fr['test_season']}): "
                 f"RMSE={fr['rmse']:.3f}, corr={fr['corr']:.3f}")

    # Compare to full model
    full_oof_path = MODELS_DIR / "mlb_oof_margin_predictions.csv"
    if full_oof_path.exists():
        full_oof = pd.read_csv(full_oof_path)
        full_rmse = root_mean_squared_error(full_oof["actual"], full_oof["predicted"])
        full_corr = np.corrcoef(full_oof["predicted"], full_oof["actual"])[0, 1]
        log.info(f"\n  Full model (with market):  RMSE={full_rmse:.3f}, corr={full_corr:.3f}")
        log.info(f"  Market-free model:         RMSE={overall_rmse:.3f}, corr={overall_corr:.3f}")
        log.info(f"  RMSE increase:             {overall_rmse - full_rmse:+.3f}")
        log.info(f"  Corr decrease:             {overall_corr - full_corr:+.3f}")

    # Backtest
    backtest_no_market(oof_df)


if __name__ == "__main__":
    main()
