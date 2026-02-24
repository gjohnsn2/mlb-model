"""
10 — MLB Profitability Backtest
==================================
Merges walk-forward OOF predictions with historical closing odds from
the training data. Uses calibrated margin-space edges to avoid the
systematic underdog bias inherent in margin→probability conversion.

Edge methodology:
  1. Convert market H2H odds → de-vigged probability → implied margin
     via margin = RMSE * Φ⁻¹(prob)
  2. Rescale model predictions to match market's mean/std (calibrate)
  3. Edge = calibrated_prediction - market_implied_margin (in runs)
  4. Bet the side the model favors when |edge| >= threshold

This eliminates the structural dog bias where compressed model predictions
mechanically create fake edges on underdogs.

Also backtests totals at -110 juice, and reports blind-dog null strategy
as a benchmark.

Inputs:
  - models/mlb_oof_margin_predictions.csv  (from 06_train_mlb_model.py)
  - models/mlb_oof_total_predictions.csv   (from 06_train_mlb_model.py)
  - data/historical/training_data_mlb_v1.csv (for actual odds)

Outputs:
  - models/mlb_backtest_report.txt
  - models/mlb_backtest_results.csv

Run: python3 10_backtest_mlb.py
"""

import sys
import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import norm
from pathlib import Path
from config import (
    MODELS_DIR, HISTORICAL_DIR,
    MLB_TOTAL_EDGE_THRESHOLD, MLB_MARGIN_MODEL_RMSE, MLB_ML_UNIT_TIERS,
    get_logger
)

log = get_logger("10_backtest_mlb")

# Betting constants for totals (flat -110)
UNIT_SIZE = 100
WIN_PAYOUT = 100
LOSS_COST = 110
BREAKEVEN_PCT = LOSS_COST / (WIN_PAYOUT + LOSS_COST)  # ~52.38%

# Edge thresholds in runs (margin space)
ML_MARGIN_THRESHOLDS = [0.0, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0]
ML_PRODUCTION_THRESHOLD = 0.5  # runs
TOTAL_THRESHOLDS = [0.5, 1.0, 1.5, 2.0, 3.0]

# Unit tiers by margin edge (runs)
ML_MARGIN_UNIT_TIERS = [
    (2.0, 3.0, "3u"),   # edge >= 2.0 runs
    (1.5, 2.0, "2u"),   # edge >= 1.5 runs
    (1.0, 1.5, "1.5u"), # edge >= 1.0 runs
    (0.5, 1.0, "1u"),   # edge >= 0.5 runs
]


def american_to_implied_prob(odds):
    """Convert American odds to implied probability (no vig removal)."""
    if pd.isna(odds) or odds == 0:
        return np.nan
    if odds < 0:
        return abs(odds) / (abs(odds) + 100)
    else:
        return 100 / (odds + 100)


def american_to_decimal(odds):
    """Convert American odds to decimal odds (payout per $1 risked, including stake)."""
    if pd.isna(odds) or odds == 0:
        return np.nan
    if odds < 0:
        return 1 + 100 / abs(odds)
    else:
        return 1 + odds / 100


def load_oof_predictions():
    """Load walk-forward OOF predictions for margin and total."""
    margin_path = MODELS_DIR / "mlb_oof_margin_predictions.csv"
    total_path = MODELS_DIR / "mlb_oof_total_predictions.csv"

    margin_oof = None
    total_oof = None

    if margin_path.exists():
        margin_oof = pd.read_csv(margin_path)
        margin_oof["date"] = margin_oof["date"].astype(str).str[:10]
        log.info(f"Loaded margin OOF: {len(margin_oof)} games, "
                 f"seasons {margin_oof['season'].min()}-{margin_oof['season'].max()}")
    else:
        log.warning(f"Margin OOF not found: {margin_path}")

    if total_path.exists():
        total_oof = pd.read_csv(total_path)
        total_oof["date"] = total_oof["date"].astype(str).str[:10]
        log.info(f"Loaded total OOF: {len(total_oof)} games")
    else:
        log.warning(f"Total OOF not found: {total_path}")

    return margin_oof, total_oof


def load_training_data_for_odds():
    """Load training data to get actual H2H odds and totals."""
    path = HISTORICAL_DIR / "training_data_mlb_v2.csv"
    if not path.exists():
        log.error(f"Training data not found: {path}")
        sys.exit(1)

    df = pd.read_csv(path)
    log.info(f"Loaded training data for odds: {len(df)} games")
    return df


def match_with_odds(oof_df, training_df):
    """Join OOF predictions with odds data by game_pk."""
    odds_cols = ["game_pk", "consensus_h2h_home", "consensus_h2h_away",
                 "consensus_total", "consensus_spread", "num_books",
                 "actual_margin", "actual_total",
                 "home_team", "away_team", "is_7_inning_dh"]

    available_cols = [c for c in odds_cols if c in training_df.columns]
    odds_data = training_df[available_cols].copy()

    merged = oof_df.merge(odds_data, on="game_pk", how="left",
                          suffixes=("", "_train"))

    # Filter corrupt H2H lines: |ML| < 100 produces absurd payouts (e.g. -1 → 101x)
    # These are data artifacts from Odds API median calculation (~0.9% of games)
    MIN_ML = 100
    for col in ["consensus_h2h_home", "consensus_h2h_away"]:
        if col in merged.columns:
            corrupt = merged[col].notna() & (merged[col].abs() < MIN_ML)
            if corrupt.any():
                log.info(f"  Filtering {corrupt.sum()} corrupt {col} values (|ML| < {MIN_ML})")
                merged.loc[corrupt, "consensus_h2h_home"] = np.nan
                merged.loc[corrupt, "consensus_h2h_away"] = np.nan

    has_h2h = merged["consensus_h2h_home"].notna().sum()
    has_total = merged["consensus_total"].notna().sum() if "consensus_total" in merged.columns else 0
    log.info(f"Matched odds: {has_h2h}/{len(merged)} with H2H, "
             f"{has_total}/{len(merged)} with totals")

    return merged


def calibrate_predictions(df, margin_rmse):
    """
    Calibrate model predictions and compute margin-space edges.

    Problem: Model predictions are compressed toward 0 (std ~1.86) compared
    to market implied margins (std ~2.37). This creates systematic underdog
    bias when comparing probabilities.

    Fix: Rescale model predictions to match the market's mean and std,
    then compute edges in margin space. This eliminates the compression
    artifact while preserving the model's ranking of games.
    """
    h2h_mask = df["consensus_h2h_home"].notna() & df["consensus_h2h_away"].notna()
    has_odds = df[h2h_mask].copy()

    # De-vig market probabilities
    raw_home = has_odds["consensus_h2h_home"].apply(american_to_implied_prob)
    raw_away = has_odds["consensus_h2h_away"].apply(american_to_implied_prob)
    total_vig = raw_home + raw_away
    valid = total_vig.notna() & (total_vig > 0)
    has_odds.loc[valid, "market_home_prob"] = raw_home[valid] / total_vig[valid]

    # Convert market prob to implied margin: margin = RMSE * Φ⁻¹(prob)
    has_odds["market_implied_margin"] = margin_rmse * norm.ppf(
        has_odds["market_home_prob"].clip(0.001, 0.999)
    )

    # Rescale model predictions to match market distribution
    model_mean = has_odds["predicted"].mean()
    model_std = has_odds["predicted"].std()
    market_mean = has_odds["market_implied_margin"].mean()
    market_std = has_odds["market_implied_margin"].std()

    has_odds["calibrated_pred"] = (
        (has_odds["predicted"] - model_mean) / model_std * market_std + market_mean
    )

    # Calibrated edge in runs
    has_odds["margin_edge"] = has_odds["calibrated_pred"] - has_odds["market_implied_margin"]

    log.info(f"Calibration: model std {model_std:.3f} → market std {market_std:.3f}")
    log.info(f"  Edge distribution: mean={has_odds['margin_edge'].mean():.3f}, "
             f"std={has_odds['margin_edge'].std():.3f}")
    home_pct = (has_odds["margin_edge"] > 0).mean() * 100
    log.info(f"  Side balance: {home_pct:.1f}% home / {100-home_pct:.1f}% away")

    return has_odds


def simulate_ml_bets(df, threshold):
    """
    Simulate moneyline betting using calibrated margin-space edges.

    Edge = calibrated_prediction - market_implied_margin (in runs).
    Bet the side the model favors when |edge| >= threshold.
    Profit computed from actual H2H American odds (variable payouts).
    """
    bets = []

    for _, row in df.iterrows():
        edge = row["margin_edge"]
        if pd.isna(edge) or abs(edge) < threshold:
            continue

        if edge > 0:
            # Model likes home more than market
            side = "HOME_ML"
            odds_used = row["consensus_h2h_home"]
            won = row["actual"] > 0
        else:
            # Model likes away more than market
            side = "AWAY_ML"
            odds_used = row["consensus_h2h_away"]
            won = row["actual"] < 0

        dec = american_to_decimal(odds_used)
        if pd.isna(dec):
            continue

        # Is this a dog or fav bet?
        is_dog = (side == "HOME_ML" and row["market_home_prob"] < 0.5) or \
                 (side == "AWAY_ML" and row["market_home_prob"] >= 0.5)

        # Unit tiers by margin edge
        ml_units = 1.0
        for tier_min, tier_units, _ in ML_MARGIN_UNIT_TIERS:
            if abs(edge) >= tier_min:
                ml_units = tier_units
                break

        bet_risk = UNIT_SIZE * ml_units

        # Push on exact tie
        push = (row["actual"] == 0)
        if push:
            profit = 0
        elif won:
            profit = round(bet_risk * (dec - 1))
        else:
            profit = -bet_risk

        bets.append({
            "date": row["date"],
            "game_pk": row["game_pk"],
            "home_team": row.get("home_team", row.get("home_team_train", "")),
            "away_team": row.get("away_team", row.get("away_team_train", "")),
            "season": row.get("season"),
            "calibrated_pred": round(row["calibrated_pred"], 3),
            "market_implied_margin": round(row["market_implied_margin"], 3),
            "margin_edge": round(edge, 3),
            "side": side,
            "is_dog": is_dog,
            "odds_used": int(odds_used),
            "ml_units": ml_units,
            "actual_margin": row["actual"],
            "won": won,
            "push": push,
            "profit": profit,
            "bet_type": "ml",
        })

    return pd.DataFrame(bets)


def simulate_blind_dog(df):
    """
    Null strategy: always bet the underdog at actual odds.
    Used as a benchmark to measure model's incremental value.
    """
    bets = []

    for _, row in df.iterrows():
        if pd.isna(row.get("consensus_h2h_home")) or pd.isna(row.get("consensus_h2h_away")):
            continue

        market_home_prob = row.get("market_home_prob", 0.5)
        if pd.isna(market_home_prob):
            continue

        if market_home_prob < 0.5:
            # Home is underdog
            odds_used = row["consensus_h2h_home"]
            won = row["actual"] > 0
        else:
            # Away is underdog
            odds_used = row["consensus_h2h_away"]
            won = row["actual"] < 0

        dec = american_to_decimal(odds_used)
        if pd.isna(dec):
            continue

        push = (row["actual"] == 0)
        if push:
            profit = 0
        elif won:
            profit = round(UNIT_SIZE * (dec - 1))
        else:
            profit = -UNIT_SIZE

        bets.append({
            "season": row.get("season"),
            "won": won,
            "push": push,
            "profit": profit,
        })

    return pd.DataFrame(bets)


def simulate_total_bets(df, threshold):
    """
    Simulate over/under betting at -110 juice.

    Edge = model_total - consensus_total.
    Positive = Over, Negative = Under.
    """
    bets = []

    for _, row in df.iterrows():
        if pd.isna(row.get("consensus_total")) or pd.isna(row["predicted"]):
            continue

        model_total = row["predicted"]
        mkt_total = row["consensus_total"]
        actual_total = row["actual"]

        edge = model_total - mkt_total

        if abs(edge) < threshold:
            continue

        if edge > 0:
            side = "OVER"
            won = actual_total > mkt_total
            push = actual_total == mkt_total
        else:
            side = "UNDER"
            won = actual_total < mkt_total
            push = actual_total == mkt_total

        profit = WIN_PAYOUT if won else (-LOSS_COST if not push else 0)

        bets.append({
            "date": row["date"],
            "game_pk": row["game_pk"],
            "home_team": row.get("home_team", row.get("home_team_train", "")),
            "away_team": row.get("away_team", row.get("away_team_train", "")),
            "season": row.get("season"),
            "model_total": round(model_total, 2),
            "consensus_total": mkt_total,
            "edge": round(edge, 1),
            "side": side,
            "actual_total": actual_total,
            "won": won,
            "push": push,
            "profit": profit,
            "bet_type": "total",
        })

    return pd.DataFrame(bets)


def compute_ml_stats(bets_df):
    """Compute stats for ML bets (variable odds, tiered units).

    P-value tests whether observed wins exceed expected wins under the null
    that each bet wins with its de-vigged market implied probability.
    Under this null, expected profit = 0 (market is efficient).
    """
    if bets_df.empty:
        return {"n_bets": 0, "wins": 0, "losses": 0,
                "win_pct": 0, "profit": 0, "roi": 0, "p_value": 1.0,
                "dog_pct": 0}

    non_push = bets_df[~bets_df["push"]]
    wins = int(non_push["won"].sum())
    losses = len(non_push) - wins
    n_bets = wins + losses
    win_pct = wins / n_bets * 100 if n_bets > 0 else 0

    profit = bets_df["profit"].sum()
    if "ml_units" in bets_df.columns:
        total_risked = (bets_df["ml_units"] * UNIT_SIZE).sum()
    else:
        total_risked = n_bets * UNIT_SIZE
    roi = profit / total_risked * 100 if total_risked > 0 else 0

    # Dog percentage
    dog_pct = bets_df["is_dog"].mean() * 100 if "is_dog" in bets_df.columns else 0

    # P-value: test whether mean per-bet profit is significantly > 0.
    # This properly handles variable odds + variable unit sizing.
    # z = mean(profits) * sqrt(n) / std(profits), p = 1 - Φ(z)
    p_value = 1.0
    if n_bets > 0:
        profits = bets_df["profit"].values
        profit_mean = profits.mean()
        profit_std = profits.std(ddof=1)
        if profit_std > 0:
            z = profit_mean * np.sqrt(len(profits)) / profit_std
            p_value = 1 - norm.cdf(z)

    return {
        "n_bets": n_bets, "wins": wins, "losses": losses,
        "win_pct": win_pct, "profit": profit, "roi": roi, "p_value": p_value,
        "dog_pct": dog_pct,
    }


def compute_total_stats(bets_df):
    """Compute stats for total bets (flat -110)."""
    if bets_df.empty:
        return {"n_bets": 0, "wins": 0, "losses": 0, "pushes": 0,
                "win_pct": 0, "profit": 0, "roi": 0, "p_value": 1.0}

    non_push = bets_df[~bets_df["push"]]
    wins = int(non_push["won"].sum())
    losses = len(non_push) - wins
    pushes = int(bets_df["push"].sum())
    n_bets = wins + losses
    win_pct = wins / n_bets * 100 if n_bets > 0 else 0

    profit = bets_df["profit"].sum()
    total_risked = n_bets * LOSS_COST
    roi = profit / total_risked * 100 if total_risked > 0 else 0

    if n_bets > 0:
        p_value = stats.binomtest(wins, n_bets, BREAKEVEN_PCT, alternative="greater").pvalue
    else:
        p_value = 1.0

    return {
        "n_bets": n_bets, "wins": wins, "losses": losses, "pushes": pushes,
        "win_pct": win_pct, "profit": profit, "roi": roi, "p_value": p_value,
    }


def compute_null_stats(bets_df):
    """Compute stats for blind-dog null strategy."""
    if bets_df.empty:
        return {"n_bets": 0, "wins": 0, "losses": 0,
                "win_pct": 0, "profit": 0, "roi": 0}

    non_push = bets_df[~bets_df["push"]]
    wins = int(non_push["won"].sum())
    losses = len(non_push) - wins
    n_bets = wins + losses
    win_pct = wins / n_bets * 100 if n_bets > 0 else 0
    profit = bets_df["profit"].sum()
    total_risked = n_bets * UNIT_SIZE
    roi = profit / total_risked * 100 if total_risked > 0 else 0

    return {
        "n_bets": n_bets, "wins": wins, "losses": losses,
        "win_pct": win_pct, "profit": profit, "roi": roi,
    }


def build_report(ml_results, total_results, margin_calibrated, total_matched,
                 margin_rmse, ml_bets_prod, total_bets_prod, null_stats):
    """Build human-readable backtest report."""
    lines = []
    lines.append("MLB PROFITABILITY BACKTEST REPORT")
    lines.append("=" * 70)
    lines.append(f"Generated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')}")
    lines.append(f"Margin model RMSE: {margin_rmse:.2f}")
    lines.append(f"Edge method: Calibrated margin-space (model rescaled to market std)")
    lines.append(f"ML margin threshold (production): {ML_PRODUCTION_THRESHOLD} runs")
    lines.append(f"Total edge threshold (production): {MLB_TOTAL_EDGE_THRESHOLD} runs")
    lines.append("")

    if margin_calibrated is not None:
        n_total = len(margin_calibrated)
        lines.append(f"ML games: {n_total} OOF games with H2H odds")
    if total_matched is not None:
        n_total_t = len(total_matched)
        n_ot = total_matched["consensus_total"].notna().sum()
        lines.append(f"Total coverage: {n_ot}/{n_total_t} OOF games with total line ({n_ot/n_total_t*100:.1f}%)")
    lines.append("")

    # ── ML BETS ──
    if ml_results:
        lines.append(f"{'='*70}")
        lines.append("MONEYLINE BETS (calibrated margin-space edges, actual H2H odds)")
        lines.append(f"{'='*70}")
        lines.append(f"{'Threshold':>10} | {'Bets':>5} | {'W-L':>10} | "
                     f"{'Win%':>6} | {'Profit':>9} | {'ROI':>7} | {'Dog%':>5} | {'p-val':>7}")
        lines.append("-" * 70)
        for thresh, s in sorted(ml_results.items()):
            marker = " <--" if thresh == ML_PRODUCTION_THRESHOLD else ""
            wl = f"{s['wins']}-{s['losses']}"
            lines.append(
                f"  >= {thresh:>4.2f} | {s['n_bets']:>5} | {wl:>10} | "
                f"{s['win_pct']:>5.1f}% | ${s['profit']:>+8.0f} | "
                f"{s['roi']:>+6.1f}% | {s['dog_pct']:>4.0f}% | "
                f"{s['p_value']:>7.4f}{marker}"
            )
        lines.append("")

    # ── BLIND DOG NULL ──
    if null_stats:
        lines.append(f"{'='*70}")
        lines.append("BLIND DOG NULL STRATEGY (always bet underdog, benchmark)")
        lines.append(f"{'='*70}")
        lines.append(f"{'Season':>7} | {'Bets':>5} | {'W-L':>10} | "
                     f"{'Win%':>6} | {'Profit':>9} | {'ROI':>7}")
        lines.append("-" * 55)
        for season, ns in sorted(null_stats.items()):
            wl = f"{ns['wins']}-{ns['losses']}"
            lines.append(
                f"  {int(season):>5} | {ns['n_bets']:>5} | {wl:>10} | "
                f"{ns['win_pct']:>5.1f}% | ${ns['profit']:>+8.0f} | "
                f"{ns['roi']:>+6.1f}%"
            )
        lines.append("")

    # ── TOTAL BETS ──
    if total_results:
        lines.append(f"{'='*70}")
        lines.append("TOTAL BETS (over/under at -110)")
        lines.append(f"{'='*70}")
        lines.append(f"{'Threshold':>10} | {'Bets':>5} | {'W-L-P':>10} | "
                     f"{'Win%':>6} | {'Profit':>9} | {'ROI':>7} | {'p-val':>7}")
        lines.append("-" * 70)
        for thresh, s in sorted(total_results.items()):
            marker = " <--" if thresh == MLB_TOTAL_EDGE_THRESHOLD else ""
            wlp = f"{s['wins']}-{s['losses']}-{s['pushes']}"
            lines.append(
                f"  >= {thresh:>4.1f} | {s['n_bets']:>5} | {wlp:>10} | "
                f"{s['win_pct']:>5.1f}% | ${s['profit']:>+8.0f} | "
                f"{s['roi']:>+6.1f}% | {s['p_value']:>7.4f}{marker}"
            )
        lines.append("")

    # ── ML BY SEASON + NULL COMPARISON ──
    if ml_bets_prod is not None and not ml_bets_prod.empty:
        lines.append(f"{'='*70}")
        lines.append(f"ML BY SEASON (>= {ML_PRODUCTION_THRESHOLD} runs) vs BLIND DOG NULL")
        lines.append(f"{'='*70}")
        lines.append(f"{'Season':>7} | {'Bets':>5} | {'W-L':>10} | "
                     f"{'Win%':>6} | {'ROI':>7} | {'Dog%':>5} | {'Null ROI':>9} | {'Lift':>7}")
        lines.append("-" * 70)
        for season in sorted(ml_bets_prod["season"].dropna().unique()):
            sb = ml_bets_prod[ml_bets_prod["season"] == season]
            ss = compute_ml_stats(sb)
            ns = null_stats.get(season, {})
            null_roi = ns.get("roi", 0)
            lift = ss["roi"] - null_roi
            if ss["n_bets"] > 0:
                wl = f"{ss['wins']}-{ss['losses']}"
                lines.append(
                    f"  {int(season):>5} | {ss['n_bets']:>5} | {wl:>10} | "
                    f"{ss['win_pct']:>5.1f}% | {ss['roi']:>+6.1f}% | "
                    f"{ss['dog_pct']:>4.0f}% | {null_roi:>+8.1f}% | "
                    f"{lift:>+6.1f}pp"
                )
        lines.append("")

    # ── TOTAL BY SEASON ──
    if total_bets_prod is not None and not total_bets_prod.empty:
        lines.append(f"{'='*70}")
        lines.append(f"TOTAL BY SEASON (edge >= {MLB_TOTAL_EDGE_THRESHOLD} runs)")
        lines.append(f"{'='*70}")
        lines.append(f"{'Season':>7} | {'Bets':>5} | {'W-L-P':>10} | "
                     f"{'Win%':>6} | {'Profit':>9} | {'ROI':>7}")
        lines.append("-" * 55)
        for season in sorted(total_bets_prod["season"].dropna().unique()):
            sb = total_bets_prod[total_bets_prod["season"] == season]
            ss = compute_total_stats(sb)
            if ss["n_bets"] > 0:
                wlp = f"{ss['wins']}-{ss['losses']}-{ss['pushes']}"
                lines.append(
                    f"  {int(season):>5} | {ss['n_bets']:>5} | {wlp:>10} | "
                    f"{ss['win_pct']:>5.1f}% | ${ss['profit']:>+8.0f} | "
                    f"{ss['roi']:>+6.1f}%"
                )
        lines.append("")

    # ── MODEL DIAGNOSTICS ──
    if margin_calibrated is not None:
        lines.append("=" * 70)
        lines.append("MODEL DIAGNOSTICS")
        lines.append("=" * 70)
        lines.append(f"Model pred std:       {margin_calibrated['predicted'].std():.3f}")
        lines.append(f"Market margin std:    {margin_calibrated['market_implied_margin'].std():.3f}")
        lines.append(f"Calibrated pred std:  {margin_calibrated['calibrated_pred'].std():.3f}")
        lines.append(f"Actual margin std:    {margin_calibrated['actual'].std():.3f}")
        lines.append(f"Edge mean:            {margin_calibrated['margin_edge'].mean():.4f}")
        lines.append(f"Edge std:             {margin_calibrated['margin_edge'].std():.3f}")

        r_model = margin_calibrated["predicted"].corr(margin_calibrated["actual"])
        r_market = margin_calibrated["market_implied_margin"].corr(margin_calibrated["actual"])
        resid = margin_calibrated["actual"] - margin_calibrated["market_implied_margin"]
        r_resid = margin_calibrated["margin_edge"].corr(resid)
        lines.append(f"corr(model, actual):  {r_model:.3f}")
        lines.append(f"corr(market, actual): {r_market:.3f}")
        lines.append(f"corr(edge, market_residual): {r_resid:.3f} (model's value-add)")
        lines.append("")

    # ── NOTES ──
    lines.append("=" * 70)
    lines.append("NOTES")
    lines.append("=" * 70)
    lines.append("- OOF predictions are truly out-of-sample (walk-forward, per-fold Boruta)")
    lines.append("- ML edges use calibrated margin space (model rescaled to market std)")
    lines.append("- This eliminates the structural underdog bias from probability conversion")
    lines.append("- ML bets use actual H2H odds from the data (variable payouts)")
    lines.append("- 'Lift' = model ROI minus blind-dog null ROI (incremental value)")
    lines.append("- Total bets assume flat -110 juice")
    lines.append("- '<--' marks production threshold")
    lines.append(f"- Margin RMSE: {margin_rmse:.2f}")
    lines.append("- H2H odds coverage varies by season (more books in recent years)")
    lines.append("- 7-inning DH games excluded from total model")

    return "\n".join(lines)


def main():
    log.info("=" * 60)
    log.info("MLB PROFITABILITY BACKTEST (calibrated edges)")
    log.info("=" * 60)

    # Load data
    margin_oof, total_oof = load_oof_predictions()
    training_df = load_training_data_for_odds()

    if margin_oof is None and total_oof is None:
        log.error("No OOF predictions available. Run 06_train_mlb_model.py first.")
        sys.exit(1)

    # Determine actual RMSE from OOF predictions
    margin_rmse = MLB_MARGIN_MODEL_RMSE
    if margin_oof is not None:
        from sklearn.metrics import root_mean_squared_error
        margin_rmse = root_mean_squared_error(margin_oof["actual"], margin_oof["predicted"])
        log.info(f"Using OOF margin RMSE: {margin_rmse:.2f}")

    # Match games to odds
    margin_matched = None
    total_matched = None

    if margin_oof is not None:
        margin_matched = match_with_odds(margin_oof, training_df)

    if total_oof is not None:
        total_matched = match_with_odds(total_oof, training_df)

    # ── Calibrate margin predictions ──
    margin_calibrated = None
    if margin_matched is not None:
        margin_calibrated = calibrate_predictions(margin_matched, margin_rmse)

    # ── Blind-dog null strategy (benchmark) ──
    null_stats = {}
    if margin_calibrated is not None:
        log.info("\nBlind-dog null strategy:")
        null_bets = simulate_blind_dog(margin_calibrated)
        for s in sorted(margin_calibrated["season"].unique()):
            sb = null_bets[null_bets["season"] == s]
            ns = compute_null_stats(sb)
            null_stats[s] = ns
            log.info(f"  {int(s)}: {ns['wins']}-{ns['losses']} ({ns['win_pct']:.1f}%), "
                     f"ROI {ns['roi']:+.1f}%")

    # ── ML backtest across margin thresholds ──
    ml_results = {}
    ml_bets_prod = None

    if margin_calibrated is not None:
        log.info(f"\nML backtest (calibrated margin-space edges):")
        for thresh in ML_MARGIN_THRESHOLDS:
            bets = simulate_ml_bets(margin_calibrated, thresh)
            s = compute_ml_stats(bets)
            ml_results[thresh] = s
            if s["n_bets"] > 0:
                log.info(f"  ML >= {thresh:.2f}: {s['wins']}-{s['losses']} "
                         f"({s['win_pct']:.1f}%), ROI {s['roi']:+.1f}%, "
                         f"dog%={s['dog_pct']:.0f}%, p={s['p_value']:.4f}")

            if thresh == ML_PRODUCTION_THRESHOLD:
                ml_bets_prod = bets

    # ── Total backtest across thresholds ──
    total_results = {}
    total_bets_prod = None

    if total_matched is not None:
        log.info(f"\nTotal backtest:")
        for thresh in TOTAL_THRESHOLDS:
            bets = simulate_total_bets(total_matched, thresh)
            s = compute_total_stats(bets)
            total_results[thresh] = s
            if s["n_bets"] > 0:
                log.info(f"  Total >= {thresh:.1f}: {s['wins']}-{s['losses']}-{s['pushes']} "
                         f"({s['win_pct']:.1f}%), ROI {s['roi']:+.1f}%, "
                         f"p={s['p_value']:.4f}")

            if thresh == MLB_TOTAL_EDGE_THRESHOLD:
                total_bets_prod = bets

    # ── Build report ──
    report = build_report(ml_results, total_results,
                          margin_calibrated, total_matched,
                          margin_rmse, ml_bets_prod, total_bets_prod,
                          null_stats)

    report_path = MODELS_DIR / "mlb_backtest_report.txt"
    with open(report_path, "w") as f:
        f.write(report)
    log.info(f"\nSaved backtest report -> {report_path}")

    # ── Save per-game results ──
    all_bets = []
    if ml_bets_prod is not None and not ml_bets_prod.empty:
        all_bets.append(ml_bets_prod)
    if total_bets_prod is not None and not total_bets_prod.empty:
        all_bets.append(total_bets_prod)

    if all_bets:
        results_df = pd.concat(all_bets, ignore_index=True)
        results_path = MODELS_DIR / "mlb_backtest_results.csv"
        results_df.to_csv(results_path, index=False)
        log.info(f"Saved {len(results_df)} bet details -> {results_path}")

    # ── Print summary ──
    print(f"\n{'='*70}")
    print("MLB BACKTEST SUMMARY (calibrated margin-space edges)")
    print(f"{'='*70}")
    print(f"  Margin RMSE: {margin_rmse:.2f}")
    if ML_PRODUCTION_THRESHOLD in ml_results and ml_results[ML_PRODUCTION_THRESHOLD]["n_bets"] > 0:
        s = ml_results[ML_PRODUCTION_THRESHOLD]
        print(f"  ML (>= {ML_PRODUCTION_THRESHOLD} runs): "
              f"{s['wins']}-{s['losses']} ({s['win_pct']:.1f}%), "
              f"ROI {s['roi']:+.1f}%, dog%={s['dog_pct']:.0f}%, "
              f"P&L ${s['profit']:+,.0f}, p={s['p_value']:.4f}")
    if MLB_TOTAL_EDGE_THRESHOLD in total_results and total_results[MLB_TOTAL_EDGE_THRESHOLD]["n_bets"] > 0:
        s = total_results[MLB_TOTAL_EDGE_THRESHOLD]
        print(f"  Total (>= {MLB_TOTAL_EDGE_THRESHOLD} runs): "
              f"{s['wins']}-{s['losses']}-{s['pushes']} ({s['win_pct']:.1f}%), "
              f"ROI {s['roi']:+.1f}%, P&L ${s['profit']:+,.0f}, p={s['p_value']:.4f}")
    print(f"  Full report: {report_path}")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()
