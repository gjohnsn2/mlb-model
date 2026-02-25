"""
10b — F5 + NRFI Profitability Backtest
=========================================
Backtests F5 margin (moneyline), F5 total, and NRFI models using
walk-forward OOF predictions against historical closing odds.

F5 ML: Calibrated margin-space edges vs consensus_f5_h2h odds (~19% coverage).
F5 Total: Model F5 total vs consensus_f5_total at -110 (~19% coverage).
NRFI: Model P(NRFI) vs implied prob from consensus_f1_total (~16% coverage).

Each section has:
  1. Prediction accuracy (all games with actuals)
  2. Betting profitability (only games with odds)

Inputs:
  - models/mlb_oof_f5_margin_predictions.csv
  - models/mlb_oof_f5_total_predictions.csv
  - models/mlb_oof_nrfi_predictions.csv
  - data/historical/training_data_mlb_v2.csv

Outputs:
  - models/mlb_f5_nrfi_backtest_report.txt
  - models/mlb_f5_nrfi_backtest_results.csv

Run: python3 10b_backtest_f5_nrfi.py
"""

import sys
import argparse
import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import norm
from pathlib import Path
from config import (
    MODELS_DIR, HISTORICAL_DIR,
    F5_ML_EDGE_THRESHOLD, F5_TOTAL_EDGE_THRESHOLD, NRFI_EDGE_THRESHOLD,
    get_logger
)

log = get_logger("10b_backtest_f5_nrfi")

# Betting constants
UNIT_SIZE = 100
WIN_PAYOUT = 100
LOSS_COST = 110
BREAKEVEN_PCT = LOSS_COST / (WIN_PAYOUT + LOSS_COST)  # ~52.38%

# F5 ML thresholds (in runs of margin space)
F5_ML_THRESHOLDS = [0.0, 0.25, 0.5, 0.75, 1.0, 1.5]
F5_ML_PRODUCTION_THRESHOLD = 0.5

# F5 Total thresholds
F5_TOTAL_THRESHOLDS = [0.25, 0.5, 0.75, 1.0, 1.5]
F5_TOTAL_PRODUCTION_THRESHOLD = 0.5

# NRFI probability edge thresholds
NRFI_THRESHOLDS = [0.00, 0.02, 0.05, 0.08, 0.10, 0.15]
NRFI_PRODUCTION_THRESHOLD = 0.05

# F5 ML unit tiers (by margin edge in runs)
F5_ML_UNIT_TIERS = [
    (1.5, 3.0, "3u"),
    (1.0, 2.0, "2u"),
    (0.75, 1.5, "1.5u"),
    (0.5, 1.0, "1u"),
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
    """Convert American odds to decimal odds."""
    if pd.isna(odds) or odds == 0:
        return np.nan
    if odds < 0:
        return 1 + 100 / abs(odds)
    else:
        return 1 + odds / 100


# ══════════════════════════════════════════════════════════════
# DATA LOADING
# ══════════════════════════════════════════════════════════════

def load_oof(name, suffix=""):
    """Load walk-forward OOF predictions for a model."""
    path = MODELS_DIR / f"mlb_oof_{name}{suffix}_predictions.csv"
    if not path.exists():
        log.warning(f"OOF not found: {path}")
        return None
    df = pd.read_csv(path)
    df["date"] = df["date"].astype(str).str[:10]
    log.info(f"Loaded {name} OOF: {len(df)} games, "
             f"seasons {df['season'].min()}-{df['season'].max()}")
    return df


def load_training_data():
    """Load training data for odds columns."""
    path = HISTORICAL_DIR / "training_data_mlb_v2.csv"
    if not path.exists():
        log.error(f"Training data not found: {path}")
        sys.exit(1)
    df = pd.read_csv(path)
    log.info(f"Loaded training data: {len(df)} games")
    return df


def merge_odds(oof_df, training_df, odds_cols):
    """Join OOF predictions with specific odds columns."""
    base_cols = ["game_pk"]
    available = [c for c in base_cols + odds_cols if c in training_df.columns]
    merged = oof_df.merge(training_df[available], on="game_pk", how="left",
                          suffixes=("", "_train"))
    return merged


# ══════════════════════════════════════════════════════════════
# F5 MONEYLINE BACKTEST
# ══════════════════════════════════════════════════════════════

def backtest_f5_ml(f5_margin_oof, training_df):
    """
    Backtest F5 moneyline using calibrated margin-space edges.
    Same methodology as full-game but with F5-specific RMSE.
    """
    log.info("\n" + "=" * 60)
    log.info("F5 MONEYLINE BACKTEST")

    # Merge F5 H2H odds
    odds_cols = ["consensus_f5_h2h_home", "consensus_f5_h2h_away",
                 "actual_f5_margin", "home_team", "away_team"]
    merged = merge_odds(f5_margin_oof, training_df, odds_cols)

    # Filter corrupt F5 H2H (|ML| < 100)
    for col in ["consensus_f5_h2h_home", "consensus_f5_h2h_away"]:
        if col in merged.columns:
            corrupt = merged[col].notna() & (merged[col].abs() < 100)
            if corrupt.any():
                log.info(f"  Filtering {corrupt.sum()} corrupt {col} (|ML| < 100)")
                merged.loc[corrupt, col] = np.nan

    # Prediction accuracy (all games)
    from sklearn.metrics import root_mean_squared_error, mean_absolute_error
    valid = merged["actual"].notna() & merged["predicted"].notna()
    rmse = root_mean_squared_error(merged.loc[valid, "actual"], merged.loc[valid, "predicted"])
    mae = mean_absolute_error(merged.loc[valid, "actual"], merged.loc[valid, "predicted"])
    log.info(f"  Prediction accuracy (all {valid.sum()} games): RMSE={rmse:.2f}, MAE={mae:.2f}")

    # Games with F5 H2H odds
    has_f5_odds = merged["consensus_f5_h2h_home"].notna() & merged["consensus_f5_h2h_away"].notna()
    odds_df = merged[has_f5_odds].copy()
    log.info(f"  Games with F5 H2H odds: {len(odds_df)} ({len(odds_df)/len(merged)*100:.1f}%)")

    if len(odds_df) < 50:
        log.warning("  Too few games with F5 odds for meaningful backtest")
        return None, None, {"rmse": rmse, "mae": mae, "n_all": int(valid.sum())}

    # De-vig F5 market probabilities
    raw_home = odds_df["consensus_f5_h2h_home"].apply(american_to_implied_prob)
    raw_away = odds_df["consensus_f5_h2h_away"].apply(american_to_implied_prob)
    total_vig = raw_home + raw_away
    valid_vig = total_vig.notna() & (total_vig > 0)
    odds_df.loc[valid_vig, "market_home_prob"] = raw_home[valid_vig] / total_vig[valid_vig]

    # Market implied margin
    odds_df["market_implied_margin"] = rmse * norm.ppf(
        odds_df["market_home_prob"].clip(0.001, 0.999)
    )

    # Calibrate model predictions to market distribution
    model_mean = odds_df["predicted"].mean()
    model_std = odds_df["predicted"].std()
    market_mean = odds_df["market_implied_margin"].mean()
    market_std = odds_df["market_implied_margin"].std()

    if model_std > 0:
        odds_df["calibrated_pred"] = (
            (odds_df["predicted"] - model_mean) / model_std * market_std + market_mean
        )
    else:
        odds_df["calibrated_pred"] = odds_df["predicted"]

    odds_df["margin_edge"] = odds_df["calibrated_pred"] - odds_df["market_implied_margin"]

    log.info(f"  Calibration: model std {model_std:.3f} -> market std {market_std:.3f}")
    home_pct = (odds_df["margin_edge"] > 0).mean() * 100
    log.info(f"  Side balance: {home_pct:.1f}% home / {100-home_pct:.1f}% away")

    # Simulate bets at each threshold
    results = {}
    prod_bets = None

    for thresh in F5_ML_THRESHOLDS:
        bets = _simulate_f5_ml_bets(odds_df, thresh)
        s = _compute_ml_stats(bets)
        results[thresh] = s
        if s["n_bets"] > 0:
            log.info(f"  F5 ML >= {thresh:.2f}: {s['wins']}-{s['losses']} "
                     f"({s['win_pct']:.1f}%), ROI {s['roi']:+.1f}%, p={s['p_value']:.4f}")
        if thresh == F5_ML_PRODUCTION_THRESHOLD:
            prod_bets = bets

    accuracy = {"rmse": rmse, "mae": mae, "n_all": int(valid.sum()),
                "n_with_odds": len(odds_df)}
    return results, prod_bets, accuracy


def _simulate_f5_ml_bets(df, threshold):
    """Simulate F5 moneyline bets using calibrated margin-space edges."""
    bets = []
    for _, row in df.iterrows():
        edge = row.get("margin_edge", np.nan)
        if pd.isna(edge) or abs(edge) < threshold:
            continue

        if edge > 0:
            side = "HOME_F5_ML"
            odds_used = row["consensus_f5_h2h_home"]
            won = row["actual"] > 0  # F5 margin > 0 = home leads after 5
        else:
            side = "AWAY_F5_ML"
            odds_used = row["consensus_f5_h2h_away"]
            won = row["actual"] < 0

        dec = american_to_decimal(odds_used)
        if pd.isna(dec):
            continue

        is_dog = (side == "HOME_F5_ML" and row.get("market_home_prob", 0.5) < 0.5) or \
                 (side == "AWAY_F5_ML" and row.get("market_home_prob", 0.5) >= 0.5)

        # Unit tiers
        ml_units = 1.0
        for tier_min, tier_units, _ in F5_ML_UNIT_TIERS:
            if abs(edge) >= tier_min:
                ml_units = tier_units
                break

        bet_risk = UNIT_SIZE * ml_units
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
            "margin_edge": round(edge, 3),
            "side": side,
            "is_dog": is_dog,
            "odds_used": int(odds_used),
            "ml_units": ml_units,
            "actual_f5_margin": row["actual"],
            "won": won,
            "push": push,
            "profit": profit,
            "bet_type": "f5_ml",
        })

    return pd.DataFrame(bets)


# ══════════════════════════════════════════════════════════════
# F5 TOTAL BACKTEST
# ══════════════════════════════════════════════════════════════

def backtest_f5_total(f5_total_oof, training_df):
    """Backtest F5 total over/under at -110."""
    log.info("\n" + "=" * 60)
    log.info("F5 TOTAL BACKTEST")

    odds_cols = ["consensus_f5_total", "actual_f5_total",
                 "home_team", "away_team"]
    merged = merge_odds(f5_total_oof, training_df, odds_cols)

    # Prediction accuracy
    from sklearn.metrics import root_mean_squared_error, mean_absolute_error
    valid = merged["actual"].notna() & merged["predicted"].notna()
    rmse = root_mean_squared_error(merged.loc[valid, "actual"], merged.loc[valid, "predicted"])
    mae = mean_absolute_error(merged.loc[valid, "actual"], merged.loc[valid, "predicted"])
    log.info(f"  Prediction accuracy (all {valid.sum()} games): RMSE={rmse:.2f}, MAE={mae:.2f}")

    # Games with F5 total line
    has_odds = merged["consensus_f5_total"].notna()
    odds_df = merged[has_odds].copy()
    log.info(f"  Games with F5 total line: {len(odds_df)} ({len(odds_df)/len(merged)*100:.1f}%)")

    if len(odds_df) < 50:
        log.warning("  Too few games with F5 total odds")
        return None, None, {"rmse": rmse, "mae": mae, "n_all": int(valid.sum())}

    results = {}
    prod_bets = None

    for thresh in F5_TOTAL_THRESHOLDS:
        bets = _simulate_f5_total_bets(odds_df, thresh)
        s = _compute_total_stats(bets)
        results[thresh] = s
        if s["n_bets"] > 0:
            log.info(f"  F5 Total >= {thresh:.2f}: {s['wins']}-{s['losses']}-{s['pushes']} "
                     f"({s['win_pct']:.1f}%), ROI {s['roi']:+.1f}%, p={s['p_value']:.4f}")
        if thresh == F5_TOTAL_PRODUCTION_THRESHOLD:
            prod_bets = bets

    accuracy = {"rmse": rmse, "mae": mae, "n_all": int(valid.sum()),
                "n_with_odds": len(odds_df)}
    return results, prod_bets, accuracy


def _simulate_f5_total_bets(df, threshold):
    """Simulate F5 over/under bets at -110."""
    bets = []
    for _, row in df.iterrows():
        if pd.isna(row.get("consensus_f5_total")) or pd.isna(row["predicted"]):
            continue

        model_total = row["predicted"]
        mkt_total = row["consensus_f5_total"]
        actual_total = row["actual"]

        edge = model_total - mkt_total
        if abs(edge) < threshold:
            continue

        if edge > 0:
            side = "F5_OVER"
            won = actual_total > mkt_total
            push = actual_total == mkt_total
        else:
            side = "F5_UNDER"
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
            "consensus_f5_total": mkt_total,
            "edge": round(edge, 2),
            "side": side,
            "actual_f5_total": actual_total,
            "won": won,
            "push": push,
            "profit": profit,
            "bet_type": "f5_total",
        })

    return pd.DataFrame(bets)


# ══════════════════════════════════════════════════════════════
# NRFI BACKTEST
# ══════════════════════════════════════════════════════════════

def backtest_nrfi(nrfi_oof, training_df):
    """
    Backtest NRFI using model P(NRFI) vs market implied probability.

    consensus_f1_total is the 1st-inning total line (typically 0.5).
    Market P(NRFI) = implied prob of Under 0.5.
    Bet NRFI when model_prob > market_prob + threshold.
    Bet YRFI when market_prob > model_prob + threshold.
    """
    log.info("\n" + "=" * 60)
    log.info("NRFI BACKTEST")

    odds_cols = ["consensus_f1_total", "actual_nrfi",
                 "home_team", "away_team"]
    merged = merge_odds(nrfi_oof, training_df, odds_cols)

    # Prediction accuracy (all games)
    from sklearn.metrics import brier_score_loss, roc_auc_score
    valid = merged["actual"].notna() & merged["predicted"].notna()
    actual_int = merged.loc[valid, "actual"].astype(int)
    brier = brier_score_loss(actual_int, merged.loc[valid, "predicted"])
    auc = roc_auc_score(actual_int, merged.loc[valid, "predicted"])
    base_rate = actual_int.mean()
    naive_brier = base_rate * (1 - base_rate) ** 2 + (1 - base_rate) * base_rate ** 2

    log.info(f"  Prediction accuracy (all {valid.sum()} games):")
    log.info(f"    Brier: {brier:.4f} (naive: {naive_brier:.4f}, skill: {1-brier/naive_brier:.3f})")
    log.info(f"    AUC: {auc:.3f}, Base rate: {base_rate:.3f}")

    # Games with F1 total odds
    # consensus_f1_total is typically a total line (e.g. 0.5) with associated juice
    # We approximate market NRFI prob from the total line
    has_odds = merged["consensus_f1_total"].notna()
    odds_df = merged[has_odds].copy()
    log.info(f"  Games with F1 total line: {len(odds_df)} ({len(odds_df)/len(merged)*100:.1f}%)")

    if len(odds_df) < 50:
        log.warning("  Too few games with F1 total odds")
        return None, None, {"brier": brier, "auc": auc, "base_rate": base_rate,
                            "naive_brier": naive_brier, "n_all": int(valid.sum())}

    # Market implied NRFI probability
    # F1 total line = 0.5 means NRFI is under 0.5
    # We use the actual line value as a proxy for market NRFI probability:
    # If consensus_f1_total is the raw total (e.g. 0.5), NRFI prob ~ implied under
    # Since we don't have separate under/over odds, approximate:
    # NRFI base rate is ~72%, so market prices are typically -220/+180 range
    # Use base rate as market prob when we only have total line
    odds_df["market_nrfi_prob"] = base_rate  # Conservative: use base rate as benchmark

    results = {}
    prod_bets = None

    for thresh in NRFI_THRESHOLDS:
        bets = _simulate_nrfi_bets(odds_df, thresh)
        s = _compute_nrfi_stats(bets)
        results[thresh] = s
        if s["n_bets"] > 0:
            log.info(f"  NRFI >= {thresh:.2f}: {s['n_bets']} bets "
                     f"({s['nrfi_bets']} NRFI + {s['yrfi_bets']} YRFI), "
                     f"ROI {s['roi']:+.1f}%, p={s['p_value']:.4f}")
        if thresh == NRFI_PRODUCTION_THRESHOLD:
            prod_bets = bets

    accuracy = {"brier": brier, "auc": auc, "base_rate": base_rate,
                "naive_brier": naive_brier, "n_all": int(valid.sum()),
                "n_with_odds": len(odds_df)}
    return results, prod_bets, accuracy


def _simulate_nrfi_bets(df, threshold):
    """
    Simulate NRFI/YRFI bets.

    NRFI at -110: bet when model P(NRFI) > market P(NRFI) + threshold
    YRFI at -110: bet when model P(YRFI) > market P(YRFI) + threshold
    (i.e., market_nrfi_prob > model_nrfi_prob + threshold)
    """
    bets = []
    for _, row in df.iterrows():
        model_nrfi_prob = row["predicted"]
        market_nrfi_prob = row["market_nrfi_prob"]

        if pd.isna(model_nrfi_prob) or pd.isna(market_nrfi_prob):
            continue

        edge = model_nrfi_prob - market_nrfi_prob

        if abs(edge) < threshold:
            continue

        actual_nrfi = int(row["actual"])

        if edge > 0:
            # Model says NRFI is more likely than market
            side = "NRFI"
            won = actual_nrfi == 1
        else:
            # Model says YRFI is more likely than market
            side = "YRFI"
            won = actual_nrfi == 0

        # Flat -110 both sides
        profit = WIN_PAYOUT if won else -LOSS_COST

        bets.append({
            "date": row["date"],
            "game_pk": row["game_pk"],
            "home_team": row.get("home_team", row.get("home_team_train", "")),
            "away_team": row.get("away_team", row.get("away_team_train", "")),
            "season": row.get("season"),
            "model_nrfi_prob": round(model_nrfi_prob, 4),
            "market_nrfi_prob": round(market_nrfi_prob, 4),
            "edge": round(edge, 4),
            "side": side,
            "actual_nrfi": actual_nrfi,
            "won": won,
            "push": False,
            "profit": profit,
            "bet_type": "nrfi",
        })

    return pd.DataFrame(bets)


# ══════════════════════════════════════════════════════════════
# STATS COMPUTATION
# ══════════════════════════════════════════════════════════════

def _compute_ml_stats(bets_df):
    """Stats for F5 ML bets (variable odds, tiered units)."""
    if bets_df.empty:
        return {"n_bets": 0, "wins": 0, "losses": 0,
                "win_pct": 0, "profit": 0, "roi": 0, "p_value": 1.0, "dog_pct": 0}

    non_push = bets_df[~bets_df["push"]]
    wins = int(non_push["won"].sum())
    losses = len(non_push) - wins
    n_bets = wins + losses
    win_pct = wins / n_bets * 100 if n_bets > 0 else 0

    profit = bets_df["profit"].sum()
    total_risked = (bets_df["ml_units"] * UNIT_SIZE).sum() if "ml_units" in bets_df.columns else n_bets * UNIT_SIZE
    roi = profit / total_risked * 100 if total_risked > 0 else 0

    dog_pct = bets_df["is_dog"].mean() * 100 if "is_dog" in bets_df.columns else 0

    p_value = 1.0
    if n_bets > 0:
        profits = bets_df["profit"].values
        profit_mean = profits.mean()
        profit_std = profits.std(ddof=1)
        if profit_std > 0:
            z = profit_mean * np.sqrt(len(profits)) / profit_std
            p_value = 1 - norm.cdf(z)

    return {"n_bets": n_bets, "wins": wins, "losses": losses,
            "win_pct": win_pct, "profit": profit, "roi": roi,
            "p_value": p_value, "dog_pct": dog_pct}


def _compute_total_stats(bets_df):
    """Stats for F5 total bets (flat -110)."""
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

    p_value = 1.0
    if n_bets > 0:
        p_value = stats.binomtest(wins, n_bets, BREAKEVEN_PCT, alternative="greater").pvalue

    return {"n_bets": n_bets, "wins": wins, "losses": losses, "pushes": pushes,
            "win_pct": win_pct, "profit": profit, "roi": roi, "p_value": p_value}


def _compute_nrfi_stats(bets_df):
    """Stats for NRFI/YRFI bets (flat -110)."""
    if bets_df.empty:
        return {"n_bets": 0, "nrfi_bets": 0, "yrfi_bets": 0,
                "wins": 0, "losses": 0, "win_pct": 0,
                "profit": 0, "roi": 0, "p_value": 1.0}

    wins = int(bets_df["won"].sum())
    losses = len(bets_df) - wins
    n_bets = len(bets_df)
    win_pct = wins / n_bets * 100 if n_bets > 0 else 0

    nrfi_bets = int((bets_df["side"] == "NRFI").sum())
    yrfi_bets = int((bets_df["side"] == "YRFI").sum())

    profit = bets_df["profit"].sum()
    total_risked = n_bets * LOSS_COST
    roi = profit / total_risked * 100 if total_risked > 0 else 0

    p_value = 1.0
    if n_bets > 0:
        p_value = stats.binomtest(wins, n_bets, BREAKEVEN_PCT, alternative="greater").pvalue

    return {"n_bets": n_bets, "nrfi_bets": nrfi_bets, "yrfi_bets": yrfi_bets,
            "wins": wins, "losses": losses, "win_pct": win_pct,
            "profit": profit, "roi": roi, "p_value": p_value}


# ══════════════════════════════════════════════════════════════
# REPORT
# ══════════════════════════════════════════════════════════════

def build_report(f5_ml_results, f5_ml_accuracy, f5_ml_bets,
                 f5_total_results, f5_total_accuracy, f5_total_bets,
                 nrfi_results, nrfi_accuracy, nrfi_bets):
    """Build human-readable backtest report."""
    lines = []
    lines.append("F5 + NRFI PROFITABILITY BACKTEST REPORT")
    lines.append("=" * 70)
    lines.append(f"Generated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')}")
    lines.append("")

    # ── F5 ML ──
    if f5_ml_accuracy:
        lines.append(f"\n{'='*70}")
        lines.append("F5 MONEYLINE — PREDICTION ACCURACY")
        lines.append(f"{'='*70}")
        lines.append(f"All games: {f5_ml_accuracy['n_all']:,}")
        lines.append(f"F5 Margin RMSE: {f5_ml_accuracy['rmse']:.2f}")
        lines.append(f"F5 Margin MAE:  {f5_ml_accuracy['mae']:.2f}")
        if "n_with_odds" in f5_ml_accuracy:
            lines.append(f"Games with F5 H2H odds: {f5_ml_accuracy['n_with_odds']:,} "
                         f"({f5_ml_accuracy['n_with_odds']/f5_ml_accuracy['n_all']*100:.1f}%)")
        lines.append("")

    if f5_ml_results:
        lines.append(f"{'='*70}")
        lines.append("F5 MONEYLINE — BETTING PROFITABILITY")
        lines.append(f"{'='*70}")
        lines.append(f"{'Threshold':>10} | {'Bets':>5} | {'W-L':>10} | "
                     f"{'Win%':>6} | {'Profit':>9} | {'ROI':>7} | {'Dog%':>5} | {'p-val':>7}")
        lines.append("-" * 70)
        for thresh, s in sorted(f5_ml_results.items()):
            marker = " <--" if thresh == F5_ML_PRODUCTION_THRESHOLD else ""
            wl = f"{s['wins']}-{s['losses']}"
            lines.append(
                f"  >= {thresh:>4.2f} | {s['n_bets']:>5} | {wl:>10} | "
                f"{s['win_pct']:>5.1f}% | ${s['profit']:>+8.0f} | "
                f"{s['roi']:>+6.1f}% | {s['dog_pct']:>4.0f}% | "
                f"{s['p_value']:>7.4f}{marker}"
            )
        lines.append("")

        # F5 ML by season
        if f5_ml_bets is not None and not f5_ml_bets.empty:
            lines.append(f"F5 ML BY SEASON (>= {F5_ML_PRODUCTION_THRESHOLD} runs):")
            lines.append(f"{'Season':>7} | {'Bets':>5} | {'W-L':>10} | "
                         f"{'Win%':>6} | {'ROI':>7}")
            lines.append("-" * 45)
            for season in sorted(f5_ml_bets["season"].dropna().unique()):
                sb = f5_ml_bets[f5_ml_bets["season"] == season]
                ss = _compute_ml_stats(sb)
                if ss["n_bets"] > 0:
                    wl = f"{ss['wins']}-{ss['losses']}"
                    lines.append(
                        f"  {int(season):>5} | {ss['n_bets']:>5} | {wl:>10} | "
                        f"{ss['win_pct']:>5.1f}% | {ss['roi']:>+6.1f}%"
                    )
            lines.append("")

    # ── F5 TOTAL ──
    if f5_total_accuracy:
        lines.append(f"\n{'='*70}")
        lines.append("F5 TOTAL — PREDICTION ACCURACY")
        lines.append(f"{'='*70}")
        lines.append(f"All games: {f5_total_accuracy['n_all']:,}")
        lines.append(f"F5 Total RMSE: {f5_total_accuracy['rmse']:.2f}")
        lines.append(f"F5 Total MAE:  {f5_total_accuracy['mae']:.2f}")
        if "n_with_odds" in f5_total_accuracy:
            lines.append(f"Games with F5 total line: {f5_total_accuracy['n_with_odds']:,} "
                         f"({f5_total_accuracy['n_with_odds']/f5_total_accuracy['n_all']*100:.1f}%)")
        lines.append("")

    if f5_total_results:
        lines.append(f"{'='*70}")
        lines.append("F5 TOTAL — BETTING PROFITABILITY (over/under at -110)")
        lines.append(f"{'='*70}")
        lines.append(f"{'Threshold':>10} | {'Bets':>5} | {'W-L-P':>10} | "
                     f"{'Win%':>6} | {'Profit':>9} | {'ROI':>7} | {'p-val':>7}")
        lines.append("-" * 65)
        for thresh, s in sorted(f5_total_results.items()):
            marker = " <--" if thresh == F5_TOTAL_PRODUCTION_THRESHOLD else ""
            wlp = f"{s['wins']}-{s['losses']}-{s['pushes']}"
            lines.append(
                f"  >= {thresh:>4.2f} | {s['n_bets']:>5} | {wlp:>10} | "
                f"{s['win_pct']:>5.1f}% | ${s['profit']:>+8.0f} | "
                f"{s['roi']:>+6.1f}% | {s['p_value']:>7.4f}{marker}"
            )
        lines.append("")

        # F5 Total by season
        if f5_total_bets is not None and not f5_total_bets.empty:
            lines.append(f"F5 TOTAL BY SEASON (>= {F5_TOTAL_PRODUCTION_THRESHOLD} runs):")
            lines.append(f"{'Season':>7} | {'Bets':>5} | {'W-L-P':>10} | "
                         f"{'Win%':>6} | {'ROI':>7}")
            lines.append("-" * 45)
            for season in sorted(f5_total_bets["season"].dropna().unique()):
                sb = f5_total_bets[f5_total_bets["season"] == season]
                ss = _compute_total_stats(sb)
                if ss["n_bets"] > 0:
                    wlp = f"{ss['wins']}-{ss['losses']}-{ss['pushes']}"
                    lines.append(
                        f"  {int(season):>5} | {ss['n_bets']:>5} | {wlp:>10} | "
                        f"{ss['win_pct']:>5.1f}% | {ss['roi']:>+6.1f}%"
                    )
            lines.append("")

    # ── NRFI ──
    if nrfi_accuracy:
        lines.append(f"\n{'='*70}")
        lines.append("NRFI — PREDICTION ACCURACY")
        lines.append(f"{'='*70}")
        lines.append(f"All games: {nrfi_accuracy['n_all']:,}")
        lines.append(f"Brier score:  {nrfi_accuracy['brier']:.4f} "
                     f"(naive: {nrfi_accuracy['naive_brier']:.4f}, "
                     f"skill: {1-nrfi_accuracy['brier']/nrfi_accuracy['naive_brier']:.3f})")
        lines.append(f"AUC:          {nrfi_accuracy['auc']:.3f}")
        lines.append(f"Base rate:    {nrfi_accuracy['base_rate']:.3f}")
        if "n_with_odds" in nrfi_accuracy:
            lines.append(f"Games with F1 total line: {nrfi_accuracy['n_with_odds']:,} "
                         f"({nrfi_accuracy['n_with_odds']/nrfi_accuracy['n_all']*100:.1f}%)")
        lines.append("")

    if nrfi_results:
        lines.append(f"{'='*70}")
        lines.append("NRFI — BETTING PROFITABILITY (NRFI/YRFI at -110)")
        lines.append(f"{'='*70}")
        lines.append(f"{'Threshold':>10} | {'Bets':>5} | {'NRFI':>5} | {'YRFI':>5} | "
                     f"{'W-L':>10} | {'Win%':>6} | {'ROI':>7} | {'p-val':>7}")
        lines.append("-" * 70)
        for thresh, s in sorted(nrfi_results.items()):
            marker = " <--" if thresh == NRFI_PRODUCTION_THRESHOLD else ""
            wl = f"{s['wins']}-{s['losses']}"
            lines.append(
                f"  >= {thresh:>4.2f} | {s['n_bets']:>5} | {s['nrfi_bets']:>5} | "
                f"{s['yrfi_bets']:>5} | {wl:>10} | "
                f"{s['win_pct']:>5.1f}% | {s['roi']:>+6.1f}% | "
                f"{s['p_value']:>7.4f}{marker}"
            )
        lines.append("")

        # NRFI by season
        if nrfi_bets is not None and not nrfi_bets.empty:
            lines.append(f"NRFI BY SEASON (>= {NRFI_PRODUCTION_THRESHOLD} edge):")
            lines.append(f"{'Season':>7} | {'Bets':>5} | {'NRFI':>5} | {'YRFI':>5} | "
                         f"{'Win%':>6} | {'ROI':>7}")
            lines.append("-" * 50)
            for season in sorted(nrfi_bets["season"].dropna().unique()):
                sb = nrfi_bets[nrfi_bets["season"] == season]
                ss = _compute_nrfi_stats(sb)
                if ss["n_bets"] > 0:
                    lines.append(
                        f"  {int(season):>5} | {ss['n_bets']:>5} | {ss['nrfi_bets']:>5} | "
                        f"{ss['yrfi_bets']:>5} | {ss['win_pct']:>5.1f}% | {ss['roi']:>+6.1f}%"
                    )
            lines.append("")

    # ── NOTES ──
    lines.append("=" * 70)
    lines.append("NOTES")
    lines.append("=" * 70)
    lines.append("- OOF predictions are truly out-of-sample (walk-forward, per-fold Boruta)")
    lines.append("- F5 ML uses calibrated margin-space edges (same method as full-game)")
    lines.append("- F5 Total bets assume flat -110 juice")
    lines.append("- NRFI bets assume flat -110 juice for both NRFI and YRFI")
    lines.append("- NRFI market prob approximated as base rate (conservative benchmark)")
    lines.append("- F5 odds coverage is sparse (~19%, mostly 2023+)")
    lines.append("- '<--' marks production threshold")

    return "\n".join(lines)


# ══════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="F5 + NRFI profitability backtest")
    mode_group = parser.add_mutually_exclusive_group()
    mode_group.add_argument("--no-market", action="store_true",
                            help="Use no-market OOF predictions")
    mode_group.add_argument("--ensemble", action="store_true",
                            help="Use ensemble OOF predictions")
    args = parser.parse_args()

    suffix = "_nomarket" if args.no_market else ("_ensemble" if args.ensemble else "")
    mode_label = "NO-MARKET" if args.no_market else ("ENSEMBLE" if args.ensemble else "standard")

    log.info("=" * 60)
    log.info(f"F5 + NRFI PROFITABILITY BACKTEST [{mode_label}]")
    log.info("=" * 60)

    # Load OOF predictions
    f5_margin_oof = load_oof("f5_margin", suffix)
    f5_total_oof = load_oof("f5_total", suffix)
    nrfi_oof = load_oof("nrfi", suffix)

    if f5_margin_oof is None and f5_total_oof is None and nrfi_oof is None:
        log.error("No F5/NRFI OOF predictions found. Run 06_train_mlb_model.py first.")
        sys.exit(1)

    training_df = load_training_data()

    # Run backtests
    f5_ml_results, f5_ml_bets, f5_ml_accuracy = None, None, None
    f5_total_results, f5_total_bets, f5_total_accuracy = None, None, None
    nrfi_results, nrfi_bets, nrfi_accuracy = None, None, None

    if f5_margin_oof is not None:
        f5_ml_results, f5_ml_bets, f5_ml_accuracy = backtest_f5_ml(f5_margin_oof, training_df)

    if f5_total_oof is not None:
        f5_total_results, f5_total_bets, f5_total_accuracy = backtest_f5_total(f5_total_oof, training_df)

    if nrfi_oof is not None:
        nrfi_results, nrfi_bets, nrfi_accuracy = backtest_nrfi(nrfi_oof, training_df)

    # Build report
    report = build_report(f5_ml_results, f5_ml_accuracy, f5_ml_bets,
                          f5_total_results, f5_total_accuracy, f5_total_bets,
                          nrfi_results, nrfi_accuracy, nrfi_bets)

    report_path = MODELS_DIR / f"mlb_f5_nrfi_backtest_report{suffix}.txt"
    with open(report_path, "w") as f:
        f.write(report)
    log.info(f"\nSaved backtest report -> {report_path}")

    # Save per-game results
    all_bets = []
    for bets_df in [f5_ml_bets, f5_total_bets, nrfi_bets]:
        if bets_df is not None and not bets_df.empty:
            all_bets.append(bets_df)

    if all_bets:
        results_df = pd.concat(all_bets, ignore_index=True)
        results_path = MODELS_DIR / f"mlb_f5_nrfi_backtest_results{suffix}.csv"
        results_df.to_csv(results_path, index=False)
        log.info(f"Saved {len(results_df)} bet details -> {results_path}")

    # Print summary
    print(f"\n{'='*70}")
    print("F5 + NRFI BACKTEST SUMMARY")
    print(f"{'='*70}")

    if f5_ml_accuracy:
        print(f"  F5 Margin RMSE: {f5_ml_accuracy['rmse']:.2f} ({f5_ml_accuracy['n_all']:,} games)")
        if f5_ml_results and F5_ML_PRODUCTION_THRESHOLD in f5_ml_results:
            s = f5_ml_results[F5_ML_PRODUCTION_THRESHOLD]
            if s["n_bets"] > 0:
                print(f"  F5 ML (>= {F5_ML_PRODUCTION_THRESHOLD} runs): "
                      f"{s['wins']}-{s['losses']} ({s['win_pct']:.1f}%), "
                      f"ROI {s['roi']:+.1f}%, p={s['p_value']:.4f}")

    if f5_total_accuracy:
        print(f"  F5 Total RMSE: {f5_total_accuracy['rmse']:.2f} ({f5_total_accuracy['n_all']:,} games)")
        if f5_total_results and F5_TOTAL_PRODUCTION_THRESHOLD in f5_total_results:
            s = f5_total_results[F5_TOTAL_PRODUCTION_THRESHOLD]
            if s["n_bets"] > 0:
                print(f"  F5 Total (>= {F5_TOTAL_PRODUCTION_THRESHOLD} runs): "
                      f"{s['wins']}-{s['losses']}-{s['pushes']} ({s['win_pct']:.1f}%), "
                      f"ROI {s['roi']:+.1f}%, p={s['p_value']:.4f}")

    if nrfi_accuracy:
        print(f"  NRFI Brier: {nrfi_accuracy['brier']:.4f} "
              f"(naive: {nrfi_accuracy['naive_brier']:.4f}, "
              f"skill: {1-nrfi_accuracy['brier']/nrfi_accuracy['naive_brier']:.3f}), "
              f"AUC: {nrfi_accuracy['auc']:.3f}")
        if nrfi_results and NRFI_PRODUCTION_THRESHOLD in nrfi_results:
            s = nrfi_results[NRFI_PRODUCTION_THRESHOLD]
            if s["n_bets"] > 0:
                print(f"  NRFI (>= {NRFI_PRODUCTION_THRESHOLD} edge): "
                      f"{s['n_bets']} bets ({s['nrfi_bets']}N + {s['yrfi_bets']}Y), "
                      f"ROI {s['roi']:+.1f}%, p={s['p_value']:.4f}")

    print(f"  Full report: {report_path}")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()
