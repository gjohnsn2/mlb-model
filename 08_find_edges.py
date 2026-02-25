"""
08 -- Find Edges & Daily Betting Card (Margin-Space)
=====================================================
Uses calibrated margin-space edges matching 10_backtest_mlb.py methodology.

Edge methodology:
  1. De-vig market H2H odds to fair probability
  2. Convert to implied margin: margin = RMSE * phi_inv(prob)
  3. Calibrate model prediction to market scale using OOF calibration params
  4. Edge = calibrated_prediction - market_implied_margin (in runs)
  5. Bet when |edge| >= threshold

This eliminates the structural underdog bias from probability conversion.

Inputs:
  data/predictions/picks_{TODAY}.csv  (from 07_predict.py)
  data/raw/odds_{TODAY}.csv           (from 04_fetch_odds.py)

Outputs:
  data/predictions/edges_{TODAY}.csv
  data/predictions/betting_card_{TODAY}.json
  data/predictions/betting_card_{TODAY}.csv

Run: python3 08_find_edges.py
"""

import sys
import json
import pandas as pd
import numpy as np
from scipy.stats import norm
from config import (
    PREDICTIONS_DIR, RAW_DIR, TRACKING_DIR, TODAY,
    BANKROLL_START, BANKROLL_UNIT_PCT,
    DRAWDOWN_WARNING, DRAWDOWN_PAUSE,
    BOOK_DISPLAY_NAMES,
    get_logger
)

log = get_logger("08_edges")

# Edge thresholds in margin space (runs) — matching 10_backtest_mlb.py
ML_MARGIN_THRESHOLD = 1.5  # Production threshold (runs)

# Unit tiers by margin edge (runs) — matching 10_backtest_mlb.py
ML_MARGIN_UNIT_TIERS = [
    (2.0, 3.0, "3u"),   # edge >= 2.0 runs
    (1.5, 2.0, "2u"),   # edge >= 1.5 runs
    (1.0, 1.5, "1.5u"), # edge >= 1.0 runs
    (0.5, 1.0, "1u"),   # edge >= 0.5 runs
]

# Total edge threshold
TOTAL_EDGE_THRESHOLD = 1.5  # runs


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


def get_current_bankroll():
    """Load current bankroll from tracking file."""
    bankroll_path = TRACKING_DIR / "bankroll.csv"
    if bankroll_path.exists():
        try:
            df = pd.read_csv(bankroll_path)
            if len(df) > 0:
                return df.iloc[-1]["bankroll"]
        except Exception:
            pass
    return BANKROLL_START


def check_drawdown(current_bankroll):
    """Check if we're in a drawdown state."""
    bankroll_path = TRACKING_DIR / "bankroll.csv"
    peak = BANKROLL_START

    if bankroll_path.exists():
        try:
            df = pd.read_csv(bankroll_path)
            if len(df) > 0:
                peak = df["bankroll"].max()
        except Exception:
            pass

    drawdown = (peak - current_bankroll) / peak

    if drawdown >= DRAWDOWN_PAUSE:
        log.error(f"DRAWDOWN PAUSE: {drawdown:.1%} from peak "
                  f"(${peak:,.0f} -> ${current_bankroll:,.0f})")
        return "PAUSE"
    elif drawdown >= DRAWDOWN_WARNING:
        log.warning(f"DRAWDOWN WARNING: {drawdown:.1%} from peak")
        return "WARNING"
    else:
        return "OK"


def load_odds():
    """Load today's odds from the Odds API fetch."""
    odds_path = RAW_DIR / f"odds_{TODAY}.csv"
    if not odds_path.exists():
        log.warning(f"Odds not found: {odds_path}")
        return pd.DataFrame()

    df = pd.read_csv(odds_path)
    log.info(f"Loaded odds for {len(df)} games")
    return df


def match_odds_to_picks(picks_df, odds_df):
    """Match odds data to picks by game_pk or team names."""
    if odds_df.empty:
        picks_df["consensus_h2h_home"] = np.nan
        picks_df["consensus_h2h_away"] = np.nan
        picks_df["consensus_total_line"] = np.nan
        return picks_df

    for i, row in picks_df.iterrows():
        game_pk = row.get("game_pk")
        home = str(row.get("home_team", ""))
        away = str(row.get("away_team", ""))

        matched = False
        for _, o in odds_df.iterrows():
            o_pk = o.get("game_pk", o.get("game_id"))
            o_home = str(o.get("home_team", ""))
            o_away = str(o.get("away_team", ""))

            if (game_pk == o_pk) or (o_home == home and o_away == away):
                # Look for consensus H2H columns (various naming conventions)
                for h_col in ["consensus_h2h_home", "consensus_ml_home", "h2h_home"]:
                    if h_col in o.index and pd.notna(o.get(h_col)):
                        picks_df.at[i, "consensus_h2h_home"] = o[h_col]
                        break

                for a_col in ["consensus_h2h_away", "consensus_ml_away", "h2h_away"]:
                    if a_col in o.index and pd.notna(o.get(a_col)):
                        picks_df.at[i, "consensus_h2h_away"] = o[a_col]
                        break

                for t_col in ["consensus_total", "total"]:
                    if t_col in o.index and pd.notna(o.get(t_col)):
                        picks_df.at[i, "consensus_total_line"] = o[t_col]
                        break

                # Also grab per-book odds JSON if available
                for bk_col in ["book_mls_json", "book_odds_json"]:
                    if bk_col in o.index and pd.notna(o.get(bk_col)):
                        picks_df.at[i, "book_mls_json"] = o[bk_col]
                        break

                matched = True
                break

    has_h2h = picks_df["consensus_h2h_home"].notna().sum() if "consensus_h2h_home" in picks_df.columns else 0
    log.info(f"Matched odds: {has_h2h}/{len(picks_df)} with H2H")
    return picks_df


def compute_margin_edges(picks_df, margin_rmse):
    """
    Compute calibrated margin-space edges.
    Replicates 10_backtest_mlb.py:calibrate_predictions() exactly.
    """
    # Need H2H odds for de-vigging
    if "consensus_h2h_home" not in picks_df.columns:
        log.warning("No H2H odds available — cannot compute edges")
        return picks_df

    h2h_mask = picks_df["consensus_h2h_home"].notna() & picks_df["consensus_h2h_away"].notna()

    # Filter corrupt H2H lines (|ML| < 100)
    MIN_ML = 100
    for col in ["consensus_h2h_home", "consensus_h2h_away"]:
        if col in picks_df.columns:
            corrupt = picks_df[col].notna() & (picks_df[col].abs() < MIN_ML)
            if corrupt.any():
                log.warning(f"Filtering {corrupt.sum()} corrupt {col} values")
                picks_df.loc[corrupt, col] = np.nan
                h2h_mask = picks_df["consensus_h2h_home"].notna() & picks_df["consensus_h2h_away"].notna()

    if not h2h_mask.any():
        log.warning("No valid H2H odds after filtering")
        return picks_df

    # De-vig market probabilities
    raw_home = picks_df.loc[h2h_mask, "consensus_h2h_home"].apply(american_to_implied_prob)
    raw_away = picks_df.loc[h2h_mask, "consensus_h2h_away"].apply(american_to_implied_prob)
    total_vig = raw_home + raw_away
    valid = total_vig.notna() & (total_vig > 0)

    picks_df.loc[h2h_mask & valid, "market_home_prob"] = (
        raw_home[valid] / total_vig[valid]
    ).values

    # Convert market prob to implied margin: margin = RMSE * phi_inv(prob)
    prob_col = picks_df["market_home_prob"].clip(0.001, 0.999)
    picks_df["market_implied_margin"] = np.where(
        prob_col.notna(),
        margin_rmse * norm.ppf(prob_col),
        np.nan
    )

    # Calibrate model predictions to market scale
    # Use calibration params from the model bundle (pre-computed from OOF data)
    cal_model_mean = picks_df["cal_model_mean"].iloc[0] if "cal_model_mean" in picks_df.columns else 0
    cal_model_std = picks_df["cal_model_std"].iloc[0] if "cal_model_std" in picks_df.columns else 1
    cal_market_mean = picks_df["cal_market_mean"].iloc[0] if "cal_market_mean" in picks_df.columns else 0
    cal_market_std = picks_df["cal_market_std"].iloc[0] if "cal_market_std" in picks_df.columns else 1

    if cal_model_std > 0:
        picks_df["calibrated_pred"] = (
            (picks_df["raw_margin_pred"] - cal_model_mean) / cal_model_std
            * cal_market_std + cal_market_mean
        )
    else:
        picks_df["calibrated_pred"] = picks_df["raw_margin_pred"]

    # Edge in runs
    picks_df["margin_edge"] = picks_df["calibrated_pred"] - picks_df["market_implied_margin"]

    edge_valid = picks_df["margin_edge"].notna()
    if edge_valid.any():
        log.info(f"Calibration: model std {cal_model_std:.3f} -> market std {cal_market_std:.3f}")
        log.info(f"Edge distribution: mean={picks_df.loc[edge_valid, 'margin_edge'].mean():.3f}, "
                 f"std={picks_df.loc[edge_valid, 'margin_edge'].std():.3f}")
        home_pct = (picks_df.loc[edge_valid, "margin_edge"] > 0).mean() * 100
        log.info(f"Side balance: {home_pct:.0f}% home / {100-home_pct:.0f}% away")

    return picks_df


def find_edges():
    """Identify model edges vs. market for today's games."""
    picks_path = PREDICTIONS_DIR / f"picks_{TODAY}.csv"
    if not picks_path.exists():
        log.error(f"Predictions not found: {picks_path}")
        log.error("Run 07_predict.py first")
        sys.exit(1)

    df = pd.read_csv(picks_path)
    log.info(f"Loaded {len(df)} predictions")

    # Load and match odds
    odds_df = load_odds()
    df = match_odds_to_picks(df, odds_df)

    # Get margin RMSE from predictions
    margin_rmse = df["margin_rmse"].iloc[0] if "margin_rmse" in df.columns else 4.447

    # Compute calibrated edges
    df = compute_margin_edges(df, margin_rmse)

    # Check bankroll/drawdown
    current_bankroll = get_current_bankroll()
    unit_size = current_bankroll * BANKROLL_UNIT_PCT
    drawdown_status = check_drawdown(current_bankroll)

    if drawdown_status == "PAUSE":
        log.error("Betting paused due to drawdown. No card generated.")
        return None

    log.info(f"Bankroll: ${current_bankroll:,.0f} | Unit: ${unit_size:,.0f} | Status: {drawdown_status}")

    # ── Build betting card ──
    plays = []
    all_edges = []

    for _, row in df.iterrows():
        game_pk = row.get("game_pk", "")
        home_team = row.get("home_team", "")
        away_team = row.get("away_team", "")
        edge = row.get("margin_edge", np.nan)
        calibrated_pred = row.get("calibrated_pred", np.nan)
        market_margin = row.get("market_implied_margin", np.nan)
        market_prob = row.get("market_home_prob", np.nan)

        # Total edge (raw model total - consensus total)
        total_edge = np.nan
        total_direction = ""
        raw_total = row.get("raw_total_pred", np.nan)
        consensus_total = row.get("consensus_total_line", np.nan)
        if pd.notna(raw_total) and pd.notna(consensus_total):
            total_edge = raw_total - consensus_total
            total_direction = "OVER" if total_edge > 0 else "UNDER"

        edge_row = {
            "game_pk": game_pk,
            "home_team": home_team,
            "away_team": away_team,
            "calibrated_pred": round(calibrated_pred, 3) if pd.notna(calibrated_pred) else np.nan,
            "market_implied_margin": round(market_margin, 3) if pd.notna(market_margin) else np.nan,
            "margin_edge": round(edge, 3) if pd.notna(edge) else np.nan,
            "market_home_prob": round(market_prob, 4) if pd.notna(market_prob) else np.nan,
            "consensus_h2h_home": row.get("consensus_h2h_home", np.nan),
            "consensus_h2h_away": row.get("consensus_h2h_away", np.nan),
            "raw_total_pred": round(raw_total, 1) if pd.notna(raw_total) else np.nan,
            "consensus_total": consensus_total,
            "total_edge": round(total_edge, 1) if pd.notna(total_edge) else np.nan,
            "total_direction": total_direction,
        }
        all_edges.append(edge_row)

        # ── Check if ML play qualifies ──
        if pd.isna(edge) or abs(edge) < ML_MARGIN_THRESHOLD:
            continue

        if edge > 0:
            # Model likes home more than market
            side = "HOME"
            bet_team = home_team
            odds_used = row.get("consensus_h2h_home", np.nan)
            is_dog = pd.notna(market_prob) and market_prob < 0.5
        else:
            # Model likes away more than market
            side = "AWAY"
            bet_team = away_team
            odds_used = row.get("consensus_h2h_away", np.nan)
            is_dog = pd.notna(market_prob) and market_prob >= 0.5

        if pd.isna(odds_used):
            continue

        # Unit tiers
        units = 1.0
        label = "1u"
        for tier_min, tier_units, tier_label in ML_MARGIN_UNIT_TIERS:
            if abs(edge) >= tier_min:
                units = tier_units
                label = tier_label
                break

        plays.append({
            "type": "ML",
            "game_pk": game_pk,
            "team": bet_team,
            "side": side,
            "matchup": f"{away_team} @ {home_team}",
            "calibrated_pred": round(calibrated_pred, 3),
            "market_margin": round(market_margin, 3),
            "edge_runs": round(abs(edge), 3),
            "odds": int(odds_used),
            "is_dog": is_dog,
            "units": units,
            "label": label,
            "dollar_amount": round(units * unit_size, 2),
            "margin_drivers": row.get("margin_drivers", ""),
        })

    # Save edges (all games)
    edges_df = pd.DataFrame(all_edges)
    edges_path = PREDICTIONS_DIR / f"edges_{TODAY}.csv"
    edges_df.to_csv(edges_path, index=False)
    log.info(f"Saved edge analysis to {edges_path}")

    # ── Generate betting card ──
    if plays:
        plays.sort(key=lambda x: x["edge_runs"], reverse=True)

        log.info(f"\n{'='*70}")
        log.info(f"MLB BETTING CARD -- {TODAY}")
        log.info(f"{'='*70}")
        log.info(f"Bankroll: ${current_bankroll:,.0f} | Unit: ${unit_size:,.0f}")
        log.info(f"Status: {drawdown_status}")
        log.info(f"Edge method: Calibrated margin-space (threshold >= {ML_MARGIN_THRESHOLD} runs)")
        log.info(f"{'='*70}")

        total_risk = 0
        for play in plays:
            dog_tag = " (DOG)" if play["is_dog"] else ""
            log.info(f"\n  {play['label']} {play['type']}: {play['team']} ({play['side']}){dog_tag}")
            log.info(f"    {play['matchup']}")
            log.info(f"    Model: {play['calibrated_pred']:+.2f} | "
                     f"Market: {play['market_margin']:+.2f} | "
                     f"Edge: {play['edge_runs']:.2f} runs")
            log.info(f"    Odds: {play['odds']:+d} | Risk: ${play['dollar_amount']:,.0f}")
            log.info(f"    Drivers: {play['margin_drivers']}")
            total_risk += play["dollar_amount"]

        log.info(f"\n{'='*70}")
        log.info(f"Total plays: {len(plays)} | Total risk: ${total_risk:,.0f}")
        log.info(f"{'='*70}")

        # Save JSON card
        card = {
            "date": TODAY,
            "bankroll": current_bankroll,
            "unit_size": unit_size,
            "drawdown_status": drawdown_status,
            "edge_method": "calibrated_margin_space",
            "threshold_runs": ML_MARGIN_THRESHOLD,
            "plays": plays,
        }
        card_path = PREDICTIONS_DIR / f"betting_card_{TODAY}.json"
        with open(card_path, "w") as f:
            json.dump(card, f, indent=2)
        log.info(f"Saved betting card to {card_path}")

        # Save CSV card
        card_df = pd.DataFrame(plays)
        csv_path = PREDICTIONS_DIR / f"betting_card_{TODAY}.csv"
        card_df.to_csv(csv_path, index=False)
        log.info(f"Saved CSV card to {csv_path}")

    else:
        log.info(f"\nNo qualified plays for {TODAY} (threshold: {ML_MARGIN_THRESHOLD} runs)")

    return plays


if __name__ == "__main__":
    find_edges()
