"""
08 -- Find Edges & Daily Betting Card
========================================
Compares model predictions to market lines, computes per-book edges,
assigns unit tiers, and produces an actionable daily betting card.

MLB-specific: Primary market is moneylines (not spreads). The model's
predicted margin is converted to a win probability and compared to
market-implied probability from the consensus moneyline.

Edge = model_win_prob - market_implied_prob

Outputs:
  data/predictions/edges_YYYY-MM-DD.csv
  data/predictions/betting_card_YYYY-MM-DD.json
  data/predictions/betting_card_YYYY-MM-DD.csv
"""

import sys
import json
import pandas as pd
import numpy as np
from scipy.stats import norm
from config import (
    PREDICTIONS_DIR, TRACKING_DIR, TODAY,
    ML_EDGE_THRESHOLD, RUNLINE_EDGE_THRESHOLD, TOTAL_EDGE_THRESHOLD,
    ML_UNIT_TIERS, RUNLINE_UNIT_TIERS, TOTAL_UNIT_TIERS,
    BOOK_DISPLAY_NAMES,
    BANKROLL_START, BANKROLL_UNIT_PCT,
    DRAWDOWN_WARNING, DRAWDOWN_PAUSE,
    MAX_ML_PRICE, MIN_IMPLIED_PROB,
    get_logger
)

log = get_logger("08_edges")


def compute_ml_edge(model_win_prob, market_implied_prob):
    """Compute moneyline edge as model probability minus market probability."""
    if pd.isna(model_win_prob) or pd.isna(market_implied_prob):
        return np.nan
    return model_win_prob - market_implied_prob


def compute_ml_unit_size(edge_abs):
    """Map probability edge to unit size using ML_UNIT_TIERS."""
    for threshold, units, label in ML_UNIT_TIERS:
        if edge_abs >= threshold:
            return units, label
    return 0.0, "0u"


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
        log.error(f"DRAWDOWN PAUSE: {drawdown:.1%} from peak (${peak:,.0f} -> ${current_bankroll:,.0f})")
        log.error("All betting suspended until drawdown recovers below 25%")
        return "PAUSE"
    elif drawdown >= DRAWDOWN_WARNING:
        log.warning(f"DRAWDOWN WARNING: {drawdown:.1%} from peak")
        return "WARNING"
    else:
        return "OK"


def find_edges():
    """Identify model edges vs. market for today's games."""
    picks_path = PREDICTIONS_DIR / f"picks_{TODAY}.csv"
    if not picks_path.exists():
        log.error(f"Predictions not found: {picks_path}")
        log.error("Run 07_predict.py first")
        sys.exit(1)

    df = pd.read_csv(picks_path)
    log.info(f"Loaded {len(df)} predictions")

    # Check bankroll/drawdown
    current_bankroll = get_current_bankroll()
    unit_size = current_bankroll * BANKROLL_UNIT_PCT
    drawdown_status = check_drawdown(current_bankroll)

    if drawdown_status == "PAUSE":
        log.error("Betting paused due to drawdown. No card generated.")
        return None

    log.info(f"Bankroll: ${current_bankroll:,.0f} | Unit: ${unit_size:,.0f} | Status: {drawdown_status}")

    # Compute edges
    plays = []
    all_edges = []

    for _, row in df.iterrows():
        game_id = row.get("game_id", "")
        home_team = row.get("home_team", "")
        away_team = row.get("away_team", "")
        model_margin = row.get("model_margin", np.nan)
        model_total = row.get("model_total", np.nan)
        model_wp_home = row.get("model_win_prob_home", np.nan)
        model_wp_away = row.get("model_win_prob_away", np.nan)
        market_ip_home = row.get("ml_implied_prob_home", np.nan)
        market_ip_away = row.get("ml_implied_prob_away", np.nan)
        consensus_ml_home = row.get("consensus_ml_home", np.nan)
        consensus_ml_away = row.get("consensus_ml_away", np.nan)
        consensus_total = row.get("consensus_total", np.nan)

        # ML edge (home side)
        ml_edge_home = compute_ml_edge(model_wp_home, market_ip_home)
        ml_edge_away = compute_ml_edge(model_wp_away, market_ip_away)

        # Determine best side
        if not pd.isna(ml_edge_home) and not pd.isna(ml_edge_away):
            if ml_edge_home >= ml_edge_away:
                ml_edge = ml_edge_home
                ml_side = "HOME"
                ml_team = home_team
                ml_odds = consensus_ml_home
                ml_model_prob = model_wp_home
            else:
                ml_edge = ml_edge_away
                ml_side = "AWAY"
                ml_team = away_team
                ml_odds = consensus_ml_away
                ml_model_prob = model_wp_away
        else:
            ml_edge = np.nan
            ml_side = ""
            ml_team = ""
            ml_odds = np.nan
            ml_model_prob = np.nan

        # Total edge
        total_edge = np.nan
        total_direction = ""
        if not pd.isna(model_total) and not pd.isna(consensus_total):
            total_edge = model_total - consensus_total
            total_direction = "OVER" if total_edge > 0 else "UNDER"

        edge_row = {
            "game_id": game_id,
            "home_team": home_team,
            "away_team": away_team,
            "model_margin": model_margin,
            "model_total": model_total,
            "model_wp_home": model_wp_home,
            "model_wp_away": model_wp_away,
            "consensus_ml_home": consensus_ml_home,
            "consensus_ml_away": consensus_ml_away,
            "consensus_total": consensus_total,
            "ml_edge": ml_edge,
            "ml_side": ml_side,
            "ml_team": ml_team,
            "ml_odds": ml_odds,
            "total_edge": total_edge,
            "total_direction": total_direction,
        }
        all_edges.append(edge_row)

        # Check if ML play qualifies
        if not pd.isna(ml_edge) and ml_edge >= ML_EDGE_THRESHOLD:
            # Price filter: skip very heavy favorites
            if not pd.isna(ml_odds) and ml_odds < MAX_ML_PRICE:
                continue

            units, label = compute_ml_unit_size(ml_edge)
            if units > 0:
                plays.append({
                    "type": "ML",
                    "game_id": game_id,
                    "team": ml_team,
                    "side": ml_side,
                    "matchup": f"{away_team} @ {home_team}",
                    "model_prob": round(ml_model_prob, 4),
                    "market_prob": round(market_ip_home if ml_side == "HOME" else market_ip_away, 4),
                    "edge": round(ml_edge, 4),
                    "odds": ml_odds,
                    "units": units,
                    "label": label,
                    "dollar_amount": round(units * unit_size, 2),
                    "margin_shap": row.get("margin_shap", ""),
                })

    # Save edges (all games)
    edges_df = pd.DataFrame(all_edges)
    edges_path = PREDICTIONS_DIR / f"edges_{TODAY}.csv"
    edges_df.to_csv(edges_path, index=False)
    log.info(f"Saved edge analysis to {edges_path}")

    # Generate betting card
    if plays:
        plays.sort(key=lambda x: x["edge"], reverse=True)

        # Console output
        log.info(f"\n{'='*70}")
        log.info(f"MLB BETTING CARD -- {TODAY}")
        log.info(f"{'='*70}")
        log.info(f"Bankroll: ${current_bankroll:,.0f} | Unit: ${unit_size:,.0f}")
        log.info(f"Status: {drawdown_status}")
        log.info(f"{'='*70}")

        total_risk = 0
        for play in plays:
            log.info(f"\n  {play['label']} {play['type']}: {play['team']} ({play['side']})")
            log.info(f"    {play['matchup']}")
            log.info(f"    Model: {play['model_prob']:.1%} | Market: {play['market_prob']:.1%} | "
                      f"Edge: {play['edge']:.1%}")
            log.info(f"    Odds: {play['odds']:+.0f} | Risk: ${play['dollar_amount']:,.0f}")
            log.info(f"    SHAP: {play['margin_shap']}")
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
        log.info(f"\nNo qualified plays for {TODAY}")

    return plays


if __name__ == "__main__":
    find_edges()
