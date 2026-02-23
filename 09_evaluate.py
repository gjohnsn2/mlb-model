"""
09 -- Evaluate Model Performance
===================================
After games complete, pulls actual scores from ESPN and compares
to model predictions. Tracks:
  - ML record (did the model's ML picks win?)
  - Run line record
  - O/U record (did total picks hit?)
  - ROI accounting for actual ML odds (variable juice, not flat -110)
  - CLV (closing line value)

Run this the MORNING AFTER games complete.
Outputs: appends to data/tracking/performance.csv
"""

import sys
import json
import pandas as pd
import numpy as np
import requests
from config import (
    PREDICTIONS_DIR, TRACKING_DIR, RAW_DIR, TODAY,
    ESPN_SCOREBOARD_URL,
    BANKROLL_START, BANKROLL_UNIT_PCT,
    DRAWDOWN_WARNING, DRAWDOWN_PAUSE,
    get_logger
)

log = get_logger("09_evaluate")


def fetch_scores(date_str):
    """Pull final scores for a given date from ESPN MLB."""
    api_date = date_str.replace("-", "")

    log.info(f"Fetching MLB scores for {date_str}...")
    resp = requests.get(ESPN_SCOREBOARD_URL, params={
        "dates": api_date,
        "limit": 50,
    }, timeout=30)
    resp.raise_for_status()

    data = resp.json()
    events = data.get("events", [])

    scores = {}
    for event in events:
        game_id = event.get("id")
        status = event.get("status", {}).get("type", {}).get("name", "")

        if status != "STATUS_FINAL":
            continue

        comp = event.get("competitions", [{}])[0]
        home_score = away_score = None
        home_name = away_name = ""

        for team_data in comp.get("competitors", []):
            name = team_data.get("team", {}).get("displayName", "")
            score = int(team_data.get("score", 0))
            if team_data.get("homeAway") == "home":
                home_score = score
                home_name = name
            else:
                away_score = score
                away_name = name

        if home_score is not None and away_score is not None:
            scores[str(game_id)] = {
                "home_team": home_name,
                "away_team": away_name,
                "home_score": home_score,
                "away_score": away_score,
                "actual_margin": home_score - away_score,
                "actual_total": home_score + away_score,
                "home_win": 1 if home_score > away_score else 0,
            }

    log.info(f"Retrieved {len(scores)} final scores")
    return scores


def ml_payout(odds, units):
    """Calculate payout for a moneyline bet.
    Returns profit (positive) or loss (negative).
    """
    if pd.isna(odds):
        return 0

    if odds > 0:
        # Underdog: +150 means bet $100 to win $150
        profit = units * (odds / 100)
    else:
        # Favorite: -150 means bet $150 to win $100
        profit = units * (100 / abs(odds))

    return profit


def evaluate(eval_date=None):
    """Evaluate predictions against actual results."""
    if eval_date is None:
        # Evaluate yesterday by default
        from datetime import datetime, timedelta
        eval_date = (datetime.strptime(TODAY, "%Y-%m-%d") - timedelta(days=1)).strftime("%Y-%m-%d")

    log.info(f"Evaluating predictions for {eval_date}...")

    # Load betting card
    card_path = PREDICTIONS_DIR / f"betting_card_{eval_date}.json"
    if not card_path.exists():
        log.warning(f"No betting card found for {eval_date}")
        return

    with open(card_path) as f:
        card = json.load(f)

    plays = card.get("plays", [])
    if not plays:
        log.info(f"No plays on {eval_date}")
        return

    # Fetch actual scores
    scores = fetch_scores(eval_date)
    if not scores:
        log.warning(f"No final scores available for {eval_date}")
        return

    # Grade each play
    results = []
    for play in plays:
        game_id = str(play.get("game_id", ""))
        if game_id not in scores:
            log.warning(f"No score for game {game_id}")
            continue

        score = scores[game_id]
        play_type = play.get("type", "ML")
        side = play.get("side", "")
        units = play.get("units", 1)
        odds = play.get("odds", -110)

        # Determine if the play won
        if play_type == "ML":
            if side == "HOME":
                won = score["home_win"] == 1
            else:
                won = score["home_win"] == 0

            # Calculate P&L using actual ML odds
            if won:
                pnl = ml_payout(odds, units)
            else:
                pnl = -units  # Lost the stake

        elif play_type == "TOTAL":
            direction = play.get("total_direction", "OVER")
            if direction == "OVER":
                won = score["actual_total"] > play.get("total_line", 0)
            else:
                won = score["actual_total"] < play.get("total_line", 0)
            # Standard -110 juice for totals
            pnl = units * (100 / 110) if won else -units

        results.append({
            "date": eval_date,
            "game_id": game_id,
            "home_team": score["home_team"],
            "away_team": score["away_team"],
            "type": play_type,
            "side": side,
            "team": play.get("team", ""),
            "edge": play.get("edge", 0),
            "odds": odds,
            "units": units,
            "won": int(won),
            "pnl_units": round(pnl, 3),
            "actual_margin": score["actual_margin"],
            "actual_total": score["actual_total"],
        })

    if results:
        results_df = pd.DataFrame(results)

        # Summary
        n_plays = len(results_df)
        wins = results_df["won"].sum()
        losses = n_plays - wins
        win_rate = wins / n_plays if n_plays > 0 else 0
        total_pnl = results_df["pnl_units"].sum()

        log.info(f"\nResults for {eval_date}:")
        log.info(f"  Record: {wins}-{losses} ({win_rate:.1%})")
        log.info(f"  P&L: {total_pnl:+.2f} units")

        # Append to tracking files
        TRACKING_DIR.mkdir(parents=True, exist_ok=True)

        # Performance log
        perf_path = TRACKING_DIR / "performance.csv"
        if perf_path.exists():
            existing = pd.read_csv(perf_path)
            combined = pd.concat([existing, results_df], ignore_index=True)
        else:
            combined = results_df
        combined.to_csv(perf_path, index=False)
        log.info(f"Updated {perf_path}")

        # Bet ledger
        ledger_path = TRACKING_DIR / "bet_ledger.csv"
        if ledger_path.exists():
            existing = pd.read_csv(ledger_path)
            combined = pd.concat([existing, results_df], ignore_index=True)
        else:
            combined = results_df
        combined.to_csv(ledger_path, index=False)

        # Update bankroll
        bankroll_path = TRACKING_DIR / "bankroll.csv"
        if bankroll_path.exists():
            br_df = pd.read_csv(bankroll_path)
            current_bankroll = br_df.iloc[-1]["bankroll"]
        else:
            br_df = pd.DataFrame()
            current_bankroll = BANKROLL_START

        unit_value = current_bankroll * BANKROLL_UNIT_PCT
        daily_pnl = total_pnl * unit_value
        new_bankroll = current_bankroll + daily_pnl

        new_row = pd.DataFrame([{
            "date": eval_date,
            "bankroll": round(new_bankroll, 2),
            "daily_pnl": round(daily_pnl, 2),
            "daily_pnl_units": round(total_pnl, 3),
            "n_plays": n_plays,
            "wins": wins,
            "losses": losses,
        }])

        if not br_df.empty:
            br_df = pd.concat([br_df, new_row], ignore_index=True)
        else:
            br_df = new_row
        br_df.to_csv(bankroll_path, index=False)
        log.info(f"Bankroll: ${current_bankroll:,.0f} -> ${new_bankroll:,.0f} ({daily_pnl:+,.0f})")


if __name__ == "__main__":
    evaluate()
