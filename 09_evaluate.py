"""
09 -- Evaluate Model Performance
===================================
After games complete, pulls actual scores from the MLB Stats API and compares
to model predictions. Tracks:
  - ML record (did the model's ML picks win?)
  - O/U record (did total picks hit?)
  - ROI accounting for actual ML odds (variable juice, not flat -110)

Run this the MORNING AFTER games complete.
Outputs: appends to data/tracking/performance.csv

Run:
  python3 09_evaluate.py                     # Evaluate yesterday's picks
  MLB_DATE=2025-04-16 python3 09_evaluate.py # Evaluate 2025-04-15's picks
"""

import sys
import json
import pandas as pd
import numpy as np
import requests
from config import (
    PREDICTIONS_DIR, TRACKING_DIR, TODAY,
    BANKROLL_START, BANKROLL_UNIT_PCT,
    MLB_API_BASE,
    get_logger
)

log = get_logger("09_evaluate")


def fetch_scores(date_str):
    """Pull final scores for a given date from MLB Stats API.

    Returns dict keyed by game_pk (int) with score info.
    """
    log.info(f"Fetching MLB scores for {date_str}...")

    try:
        resp = requests.get(
            f"{MLB_API_BASE}/schedule",
            params={
                "date": date_str,
                "sportId": 1,
                "hydrate": "linescore",
            },
            timeout=30,
        )
        resp.raise_for_status()
        data = resp.json()
    except Exception as e:
        log.error(f"Failed to fetch scores: {e}")
        return {}

    dates = data.get("dates", [])
    if not dates:
        log.warning(f"No games found for {date_str}")
        return {}

    games = dates[0].get("games", [])
    scores = {}
    for game in games:
        status = game.get("status", {}).get("detailedState", "")
        if "Final" not in status and "Game Over" not in status:
            continue

        game_pk = game.get("gamePk")
        game_type = game.get("gameType", "R")
        if game_type not in ("R", "F", "D", "L", "W"):
            continue

        teams = game.get("teams", {})
        home_data = teams.get("home", {})
        away_data = teams.get("away", {})

        home_score = home_data.get("score")
        away_score = away_data.get("score")
        home_name = home_data.get("team", {}).get("name", "")
        away_name = away_data.get("team", {}).get("name", "")

        if home_score is not None and away_score is not None:
            scores[game_pk] = {
                "home_team": home_name,
                "away_team": away_name,
                "home_score": int(home_score),
                "away_score": int(away_score),
                "actual_margin": int(home_score) - int(away_score),
                "actual_total": int(home_score) + int(away_score),
                "home_win": 1 if int(home_score) > int(away_score) else 0,
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

    # Fetch actual scores (keyed by game_pk)
    scores = fetch_scores(eval_date)
    if not scores:
        log.warning(f"No final scores available for {eval_date}")
        return

    # Grade each play
    results = []
    for play in plays:
        game_pk = play.get("game_pk")

        # Match by game_pk (int)
        score = scores.get(game_pk)
        if score is None:
            # Try string/int coercion
            score = scores.get(int(game_pk)) if game_pk is not None else None
        if score is None:
            # Fallback: match by team name
            for pk, s in scores.items():
                if s["home_team"] == play.get("matchup", "").split(" @ ")[-1]:
                    score = s
                    game_pk = pk
                    break
        if score is None:
            log.warning(f"No score for game_pk {game_pk} ({play.get('matchup', '?')})")
            continue

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

        else:
            continue

        result_tag = "WIN" if won else "LOSS"
        log.info(f"  {play.get('team', '?')} ({side}) {play_type}: "
                 f"{result_tag} | {score['away_score']}-{score['home_score']} | "
                 f"P&L: {pnl:+.2f}u")

        results.append({
            "date": eval_date,
            "game_pk": game_pk,
            "home_team": score["home_team"],
            "away_team": score["away_team"],
            "type": play_type,
            "side": side,
            "team": play.get("team", ""),
            "edge_runs": play.get("edge_runs", 0),
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
        total_risked = results_df["units"].sum()
        roi = total_pnl / total_risked * 100 if total_risked > 0 else 0

        log.info(f"\nResults for {eval_date}:")
        log.info(f"  Record: {wins}-{losses} ({win_rate:.1%})")
        log.info(f"  P&L: {total_pnl:+.2f} units | ROI: {roi:+.1f}%")

        # Append to tracking files
        TRACKING_DIR.mkdir(parents=True, exist_ok=True)

        # Performance log
        perf_path = TRACKING_DIR / "performance.csv"
        if perf_path.exists():
            existing = pd.read_csv(perf_path)
            # Avoid duplicates for the same date
            existing = existing[existing["date"] != eval_date]
            combined = pd.concat([existing, results_df], ignore_index=True)
        else:
            combined = results_df
        combined.to_csv(perf_path, index=False)
        log.info(f"Updated {perf_path}")

        # Bet ledger
        ledger_path = TRACKING_DIR / "bet_ledger.csv"
        if ledger_path.exists():
            existing = pd.read_csv(ledger_path)
            existing = existing[existing["date"] != eval_date]
            combined = pd.concat([existing, results_df], ignore_index=True)
        else:
            combined = results_df
        combined.to_csv(ledger_path, index=False)

        # Update bankroll
        bankroll_path = TRACKING_DIR / "bankroll.csv"
        if bankroll_path.exists():
            br_df = pd.read_csv(bankroll_path)
            # Remove any existing entry for this date
            br_df = br_df[br_df["date"] != eval_date]
            current_bankroll = br_df.iloc[-1]["bankroll"] if len(br_df) > 0 else BANKROLL_START
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

    else:
        log.warning(f"Could not grade any plays for {eval_date}")


if __name__ == "__main__":
    evaluate()
