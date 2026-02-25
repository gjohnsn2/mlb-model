#!/bin/bash
# ===================================================================
# MLB Daily Betting Model -- Pipeline Orchestrator (Lasso)
# ===================================================================
# Runs the daily Lasso prediction pipeline.
#
# Usage:
#   ./run_daily.sh predict      # Full prediction pipeline (afternoon)
#   ./run_daily.sh evaluate     # Grade yesterday's picks (morning)
#   ./run_daily.sh update       # Append yesterday's data to historical files (morning)
#   ./run_daily.sh train        # Retrain production Lasso model (weekly/monthly)
#   ./run_daily.sh build        # Rebuild full historical training data
#   ./run_daily.sh backtest     # Run profitability backtest (walk-forward)
#
# Date override:
#   ./run_daily.sh --date 2026-04-15 predict
#   MLB_DATE=2026-04-15 ./run_daily.sh predict
#
# Typical daily workflow:
#   Morning (after yesterday's games complete):
#     ./run_daily.sh evaluate     # Grade yesterday's picks
#     ./run_daily.sh update       # Append yesterday's data
#
#   Afternoon (after lineups posted, ~2-3 hours before first pitch):
#     ./run_daily.sh predict      # Full prediction pipeline
#
# Prerequisites:
#   - Set environment variables (or edit config.py):
#     export ODDS_API_KEY='your_key_here'
#   - Install packages:
#     pip install scikit-learn statsapi pybaseball requests
#   - Run once: python3 06c_train_production_lasso.py
# ===================================================================

set -e  # Exit on any error
cd "$(dirname "$0")"

# Parse --date argument
if [ "$1" = "--date" ]; then
    export MLB_DATE="$2"
    shift 2
fi

DATE=${MLB_DATE:-$(date +%Y-%m-%d)}
export MLB_DATE="$DATE"
MODE=${1:-"predict"}

echo ""
echo "+=================================================+"
echo "|  MLB Lasso Model -- $DATE                  |"
echo "|  Mode: $MODE                                      |"
echo "+=================================================+"
echo ""

if [ "$MODE" = "predict" ]; then
    echo "> Step 1/6: Fetching today's schedule..."
    python3 03_fetch_schedule.py
    echo "  Done"
    echo ""

    echo "> Step 2/6: Fetching lineups..."
    python3 02e_scrape_lineups.py || echo "  Warning: Lineups failed (non-fatal, using probable SPs)"
    echo ""

    echo "> Step 3/6: Fetching odds..."
    python3 04_fetch_odds.py || echo "  Warning: Odds fetch failed (non-fatal, edges will be skipped)"
    echo ""

    echo "> Step 4/6: Building features from historical data..."
    python3 05_build_features.py
    echo "  Done"
    echo ""

    echo "> Step 5/6: Running Lasso predictions..."
    python3 07_predict.py
    echo "  Done"
    echo ""

    echo "> Step 6/6: Finding margin-space edges..."
    python3 08_find_edges.py
    echo "  Done"

elif [ "$MODE" = "evaluate" ]; then
    echo "> Evaluating yesterday's picks against results..."
    python3 09_evaluate.py
    echo "  Done"

elif [ "$MODE" = "update" ]; then
    echo "> Appending yesterday's results to historical data..."
    python3 14_update_daily_data.py
    echo "  Done"

elif [ "$MODE" = "train" ]; then
    echo "> Training production Lasso model on all historical data..."
    python3 06c_train_production_lasso.py
    echo "  Done"

elif [ "$MODE" = "build" ]; then
    echo "> Rebuilding historical training data..."
    python3 00_build_mlb_historical.py
    echo "  Done"

elif [ "$MODE" = "backtest" ]; then
    echo "> Running walk-forward experiments + backtest..."
    echo ""
    echo "> Step 1/3: Walk-forward XGBoost (no-market)..."
    python3 06_train_mlb_model.py --no-market
    echo "  Done"
    echo ""

    echo "> Step 2/3: Walk-forward Ridge/Lasso (no-market)..."
    python3 06c_ridge_lasso_experiment.py --no-market
    echo "  Done"
    echo ""

    echo "> Step 3/3: Profitability backtest (Lasso no-market)..."
    python3 10_backtest_mlb.py --no-market
    echo "  Done"

elif [ "$MODE" = "full" ]; then
    echo "> Running full pipeline: update + predict"
    echo ""

    echo "> Step 1: Appending yesterday's data..."
    python3 14_update_daily_data.py
    echo "  Done"
    echo ""

    echo "> Step 2: Fetching today's schedule..."
    python3 03_fetch_schedule.py
    echo "  Done"
    echo ""

    echo "> Step 3: Fetching lineups..."
    python3 02e_scrape_lineups.py || echo "  Warning: Lineups failed (non-fatal)"
    echo ""

    echo "> Step 4: Fetching odds..."
    python3 04_fetch_odds.py
    echo "  Done"
    echo ""

    echo "> Step 5: Building features..."
    python3 05_build_features.py
    echo "  Done"
    echo ""

    echo "> Step 6: Running Lasso predictions..."
    python3 07_predict.py
    echo "  Done"
    echo ""

    echo "> Step 7: Finding edges..."
    python3 08_find_edges.py
    echo "  Done"

else
    echo "Unknown mode: $MODE"
    echo ""
    echo "Usage: ./run_daily.sh [predict|evaluate|update|train|build|backtest|full]"
    echo ""
    echo "  predict   - Full prediction pipeline (schedule + lineups + odds + features + predict + edges)"
    echo "  evaluate  - Grade yesterday's picks"
    echo "  update    - Append yesterday's game data to historical files"
    echo "  train     - Retrain production Lasso model"
    echo "  build     - Rebuild full historical training data"
    echo "  backtest  - Run walk-forward experiments + profitability backtest"
    echo "  full      - Update data + full prediction pipeline"
    exit 1
fi

echo ""
echo "+=================================================+"
echo "|  Done! Check data/predictions/ for output.      |"
echo "+=================================================+"
echo ""
