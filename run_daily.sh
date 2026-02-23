#!/bin/bash
# ===================================================================
# MLB Daily Betting Model -- Pipeline Orchestrator
# ===================================================================
# Runs the full daily pipeline: scrape -> features -> predict -> edges
#
# Usage:
#   ./run_daily.sh           # Full pipeline
#   ./run_daily.sh predict   # Skip scraping, just re-predict (uses cached data)
#   ./run_daily.sh evaluate  # Evaluate yesterday's picks against results
#   ./run_daily.sh train     # Retrain models (weekly)
#   ./run_daily.sh features  # Run Boruta feature selection + retrain
#   ./run_daily.sh build     # Rebuild historical training data
#   ./run_daily.sh backtest  # Run profitability backtest on walk-forward predictions
#   ./run_daily.sh monitor   # Run monitoring checks on tracking data
#
# Prerequisites:
#   - Set environment variables (or edit config.py):
#     export ODDS_API_KEY='your_key_here'
#     export WEATHER_API_KEY='your_key_here'  (optional)
#
#   - Install packages:
#     pip install scikit-learn xgboost shap boruta tabulate pybaseball
# ===================================================================

set -e  # Exit on any error
cd "$(dirname "$0")"

# Parse --date argument: ./run_daily.sh --date 2026-04-15 full
# Or use MLB_DATE env var: MLB_DATE=2026-04-15 ./run_daily.sh full
if [ "$1" = "--date" ]; then
    export MLB_DATE="$2"
    shift 2
fi

DATE=${MLB_DATE:-$(date +%Y-%m-%d)}
export MLB_DATE="$DATE"
MODE=${1:-"full"}

echo ""
echo "+=================================================+"
echo "|  MLB Daily Model -- $DATE                   |"
echo "|  Mode: $MODE                                      |"
echo "+=================================================+"
echo ""

if [ "$MODE" = "full" ]; then
    echo "> Step 1/10: Scraping FanGraphs team + pitcher stats..."
    python3 01_scrape_fangraphs.py
    echo "  Done: FanGraphs complete"
    echo ""

    echo "> Step 2/10: Scraping Statcast/Baseball Savant..."
    python3 01b_scrape_statcast.py || echo "  Warning: Statcast failed (non-fatal)"
    echo ""

    echo "> Step 3/10: Scraping park factors..."
    python3 01c_scrape_park_factors.py || echo "  Warning: Park factors failed (non-fatal)"
    echo ""

    echo "> Step 4/10: Scraping pitcher game logs..."
    python3 02b_scrape_pitcher_logs.py
    echo "  Done: Pitcher logs complete"
    echo ""

    echo "> Step 5/10: Scraping bullpen usage..."
    python3 02c_scrape_bullpen.py || echo "  Warning: Bullpen scrape failed (non-fatal)"
    echo ""

    echo "> Step 6/10: Fetching schedule + lineups..."
    python3 03_fetch_schedule.py
    python3 02e_scrape_lineups.py || echo "  Warning: Lineups failed (non-fatal)"
    echo "  Done: Schedule complete"
    echo ""

    echo "> Step 7/10: Fetching weather..."
    python3 02d_scrape_weather.py || echo "  Warning: Weather failed (non-fatal)"
    echo ""

    echo "> Step 8/10: Fetching odds..."
    python3 04_fetch_odds.py
    echo "  Done: Odds complete"
    echo ""

    echo "> Step 9/10: Building features + predictions..."
    python3 05_build_features.py
    python3 07_predict.py
    echo "  Done: Predictions complete"
    echo ""

    echo "> Step 10/10: Finding edges..."
    python3 08_find_edges.py
    echo "  Done: Edge analysis complete"

elif [ "$MODE" = "predict" ]; then
    echo "> Skipping scraping -- using cached data"
    echo ""

    echo "> Building features..."
    python3 05_build_features.py
    echo "  Done: Features complete"
    echo ""

    echo "> Running predictions..."
    python3 07_predict.py
    echo "  Done: Predictions complete"
    echo ""

    echo "> Finding edges..."
    python3 08_find_edges.py
    echo "  Done: Edge analysis complete"

elif [ "$MODE" = "evaluate" ]; then
    echo "> Evaluating predictions against actual results..."
    python3 09_evaluate.py
    echo "  Done: Evaluation complete"
    echo ""
    echo "> Running monitoring checks..."
    python3 11_monitor.py || echo "  Warning: Monitor failed (non-fatal)"
    echo "  Done: Monitoring complete"

elif [ "$MODE" = "train" ]; then
    echo "> Training models..."
    python3 06_train_model.py
    echo "  Done: Training complete"

elif [ "$MODE" = "features" ]; then
    echo "> Running Boruta feature selection..."
    python3 05b_select_features.py
    echo "  Done: Feature selection complete"
    echo ""
    echo "> Retraining with selected features..."
    python3 06_train_model.py
    echo "  Done: Retraining complete"

elif [ "$MODE" = "build" ]; then
    echo "> Rebuilding historical training data..."
    python3 00_build_historical.py
    echo "  Done: Historical data rebuilt"

elif [ "$MODE" = "backtest" ]; then
    echo "> Running profitability backtest..."
    python3 10_backtest.py
    echo "  Done: Backtest complete"

elif [ "$MODE" = "monitor" ]; then
    echo "> Running monitoring checks..."
    python3 11_monitor.py
    echo "  Done: Monitoring complete"

elif [ "$MODE" = "tune" ]; then
    echo "> Running hyperparameter tuning (this may take a while)..."
    python3 06b_tune_hyperparams.py
    echo "  Done: Tuning complete"

else
    echo "Unknown mode: $MODE"
    echo "Usage: ./run_daily.sh [full|predict|evaluate|train|features|build|backtest|monitor|tune]"
    exit 1
fi

echo ""
echo "+=================================================+"
echo "|  Done! Check data/predictions/ for output.      |"
echo "+=================================================+"
echo ""
