# MLB Model — Backtests Directory

## Purpose
Stores backtest results from `10_backtest.py` — the historical profitability
analysis that determines whether the model has a real edge.

## Key Differences from CBB Backtest
- **Variable juice**: MLB moneylines have variable odds (-110 to -250+).
  P&L must use actual odds, not flat -110. This is the most critical difference.
- **Market efficiency**: MLB is a more efficient market than CBB. Expected
  edges will be smaller (3-8% vs. CBB's 10-15%).
- **Sample size per season**: 2,430 games (vs. ~5,500 CBB). Adequate for
  statistical power but each season has fewer bet opportunities.
- **Starting pitcher**: SP information quality varies. Early-season SPs have
  fewer starts and noisier statistics.

## Backtest Rules (Non-Negotiable)
1. No look-ahead bias. Pre-game data only.
2. Confirmed starting pitcher must be the actual starter.
3. Consensus (median) lines from The Odds API.
4. Walk-forward folds with per-fold Boruta.
5. P&L uses actual moneyline odds (not flat -110).
6. Postponed and suspended games excluded.
7. Leakage check in `00_build_historical.py` exits on violation.
