# MLB Model — Tracking Directory

## Files
- `bankroll.csv`: Daily bankroll state (date, bankroll, daily_pnl, n_plays, wins, losses)
- `bet_ledger.csv`: Per-bet details (game_id, type, side, odds, units, won, pnl_units)
- `performance.csv`: Game-level evaluation (predictions vs. actual results)
- `monitoring.csv`: Rolling metrics, PSI scores, alert history

## Backtest Methodology
- Walk-forward validation with per-fold Boruta feature selection
- No look-ahead: features use only pre-game data
- Historical odds from The Odds API (consensus = median across books)
- P&L calculated with ACTUAL ML odds (not flat -110) — this is critical for MLB
  because moneyline juice is highly variable (-110 to -250+)
- Pushes excluded from win rate (rare in ML, more common in run lines)

## Evaluation Mechanics
- `09_evaluate.py` runs the morning after games complete
- Fetches final scores from ESPN MLB scoreboard
- Grades each play from the betting card against actual results
- Updates bankroll.csv, bet_ledger.csv, and performance.csv
- `11_monitor.py` computes rolling metrics and fires alerts

## ML P&L Calculation
MLB P&L requires actual moneyline odds (not flat -110):
- Underdog (+150): Win $150 on $100 bet. Profit = units * (odds/100)
- Favorite (-150): Win $100 on $150 bet. Profit = units * (100/abs(odds))
- Loss: -units regardless of odds
- ROI must account for the asymmetry (favorites cost more to lose)

## Red Flags to Watch
- Win rate below 52% over 100+ bets (below breakeven for typical ML juice)
- Edge-calibration slope not significant (p > 0.10)
- SP scratch rate affecting more than 10% of bets
- Systematic bias in weather-adjusted totals
- Conference/division-specific underperformance
