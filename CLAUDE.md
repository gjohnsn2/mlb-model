# MLB Gambling Model — Project CLAUDE.md

## Maintenance Protocol
After any session with significant decisions, data quirks, or architecture changes —
update this file and relevant memory files before closing. Add a dated entry to the
Change Log at the bottom.

## What This Project Is
A Major League Baseball gambling model targeting full-game moneyline and totals.
Built using XGBoost + walk-forward validation + Boruta feature selection.

**Current phase**: RESEARCH — Ridge/Lasso finds real baseball signal.
Rate stat bug fixed (all baseball features were NaN before). Now 10 baseball
features survive Boruta (XGBoost) and 39 survive Lasso. Ridge/Lasso beats XGBoost
on no-market features: +7.8% ROI at >=1.5 threshold (p=0.005).
Caution: 2025 season is -14.6% at that threshold (8/9 seasons positive).

## Project Structure
```
mlb-model/
├── config.py                      # Central config (paths, credentials, features, params)
├── feature_engine.py              # Feature engine (template, shared utility)
├── 00_build_mlb_historical.py     # Build training data from historical archives
├── 06_train_mlb_model.py          # XGBoost walk-forward training + Boruta
├── 06c_ridge_lasso_experiment.py  # Ridge/Lasso walk-forward (the breakthrough)
├── 10_backtest_mlb.py             # Historical profitability backtest
├── 10b_backtest_f5_nrfi.py        # F5 + NRFI backtest
├── 11_segmented_backtest.py       # Segmented backtest (find profitable subsets)
├── 12c_fetch_mlb_pinnacle.py      # Fetch Pinnacle H2H + totals (eu region)
├── 12d_validate_mlb_pinnacle.py   # Pinnacle validation analysis
├── scripts/
│   ├── fetch_historical_games.py  # Fetch game results from MLB Stats API
│   ├── fetch_bullpen_data.py      # Fetch per-reliever game logs (169K rows)
│   ├── fetch_batter_data.py       # Fetch per-batter boxscore data (598K rows)
│   ├── fetch_player_handedness.py # Fetch bat/pitch handedness (4K players)
│   ├── fetch_historical_statcast.py # Fetch Statcast metrics per pitcher (incl. handedness splits, pitch mix)
│   ├── fetch_batter_pitch_type_stats.py # Fetch batter pitch-type stats (2019-2025)
│   ├── merge_pitcher_logs.py      # Merge MLB API + Statcast pitcher logs
│   ├── integrate_historical_mlb_odds.py # Integrate Sports-Statistics odds
│   └── experiment_market_free_mlb.py    # Market-free feature experiment
├── data/
│   ├── mlb_stadium_locations.csv  # Static: 48 venues with lat/lon/timezone
│   ├── park_factors.csv           # Static reference (template)
│   ├── venue_locations.csv        # Static reference (template)
│   └── historical/                # Training data + archives (git-ignored)
├── models/
│   ├── mlb_selected_features.json # Boruta-selected features
│   ├── mlb_training_metrics.json  # Walk-forward RMSE, fold metrics
│   ├── mlb_walkforward_report.txt # Walk-forward fold report
│   ├── mlb_backtest_report.txt    # Backtest summary
│   ├── mlb_pinnacle_validation_report.txt # Pinnacle validation
│   └── trained/                   # .pkl model files (git-ignored)
├── 00_build_historical.py         # Template (original bootstrap)
├── 01_scrape_fangraphs.py         # Template for future daily pipeline
│   ... (01b-12b template scripts)
├── run_daily.sh                   # Pipeline orchestrator (template)
└── requirements.txt
```

## Key Commands
```bash
# Full rebuild + train + backtest pipeline
python3 00_build_mlb_historical.py    # Rebuild features (25K games)
python3 06_train_mlb_model.py         # Walk-forward + Boruta (XGBoost)
python3 06_train_mlb_model.py --no-market  # Baseball-only XGBoost
python3 06c_ridge_lasso_experiment.py --no-market  # Ridge/Lasso (best model)
python3 10_backtest_mlb.py            # Profitability backtest
python3 10_backtest_mlb.py --no-market # Backtest no-market OOF predictions

# Data fetching (run once, resumable)
python3 scripts/fetch_historical_games.py     # ~4 hours
python3 scripts/fetch_bullpen_data.py         # ~36 minutes
python3 scripts/fetch_batter_data.py          # ~35 minutes
python3 scripts/fetch_player_handedness.py    # ~37 seconds
python3 scripts/fetch_historical_statcast.py  # ~2 hours

# Pinnacle validation
python3 12c_fetch_mlb_pinnacle.py     # Fetch Pinnacle odds (~25K credits)
python3 12d_validate_mlb_pinnacle.py  # Analyze edge vs Pinnacle
```

## Data Sources
| Source | Auth | Script | Notes |
|--------|------|--------|-------|
| **MLB Stats API** | None (free) | `scripts/fetch_*` | Schedule, boxscores, linescore, SPs, umpires, weather |
| **Statcast** | None (pybaseball) | `scripts/fetch_historical_statcast.py` | xwOBA, barrel%, whiff rate per pitcher |
| **Odds API** | `ODDS_API_KEY` | `12c_` | H2H, totals from 15+ books + Pinnacle (eu region) |
| **Sports-Statistics** | None | `scripts/integrate_historical_mlb_odds.py` | Single-book odds 2015-2021 (backfill) |

## Core Modeling Decisions
- **Best model**: Ridge/Lasso on baseball-only features (no market inputs)
- **XGBoost**: Still used for Boruta feature selection; Ridge/Lasso outperforms it
- **Targets**: Home margin and combined score
- **Training**: 25,611 games (2015-2025, 11 seasons), 91 candidate features
- **Feature selection**: Boruta for XGBoost, L1 penalty for Lasso (39 features survive)
- **Validation**: Walk-forward 9-fold (test seasons 2017-2025), per-fold selection
- **Sample weighting**: Exponential decay (half_life=3yr)
- **Edge thresholds**: ML margin edge >=1.5 runs (calibrated margin space)
- **Sign convention**: Positive margin = home team wins

## Current Performance (2026-02-24)

### Ridge/Lasso No-Market (Best Model)
| Metric | XGBoost | Ridge | Lasso |
|--------|---------|-------|-------|
| RMSE | 4.524 | 4.447 | 4.447 |
| corr(model, actual) | 0.111 | 0.176 | 0.179 |
| corr(edge, mkt_resid) | 0.049 | 0.065 | 0.064 |
| ROI >= 0.5 | -2.3% | -0.0% | +0.3% |
| ROI >= 1.5 | -1.6% | +6.2% (p=.018) | **+7.8% (p=.005)** |
| ROI >= 2.0 | +1.5% | +12.4% (p=.004) | +12.7% (p=.004) |

### Lasso Stable Features (9/9 folds, no-market)
`sp_season_ip_diff`, `sp_k_pct_diff`, `bb_rate_diff`, `team_run_diff_10_diff`,
`bullpen_whip_diff`, `lineup_power_diff`, `star_missing_ops_diff`,
`lineup_top_heavy_diff`, `lineup_bb_k_ratio_diff`

**39 features survive Lasso (>= 4/9 folds)** vs 8 for XGBoost+Boruta.
Signal is real but linear — XGBoost can't find strong enough splits.

### Lasso No-Market ROI by Season (>= 1.5 runs)
| Season | Bets | ROI |
|--------|------|-----|
| 2017-2019 | 673 | +9.5% |
| 2020 | 114 | +16.8% |
| 2021 | 242 | +14.3% |
| 2022 | 226 | +11.7% |
| 2023 | 283 | +7.0% |
| 2024 | 282 | +2.5% |
| **2025** | **244** | **-14.6%** |

**CAUTION**: 2025 is strongly negative. 8/9 seasons positive but latest season fails.

### XGBoost Market-Only (Reference)
Still beats Pinnacle (+6.9% ROI) but NOT clean consensus (-2.9%).
The edge is consensus-vs-Pinnacle disagreement, not model alpha.

## 91 Candidate Features (No-Market)
| Category | Count | Boruta (XGB) | Lasso (>=4/9) |
|----------|-------|-------------|---------------|
| SP season stats | 15 | 3 | 8 |
| SP velocity/command (Statcast) | 5 | 0 | 2 |
| SP recency (last 3) | 10 | 1 | 5 |
| Batting | 6 | 1 | 4 |
| Context (park, ump, weather, dome) | 7 | 2 | 3 |
| Rest/Workload | 3 | 1 | 2 |
| Momentum | 2 | 1 | 2 |
| Bullpen | 5 | 0 | 3 |
| Bullpen availability | 4 | 0 | 1 |
| Travel & fatigue | 5 | 0 | 1 |
| Schedule context | 3 | 0 | 1 |
| Lineup composition | 11 | 3 | 5 |
| Opponent-adjusted | 4 | 0 | 2 |
| Handedness splits | 4 | 0 | 0 |
| Pitch-type matchups | 5 | 0 | 0 |
| Interaction features | 4 | 0 | 0 |

## Dead Ends (Experiments Tried and Discarded)
- **Residual target Ridge (06d)**: Trained Ridge/Lasso to predict `actual_margin - market_implied_margin`
  directly (the market's error). Hypothesis was that skipping calibration would be cleaner.
  Result: predictions collapsed to ~0 (std=0.075 vs actual residual std=4.4). Model correctly
  learns market errors are unpredictable, so it predicts nothing. Baseline Ridge (06c) with
  calibration has `corr(edge, residual) = 0.100` vs residual model's 0.013. Calibration
  *amplifies* weak signal; direct residual prediction *suppresses* it. Script deleted.
- **F5 NRFI model**: Brier skill score -0.009 (worse than naive base rate). Not viable.
- **Segmented backtest**: No profitable segments for XGBoost at any threshold.
- **Handedness splits / pitch-type matchups / interaction features**: 0 survived either Boruta or Lasso.

## Known Data Quirks
- **Corrupt consensus H2H**: ~100 games with |ML| < 100 (values 0, -1, -2). Creates
  absurd payouts. Filter: `abs(consensus_h2h) >= 100`. Previously inflated ROI from -2.9% to +62.7%.
- **7-inning doubleheaders**: 2020-2022 only. Excluded from total model.
- **Odds clipping**: H2H values >|1000| set to NaN.
- **Sports-Statistics backfill**: Single-book odds for 2015-2021 (13K games).
- **Park factor normalization**: Uses home/road split (venue avg / home team's road avg)
  to isolate venue effect from team quality. Umpire factor is park-adjusted.
- **Stadium locations**: 48 venues including neutral sites (Tokyo, London, Monterrey).
- **Player handedness**: 4,019 players (65% R bat, 30% L, 6% S; 79% R pitch, 21% L).

## Backtest Rules (Non-Negotiable)
- No look-ahead. Pre-game data only.
- Consensus (median) lines with variable ML juice.
- Walk-forward 9-fold with per-fold Boruta.
- |ML| >= 100 filter on consensus H2H (corrupt data exclusion).
- Leakage check in `00_build_mlb_historical.py`.

## Historical Data Files
| File | Rows | Description |
|------|------|-------------|
| `game_results_all.csv` | 25,611 | Games with scores, F5, NRFI, SPs, venue, weather, umpires |
| `pitcher_game_logs_mlbapi.csv` | 51,222 | SP counting stats per game |
| `team_batting_game_logs.csv` | 51,222 | Team batting per game |
| `bullpen_game_logs.csv` | 169,050 | Per-reliever per-game stats |
| `statcast_pitcher_games.csv` | ~50K | Statcast metrics (xwOBA, barrel%, etc.) |
| `batter_game_logs.csv` | 597,714 | Per-batter per-game stats |
| `player_handedness.csv` | 4,019 | Bat side (L/R/S) + pitch hand (L/R) |
| `historical_mlb_odds.csv` | ~25K | Odds API data (2020+) |
| `sports_statistics_mlb_odds.csv` | ~13K | Single-book backfill (2015-2021) |
| `pinnacle_mlb_odds.csv` | 19,127 | Pinnacle H2H + totals (76% H2H coverage) |
| `training_data_mlb_v2.csv` | 25,611 | Assembled training matrix (69+ features) |

## Change Log
| Date | Change | Impact |
|------|--------|--------|
| 2026-02-22 | Repository created | Initial structure |
| 2026-02-23 | Data source audit, MLB Stats API as backbone | Pipeline architecture |
| 2026-02-23 | Historical data fetch (25K games, 51K pitcher logs, 51K batting logs) | Training data |
| 2026-02-23 | Feature engineering: 53 candidate features (SP, batting, market, context, rest, momentum) | First model |
| 2026-02-23 | Walk-forward training + Boruta: only 3 market features survive | Model is market regurgitator |
| 2026-02-23 | Backtest: -2.9% ROI vs consensus (corrupt data fix: was +62.7%) | No edge |
| 2026-02-23 | Pinnacle validation: +6.9% ROI vs Pinnacle (p=0.000) | Edge vs Pinnacle only |
| 2026-02-23 | Bullpen data fetch (169K rows) + 3 bullpen features | None survived Boruta |
| 2026-02-23 | Player-level features: batter data (598K), handedness (4K), 20 new features | None survived Boruta |
| 2026-02-23 | Bullpen availability, travel/fatigue, schedule context, lineup composition | 69 total candidates, 0 baseball features survive |
| 2026-02-23 | Park factor fix: home/road normalization, umpire factor: park-adjusted | Methodological correctness |
| 2026-02-24 | Fixed rate stat bug: ERA/FIP/WHIP/K% were never computed (all NaN) | All baseball features now populated |
| 2026-02-24 | Expanded to 91 candidate features (Statcast velo/command, opponent-adj, handedness, pitch-type, interactions) | More signal candidates |
| 2026-02-24 | Ridge/Lasso breakthrough: +7.8% ROI at >=1.5 (Lasso, no-market, p=0.005) | Real baseball signal found |
| 2026-02-24 | Residual target Ridge experiment — tried and discarded (calibration is better) | Dead end documented |
