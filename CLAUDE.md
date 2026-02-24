# MLB Gambling Model — Project CLAUDE.md

## Maintenance Protocol
After any session with significant decisions, data quirks, or architecture changes —
update this file and relevant memory files before closing. Add a dated entry to the
Change Log at the bottom.

## What This Project Is
A Major League Baseball gambling model targeting full-game moneyline and totals.
Built using the same XGBoost + walk-forward + Boruta architecture as the CBB model.

**Current phase**: RESEARCH COMPLETE — model is purely market-driven.
No baseball feature (SP, batting, bullpen, park, weather, lineup) survives Boruta.
Model beats Pinnacle (+6.9% ROI) but NOT clean consensus (-2.9%).
Betting is paused pending discovery of non-market features that survive selection.

## Project Structure
```
mlb-model/
├── config.py                      # Central config (paths, credentials, features, params)
├── feature_engine.py              # Feature engine (template, shared utility)
├── 00_build_mlb_historical.py     # Build training data from historical archives
├── 06_train_mlb_model.py          # XGBoost walk-forward training + Boruta
├── 10_backtest_mlb.py             # Historical profitability backtest
├── 12c_fetch_mlb_pinnacle.py      # Fetch Pinnacle H2H + totals (eu region)
├── 12d_validate_mlb_pinnacle.py   # Pinnacle validation analysis
├── scripts/
│   ├── fetch_historical_games.py  # Fetch game results from MLB Stats API
│   ├── fetch_bullpen_data.py      # Fetch per-reliever game logs (169K rows)
│   ├── fetch_batter_data.py       # Fetch per-batter boxscore data (598K rows)
│   ├── fetch_player_handedness.py # Fetch bat/pitch handedness (4K players)
│   ├── fetch_historical_statcast.py # Fetch Statcast metrics per pitcher
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
python3 06_train_mlb_model.py         # Walk-forward + Boruta
python3 10_backtest_mlb.py            # Profitability backtest

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
- **Model**: XGBoost Regressor (separate margin + total models)
- **Targets**: Home margin and combined score
- **Training**: 25,611 games (2015-2025, 11 seasons)
- **Feature selection**: Boruta (100 iterations, alpha=0.05)
- **Validation**: Walk-forward 9-fold (test seasons 2017-2025), per-fold Boruta
- **Sample weighting**: Exponential decay (half_life=3yr)
- **Edge thresholds**: ML prob edge >=5%, Total edge >=1.5 runs
- **Sign convention**: Positive margin = home team wins

## Current Performance (2026-02-23)

### Walk-Forward Results
| Model | RMSE | MAE | OOF Samples | Features |
|-------|------|-----|-------------|----------|
| **Margin** | 4.49 | 3.48 | 20,688 | 4 |
| **Total** | 4.37 | 3.41 | 20,319 | 3 |

### Selected Features (Boruta)
**Margin**: `market_implied_prob` (9/9), `market_logit` (9/9), `num_books` (5/9), `consensus_total` (4/9)
**Total**: `consensus_total` (9/9), `temp` (7/9), `num_books` (4/9)

**CRITICAL**: Model is purely market-driven. Zero of 69 baseball features survive Boruta.
`lineup_k_rate_diff` appeared in 3/9 total folds (closest, but not enough).

### Backtest (Clean Consensus, |ML| >= 100 filter)
| Threshold | Bets | W-L | Win% | ROI |
|-----------|------|-----|------|-----|
| >= 0.00 | 18,065 | 8766-9299 | 48.5% | -2.8% |
| >= 0.50 (prod) | 6,983 | 3260-3723 | 46.7% | **-2.9%** |
| >= 1.00 | 2,234 | 951-1283 | 42.6% | -2.5% |

**No edge against clean consensus at any threshold.**

### Pinnacle Validation
| Threshold | Consensus ROI | Pinnacle ROI |
|-----------|--------------|--------------|
| >= 0.00 | -2.9% | +3.5% |
| >= 0.50 (prod) | **-3.7%** | **+6.9%** (p=0.000) |
| >= 1.00 | -6.6% | +18.0% |

**Model beats Pinnacle but NOT clean consensus.** The edge is consensus-vs-Pinnacle
disagreement flowing through a consensus-correlated model.

Pinnacle ROI by season (>= 0.5 runs):
| Season | Bets | Win% | ROI |
|--------|------|------|-----|
| 2020 | 289 | 51.9% | +15.1% |
| 2021 | 667 | 51.7% | +12.9% |
| 2022 | 698 | 51.4% | +10.4% |
| 2023 | 711 | 53.4% | +9.1% |
| **2024** | **646** | **44.6%** | **-7.6%** |
| 2025 | 631 | 50.7% | +5.4% |

## 69 Candidate Features Tested
| Category | Count | Survived |
|----------|-------|----------|
| SP season stats | 14 | 0 |
| SP recency (last 3) | 10 | 0 |
| Batting | 6 | 0 |
| Market-derived | 4 | **4** (margin) / **2** (total) |
| Context (park, ump, weather, dome) | 7 | **1** (temp, total only) |
| Rest/Workload | 3 | 0 |
| Momentum | 2 | 0 |
| Bullpen (aggregate) | 3 | 0 |
| Bullpen availability (game-day) | 4 | 0 |
| Travel & fatigue | 5 | 0 |
| Schedule context | 3 | 0 |
| Lineup composition | 8 | 0 |

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
| 2026-02-22 | Repository bootstrapped from CBB template | Initial structure |
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
