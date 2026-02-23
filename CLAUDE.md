# MLB Gambling Model — Project CLAUDE.md

## Maintenance Protocol
After any session with significant decisions, data quirks, or architecture changes —
update this file and relevant memory files before closing. Add a dated entry to the
Change Log at the bottom.

## What This Project Is
A Major League Baseball gambling model targeting moneylines (primary), run lines (+/- 1.5),
and totals (over/under). Identifies closing line value against consensus (median) and
Pinnacle lines from The Odds API (15+ books).
**Current phase**: INITIAL RESEARCH (data collection and feature engineering).

## Reference Implementation
This project was bootstrapped from the CBB (college basketball) model at
`/Users/grantjohnson/Desktop/cbb-model`. The CBB model achieves 65.2% spread win rate
and 65.9% ML win rate with walk-forward validation. This MLB model ports the PROCESS
(validation, auditing, monitoring, staking) but rebuilds the FEATURES and domain logic
for baseball. Realistic expectations: MLB is a more efficient market than CBB; edges
will be smaller but sample size is much larger (2,430 games/season vs ~5,500 in CBB).

## Project Structure
```
mlb-model/
├── config.py                    # Central config (paths, credentials, features, params)
├── feature_engine.py            # Single source of truth for all candidate features
├── 00_build_historical.py       # Build training data from historical archives (run once)
├── 01_scrape_fangraphs.py       # FanGraphs team/pitcher stats (pybaseball)
├── 01b_scrape_statcast.py       # Statcast/Baseball Savant (pybaseball)
├── 01c_scrape_park_factors.py   # Park factors (FanGraphs/ESPN)
├── 02_scrape_bref.py            # Baseball Reference standings + game logs
├── 02b_scrape_pitcher_logs.py   # Starting pitcher game logs (pybaseball)
├── 02c_scrape_bullpen.py        # Bullpen usage and fatigue metrics
├── 02d_scrape_weather.py        # Weather forecasts for game-day conditions
├── 02e_scrape_lineups.py        # Confirmed lineups (RotoBaller, MLB API)
├── 03_fetch_schedule.py         # ESPN/MLB API schedule
├── 04_fetch_odds.py             # Odds API (moneylines, run lines, totals)
├── 05_build_features.py         # Daily feature engineering + team name normalization
├── 05b_select_features.py       # Boruta feature selection
├── 06_train_model.py            # XGBoost training + walk-forward validation
├── 06b_tune_hyperparams.py      # Optuna hyperparameter tuning
├── 07_predict.py                # Daily predictions + SHAP explanations
├── 08_find_edges.py             # Model vs. market edge detection
├── 09_evaluate.py               # Performance grading + tracking
├── 10_backtest.py               # Historical profitability backtest
├── 11_monitor.py                # Real-time monitoring (PSI, rolling metrics)
├── 12_compare_models.py         # Production vs fold comparison
├── 12b_fetch_pinnacle.py        # Pinnacle closing line fetcher (eu region)
├── run_daily.sh                 # Pipeline orchestrator
├── utils_park_factors.py        # Park factors by venue
├── utils_weather.py             # Weather impact modeling
├── data/
│   ├── CLAUDE.md
│   ├── raw/                     # Daily scrape outputs (git-ignored)
│   ├── processed/               # Engineered feature matrices (git-ignored)
│   ├── predictions/             # Model picks + edges (git-ignored)
│   ├── tracking/                # Performance log (git-ignored)
│   ├── historical/              # Training data + archives (git-ignored)
│   ├── lines/                   # Historical line data (git-ignored)
│   ├── park_factors.csv         # Static reference: venue → park factor
│   └── venue_locations.csv      # Static reference: venue → lat/lon/elevation
├── models/
│   ├── CLAUDE.md
│   ├── configs/                 # Hyperparameter configs
│   └── trained/                 # Trained models (.pkl)
├── backtests/
│   ├── CLAUDE.md
│   └── results/
├── reports/                     # Daily SHAP charts
│   ├── weekly/
│   └── monthly/
├── docs/
│   └── feature_research.md      # Feature research document
├── notebooks/
│   └── CLAUDE.md
└── scripts/                     # One-off utility scripts
```

NOTE: Flat numbered-script pipeline (00-11, plus 12/12b one-off analysis scripts).
All feature logic in `feature_engine.py`. All team name normalization in `05_build_features.py`.

## File Organization Rules
- Pipeline scripts: numbered prefix — `01_scrape_fangraphs.py`
- Data outputs: include date — `fangraphs_2026-04-15.csv`
- Raw data → `data/raw/`, processed → `data/processed/`, historical → `data/historical/`
- Model artifacts → `models/trained/`, daily SHAP → `reports/`
- Never commit credentials; use env vars
- `.gitignore` excludes: `.env`, `__pycache__`, `*.pkl`, `data/raw/`, `data/processed/`,
  `data/predictions/`, `data/tracking/`, `data/historical/`, `data/lines/`, `reports/`,
  `models/trained/*.pkl`

## Key Commands
- Full daily pipeline: `source .env && ./run_daily.sh full`
- Re-predict (cached data): `./run_daily.sh predict`
- Evaluate yesterday: `./run_daily.sh evaluate`
- Retrain models: `./run_daily.sh train`
- Boruta + retrain: `./run_daily.sh features`
- Rebuild historical: `./run_daily.sh build` (slow — multiple seasons)

## Data Sources
| Source | Auth | Script | Notes |
|--------|------|--------|-------|
| **FanGraphs** | None (pybaseball) | `01_`, `01b_` | Team stats, pitcher stats, leaderboards |
| **Statcast/Savant** | None (pybaseball) | `01b_` | Pitch-level data, xStats, barrel rates |
| **Baseball Reference** | None | `02_`, `02b_` | Standings, game logs, splits |
| **ESPN** | None | `03_` | Schedule, scores, lineup status |
| **MLB Stats API** | None | `02e_`, `03_` | Official schedule, confirmed lineups |
| **Odds API** | `ODDS_API_KEY` (~$80/mo) | `04_` | Moneylines, run lines, totals from 15+ books |
| **Weather API** | `WEATHER_API_KEY` (optional) | `02d_` | Game-day weather forecasts |

## MLB-Specific Market Structure
- **Moneyline** (PRIMARY): No spread; betting on outright winner. Variable juice (-110 to -250+).
  This is the dominant MLB market — the model's primary target.
- **Run Line** (+/- 1.5): Standard 1.5-run spread. Favorite at -1.5, underdog at +1.5.
  Less liquid than moneyline but provides alternative when ML price is too steep.
- **Totals** (O/U): Combined runs scored. Typically 7.0-10.5. Weather and park factors
  are critical for totals.
- **First 5 Innings (F5)**: Isolates starting pitcher matchup, removes bullpen noise.
  Strong candidate for a separate model.

## Core Modeling Decisions (Ported from CBB, Adapted for MLB)
- **Model**: XGBoost Regressor (separate margin model + total model)
- **Targets**: Home run margin (for ML/run line) and combined runs (for totals)
- **Training**: TBD — target 5+ seasons (2019-2026), ~12,000+ games
- **Feature selection**: Boruta (100 iterations, alpha=0.05). Confirmed + tentative used.
- **Validation**: Walk-forward folds (test seasons TBD), per-fold Boruta
- **Calibration**: Tail-aware isotonic (core 5th-95th percentile + linear tails)
- **Edge thresholds**: TBD after initial backtesting (start with ~3% ML edge minimum)
- **Margin-to-ML conversion**: margin→win prob via normal CDF: `P(win) = Phi(margin/RMSE)`
- **Sign convention**: Positive margin = home team wins by that many runs

## MLB Season Structure
- **Regular season**: 162 games per team, 2,430 total games per season
- **Opening Day**: Late March / Early April
- **All-Star Break**: Mid-July (~3 days off)
- **Regular season ends**: Late September / Early October
- **Postseason**: October (Wild Card, Division Series, LCS, World Series)
- **Daily cadence**: Games nearly every day April through September (unlike CBB's weekly clusters)
- **Double-headers**: Occasional, same-day split-squad — handle as separate games
- **30 teams**, 2 leagues (AL/NL), 6 divisions

## What Transfers from CBB
- XGBoost + walk-forward validation architecture
- Boruta feature selection methodology
- Tail-aware isotonic calibration
- Edge detection framework (08_find_edges.py)
- Staking/bankroll management system
- Monitoring system (PSI, rolling metrics, alerts)
- Agent infrastructure (forensic-auditor, pressure-test-runner, live-monitor)
- Backtest rigor (no look-ahead, consensus + Pinnacle lines, flat juice baseline)

## What Must Be Rebuilt for MLB
- **Features**: Starting pitcher is THE key feature (no CBB analog). Park factors,
  weather, bullpen fatigue, platoon splits, travel/rest.
- **Team name normalization**: MLB has 30 teams (simpler than CBB's 350+) but still
  need Odds API / ESPN / FanGraphs / Baseball Reference crosswalk.
- **Market structure**: ML is dominant (not spreads). Run line is secondary.
  Variable juice changes the ROI calculation fundamentally.
- **Data sources**: FanGraphs replaces KenPom/Torvik. Statcast replaces four factors.
  Weather is new and critical.
- **Season cadence**: Daily pipeline runs March-October (not November-April).
- **Sample size**: 2,430 games/year (vs ~5,500 CBB) but still much larger per-team
  sample (162 vs ~30 games per team).

## Feature Categories (Planned — Requires Research)
| Category | Example Features | MLB Justification |
|----------|-----------------|-------------------|
| Starting Pitcher | ERA, FIP, xFIP, K%, BB%, WHIP, xwOBA, pitch mix | Single most important factor in MLB |
| Pitcher Recency | Last 3/5 starts ERA, K-BB%, IP trend, pitch count trend | Form and fatigue matter enormously |
| Bullpen | Bullpen ERA, leverage index usage, days since last use | Game-finishing capability |
| Team Batting | wRC+, OPS, wOBA, K%, BB%, ISO, BABIP | Team offensive strength |
| Platoon Splits | Team wRC+ vs LHP, Team wRC+ vs RHP, SP handedness | Lefty/righty matchups are exploitable |
| Park Factors | Venue run factor, HR factor, 2B/3B factor | Coors Field can add 2+ runs to a total |
| Situational | Home/Away, rest days, travel distance, day/night | Contextual modifiers |
| Weather | Temperature, wind speed/direction, humidity, precipitation | Affects ball carry and total runs |
| Rolling Form | Team win% last 10/20, run differential last 10/20 | Hot/cold streaks |
| Market | Consensus ML, implied probability, line movement | Market wisdom as a feature |

## Staking & Risk Rules (Adapted from CBB)
- Starting bankroll: TBD (match CBB or separate)
- Staking: 1% of current bankroll per unit (dynamic sizing)
- ML edge tiers: TBD after backtesting (likely 1u/1.5u/2u/3u by prob edge)
- Run line edge tiers: TBD
- Total edge tiers: TBD
- Drawdown warning: 15% from peak
- Drawdown pause: 25% from peak (auto-halt)
- Track separately from CBB — different sport, different bankroll

## Known Data Quirks
- **Team names**: ESPN/FanGraphs/Baseball Reference/Odds API all differ slightly.
  MLB has only 30 teams — crosswalk is much simpler than CBB's 350+.
- **Pitcher changes**: Announced SP can be scratched last-minute. Must check for
  confirmed starters before finalizing predictions.
- **Rain delays/postponements**: Games can be postponed or suspended. Pipeline
  must handle gracefully.
- **Double-headers**: 7-inning games (through 2022), 9-inning (2023+). Historical
  data needs to flag these.
- **Season convention**: Calendar year (2026 = 2026 season). Simple.
- **All-Star Break**: Mid-season gap. Don't predict All-Star Game.
- **Interleague play**: AL vs NL games — different competitive dynamics.
- **DH rule**: Universal DH since 2022. Historical data before 2022 has NL pitchers
  batting — need to account for this.

## Current Performance
Not yet established — project is in INITIAL RESEARCH phase.

## Backtest Rules (Non-Negotiable, Ported from CBB)
- No look-ahead. Pre-game data only (ratings, stats from BEFORE game date).
- Confirmed starting pitcher must be available pre-game.
- Consensus (median) + Pinnacle lines. Variable ML juice (not flat -110).
- Walk-forward folds with per-fold Boruta.
- Leakage check in `00_build_historical.py` (exits on violation).

## Change Log
| Date | Change | Impact |
|------|--------|--------|
| 2026-02-22 | Repository bootstrapped from CBB template | Initial structure + pipeline stubs |
