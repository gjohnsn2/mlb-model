# MLB Gambling Model — Project CLAUDE.md

## Maintenance Protocol
After any session with significant decisions, data quirks, or architecture changes —
update this file and relevant memory files before closing. Add a dated entry to the
Change Log at the bottom.

## What This Project Is
A Major League Baseball gambling model targeting **multiple markets** where alpha may exist.
Unlike CBB (one clear edge on spreads), MLB has a much larger surface area of derivative
markets that receive less sharp attention. The model explores all of them and lets the data
decide where the edge is.

**Markets under investigation:**
- Full-game moneyline (ML)
- Full-game total (O/U)
- Run line (+/- 1.5)
- First 5 innings moneyline (F5 ML)
- First 5 innings total (F5 O/U)
- No Run First Inning / Yes Run First Inning (NRFI/YRFI)
- Team totals (individual team O/U)
- Alternate run lines (-1.5, +1.5, -2.5)

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
├── 01_fetch_games.py            # MLB Stats API: schedule, boxscores, linescores, SP IDs, umpires, weather
├── 01b_scrape_statcast.py       # Statcast/pybaseball: pitch-level data → per-game SP advanced metrics
├── 01c_scrape_park_factors.py   # Park factors (baseballr/FanGraphs or static CSV)
├── 02_scrape_bref.py            # Baseball Reference: standings, team game logs
├── 02b_build_pitcher_logs.py    # Merge MLB Stats API SP lines + Statcast advanced → pitcher_logs_all.csv
├── 02c_scrape_bullpen.py        # Bullpen usage and fatigue metrics
├── 02d_scrape_weather.py        # Open-Meteo forecasts (pre-game) — backtesting uses MLB API actual weather
├── 02e_scrape_lineups.py        # Probable pitchers + lineups (MLB Stats API + Rotowire for early confirms)
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

### Source Overview
| Source | Auth | Cost | Script | Primary Data |
|--------|------|------|--------|-------------|
| **MLB Stats API** | None | Free | `01_`, `02e_`, `03_` | Schedule, linescore (inning-by-inning), boxscore, lineups, umpires, weather, probable SP |
| **Statcast / Savant** | None | Free | `01b_` | Pitch-level data, xStats, barrel rates, exit velo, launch angle |
| **FanGraphs** | None | Free* | `01c_` | Team/pitcher leaderboards, park factors, splits, projections |
| **Baseball Reference** | None | Free | `02_`, `02b_` | Standings, game logs, pitcher logs, splits |
| **Retrosheet** | None | Free | `00_` | Historical play-by-play (1898-2025), box scores, game logs |
| **Odds API** | `ODDS_API_KEY` | $59-119/mo | `04_` | ML, run line, totals, F5, 1st-inning, props from 15+ books |
| **Open-Meteo** | None | Free | `02d_` | Forecasts + historical weather archive (no API key needed) |
| **Rotowire** | None | Free† | `02e_` | Early lineup confirmations, probable pitchers (2-4 hrs pre-game) |

*FanGraphs membership ($80/yr) optional for Splits Tool access. Programmatic via baseballr (R) or scraping.
†Rotowire full SP grid requires paid sub; basic lineup page is free.

### MLB Stats API — The Motherlode (FREE)
Base URL: `https://statsapi.mlb.com/api/v1/`
Python wrapper: `pip install MLB-StatsAPI` (toddrob99)

This is MLB's own production API — powers MLB.com and the MLB app. No auth required.
Single most important data source for this project.

| Endpoint | URL Pattern | Critical Data |
|----------|-------------|---------------|
| Schedule | `/schedule?date=YYYY-MM-DD` | Game PKs, status, teams, start times |
| Game Feed | `/game/{gamePk}/feed/live` | **Everything in one call**: linescore, boxscore, plays, umpires, weather, venue, probable pitchers |
| Linescore | `/game/{gamePk}/linescore` | **Inning-by-inning runs** — key for F5 + NRFI targets |
| Boxscore | `/game/{gamePk}/boxscore` | Full player stats, batting order, pitcher lines |
| Play-by-Play | `/game/{gamePk}/playByPlay` | Every play/pitch in sequence |

**Why this matters**: The linescore endpoint gives inning-by-inning runs for every game.
- F5 scores = sum innings 1-5 per team
- NRFI label = `innings[0].home.runs == 0 AND innings[0].away.runs == 0`
- Game-time weather from `gameData.weather` (actual, not forecast — better for backtesting)
- HP umpire from `gameData.officials`

Historical depth: Detailed game feeds back to ~2005. Schedule data goes further.
Rate limits: Undocumented but tolerant of reasonable usage. No key required.

### Statcast / Baseball Savant (FREE)
Access: `pip install pybaseball` → `statcast(start_dt, end_dt)`

Pitch-level data for every MLB pitch. Key for SP quality metrics and first-inning analysis.
- **Fields**: velocity, spin rate, exit velo, launch angle, xBA, xSLG, xwOBA, barrel%,
  hard-hit%, pitch type, zone, `inning`, `inning_topbot`, `at_bat_number`
- **First-inning filtering**: `inning == 1` gives all pitches in first innings →
  compute SP first-inning K%, whiff rate, barrel rate for NRFI features
- **Historical depth**: PITCHf/x from 2008, full Statcast from 2015, swing tracking from 2024
- **Limits**: 30K rows per query; pybaseball auto-chunks into 5-day windows

### FanGraphs (FREE website, $80/yr membership optional)
Access: `baseballr` (R, more reliable) or scraping. **pybaseball FanGraphs functions are
currently broken** as of late 2025 due to website changes.

| Data | Access | Notes |
|------|--------|-------|
| Pitcher/batter leaderboards | baseballr `fg_pitcher_leaders()` | 330+ metrics: FIP, xFIP, SIERA, K%, BB%, etc. |
| Park factors | baseballr `fg_park()`, `fg_park_hand()` | 1yr/3yr/5yr regressed, per-event (HR, K, BB, GB), LHB/RHB splits |
| **SP first-inning splits** | Splits Tool (web) or scraping | 1st-inning ERA, K%, BB%, HR/FB%, wOBA-against — **critical for NRFI** |
| Projections | Steamer, ZiPS, ATC, THE BAT | Pre-season and in-season updated |
| Plate discipline | Leaderboards | O-Swing%, Z-Swing%, SwStr% |

**Key limitation**: No official API. baseballr (R) is the most reliable programmatic access.
For Python pipeline, options are: (1) R subprocess calls via baseballr, (2) Selenium scraping,
(3) the `pybaseballstats` package (newer alternative, Polars-based, updated Feb 2026).

### Baseball Reference (FREE)
Access: pybaseball `schedule_and_record()`, `pitching_stats_range()`, or BeautifulSoup
(static HTML, no JS rendering needed).

- Game logs, box scores, standings, transactions
- Pitcher by-inning splits (including 1st inning) — backup for FanGraphs
- Historical depth: Complete MLB history from 1876
- **Rate limits**: Will throttle aggressive scrapers. Use 3-5 second delays.

### Retrosheet (FREE, including commercial use)
Access: Bulk CSV downloads from retrosheet.org. Also via pybaseball `retrosheet` module.

- Play-by-play event files from 1898-2025
- Box scores, game logs, inning-by-inning scores
- **Key for historical training data**: Deepest NRFI and F5 source (inning-by-inning
  scoring going back 100+ years)
- 7 master CSV files covering all games

### The Odds API (PAID — shared with CBB)
Already in use for CBB model. Same API key and credit pool.

**MLB-specific market keys** (verified from docs):
| Market Key | Description | Model |
|------------|-------------|-------|
| `h2h` | Full-game moneyline | margin_model |
| `spreads` | Full-game run line | margin_model |
| `totals` | Full-game over/under | total_model |
| `h2h_1st_5_innings` | **F5 moneyline** | f5_margin_model |
| `spreads_1st_5_innings` | **F5 run line** | f5_margin_model |
| `totals_1st_5_innings` | **F5 over/under** | f5_total_model |
| `totals_1st_1_innings` | **1st-inning O/U** (NRFI proxy @ 0.5) | nrfi_model |
| `h2h_1st_1_innings` | 1st-inning moneyline | nrfi_model |
| `h2h_3_way_1st_1_innings` | 1st-inning 3-way (includes draw = NRFI) | nrfi_model |
| `h2h_1st_3_innings` | 1st 3 innings ML | future exploration |
| `h2h_1st_7_innings` | 1st 7 innings ML | future exploration |
| `batter_home_runs` | Batter HR props | future exploration |
| `pitcher_strikeouts` | Pitcher K props | future exploration |

**NOTE**: The original config used `h2h_h1` / `totals_h1` — these are **wrong**. Correct
MLB keys use the `_1st_N_innings` format. Config must be updated.

**No dedicated team totals market key** for MLB as of current docs. Gap.

**Pricing**: $59/mo (100K credits) likely sufficient for daily MLB + CBB.
Each call = 1 credit/region/market. Historical odds = 10 credits/region/market.
Historical F5/NRFI odds available from mid-2020.

### Open-Meteo (FREE — replaces Weather API)
No API key required. No signup needed.

- Forecasts: 1-16 day temperature, humidity, wind speed/direction, precipitation, cloud cover
- **Historical weather archive**: ERA5 reanalysis data from 1940 — free backtesting weather
- Resolution: 1-11 km
- Historical Forecast API: archived model forecasts (what would the forecast have been?)
- **Replaces**: OpenWeatherMap ($40/mo for historical) and Visual Crossing ($35/mo)

For completed games, MLB Stats API `gameData.weather` gives actual game-time conditions
(more accurate than any forecast). Use Open-Meteo for pre-game predictions only.

### Umpire Data (FREE — multiple sources)
| Source | Data | Access |
|--------|------|--------|
| MLB Stats API | HP umpire assignment per game | `gameData.officials` in game feed |
| Statcast | Umpire ID per pitch | Cross-ref with game_pk |
| UmpScores | Umpire ratings and tendencies | umpscorecards.com (free) |
| Swish Analytics | K Boost, BB Boost per umpire | swishanalytics.com (free) |

Build umpire tendency metrics from Statcast data (called strike rate, K%/BB% boost)
or scrape pre-computed from UmpScores/Swish.

### Lineup Confirmation (FREE)
| Source | Timing | Access |
|--------|--------|--------|
| Rotowire | 2-4 hrs pre-game (earliest) | Scrape `rotowire.com/baseball/daily-lineups.php` |
| Rotogrinders | Similar timing | Scrape `rotogrinders.com/lineups/mlb` |
| MLB Stats API | ~1 hr pre-game (official submission) | Game feed `liveData.boxscore` |

Pipeline should check Rotowire for early SP confirmation, then verify via MLB Stats API
closer to game time. SP scratches are the #1 source of stale predictions.

### Cost Summary
| Source | Cost | Necessity |
|--------|------|-----------|
| MLB Stats API | Free | **Essential** — primary data backbone |
| Statcast/pybaseball | Free | **Essential** — SP quality metrics |
| Baseball Reference | Free | **Essential** — game logs, standings |
| Retrosheet | Free | **Important** — deep historical training data |
| Open-Meteo | Free | **Important** — weather forecasts (free beats paid alternatives) |
| Umpire sources | Free | **Nice-to-have** — NRFI feature, Phase 3 |
| **The Odds API** | **$59-119/mo** | **Essential** — odds for all markets (shared with CBB) |
| FanGraphs membership | $80/yr | **Nice-to-have** — Splits Tool, projections |
| **Total** | **~$60-120/mo + $80/yr** | Most data is free; Odds API is the only real cost |

## Multi-Market Architecture
MLB offers far more market surface area than CBB. Rather than assuming where the edge is,
we build separate models per market and let backtesting reveal what's profitable.

### Market → Model Mapping
**PRIMARY (Phase 1):**
| Market | Model Target | Key Thesis | Feature Emphasis |
|--------|-------------|------------|------------------|
| **Full-game ML** | Home run margin | Core win prediction | Everything — SP, bullpen, batting, park, weather |
| **Full-game Total** | Combined runs | Scoring environment | Park, weather, SP, bullpen, pace |
| **F5 ML** | Home margin thru 5 inn | SP matchup isolated — no bullpen noise | **SP stats only**, batting vs SP hand, park |
| **F5 Total** | Combined runs thru 5 inn | SP-driven scoring, cleaner signal | SP stats, park, weather, batting |

**SECONDARY (Phase 2+, separate effort):**
| Market | Model Target | Key Thesis | Feature Emphasis |
|--------|-------------|------------|------------------|
| **Run Line (+/-1.5)** | Home run margin | Derived from margin model (like CBB spread) | Same as ML; margin distribution matters |
| **NRFI/YRFI** | P(0 runs in 1st inning) | Binary classification, niche market — may be separate model | SP 1st-inning splits, leadoff hitter stats, umpire zone |
| **Team Total** | Individual team runs | Asymmetric — one team's offense vs opponent's pitching | SP matchup, batting, park (one-sided) |
| **Alt Run Lines** | Home run margin | Heavier juice but wider edges on big opinions | Same as ML; calibration of margin tails matters |

### Why Multiple Markets?
1. **Derivative markets get less sharp action** — books copy full-game lines from Pinnacle
   but may set F5/NRFI/team totals with less precision.
2. **Different features matter** — F5 is pure SP; NRFI is first-inning micro; bullpen is
   irrelevant for F5 but critical for full-game.
3. **Correlation benefits** — if we identify an SP edge, that edge propagates across
   full-game ML, F5 ML, F5 total, and NRFI simultaneously. One insight, multiple bets.
4. **Market efficiency hierarchy** (expected, softest → sharpest):
   NRFI > Team Totals > F5 Total > F5 ML > Full-game Total > Alt RL > Run Line > Full-game ML

### Implementation Plan
- **Phase 1**: Full-game margin + total models (standard, mirrors CBB architecture)
- **Phase 2**: F5 models (requires F5 odds from Odds API + 5-inning box scores)
- **Phase 3**: NRFI model (requires 1st-inning scoring data + SP first-inning splits)
- **Phase 4**: Team totals, alt run lines (derived from existing models with better calibration)

### Odds API Market Keys (Verified)
| Market | Odds API `markets` param | Model | Notes |
|--------|--------------------------|-------|-------|
| Full-game ML | `h2h` | margin | Primary, always available |
| Run line | `spreads` | margin | Usually +/- 1.5 |
| Full-game total | `totals` | total | Standard O/U |
| **F5 ML** | `h2h_1st_5_innings` | f5_margin | **NOT** `h2h_h1` |
| **F5 run line** | `spreads_1st_5_innings` | f5_margin | F5 spread |
| **F5 total** | `totals_1st_5_innings` | f5_total | **NOT** `totals_h1` |
| **1st-inning O/U** | `totals_1st_1_innings` | nrfi | O/U 0.5 = NRFI/YRFI proxy |
| 1st-inning ML | `h2h_1st_1_innings` | nrfi | Who scores first |
| 1st-inning 3-way | `h2h_3_way_1st_1_innings` | nrfi | Includes draw (= NRFI) |
| Alt run lines | `alternate_spreads` | margin | -1.5, -2.5, etc. |
| Alt totals | `alternate_totals` | total | Different total lines |
| Batter HR props | `batter_home_runs` | future | Player props |
| Pitcher K props | `pitcher_strikeouts` | future | Player props |

**No dedicated team totals market key** in Odds API as of current docs.
Team totals may require scraping from individual sportsbook sites.

Also available: `_1st_3_innings` and `_1st_7_innings` variants for all market types.

## Core Modeling Decisions (Ported from CBB, Adapted for MLB)
- **Model**: XGBoost (separate model per market — see Multi-Market Architecture above)
- **Planned models**:
  - `margin_model` — Regressor, predicts home run margin (→ ML, run line, alt RL)
  - `total_model` — Regressor, predicts combined runs (→ full-game O/U)
  - `f5_margin_model` — Regressor, predicts home margin thru 5 innings (→ F5 ML)
  - `f5_total_model` — Regressor, predicts combined runs thru 5 innings (→ F5 O/U)
  - `nrfi_model` — Classifier, predicts P(0 runs in 1st inning) (→ NRFI/YRFI)
  - `team_total_model` — Regressor, predicts individual team runs (→ team O/U)
- **Training**: TBD — target 5+ seasons (2019-2026), ~12,000+ games
- **Feature selection**: Boruta (100 iterations, alpha=0.05). Confirmed + tentative used.
  Per-model Boruta — F5 model will likely drop all bullpen features automatically.
- **Validation**: Walk-forward folds (test seasons TBD), per-fold Boruta
- **Calibration**: Tail-aware isotonic (core 5th-95th percentile + linear tails)
- **Edge thresholds**: TBD after initial backtesting (start with ~3% ML edge minimum)
- **Margin-to-ML conversion**: margin→win prob via normal CDF: `P(win) = Phi(margin/RMSE)`
- **Sign convention**: Positive margin = home team wins by that many runs
- **Open questions**: Which markets have edge? Don't assume — backtest everything, kill
  markets that don't clear breakeven at any threshold.

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
  weather, bullpen fatigue, platoon splits, travel/rest. NRFI needs first-inning
  splits, leadoff hitter stats, and umpire tendencies — a completely separate domain.
- **Team name normalization**: MLB has 30 teams (simpler than CBB's 350+) but still
  need Odds API / ESPN / FanGraphs / Baseball Reference crosswalk.
- **Market structure**: Multiple models for multiple markets. Not just "one model,
  derive bets" like CBB. F5 and NRFI are fundamentally different prediction tasks.
- **Data sources**: FanGraphs replaces KenPom/Torvik. Statcast replaces four factors.
  Weather is new and critical. First-inning data needed for NRFI.
- **Season cadence**: Daily pipeline runs March-October (not November-April).
- **Sample size**: 2,430 games/year (vs ~5,500 CBB) but still much larger per-team
  sample (162 vs ~30 games per team).
- **Odds collection**: Must fetch F5 (`h2h_1st_5_innings`, `totals_1st_5_innings`) and
  1st-inning (`totals_1st_1_innings`) markets. Team totals not in Odds API — gap.

## Data Architecture — How Stats Flow Into Features

The most important thing to understand: we need **per-game pitcher lines** to build
cumulative stats, NOT season-level snapshots. Season-level FanGraphs pulls give you
one row per pitcher per season — useless for computing "ERA as of April 20th."

### Layer 1: Game Results + SP Identification (MLB Stats API)
For every historical game, the MLB Stats API `/game/{gamePk}/feed/live` gives:
- `home_sp_id`, `away_sp_id` — who started
- SP pitching line: IP, H, R, ER, K, BB, HR, pitches thrown
- Linescore: inning-by-inning runs → **F5 targets** (sum innings 1-5) + **NRFI label**
- Full boxscore: all player stats, batting order
- Umpire, weather, venue

This is the backbone. One API call per game. For ~2,430 games/season x 7 seasons = ~17K calls.
With MLB-StatsAPI Python wrapper: `statsapi.boxscore_data(game_pk)`.

### Layer 2: Advanced Pitcher Metrics (Statcast via pybaseball)
Statcast pitch-level data adds what box scores don't have:
- xwOBA, barrel%, hard-hit%, exit velo against
- Pitch velocity, spin rate, movement
- Whiff rate, chase rate, zone%

Pull via `statcast(start_dt, end_dt)` → group by `pitcher` (ID) + `game_pk` →
aggregate per-game advanced metrics. Merge with Layer 1 on (pitcher_id, game_pk).

Available from 2015 (full Statcast). 2008-2014 has PITCHf/x (velocity/movement only).

### Layer 3: Feature Computation (feature_engine.py)
`StartingPitcherComputer` takes the merged pitcher log table and computes:
- **Cumulative stats**: ERA-to-date, FIP-to-date, K%-to-date, xwOBA-to-date
- **Recency stats**: ERA last 3 starts, K% last 3 starts, IP trend
- **Workload**: Season IP, days rest, pitch count trend
- All strictly using `date < game_date` (no lookahead)

### How FIP is Computed (example)
FIP = ((13*HR + 3*BB - 2*K) / IP) + constant (~3.1)
We compute this FROM the raw game logs, NOT from FanGraphs. Same for WHIP, K%, BB%.
Only xStats (xwOBA, xFIP, barrel%) require Statcast pitch-level data.

### Pipeline Flow
```
Historical:
  MLB Stats API (all games) → game_results_all.csv (results + linescore + SP IDs)
                             → pitcher_game_logs.csv (SP line per game)
  Statcast (pybaseball)     → statcast_pitcher_games.csv (advanced per-game)
  Merge on (pitcher_id, game_pk) → pitcher_logs_all.csv (complete SP log)
  00_build_historical.py reads pitcher_logs_all.csv → StartingPitcherComputer

Daily:
  MLB Stats API /schedule   → today's games + probable pitchers
  Look up each SP's cumulative stats from historical pitcher logs
  05_build_features.py      → feature matrix for today's games
```

### Target Variables per Model
| Model | Target | Source |
|-------|--------|--------|
| margin_model | `home_score - away_score` | MLB Stats API boxscore |
| total_model | `home_score + away_score` | MLB Stats API boxscore |
| f5_margin_model | `home_runs_thru_5 - away_runs_thru_5` | MLB Stats API linescore (sum innings 1-5) |
| f5_total_model | `home_runs_thru_5 + away_runs_thru_5` | MLB Stats API linescore (sum innings 1-5) |

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
  confirmed starters before finalizing predictions. Rotowire has early (2-4 hr) lineups;
  MLB Stats API has official (~1 hr pre-game).
- **Rain delays/postponements**: Games can be postponed or suspended. Pipeline
  must handle gracefully.
- **Double-headers**: 7-inning games (through 2022), 9-inning (2023+). Historical
  data needs to flag these.
- **Season convention**: Calendar year (2026 = 2026 season). Simple.
- **All-Star Break**: Mid-season gap. Don't predict All-Star Game.
- **Interleague play**: AL vs NL games — different competitive dynamics.
- **DH rule**: Universal DH since 2022. Historical data before 2022 has NL pitchers
  batting — need to account for this.
- **pybaseball FanGraphs broken**: As of late 2025, pybaseball's FanGraphs functions are
  non-functional due to website changes. Use `baseballr` (R) or `pybaseballstats` (newer
  Python alternative, Polars-based) instead.
- **Statcast query limits**: 30K rows per request. pybaseball auto-chunks into 5-day
  windows. Large historical pulls are slow.
- **Odds API market keys**: MLB uses `_1st_N_innings` format (e.g., `h2h_1st_5_innings`),
  NOT the `h2h_h1`/`totals_h1` format. The CBB-style keys are wrong for MLB.
- **No team totals in Odds API**: As of current docs, no dedicated team total market key
  for MLB. May need alternative source or compute from individual book scraping.
- **MLB Stats API weather**: `gameData.weather` in game feed gives actual game-time
  conditions (temperature, wind, condition). More accurate than forecast for backtesting.
  Use Open-Meteo for pre-game predictions only.
- **Baseball Reference throttling**: Will block aggressive scrapers. Use 3-5 second delays.

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
| 2026-02-22 | Expanded to multi-market architecture (F5, NRFI, team totals, alt RL) | 6+ model targets, derivative market research plan |
| 2026-02-23 | Comprehensive data source audit — MLB Stats API as backbone, fixed Odds API market keys, Open-Meteo replaces paid weather, pybaseball FanGraphs broken | Only real cost is Odds API ($59-119/mo shared w/ CBB) |
| 2026-02-23 | Data architecture doc: game-by-game SP stats flow (MLB API boxscore → Statcast advanced → cumulative). Priority: full-game + F5 ML/total. NRFI secondary. | Rewrote pipeline script responsibilities |
