# MLB Model — Data Directory

## Structure
```
data/
├── raw/           # Daily scrape outputs (date-stamped, git-ignored)
├── processed/     # Engineered feature matrices (date-stamped, git-ignored)
├── predictions/   # Model picks + edges (date-stamped, git-ignored)
├── tracking/      # Performance log, bankroll, bet ledger (git-ignored)
├── historical/    # Training data + historical archives (git-ignored)
├── lines/         # Historical line data from Odds API (git-ignored)
├── park_factors.csv     # Static: venue park factors
└── venue_locations.csv  # Static: venue lat/lon/elevation
```

## Data Sources
| Source | Script | Format | Refresh |
|--------|--------|--------|---------|
| FanGraphs (pybaseball) | 01_, 01b_ | CSV | Daily |
| Baseball Reference (pybaseball) | 02_, 02b_ | CSV | Daily |
| Statcast/Savant (pybaseball) | 01b_ | CSV | Daily |
| ESPN | 03_ | JSON->CSV | Daily |
| MLB Stats API | 02e_, 03_ | JSON->CSV | Daily |
| The Odds API | 04_ | JSON->CSV | Daily |
| Weather API | 02d_ | JSON->CSV | Daily |

## File Naming Convention
- Raw scrapes: `{source}_{TODAY}.csv` (e.g., `fangraphs_team_batting_2026-04-15.csv`)
- Features: `features_{TODAY}.csv`
- Predictions: `picks_{TODAY}.csv`, `edges_{TODAY}.csv`, `betting_card_{TODAY}.csv`

## Team Name Normalization
MLB has 30 teams. The canonical name map is in `05_build_features.py`.
Sources use different conventions:
- ESPN: Full display names ("Arizona Diamondbacks")
- Odds API: Varies ("Arizona D-backs" sometimes)
- FanGraphs: Abbreviations ("ARI") or full names
- Baseball Reference: Abbreviations ("ARI")

The crosswalk is manageable with only 30 MLB teams.

## Key Data Quirks
- **Double-headers**: Two games same day, same teams — must be tracked as separate games with separate game_ids
- **Postponements**: Rain delays can cause games to be rescheduled. Exclude from evaluation if not played.
- **SP scratches**: Starting pitcher may change after lines are posted. Must verify before betting.
- **7-inning double-headers**: Existed 2020-2022 only. All 9 innings from 2023+.
- **Universal DH**: Since 2022. Historical data before 2022 has NL pitchers batting.
- **Season = calendar year**: 2026 season = 2026.
