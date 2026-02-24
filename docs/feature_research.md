# MLB Feature Research Document

## Status: INITIAL RESEARCH

This document tracks the feature research process for the MLB betting model.
Every feature must be justified for baseball specifically.

## Data Sources Available

### Free / No Auth Required
| Source | Access Method | Key Data |
|--------|-------------|----------|
| **pybaseball** (FanGraphs) | `pip install pybaseball` | Team stats, pitcher stats, leaderboards, park factors |
| **pybaseball** (Statcast) | `pip install pybaseball` | Pitch-level data, xStats, barrel rates, exit velocity |
| **pybaseball** (Baseball Reference) | `pip install pybaseball` | Game logs, standings, schedule/record |
| **MLB Stats API** | REST API (no key) | Schedule, lineups, probable pitchers, game results |
| **ESPN API** | REST API (no key) | Schedule, scores, basic team info |

### Paid / Auth Required
| Source | Cost | Key Data |
|--------|------|----------|
| **The Odds API** | ~$80/mo | Moneylines, run lines, totals from 15+ books |
| **Weather API** | Free tier available | Game-day temperature, wind, humidity, precipitation |

## Feature Categories

### Category 1: Starting Pitcher (HIGHEST PRIORITY)
The starting pitcher is the single most important factor in MLB game outcomes.
No other sport has a single player with this much influence on the result.

**Season-to-date stats:**
- ERA, FIP, xFIP, WHIP: Standard pitching quality metrics
- K%, BB%, K-BB%: Strikeout and walk rates (FIP-independent)
- HR/9: Home run rate (park-adjusted ideally)
- xwOBA, hard-hit%, barrel%: Statcast quality-of-contact metrics

**Recency (last 3-5 starts):**
- ERA, FIP, K-BB% over recent starts
- IP trend: Is the pitcher going deeper into games?
- Pitch count trend: Is the pitch count increasing (fatigue)?

**Rest and workload:**
- Days since last start (optimal: 4-5 days)
- Season IP total (fatigue accumulation)
- Innings pitched in last start

**Research questions:**
- [ ] Are xStats (xFIP, xwOBA) better predictors than traditional stats?
- [ ] What's the minimum number of starts for reliable SP features?
- [ ] How much do SP stats stabilize vs. regress as sample grows?
- [ ] Do lefty/righty splits on SP matter enough to model?

### Category 2: Bullpen
The bullpen finishes ~35-40% of total innings in modern MLB.
Bullpen fatigue is one of the most underpriced factors.

**Team-level bullpen stats:**
- Bullpen ERA, FIP, WHIP, K%
- Recent workload: IP in last 3 days, pitches in last 3 days
- High-leverage reliever availability

**Research questions:**
- [ ] What's the best way to measure bullpen fatigue?
- [ ] Does closer availability (1 day off vs. 3 days off) matter for ML?
- [ ] Are bullpen stats predictive enough to move the line?

### Category 3: Team Batting
Team-level offensive metrics. Unlike pitcher features, these aggregate
across the whole lineup and change slowly.

**Season-to-date:**
- wRC+, OPS, wOBA: Offensive quality (park-adjusted ideally)
- ISO: Isolated power
- K%, BB%: Plate discipline
- BABIP: Luck indicator (high BABIP may regress)

**Recency:**
- wRC+ last 14 days
- Runs scored last 10 games

**Platoon splits:**
- Team wRC+ vs. LHP
- Team wRC+ vs. RHP
- Advantage when facing opposite-handed SP

**Research questions:**
- [ ] How stable are team batting stats week-to-week?
- [ ] Do platoon splits add predictive value over aggregate stats?
- [ ] Is lineup-level wRC+ (actual starters) better than team-level?

### Category 4: Park Factors
Park factors are well-established and relatively stable.
Coors Field is the most extreme case (+38% run factor).

**Available:**
- Run factor (overall scoring environment)
- HR factor (home run rate multiplier)
- Hit factor (batting average multiplier)

**Research questions:**
- [ ] Should park factors be static (annual) or computed rolling?
- [ ] Does the model overweight park factors for Coors? (common error)
- [ ] How do park factors interact with weather for outdoor parks?

### Category 5: Situational/Context
Game context that affects outcomes beyond team quality.

**Available:**
- Home/away flag
- Rest days (off-days between series)
- Travel distance
- Day vs. night game
- Series game number (game 1 vs. game 3 of a series)
- Interleague flag (AL vs. NL)
- Season phase (April cold starts vs. September pennant race)

**Research questions:**
- [ ] Is MLB home field advantage (~54%) stable enough to model?
- [ ] Do rest days matter in MLB? (less than NBA/NFL likely)
- [ ] Is there a "getaway day" effect (game 3 afternoon before travel)?

### Category 6: Weather
Weather directly affects ball carry and total runs scored.

**Available (outdoor parks only):**
- Temperature
- Wind speed and direction (relative to park orientation)
- Humidity
- Precipitation probability

**Research questions:**
- [ ] What's the magnitude of temperature effect? (~0.3 runs per 10F from research)
- [ ] How to compute wind direction factor per park? (need park orientations)
- [ ] Is humidity effect real or negligible?
- [ ] How accurate are weather forecasts 6+ hours before gametime?

### Category 7: Market-Derived
Market consensus as a feature (the market knows things the model doesn't).

**Available:**
- Consensus moneyline (median across books)
- Implied win probability
- Run line (typically +/- 1.5)
- Consensus total (over/under)
- Number of books posting lines

**Research questions:**
- [ ] Does the consensus ML add predictive value to the model? (likely yes, based on general modeling experience)
- [ ] How correlated are model predictions with market implied probs?
- [ ] Can line movement be captured as a feature?

### Category 8: Matchup-Specific
Specific pitcher-vs-team and lineup-vs-pitcher interactions.

**Available (harder to compute):**
- SP career stats vs. this specific team
- Team batting stats vs. this SP's handedness
- Batter-pitcher historical matchup OPS
- Lineup wRC+ (actual starters, not team season average)

**Research questions:**
- [ ] Is batter-pitcher history predictive with small samples?
- [ ] Is platoon advantage (L/R matchup) already captured by SP stats?
- [ ] Can we get confirmed lineup data reliably enough to compute lineup wRC+?

## Phase 1 Priority (Build First)
1. Starting pitcher season stats (ERA, FIP, K%, etc.)
2. Team batting season stats (wRC+, OPS, wOBA)
3. Park factors (static reference)
4. Market features (consensus ML, implied probability)
5. Basic situational (home/away, season phase)

## Phase 2 (Add After Phase 1 Backtest)
6. Starting pitcher recency (last 3 starts)
7. Bullpen aggregate stats
8. Weather features
9. Rolling form (win% last 10/20)
10. Platoon splits

## Phase 3 (Derivative Markets)
11. **F5 model features** — SP-only feature set, no bullpen
12. **NRFI features** — SP 1st-inning splits, leadoff hitter stats, umpire K-zone
13. **Team total features** — asymmetric (one team's offense vs opponent SP+bullpen)

## Phase 4 (Advanced, If Edge Exists in Any Market)
14. Lineup-level features (confirmed lineup wRC+ vs team average)
15. Batter-pitcher matchup history
16. Umpire tendencies (strike zone, run environment)
17. Travel and rest patterns
18. Bullpen fatigue modeling (complex)

---

## Derivative Market Research

### First 5 Innings (F5) — ML & Total
F5 markets isolate the starting pitcher matchup. The bullpen is irrelevant.
Books may set F5 lines with less precision than full-game lines.

**Key hypotheses:**
- [ ] F5 markets are softer than full-game because less sharp money targets them
- [ ] SP-only features should have MORE predictive power for F5 than full-game
- [ ] Bullpen-dependent features (fatigue, closer avail) are pure noise for F5
- [ ] Park factors still matter (5 innings is enough for park effects to manifest)

**Data requirements:**
- 5-inning box scores (score after 5 innings) for historical games
- F5 odds from Odds API (`h2h_h1`, `totals_h1` market keys)
- SP pitch count / efficiency metrics (pitchers who go 5+ vs those pulled early)

**Modeling approach:**
- Separate F5 margin and F5 total models
- Feature set: SP stats + batting stats + park factors + weather (NO bullpen)
- Boruta will confirm — expect bullpen features auto-rejected
- Separate walk-forward validation and backtest

### NRFI / YRFI (No Run First Inning)
Binary outcome: did either team score in the top or bottom of the 1st inning?
Historically ~55-60% of MLB games have a scoreless 1st inning (NRFI).

**Key hypotheses:**
- [ ] SP first-inning splits are more predictive than full-season ERA
- [ ] Leadoff hitter OBP and top-of-order quality drive 1st-inning scoring
- [ ] Umpire tendencies (wide zone = more Ks = more NRFI) may add signal
- [ ] NRFI market is less efficient because it's a "prop" — recreational money dominates
- [ ] Park factor matters less for 1 inning (smaller sample for park to manifest)

**Data requirements:**
- First-inning scoring data per game (did runs score in top 1st? bottom 1st?)
- SP first-inning ERA / OBP-against / K-rate (split by inning)
- Leadoff hitter stats (OBP, K%, SB tendency)
- Batting order positions 1-3 stats
- NRFI/YRFI odds (may need to scrape from individual books — not standard in Odds API)

**Modeling approach:**
- XGBoost binary classifier (objective: binary:logistic)
- Target: 1 = NRFI (no runs in first inning), 0 = YRFI
- Very different feature set from margin/total models
- Small feature space — avoid overfitting on niche stats

### Team Totals
Individual team over/under (e.g., Yankees over 4.5 runs). Asymmetric model —
one team's offense vs the opposing team's pitching.

**Key hypotheses:**
- [ ] Team total markets are softer than game totals (less liquid, less sharp action)
- [ ] Can exploit mismatches where full-game total is right but allocation is wrong
  (e.g., total 8.5 but model says 6-2 split, not 4.5-4)
- [ ] Weather/park effects may be priced into game total but not team totals correctly

**Data requirements:**
- Individual team scoring per game (already in game results)
- Team total odds from Odds API or individual books
- Same features as full-game but applied asymmetrically

**Modeling approach:**
- Predict each team's runs separately, not the margin
- Compare team prediction to team total line
- May cannibalize full-game total model or complement it

### Alternate Run Lines
Non-standard spreads like -1.5, +1.5, -2.5, etc. with heavier juice.

**Key hypotheses:**
- [ ] When margin model has high confidence, alt run lines offer better EV
- [ ] -1.5 for favorites the model loves; +2.5 for dogs the model likes
- [ ] No separate model needed — derived from margin model's distribution calibration
- [ ] Juice is heavier but edges may be wider (books set alts from full-game line)

**Data requirements:**
- Alternate run line odds from Odds API (`alternate_spreads`)
- Good calibration of margin model tails (isotonic important here)

**Modeling approach:**
- No new model — use margin model's predicted distribution
- P(home wins by > 1.5) from calibrated margin distribution
- Compare to implied probability from alt run line odds
