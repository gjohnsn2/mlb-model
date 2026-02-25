"""
15 — Monte Carlo Bankroll Simulation & Ruin Analysis
=====================================================
Generates Lasso bet-level P&L from OOF predictions, then runs season-block
bootstrap Monte Carlo to estimate ruin probabilities and drawdown distributions.

Method: Season-block bootstrap (draw N seasons with replacement from 2017-2025).
This preserves intra-season streaks and correlation structure — more conservative
than shuffling individual bets.

Inputs:
  - models/mlb_oof_margin_lasso_nomarket_predictions.csv (Lasso OOF)
  - data/historical/training_data_mlb_v2.csv (for odds)

Outputs:
  - models/montecarlo_ruin_report.txt
  - models/montecarlo_ruin_report.png

Run: python3 15_montecarlo_ruin.py [--seed 42] [--paths 10000] [--bankroll 25000]
"""

import argparse
import numpy as np
import pandas as pd
from scipy.stats import norm
from pathlib import Path

from config import MODELS_DIR, HISTORICAL_DIR, get_logger

# Import backtest helpers
from importlib.machinery import SourceFileLoader
_backtest = SourceFileLoader("backtest", str(Path(__file__).parent / "10_backtest_mlb.py")).load_module()
american_to_implied_prob = _backtest.american_to_implied_prob
american_to_decimal = _backtest.american_to_decimal

log = get_logger("15_montecarlo_ruin")

# Constants matching 10_backtest_mlb.py exactly
ML_MARGIN_THRESHOLD = 1.5  # production threshold in runs
ML_MARGIN_UNIT_TIERS = [
    (2.0, 3.0, "3u"),
    (1.5, 2.0, "2u"),
    (1.0, 1.5, "1.5u"),
    (0.5, 1.0, "1u"),
]
UNIT_SIZE = 100  # base unit for fixed-unit P&L validation


def generate_lasso_bets():
    """
    Generate Lasso bet-level P&L from OOF predictions.
    Replicates 10_backtest_mlb.py calibration + bet simulation exactly.
    """
    # Load Lasso OOF predictions
    oof_path = MODELS_DIR / "mlb_oof_margin_lasso_nomarket_predictions.csv"
    oof = pd.read_csv(oof_path)
    oof["date"] = oof["date"].astype(str).str[:10]
    log.info(f"Loaded Lasso OOF: {len(oof)} games, seasons {oof['season'].min()}-{oof['season'].max()}")

    # Load training data for odds
    training = pd.read_csv(HISTORICAL_DIR / "training_data_mlb_v2.csv")

    # Merge OOF with odds
    odds_cols = ["game_pk", "consensus_h2h_home", "consensus_h2h_away",
                 "actual_margin", "actual_total", "home_team", "away_team"]
    available = [c for c in odds_cols if c in training.columns]
    merged = oof.merge(training[available], on="game_pk", how="left", suffixes=("", "_train"))

    # Filter corrupt H2H
    for col in ["consensus_h2h_home", "consensus_h2h_away"]:
        corrupt = merged[col].notna() & (merged[col].abs() < 100)
        if corrupt.any():
            log.info(f"  Filtering {corrupt.sum()} corrupt {col} values")
            merged.loc[corrupt, "consensus_h2h_home"] = np.nan
            merged.loc[corrupt, "consensus_h2h_away"] = np.nan

    # Calibrate: de-vig → implied margin → rescale
    h2h_mask = merged["consensus_h2h_home"].notna() & merged["consensus_h2h_away"].notna()
    df = merged[h2h_mask].copy()

    from sklearn.metrics import root_mean_squared_error
    margin_rmse = root_mean_squared_error(df["actual"], df["predicted"])

    raw_home = df["consensus_h2h_home"].apply(american_to_implied_prob)
    raw_away = df["consensus_h2h_away"].apply(american_to_implied_prob)
    total_vig = raw_home + raw_away
    valid = total_vig.notna() & (total_vig > 0)
    df.loc[valid, "market_home_prob"] = raw_home[valid] / total_vig[valid]
    df["market_implied_margin"] = margin_rmse * norm.ppf(df["market_home_prob"].clip(0.001, 0.999))

    model_mean = df["predicted"].mean()
    model_std = df["predicted"].std()
    market_mean = df["market_implied_margin"].mean()
    market_std = df["market_implied_margin"].std()
    df["calibrated_pred"] = (df["predicted"] - model_mean) / model_std * market_std + market_mean
    df["margin_edge"] = df["calibrated_pred"] - df["market_implied_margin"]

    log.info(f"Calibration: model std {model_std:.3f} → market std {market_std:.3f}")
    log.info(f"Margin RMSE: {margin_rmse:.3f}")

    # Generate bets at >= 1.5 threshold
    bets = []
    for _, row in df.iterrows():
        edge = row["margin_edge"]
        if pd.isna(edge) or abs(edge) < ML_MARGIN_THRESHOLD:
            continue

        if edge > 0:
            odds_used = row["consensus_h2h_home"]
            won = row["actual"] > 0
        else:
            odds_used = row["consensus_h2h_away"]
            won = row["actual"] < 0

        dec = american_to_decimal(odds_used)
        if pd.isna(dec):
            continue

        # Unit tiers
        ml_units = 1.0
        for tier_min, tier_units, _ in ML_MARGIN_UNIT_TIERS:
            if abs(edge) >= tier_min:
                ml_units = tier_units
                break

        push = (row["actual"] == 0)
        bet_risk = UNIT_SIZE * ml_units
        if push:
            profit = 0
        elif won:
            profit = round(bet_risk * (dec - 1))
        else:
            profit = -bet_risk

        bets.append({
            "date": row["date"],
            "game_pk": row["game_pk"],
            "season": int(row["season"]),
            "won": won,
            "push": push,
            "odds_used": int(odds_used),
            "decimal_odds": dec,
            "ml_units": ml_units,
            "margin_edge": round(edge, 3),
            "profit_fixed": profit,  # fixed $100/unit
        })

    bets_df = pd.DataFrame(bets)
    bets_df["date"] = pd.to_datetime(bets_df["date"])
    bets_df = bets_df.sort_values("date").reset_index(drop=True)

    # Validate against known ROI
    total_risked = (bets_df["ml_units"] * UNIT_SIZE).sum()
    total_profit = bets_df["profit_fixed"].sum()
    roi = total_profit / total_risked * 100
    n_won = bets_df[~bets_df["push"]]["won"].sum()
    n_lost = len(bets_df[~bets_df["push"]]) - n_won
    log.info(f"Lasso bets at >= {ML_MARGIN_THRESHOLD}: {len(bets_df)} bets, "
             f"{n_won}-{n_lost}, ROI {roi:+.1f}%, P&L ${total_profit:+,.0f}")

    return bets_df


def run_montecarlo(bets_df, n_paths=10000, bankroll_start=25000, unit_pct=0.01,
                   max_seasons=5, seed=42):
    """
    Season-block bootstrap Monte Carlo simulation.

    Each path draws max_seasons seasons with replacement from available seasons.
    Within each season, bets are replayed in chronological order with proportional
    unit sizing (unit_value = current_bankroll * unit_pct).
    """
    rng = np.random.default_rng(seed)

    # Group bets by season
    seasons = sorted(bets_df["season"].unique())
    season_bets = {s: bets_df[bets_df["season"] == s].reset_index(drop=True) for s in seasons}
    log.info(f"Monte Carlo: {n_paths} paths, {max_seasons} seasons each, "
             f"drawing from {len(seasons)} seasons")

    # Pre-extract arrays for speed
    season_arrays = {}
    for s, sdf in season_bets.items():
        season_arrays[s] = {
            "won": sdf["won"].values,
            "push": sdf["push"].values,
            "decimal_odds": sdf["decimal_odds"].values,
            "ml_units": sdf["ml_units"].values,
            "n_bets": len(sdf),
        }

    # Results storage
    # Track bankroll at end of each season-slot for each path
    bankroll_paths = np.zeros((n_paths, max_seasons + 1))  # +1 for starting point
    bankroll_paths[:, 0] = bankroll_start
    max_drawdown = np.zeros(n_paths)
    peak_bankroll = np.full(n_paths, bankroll_start)
    hard_ruin = np.zeros(n_paths, dtype=bool)
    hard_ruin_season = np.full(n_paths, -1)  # which season-slot ruin happened
    drawdown_25_hit = np.zeros(n_paths, dtype=bool)
    drawdown_25_season = np.full(n_paths, -1)
    drawdown_50_hit = np.zeros(n_paths, dtype=bool)
    drawdown_50_season = np.full(n_paths, -1)
    longest_losing_streak = np.zeros(n_paths, dtype=int)
    seasons_drawn = np.zeros((n_paths, max_seasons), dtype=int)

    # Also track detailed daily bankroll for fan chart (subsample paths)
    # Store bet-by-bet bankroll for first 10K paths but only record season-end
    # For fan chart, we'll track cumulative bet-by-bet for all paths
    total_bets_per_season = {s: arr["n_bets"] for s, arr in season_arrays.items()}
    max_bets = sum(max(total_bets_per_season.values()) for _ in range(max_seasons))

    # For the fan chart, store bet-level bankroll for all paths
    # This could be huge, so we store only at season boundaries + sample for detail
    bet_level_subsample = min(500, n_paths)  # detailed paths for fan chart
    max_total_bets = max_seasons * max(total_bets_per_season.values())
    detailed_bankrolls = np.full((bet_level_subsample, max_total_bets + 1), np.nan)
    detailed_bankrolls[:, 0] = bankroll_start

    season_choices = np.array(seasons)

    for i in range(n_paths):
        bankroll = bankroll_start
        peak = bankroll_start
        max_dd = 0.0
        current_streak = 0
        longest_streak = 0
        bet_idx = 0  # for detailed tracking
        ruined = False

        drawn = rng.choice(season_choices, size=max_seasons, replace=True)
        seasons_drawn[i] = drawn

        for slot, s in enumerate(drawn):
            arr = season_arrays[s]

            for j in range(arr["n_bets"]):
                if bankroll <= 0:
                    ruined = True
                    if not hard_ruin[i]:
                        hard_ruin[i] = True
                        hard_ruin_season[i] = slot
                    break

                unit_value = bankroll * unit_pct
                risk = unit_value * arr["ml_units"][j]

                if arr["push"][j]:
                    payout = 0.0
                elif arr["won"][j]:
                    payout = risk * (arr["decimal_odds"][j] - 1)
                    current_streak = 0
                else:
                    payout = -risk
                    current_streak += 1
                    longest_streak = max(longest_streak, current_streak)

                bankroll += payout

                # Track peak and drawdown
                if bankroll > peak:
                    peak = bankroll
                dd = (peak - bankroll) / peak if peak > 0 else 0
                max_dd = max(max_dd, dd)

                # Check drawdown thresholds
                if dd >= 0.25 and not drawdown_25_hit[i]:
                    drawdown_25_hit[i] = True
                    drawdown_25_season[i] = slot
                if dd >= 0.50 and not drawdown_50_hit[i]:
                    drawdown_50_hit[i] = True
                    drawdown_50_season[i] = slot

                # Detailed tracking
                if i < bet_level_subsample:
                    bet_idx += 1
                    if bet_idx < detailed_bankrolls.shape[1]:
                        detailed_bankrolls[i, bet_idx] = bankroll

            if ruined:
                bankroll_paths[i, slot + 1:] = 0
                break

            bankroll_paths[i, slot + 1] = bankroll
            peak_bankroll[i] = peak

        if not ruined:
            # Fill remaining season slots if we didn't break
            for s_fill in range(int(drawn.shape[0])):
                if bankroll_paths[i, s_fill + 1] == 0 and s_fill + 1 > 0:
                    pass  # already set

        max_drawdown[i] = max_dd
        longest_losing_streak[i] = longest_streak

    results = {
        "bankroll_paths": bankroll_paths,
        "max_drawdown": max_drawdown,
        "hard_ruin": hard_ruin,
        "hard_ruin_season": hard_ruin_season,
        "drawdown_25_hit": drawdown_25_hit,
        "drawdown_25_season": drawdown_25_season,
        "drawdown_50_hit": drawdown_50_hit,
        "drawdown_50_season": drawdown_50_season,
        "longest_losing_streak": longest_losing_streak,
        "seasons_drawn": seasons_drawn,
        "detailed_bankrolls": detailed_bankrolls[:bet_level_subsample],
        "peak_bankroll": peak_bankroll,
        "n_paths": n_paths,
        "max_seasons": max_seasons,
        "bankroll_start": bankroll_start,
        "unit_pct": unit_pct,
    }

    return results


def kelly_analysis(bets_df):
    """
    Compute Kelly optimal fraction and compare to current 1% unit sizing.
    Uses average bet characteristics from historical data.
    """
    non_push = bets_df[~bets_df["push"]].copy()
    win_rate = non_push["won"].mean()

    # Average decimal odds when winning vs losing
    avg_dec_odds = non_push["decimal_odds"].mean()
    avg_win_payout = non_push[non_push["won"]]["decimal_odds"].mean() - 1  # net payout per $1

    # Kelly: f* = (bp - q) / b where b = avg net payout, p = win prob, q = 1-p
    b = avg_win_payout
    p = win_rate
    q = 1 - p
    kelly_fraction = (b * p - q) / b if b > 0 else 0
    kelly_fraction = max(0, kelly_fraction)

    # Half-Kelly and quarter-Kelly (common conservative adjustments)
    half_kelly = kelly_fraction / 2
    quarter_kelly = kelly_fraction / 4

    # Current sizing: ~1% per unit, avg ~2.2u per bet at >= 1.5
    avg_units = non_push["ml_units"].mean()
    current_fraction = 0.01 * avg_units  # fraction of bankroll per bet

    return {
        "win_rate": win_rate,
        "avg_decimal_odds": avg_dec_odds,
        "avg_win_payout": avg_win_payout,
        "kelly_fraction": kelly_fraction,
        "half_kelly": half_kelly,
        "quarter_kelly": quarter_kelly,
        "current_fraction": current_fraction,
        "avg_units": avg_units,
    }


def build_report(bets_df, mc_results, kelly):
    """Build human-readable Monte Carlo report."""
    lines = []
    lines.append("MONTE CARLO BANKROLL SIMULATION & RUIN ANALYSIS")
    lines.append("=" * 70)
    lines.append(f"Generated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')}")
    lines.append(f"Model: Lasso no-market | Threshold: >= {ML_MARGIN_THRESHOLD} runs")
    lines.append(f"Bankroll: ${mc_results['bankroll_start']:,.0f} | "
                 f"Unit sizing: {mc_results['unit_pct']*100:.0f}% of bankroll per unit")
    lines.append(f"Monte Carlo: {mc_results['n_paths']:,} paths, "
                 f"{mc_results['max_seasons']} seasons each (block bootstrap)")
    lines.append("")

    # ── Historical bet summary ──
    lines.append("=" * 70)
    lines.append("HISTORICAL LASSO BET-LEVEL SUMMARY (>= 1.5 runs, fixed $100/unit)")
    lines.append("=" * 70)
    non_push = bets_df[~bets_df["push"]]
    total_risked = (bets_df["ml_units"] * UNIT_SIZE).sum()
    total_profit = bets_df["profit_fixed"].sum()
    roi = total_profit / total_risked * 100

    lines.append(f"Total bets: {len(bets_df)}")
    lines.append(f"Record: {int(non_push['won'].sum())}-{int((~non_push['won']).sum())}")
    lines.append(f"Win rate: {non_push['won'].mean()*100:.1f}%")
    lines.append(f"Total risked: ${total_risked:,.0f}")
    lines.append(f"Total profit: ${total_profit:+,.0f}")
    lines.append(f"ROI: {roi:+.1f}%")
    lines.append("")

    # Per-season breakdown
    lines.append(f"{'Season':>7} | {'Bets':>5} | {'W-L':>7} | {'Win%':>6} | "
                 f"{'Risked':>9} | {'Profit':>9} | {'ROI':>7}")
    lines.append("-" * 65)
    for season in sorted(bets_df["season"].unique()):
        sb = bets_df[bets_df["season"] == season]
        snp = sb[~sb["push"]]
        sr = (sb["ml_units"] * UNIT_SIZE).sum()
        sp = sb["profit_fixed"].sum()
        sroi = sp / sr * 100 if sr > 0 else 0
        w = int(snp["won"].sum())
        l = int((~snp["won"]).sum())
        lines.append(f"  {season:>5} | {len(sb):>5} | {w:>3}-{l:<3} | "
                     f"{w/(w+l)*100 if (w+l)>0 else 0:>5.1f}% | "
                     f"${sr:>8,.0f} | ${sp:>+8,.0f} | {sroi:>+6.1f}%")
    lines.append("")

    # Per-bet profit distribution
    lines.append("=" * 70)
    lines.append("PER-BET PROFIT DISTRIBUTION (fixed $100/unit)")
    lines.append("=" * 70)
    profits = bets_df["profit_fixed"]
    lines.append(f"Mean:   ${profits.mean():>+8.1f}")
    lines.append(f"Median: ${profits.median():>+8.1f}")
    lines.append(f"Std:    ${profits.std():>8.1f}")
    lines.append(f"Min:    ${profits.min():>+8.0f}")
    lines.append(f"Max:    ${profits.max():>+8.0f}")
    from scipy.stats import skew
    lines.append(f"Skew:   {skew(profits):.3f}")
    lines.append("")

    # ── Monte Carlo results ──
    bp = mc_results["bankroll_paths"]
    start = mc_results["bankroll_start"]
    n_paths = mc_results["n_paths"]
    max_s = mc_results["max_seasons"]

    lines.append("=" * 70)
    lines.append("MONTE CARLO RESULTS (proportional unit sizing)")
    lines.append("=" * 70)
    lines.append("")

    # Ruin probabilities at different horizons
    lines.append("── Ruin Probabilities ──")
    lines.append(f"{'Horizon':>12} | {'Hard Ruin':>12} | {'25% DD Pause':>14} | {'50% Drawdown':>14}")
    lines.append("-" * 60)
    for horizon in [1, 2, 3, 5]:
        if horizon > max_s:
            continue
        # Hard ruin: bankroll <= 0 within first `horizon` seasons
        hr = mc_results["hard_ruin"] & (mc_results["hard_ruin_season"] < horizon)
        p_hr = hr.sum() / n_paths * 100

        # 25% drawdown pause
        dd25 = mc_results["drawdown_25_hit"] & (mc_results["drawdown_25_season"] < horizon)
        p_dd25 = dd25.sum() / n_paths * 100

        # 50% drawdown
        dd50 = mc_results["drawdown_50_hit"] & (mc_results["drawdown_50_season"] < horizon)
        p_dd50 = dd50.sum() / n_paths * 100

        lines.append(f"  {horizon:>2} season{'s' if horizon > 1 else ' '} | "
                     f"{p_hr:>10.2f}%  | {p_dd25:>12.1f}%  | {p_dd50:>12.2f}%")
    lines.append("")

    # Expected bankroll at each horizon
    lines.append("── Expected Bankroll by Horizon ──")
    lines.append(f"{'Horizon':>12} | {'5th %ile':>10} | {'25th %ile':>10} | "
                 f"{'Median':>10} | {'75th %ile':>10} | {'95th %ile':>10}")
    lines.append("-" * 75)
    for h in range(1, max_s + 1):
        vals = bp[:, h]
        p5, p25, p50, p75, p95 = np.percentile(vals, [5, 25, 50, 75, 95])
        lines.append(f"  {h:>2} season{'s' if h > 1 else ' '} | "
                     f"${p5:>9,.0f} | ${p25:>9,.0f} | "
                     f"${p50:>9,.0f} | ${p75:>9,.0f} | ${p95:>9,.0f}")
    lines.append("")

    # Max drawdown distribution
    lines.append("── Max Drawdown Distribution ──")
    dd = mc_results["max_drawdown"]
    lines.append(f"5th percentile:  {np.percentile(dd, 5)*100:.1f}%")
    lines.append(f"25th percentile: {np.percentile(dd, 25)*100:.1f}%")
    lines.append(f"Median:          {np.percentile(dd, 50)*100:.1f}%")
    lines.append(f"75th percentile: {np.percentile(dd, 75)*100:.1f}%")
    lines.append(f"95th percentile: {np.percentile(dd, 95)*100:.1f}%")
    lines.append(f"Max:             {dd.max()*100:.1f}%")
    lines.append("")

    # Longest losing streak
    lines.append("── Longest Losing Streak ──")
    ls = mc_results["longest_losing_streak"]
    lines.append(f"Median:          {int(np.percentile(ls, 50))}")
    lines.append(f"75th percentile: {int(np.percentile(ls, 75))}")
    lines.append(f"95th percentile: {int(np.percentile(ls, 95))}")
    lines.append(f"Max:             {int(ls.max())}")
    lines.append("")

    # ── Kelly Analysis ──
    lines.append("=" * 70)
    lines.append("KELLY CRITERION ANALYSIS")
    lines.append("=" * 70)
    lines.append(f"Historical win rate:    {kelly['win_rate']*100:.1f}%")
    lines.append(f"Avg decimal odds:       {kelly['avg_decimal_odds']:.3f}")
    lines.append(f"Avg win net payout:     {kelly['avg_win_payout']:.3f}")
    lines.append(f"Avg units per bet:      {kelly['avg_units']:.1f}u")
    lines.append(f"Full Kelly fraction:    {kelly['kelly_fraction']*100:.2f}% of bankroll per bet")
    lines.append(f"Half Kelly:             {kelly['half_kelly']*100:.2f}%")
    lines.append(f"Quarter Kelly:          {kelly['quarter_kelly']*100:.2f}%")
    lines.append(f"Current sizing:         {kelly['current_fraction']*100:.2f}% "
                 f"({mc_results['unit_pct']*100:.0f}% x {kelly['avg_units']:.1f}u avg)")

    ratio = kelly["current_fraction"] / kelly["kelly_fraction"] if kelly["kelly_fraction"] > 0 else float("inf")
    if ratio > 1:
        lines.append(f"Assessment: OVERSIZED — current is {ratio:.1f}x full Kelly")
    elif ratio > 0.5:
        lines.append(f"Assessment: Aggressive — current is {ratio:.1f}x full Kelly (~half Kelly)")
    elif ratio > 0.25:
        lines.append(f"Assessment: Conservative — current is {ratio:.1f}x full Kelly (~quarter Kelly)")
    else:
        lines.append(f"Assessment: Very conservative — current is {ratio:.2f}x full Kelly")
    lines.append("")

    # ── Notes ──
    lines.append("=" * 70)
    lines.append("NOTES")
    lines.append("=" * 70)
    lines.append("- Season-block bootstrap preserves intra-season streaks and correlation")
    lines.append("- Proportional sizing: unit_value = bankroll * 1%, risk = units * unit_value")
    lines.append("- Hard ruin = bankroll <= $0 (impossible with proportional sizing unless")
    lines.append("  a single bet can lose 100%, which can't happen with 1-3% risk per bet)")
    lines.append("- 25% drawdown pause = model stops betting (functional ruin)")
    lines.append("- 50% drawdown = psychological ruin threshold")
    lines.append("- Kelly assumes i.i.d. bets; actual bets have correlation within seasons")
    lines.append("- 2025 season (-14.6% ROI) is included in the bootstrap pool")

    return "\n".join(lines)


def make_plots(bets_df, mc_results, save_path):
    """Generate 4-panel Monte Carlo visualization."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Monte Carlo Bankroll Simulation — Lasso No-Market (>= 1.5 runs)",
                 fontsize=14, fontweight="bold")

    start = mc_results["bankroll_start"]
    max_s = mc_results["max_seasons"]
    bp = mc_results["bankroll_paths"]

    # ── Panel 1: Bankroll trajectory fan chart ──
    ax = axes[0, 0]
    x = np.arange(max_s + 1)
    labels = ["Start"] + [f"S{i}" for i in range(1, max_s + 1)]

    # Percentile bands
    p5 = np.percentile(bp, 5, axis=0)
    p25 = np.percentile(bp, 25, axis=0)
    p50 = np.percentile(bp, 50, axis=0)
    p75 = np.percentile(bp, 75, axis=0)
    p95 = np.percentile(bp, 95, axis=0)

    ax.fill_between(x, p5, p95, alpha=0.15, color="steelblue", label="5-95th %ile")
    ax.fill_between(x, p25, p75, alpha=0.3, color="steelblue", label="25-75th %ile")
    ax.plot(x, p50, color="steelblue", linewidth=2, label="Median")
    ax.axhline(start, color="gray", linestyle="--", alpha=0.5, label=f"Start (${start:,.0f})")
    ax.axhline(start * 0.75, color="orange", linestyle="--", alpha=0.5, label="25% DD pause")
    ax.axhline(start * 0.5, color="red", linestyle="--", alpha=0.5, label="50% DD")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Bankroll ($)")
    ax.set_title("Bankroll Trajectory (Season-Block Bootstrap)")
    ax.legend(fontsize=7, loc="upper left")
    ax.grid(True, alpha=0.3)

    # Format y-axis with dollar signs
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"${v:,.0f}"))

    # ── Panel 2: Max drawdown histogram ──
    ax = axes[0, 1]
    dd = mc_results["max_drawdown"] * 100
    ax.hist(dd, bins=50, color="coral", edgecolor="white", alpha=0.8)
    ax.axvline(np.median(dd), color="red", linestyle="--", linewidth=2,
               label=f"Median: {np.median(dd):.1f}%")
    ax.axvline(25, color="orange", linestyle="--", linewidth=1.5, label="25% pause")
    ax.axvline(50, color="darkred", linestyle="--", linewidth=1.5, label="50% ruin")
    ax.set_xlabel("Max Drawdown (%)")
    ax.set_ylabel("Count")
    ax.set_title(f"Max Drawdown Distribution ({mc_results['max_seasons']}-Season Horizon)")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # ── Panel 3: Probability of ruin curve ──
    ax = axes[1, 0]
    horizons = list(range(1, max_s + 1))
    n_paths = mc_results["n_paths"]

    p_hard = []
    p_dd25 = []
    p_dd50 = []
    for h in horizons:
        hr = mc_results["hard_ruin"] & (mc_results["hard_ruin_season"] < h)
        p_hard.append(hr.sum() / n_paths * 100)

        d25 = mc_results["drawdown_25_hit"] & (mc_results["drawdown_25_season"] < h)
        p_dd25.append(d25.sum() / n_paths * 100)

        d50 = mc_results["drawdown_50_hit"] & (mc_results["drawdown_50_season"] < h)
        p_dd50.append(d50.sum() / n_paths * 100)

    ax.plot(horizons, p_dd25, "o-", color="orange", linewidth=2, label="25% DD pause")
    ax.plot(horizons, p_dd50, "s-", color="red", linewidth=2, label="50% drawdown")
    ax.plot(horizons, p_hard, "^-", color="darkred", linewidth=2, label="Hard ruin ($0)")
    ax.set_xlabel("Seasons")
    ax.set_ylabel("Probability (%)")
    ax.set_title("Cumulative Ruin Probability by Horizon")
    ax.legend(fontsize=8)
    ax.set_xticks(horizons)
    ax.grid(True, alpha=0.3)

    # ── Panel 4: Season P&L distribution (box plot from historical) ──
    ax = axes[1, 1]
    seasons = sorted(bets_df["season"].unique())
    season_rois = []
    season_labels = []
    for s in seasons:
        sb = bets_df[bets_df["season"] == s]
        risked = (sb["ml_units"] * UNIT_SIZE).sum()
        profit = sb["profit_fixed"].sum()
        roi = profit / risked * 100 if risked > 0 else 0
        season_rois.append(roi)
        season_labels.append(str(s))

    colors = ["green" if r > 0 else "red" for r in season_rois]
    bars = ax.bar(season_labels, season_rois, color=colors, edgecolor="white", alpha=0.8)
    ax.axhline(0, color="black", linewidth=0.5)
    ax.axhline(np.mean(season_rois), color="steelblue", linestyle="--",
               label=f"Mean: {np.mean(season_rois):+.1f}%")
    ax.set_xlabel("Season")
    ax.set_ylabel("ROI (%)")
    ax.set_title("Historical Season ROI (Lasso >= 1.5 runs)")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3, axis="y")

    # Add value labels on bars
    for bar, roi in zip(bars, season_rois):
        y = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, y + (0.5 if y >= 0 else -1.5),
                f"{roi:+.1f}%", ha="center", va="bottom" if y >= 0 else "top",
                fontsize=7)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    log.info(f"Saved plot -> {save_path}")


def main():
    parser = argparse.ArgumentParser(description="Monte Carlo bankroll simulation")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--paths", type=int, default=10000, help="Number of MC paths")
    parser.add_argument("--bankroll", type=int, default=25000, help="Starting bankroll ($)")
    parser.add_argument("--seasons", type=int, default=5, help="Seasons per path")
    parser.add_argument("--unit-pct", type=float, default=0.01, help="Fraction of bankroll per unit")
    args = parser.parse_args()

    log.info("=" * 60)
    log.info("MONTE CARLO BANKROLL SIMULATION & RUIN ANALYSIS")
    log.info("=" * 60)

    # Part 1: Generate Lasso bet-level P&L
    log.info("\n── Part 1: Generating Lasso bet-level P&L ──")
    bets_df = generate_lasso_bets()

    # Part 2: Monte Carlo simulation
    log.info("\n── Part 2: Running Monte Carlo simulation ──")
    mc_results = run_montecarlo(
        bets_df,
        n_paths=args.paths,
        bankroll_start=args.bankroll,
        unit_pct=args.unit_pct,
        max_seasons=args.seasons,
        seed=args.seed,
    )

    # Kelly analysis
    kelly = kelly_analysis(bets_df)

    # Part 3: Report + plots
    log.info("\n── Part 3: Generating report and plots ──")

    report = build_report(bets_df, mc_results, kelly)
    report_path = MODELS_DIR / "montecarlo_ruin_report.txt"
    with open(report_path, "w") as f:
        f.write(report)
    log.info(f"Saved report -> {report_path}")

    plot_path = MODELS_DIR / "montecarlo_ruin_report.png"
    make_plots(bets_df, mc_results, plot_path)

    # Print summary to console
    print(f"\n{'='*70}")
    print("MONTE CARLO SUMMARY")
    print(f"{'='*70}")

    non_push = bets_df[~bets_df["push"]]
    total_risked = (bets_df["ml_units"] * UNIT_SIZE).sum()
    total_profit = bets_df["profit_fixed"].sum()
    roi = total_profit / total_risked * 100
    print(f"  Historical: {len(bets_df)} bets, "
          f"{int(non_push['won'].sum())}-{int((~non_push['won']).sum())}, "
          f"ROI {roi:+.1f}%")

    bp = mc_results["bankroll_paths"]
    final = bp[:, -1]
    print(f"\n  {args.seasons}-Season Bankroll (${args.bankroll:,.0f} start):")
    for pct, label in [(5, "5th"), (25, "25th"), (50, "Median"), (75, "75th"), (95, "95th")]:
        print(f"    {label:>6}: ${np.percentile(final, pct):>10,.0f}")

    dd25_total = mc_results["drawdown_25_hit"].sum() / mc_results["n_paths"] * 100
    dd50_total = mc_results["drawdown_50_hit"].sum() / mc_results["n_paths"] * 100
    hr_total = mc_results["hard_ruin"].sum() / mc_results["n_paths"] * 100
    print(f"\n  Ruin over {args.seasons} seasons:")
    print(f"    P(25% DD pause): {dd25_total:.1f}%")
    print(f"    P(50% drawdown): {dd50_total:.2f}%")
    print(f"    P(hard ruin):    {hr_total:.2f}%")

    print(f"\n  Kelly: current sizing = {kelly['current_fraction']*100:.2f}% "
          f"(vs {kelly['kelly_fraction']*100:.2f}% full Kelly)")

    print(f"\n  Report: {report_path}")
    print(f"  Plot:   {plot_path}")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()
