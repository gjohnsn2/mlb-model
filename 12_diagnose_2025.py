"""
12 — 2025 Season Diagnostic Deep Dive
======================================
The Lasso no-market model shows -14.6% ROI in 2025 at >=1.5 threshold
when 8/9 other seasons are positive. This script performs a comprehensive
breakdown across 10 diagnostic axes to understand what broke.

Each section compares 2025 vs three reference eras:
  - 2017-2019 (early history, pre-COVID)
  - 2021-2024 (recent history, post-COVID)
  - 2017-2024 (all prior combined)

Inputs:
  - models/mlb_oof_margin_lasso_nomarket_predictions.csv
  - data/historical/training_data_mlb_v2.csv

Outputs:
  - models/mlb_2025_diagnostic_report.txt
  - Stdout: executive summary

Run: python3 12_diagnose_2025.py    # ~10 sec, read-only
"""

import warnings
import importlib
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=RuntimeWarning, module="numpy")
from scipy import stats as sp_stats
from scipy.stats import norm, ks_2samp, pearsonr
from sklearn.metrics import root_mean_squared_error
from config import MODELS_DIR, HISTORICAL_DIR, get_logger

log = get_logger("12_diagnose_2025")

# Import backtest functions
bt = importlib.import_module("10_backtest_mlb")

# Lasso stable features (9/9 folds, from mlb_ridge_lasso_report_nomarket.txt)
LASSO_STABLE_9 = [
    "sp_season_ip_diff", "team_run_diff_10_diff", "star_missing_ops_diff",
    "lineup_bb_k_ratio_diff", "bb_rate_diff", "sp_k_pct_diff",
    "lineup_power_diff", "bullpen_whip_diff", "lineup_top_heavy_diff",
]

# Primary threshold where the -14.6% happens
PRIMARY_THRESHOLD = 1.5
THRESHOLD_SWEEP = [0.5, 1.0, 1.5, 2.0]

# Era definitions
ERAS = {
    "2017-2019": (2017, 2019),
    "2021-2024": (2021, 2024),
    "2017-2024": (2017, 2024),
    "2025":      (2025, 2025),
}


def era_mask(df, era_name):
    """Return boolean mask for an era."""
    lo, hi = ERAS[era_name]
    return (df["season"] >= lo) & (df["season"] <= hi)


def safe_corr(x, y):
    """Pearson correlation, returning (r, p) or (NaN, NaN) on failure."""
    mask = x.notna() & y.notna()
    if mask.sum() < 10:
        return np.nan, np.nan
    try:
        r, p = pearsonr(x[mask], y[mask])
        return r, p
    except Exception:
        return np.nan, np.nan


def bet_stats_for_subset(df, threshold):
    """Simulate ML bets and return stats dict for a subset."""
    bets = bt.simulate_ml_bets(df, threshold)
    return bt.compute_ml_stats(bets), bets


def format_stats_line(label, s, width=18):
    """Format a stats dict into a compact line."""
    if s["n_bets"] == 0:
        return f"  {label:<{width}}  no bets"
    wl = f"{s['wins']}-{s['losses']}"
    return (f"  {label:<{width}}  {s['n_bets']:>4} bets  {wl:>8}  "
            f"Win% {s['win_pct']:>5.1f}  ROI {s['roi']:>+7.1f}%  "
            f"Dog% {s['dog_pct']:>4.0f}  p={s['p_value']:.3f}")


# ═══════════════════════════════════════════════════════════════
# Section builders — each returns a list of report lines
# ═══════════════════════════════════════════════════════════════

def section_mlb_environment(training_df, lines):
    """Section 0: What actually happened in MLB — on-field reality."""
    lines.append("=" * 80)
    lines.append("0. MLB ENVIRONMENT — WHAT ACTUALLY HAPPENED ON THE FIELD")
    lines.append("=" * 80)

    # Add season column
    df = training_df.copy()
    df["date"] = pd.to_datetime(df["date"])
    df["season"] = df["date"].dt.year

    # ── A. Scoring environment ──
    lines.append("\n  A. SCORING ENVIRONMENT")
    lines.append(f"  {'Season':>7} | {'R/game':>7} | {'HR rate':>7} | {'K rate':>7} | "
                 f"{'BB rate':>7} | {'BA':>7} | {'Avg total':>9} | {'Home W%':>7}")
    lines.append("  " + "-" * 78)

    for season in sorted(df["season"].unique()):
        s = df[df["season"] == season]
        rpg = (s["home_runs"] + s["away_runs"]).mean()
        # Pool home and away stats
        hr = pd.concat([s["home_hr_rate"], s["away_hr_rate"]]).mean()
        kr = pd.concat([s["home_k_rate"], s["away_k_rate"]]).mean()
        bbr = pd.concat([s["home_bb_rate"], s["away_bb_rate"]]).mean()
        ba = pd.concat([s["home_batting_avg"], s["away_batting_avg"]]).mean()
        avg_total = s["actual_total"].mean()
        home_w = (s["actual_margin"] > 0).mean() * 100

        lines.append(f"  {int(season):>7} | {rpg:>7.2f} | {hr:>7.4f} | {kr:>7.4f} | "
                     f"{bbr:>7.4f} | {ba:>7.4f} | {avg_total:>9.2f} | {home_w:>6.1f}%")

    # ── B. Starting pitcher usage ──
    lines.append("\n  B. STARTING PITCHER USAGE")
    lines.append(f"  {'Season':>7} | {'Avg SP IP':>9} | {'SP IP std':>9} | "
                 f"{'Avg starts':>10} | {'SP ERA':>7} | {'SP K%':>7} | {'SP WHIP':>7}")
    lines.append("  " + "-" * 72)

    for season in sorted(df["season"].unique()):
        s = df[df["season"] == season]
        sp_ip = pd.concat([s["home_sp_avg_ip"], s["away_sp_avg_ip"]]).mean()
        sp_ip_std = pd.concat([s["home_sp_avg_ip"], s["away_sp_avg_ip"]]).std()
        sp_starts = pd.concat([s["home_sp_starts"], s["away_sp_starts"]]).mean()
        sp_era = pd.concat([s["home_sp_era"], s["away_sp_era"]]).mean()
        sp_k = pd.concat([s["home_sp_k_pct"], s["away_sp_k_pct"]]).mean()
        sp_whip = pd.concat([s["home_sp_whip"], s["away_sp_whip"]]).mean()

        lines.append(f"  {int(season):>7} | {sp_ip:>9.2f} | {sp_ip_std:>9.2f} | "
                     f"{sp_starts:>10.1f} | {sp_era:>7.2f} | "
                     f"{sp_k:>6.3f} | {sp_whip:>7.3f}")

    # ── C. Bullpen environment ──
    lines.append("\n  C. BULLPEN ENVIRONMENT")
    lines.append(f"  {'Season':>7} | {'BP ERA':>7} | {'BP WHIP':>7} | {'BP usage':>8}")
    lines.append("  " + "-" * 42)

    for season in sorted(df["season"].unique()):
        s = df[df["season"] == season]
        bp_era = pd.concat([s["home_bullpen_era"], s["away_bullpen_era"]]).mean()
        bp_whip = pd.concat([s["home_bullpen_whip"], s["away_bullpen_whip"]]).mean()
        bp_usage = pd.concat([s["home_bullpen_usage"], s["away_bullpen_usage"]]).mean()

        lines.append(f"  {int(season):>7} | {bp_era:>7.2f} | {bp_whip:>7.3f} | {bp_usage:>8.2f}")

    # ── D. Lineup composition ──
    lines.append("\n  D. LINEUP COMPOSITION")
    lines.append(f"  {'Season':>7} | {'OPS':>7} | {'Power':>7} | {'K rate':>7} | "
                 f"{'BB/K':>7} | {'Top-heavy':>9} | {'Continuity':>10}")
    lines.append("  " + "-" * 72)

    for season in sorted(df["season"].unique()):
        s = df[df["season"] == season]
        ops = pd.concat([s["home_lineup_ops"], s["away_lineup_ops"]]).mean()
        pwr = pd.concat([s["home_lineup_power"], s["away_lineup_power"]]).mean()
        kr = pd.concat([s["home_lineup_k_rate"], s["away_lineup_k_rate"]]).mean()
        bbk = pd.concat([s["home_lineup_bb_k_ratio"], s["away_lineup_bb_k_ratio"]]).mean()
        th = pd.concat([s["home_lineup_top_heavy"], s["away_lineup_top_heavy"]]).mean()
        cont = pd.concat([s["home_lineup_continuity"], s["away_lineup_continuity"]]).mean()

        lines.append(f"  {int(season):>7} | {ops:>7.3f} | {pwr:>7.4f} | {kr:>7.4f} | "
                     f"{bbk:>7.3f} | {th:>9.3f} | {cont:>10.2f}")

    # ── E. Upset rate (dogs winning) ──
    lines.append("\n  E. UPSET RATE & PARITY")
    lines.append(f"  {'Season':>7} | {'Games':>6} | {'Dog wins':>8} | {'Dog W%':>7} | "
                 f"{'Avg |margin|':>12} | {'1-run games':>11} | {'Blowouts 5+':>12}")
    lines.append("  " + "-" * 78)

    for season in sorted(df["season"].unique()):
        s = df[df["season"] == season]
        n = len(s)
        # Dog = away team when consensus_h2h_home is negative (home favored)
        # and home team when consensus_h2h_home is positive (away favored)
        has_odds = s[s["consensus_h2h_home"].notna()].copy()
        if len(has_odds) > 0:
            # Home is favored when consensus_h2h_home < 0 (negative ML)
            home_fav = has_odds["consensus_h2h_home"] < 0
            # Dog wins when: home is fav but away wins, or away is fav but home wins
            dog_wins = ((home_fav & (has_odds["actual_margin"] < 0)) |
                        (~home_fav & (has_odds["actual_margin"] > 0)))
            non_tie = has_odds["actual_margin"] != 0
            dog_pct = dog_wins[non_tie].mean() * 100
        else:
            dog_pct = np.nan

        avg_margin = s["actual_margin"].abs().mean()
        one_run = (s["actual_margin"].abs() == 1).mean() * 100
        blowout = (s["actual_margin"].abs() >= 5).mean() * 100

        dog_str = f"{dog_pct:>6.1f}%" if not np.isnan(dog_pct) else "    N/A"
        lines.append(f"  {int(season):>7} | {n:>6} | "
                     f"{int(dog_wins.sum()) if not np.isnan(dog_pct) else 'N/A':>8} | "
                     f"{dog_str} | {avg_margin:>12.2f} | {one_run:>10.1f}% | {blowout:>11.1f}%")

    # ── F. Context: dome, weather, postseason ──
    lines.append("\n  F. GAME CONTEXT")
    lines.append(f"  {'Season':>7} | {'Dome%':>6} | {'Avg temp':>8} | "
                 f"{'DH games':>8} | {'Postseason':>10}")
    lines.append("  " + "-" * 55)

    for season in sorted(df["season"].unique()):
        s = df[df["season"] == season]
        dome_pct = s["is_dome"].mean() * 100 if "is_dome" in s.columns else np.nan
        avg_temp = s["temp"].mean() if "temp" in s.columns else np.nan
        dh = s["is_doubleheader"].sum() if "is_doubleheader" in s.columns else 0
        post = s["is_postseason"].sum() if "is_postseason" in s.columns else 0

        dome_str = f"{dome_pct:>5.1f}%" if not np.isnan(dome_pct) else "   N/A"
        temp_str = f"{avg_temp:>7.1f}F" if not np.isnan(avg_temp) else "     N/A"
        lines.append(f"  {int(season):>7} | {dome_str} | {temp_str} | "
                     f"{dh:>8} | {post:>10}")

    lines.append("")


def section_temporal(cal, lines):
    """Section 1: Monthly breakdown."""
    lines.append("=" * 80)
    lines.append("1. TEMPORAL (MONTHLY) BREAKDOWN")
    lines.append("=" * 80)

    cal["month"] = pd.to_datetime(cal["date"]).dt.month
    cal["month_name"] = pd.to_datetime(cal["date"]).dt.strftime("%b")

    for thresh in THRESHOLD_SWEEP:
        lines.append(f"\n  Threshold >= {thresh} runs:")
        lines.append(f"  {'Month':<8} | {'2017-19':>22} | {'2021-24':>22} | {'2025':>22}")
        lines.append("  " + "-" * 82)

        for m in sorted(cal["month"].unique()):
            month_label = cal.loc[cal["month"] == m, "month_name"].iloc[0]
            parts = []
            for era in ["2017-2019", "2021-2024", "2025"]:
                sub = cal[era_mask(cal, era) & (cal["month"] == m)]
                if len(sub) == 0:
                    parts.append(f"{'--':>22}")
                    continue
                s, _ = bet_stats_for_subset(sub, thresh)
                if s["n_bets"] == 0:
                    parts.append(f"{'0 bets':>22}")
                else:
                    parts.append(f"{s['n_bets']:>3}b {s['roi']:>+5.1f}% w{s['win_pct']:.0f}%")
            lines.append(f"  {month_label:<8} | {parts[0]:>22} | {parts[1]:>22} | {parts[2]:>22}")

    # Correlation by era
    lines.append(f"\n  corr(pred, actual) by month — 2025 vs prior:")
    for m in sorted(cal["month"].unique()):
        month_label = cal.loc[cal["month"] == m, "month_name"].iloc[0]
        parts = []
        for era in ["2017-2024", "2025"]:
            sub = cal[era_mask(cal, era) & (cal["month"] == m)]
            r, _ = safe_corr(sub["calibrated_pred"], sub["actual"])
            parts.append(f"{r:+.3f}" if not np.isnan(r) else "  N/A")
        lines.append(f"  {month_label:<8}  prior: {parts[0]}  2025: {parts[1]}")
    lines.append("")


def section_side_bias(cal, lines):
    """Section 2: Home vs Away."""
    lines.append("=" * 80)
    lines.append("2. SIDE BIAS (HOME vs AWAY)")
    lines.append("=" * 80)

    # Overall side balance (all games, not just bets)
    for era in ["2017-2019", "2021-2024", "2025"]:
        sub = cal[era_mask(cal, era)]
        home_pct = (sub["margin_edge"] > 0).mean() * 100
        lines.append(f"  {era}: {home_pct:.1f}% home-favoring edges "
                     f"(mean edge {sub['margin_edge'].mean():+.3f})")
    lines.append("")

    for thresh in THRESHOLD_SWEEP:
        lines.append(f"  Threshold >= {thresh} runs:")
        lines.append(f"  {'Side':<10} | {'2017-19':>22} | {'2021-24':>22} | {'2025':>22}")
        lines.append("  " + "-" * 82)

        for side_label, edge_cond in [("Home", "edge > 0"), ("Away", "edge < 0")]:
            parts = []
            for era in ["2017-2019", "2021-2024", "2025"]:
                sub = cal[era_mask(cal, era)]
                if side_label == "Home":
                    side_sub = sub[sub["margin_edge"] > 0]
                else:
                    side_sub = sub[sub["margin_edge"] < 0]
                s, _ = bet_stats_for_subset(side_sub, thresh)
                if s["n_bets"] == 0:
                    parts.append(f"{'0 bets':>22}")
                else:
                    parts.append(f"{s['n_bets']:>3}b {s['roi']:>+5.1f}% w{s['win_pct']:.0f}%")
            lines.append(f"  {side_label:<10} | {parts[0]:>22} | {parts[1]:>22} | {parts[2]:>22}")
        lines.append("")


def section_dog_vs_fav(cal, lines):
    """Section 3: Underdog vs Favorite bets."""
    lines.append("=" * 80)
    lines.append("3. DOG vs FAVORITE")
    lines.append("=" * 80)

    for thresh in THRESHOLD_SWEEP:
        lines.append(f"\n  Threshold >= {thresh} runs:")
        lines.append(f"  {'Type':<10} | {'2017-19':>22} | {'2021-24':>22} | {'2025':>22}")
        lines.append("  " + "-" * 82)

        for bet_type in ["Dog", "Favorite"]:
            parts = []
            for era in ["2017-2019", "2021-2024", "2025"]:
                sub = cal[era_mask(cal, era)]
                _, bets = bet_stats_for_subset(sub, thresh)
                if bets.empty:
                    parts.append(f"{'0 bets':>22}")
                    continue
                if bet_type == "Dog":
                    type_bets = bets[bets["is_dog"]]
                else:
                    type_bets = bets[~bets["is_dog"]]
                s = bt.compute_ml_stats(type_bets)
                if s["n_bets"] == 0:
                    parts.append(f"{'0 bets':>22}")
                else:
                    parts.append(f"{s['n_bets']:>3}b {s['roi']:>+5.1f}% w{s['win_pct']:.0f}%")
            lines.append(f"  {bet_type:<10} | {parts[0]:>22} | {parts[1]:>22} | {parts[2]:>22}")

    # Mean implied prob of dog bets
    lines.append(f"\n  Mean implied prob of model's dog bets (>= {PRIMARY_THRESHOLD} runs):")
    for era in ["2017-2019", "2021-2024", "2025"]:
        sub = cal[era_mask(cal, era)]
        _, bets = bet_stats_for_subset(sub, PRIMARY_THRESHOLD)
        if not bets.empty:
            dog_bets = bets[bets["is_dog"]]
            if not dog_bets.empty:
                # Merge market_home_prob back
                dog_merged = dog_bets.merge(
                    cal[["game_pk", "market_home_prob"]],
                    on="game_pk", how="left"
                )
                # For dog bets: implied prob = min(home_prob, 1-home_prob)
                dog_prob = dog_merged["market_home_prob"].apply(
                    lambda p: min(p, 1-p) if pd.notna(p) else np.nan
                )
                lines.append(f"  {era}: {dog_prob.mean():.3f} (n={len(dog_bets)})")
    lines.append("")


def section_sp_quality(cal, lines):
    """Section 4: SP Quality Mismatch."""
    lines.append("=" * 80)
    lines.append("4. SP QUALITY MISMATCH (|sp_season_ip_diff| quartiles)")
    lines.append("=" * 80)

    col = "sp_season_ip_diff"
    if col not in cal.columns:
        lines.append("  Feature not available in merged data.")
        lines.append("")
        return

    # Compute quartiles from 2017-2024 (stable reference)
    ref = cal[era_mask(cal, "2017-2024") & cal[col].notna()][col].abs()
    if len(ref) < 100:
        lines.append("  Insufficient reference data.")
        lines.append("")
        return

    q25, q50, q75 = ref.quantile([0.25, 0.5, 0.75])
    cal["sp_ip_quartile"] = pd.cut(
        cal[col].abs(),
        bins=[-0.001, q25, q50, q75, cal[col].abs().max() + 1],
        labels=["Q1 (close)", "Q2", "Q3", "Q4 (mismatch)"]
    )

    for thresh in [PRIMARY_THRESHOLD]:
        lines.append(f"\n  Threshold >= {thresh} runs:")
        lines.append(f"  {'Quartile':<18} | {'2017-19':>22} | {'2021-24':>22} | {'2025':>22}")
        lines.append("  " + "-" * 88)

        for q in ["Q1 (close)", "Q2", "Q3", "Q4 (mismatch)"]:
            parts = []
            for era in ["2017-2019", "2021-2024", "2025"]:
                sub = cal[era_mask(cal, era) & (cal["sp_ip_quartile"] == q)]
                s, _ = bet_stats_for_subset(sub, thresh)
                if s["n_bets"] == 0:
                    parts.append(f"{'0 bets':>22}")
                else:
                    parts.append(f"{s['n_bets']:>3}b {s['roi']:>+5.1f}% w{s['win_pct']:.0f}%")
            lines.append(f"  {q:<18} | {parts[0]:>22} | {parts[1]:>22} | {parts[2]:>22}")

    # Also show signed direction
    lines.append(f"\n  Signed sp_season_ip_diff (positive = home SP more IP):")
    lines.append(f"  {'Direction':<18} | {'2017-19':>22} | {'2021-24':>22} | {'2025':>22}")
    lines.append("  " + "-" * 88)
    for direction, mask_fn in [("Home SP better", lambda d: d[col] > 0),
                                ("Away SP better", lambda d: d[col] < 0)]:
        parts = []
        for era in ["2017-2019", "2021-2024", "2025"]:
            sub = cal[era_mask(cal, era)]
            sub = sub[mask_fn(sub)]
            s, _ = bet_stats_for_subset(sub, PRIMARY_THRESHOLD)
            if s["n_bets"] == 0:
                parts.append(f"{'0 bets':>22}")
            else:
                parts.append(f"{s['n_bets']:>3}b {s['roi']:>+5.1f}% w{s['win_pct']:.0f}%")
        lines.append(f"  {direction:<18} | {parts[0]:>22} | {parts[1]:>22} | {parts[2]:>22}")
    lines.append("")


def section_feature_distributions(cal, lines):
    """Section 5: Feature Distribution Shifts."""
    lines.append("=" * 80)
    lines.append("5. FEATURE DISTRIBUTION SHIFTS (KS tests)")
    lines.append("=" * 80)

    lines.append(f"\n  {'Feature':<28} | {'Era':>10} | {'Mean':>8} | {'Std':>8} | "
                 f"{'NaN%':>6} | {'KS stat':>8} | {'KS p':>8}")
    lines.append("  " + "-" * 95)

    for feat in LASSO_STABLE_9:
        if feat not in cal.columns:
            lines.append(f"  {feat:<28} | {'MISSING':>10}")
            continue

        ref_all = cal[era_mask(cal, "2017-2024")][feat]
        ref_early = cal[era_mask(cal, "2017-2019")][feat]
        ref_recent = cal[era_mask(cal, "2021-2024")][feat]
        s2025 = cal[era_mask(cal, "2025")][feat]

        for era_label, ref in [("2017-2019", ref_early), ("2021-2024", ref_recent),
                                ("2017-2024", ref_all), ("2025", s2025)]:
            nan_pct = ref.isna().mean() * 100
            mean = ref.mean()
            std = ref.std()

            # KS test: compare this era to all-prior
            if era_label != "2017-2024":
                valid_ref = ref_all.dropna()
                valid_era = ref.dropna()
                if len(valid_ref) > 10 and len(valid_era) > 10:
                    ks_stat, ks_p = ks_2samp(valid_ref, valid_era)
                else:
                    ks_stat, ks_p = np.nan, np.nan
            else:
                ks_stat, ks_p = np.nan, np.nan  # reference itself

            flag = " ***" if (not np.isnan(ks_p) and ks_p < 0.01) else ""
            ks_str = f"{ks_stat:.4f}" if not np.isnan(ks_stat) else "    ref"
            kp_str = f"{ks_p:.4f}" if not np.isnan(ks_p) else "    ref"

            feat_label = feat if era_label == "2017-2019" else ""
            lines.append(f"  {feat_label:<28} | {era_label:>10} | {mean:>+8.3f} | "
                         f"{std:>8.3f} | {nan_pct:>5.1f}% | {ks_str:>8} | {kp_str:>8}{flag}")
        lines.append("  " + "." * 95)

    lines.append("\n  *** = KS p < 0.01 (significant distribution shift vs 2017-2024)")
    lines.append("")


def section_feature_edge_correlation(cal, lines):
    """Section 6: Per-Feature Edge Correlation."""
    lines.append("=" * 80)
    lines.append("6. PER-FEATURE EDGE CORRELATION")
    lines.append("   corr(feature, market_residual) — features that stopped working")
    lines.append("=" * 80)

    cal["market_residual"] = cal["actual"] - cal["market_implied_margin"]

    lines.append(f"\n  {'Feature':<28} | {'2017-19':>10} | {'2021-24':>10} | "
                 f"{'2017-24':>10} | {'2025':>10} | {'Δ vs prior':>10}")
    lines.append("  " + "-" * 90)

    for feat in LASSO_STABLE_9:
        if feat not in cal.columns:
            lines.append(f"  {feat:<28} | MISSING")
            continue

        parts = []
        r_prior = np.nan
        for era in ["2017-2019", "2021-2024", "2017-2024", "2025"]:
            sub = cal[era_mask(cal, era)]
            r, _ = safe_corr(sub[feat], sub["market_residual"])
            parts.append(f"{r:>+.4f}" if not np.isnan(r) else "     N/A")
            if era == "2017-2024":
                r_prior = r

        # Delta: 2025 - prior
        r_2025, _ = safe_corr(
            cal[era_mask(cal, "2025")][feat],
            cal[era_mask(cal, "2025")]["market_residual"]
        )
        if not np.isnan(r_2025) and not np.isnan(r_prior):
            delta = r_2025 - r_prior
            delta_str = f"{delta:>+.4f}"
            # Flag sign flips or big drops
            if r_prior != 0 and np.sign(r_2025) != np.sign(r_prior):
                delta_str += " FLIP"
            elif abs(r_prior) > 0.01 and abs(r_2025) < abs(r_prior) * 0.5:
                delta_str += " DROP"
        else:
            delta_str = "     N/A"

        lines.append(f"  {feat:<28} | {parts[0]:>10} | {parts[1]:>10} | "
                     f"{parts[2]:>10} | {parts[3]:>10} | {delta_str:>10}")

    # Also show overall model correlation
    lines.append("")
    lines.append("  Overall model signal:")
    for era in ["2017-2019", "2021-2024", "2017-2024", "2025"]:
        sub = cal[era_mask(cal, era)]
        r_pred, _ = safe_corr(sub["calibrated_pred"], sub["actual"])
        r_edge, _ = safe_corr(sub["margin_edge"], sub["market_residual"])
        lines.append(f"    {era}: corr(pred, actual)={r_pred:+.3f}  "
                     f"corr(edge, mkt_resid)={r_edge:+.3f}")
    lines.append("")


def section_calibration(cal, lines):
    """Section 7: Calibration Breakdown."""
    lines.append("=" * 80)
    lines.append("7. CALIBRATION BREAKDOWN")
    lines.append("=" * 80)

    lines.append(f"\n  {'Metric':<35} | {'2017-19':>10} | {'2021-24':>10} | "
                 f"{'2017-24':>10} | {'2025':>10}")
    lines.append("  " + "-" * 85)

    metrics = [
        ("raw pred mean", lambda d: d["predicted"].mean()),
        ("raw pred std", lambda d: d["predicted"].std()),
        ("calibrated pred mean", lambda d: d["calibrated_pred"].mean()),
        ("calibrated pred std", lambda d: d["calibrated_pred"].std()),
        ("market_implied_margin mean", lambda d: d["market_implied_margin"].mean()),
        ("market_implied_margin std", lambda d: d["market_implied_margin"].std()),
        ("edge mean", lambda d: d["margin_edge"].mean()),
        ("edge std", lambda d: d["margin_edge"].std()),
        ("actual_margin mean", lambda d: d["actual"].mean()),
        ("actual_margin std", lambda d: d["actual"].std()),
        ("corr(raw_pred, actual)", lambda d: safe_corr(d["predicted"], d["actual"])[0]),
        ("corr(calib_pred, actual)", lambda d: safe_corr(d["calibrated_pred"], d["actual"])[0]),
        ("corr(edge, mkt_residual)", lambda d: safe_corr(
            d["margin_edge"], d["actual"] - d["market_implied_margin"])[0]),
        ("RMSE(pred, actual)", lambda d: root_mean_squared_error(d["actual"], d["predicted"])
            if len(d) > 0 else np.nan),
    ]

    for name, fn in metrics:
        parts = []
        for era in ["2017-2019", "2021-2024", "2017-2024", "2025"]:
            sub = cal[era_mask(cal, era)]
            try:
                val = fn(sub)
                parts.append(f"{val:>+10.4f}" if not np.isnan(val) else f"{'N/A':>10}")
            except Exception:
                parts.append(f"{'err':>10}")
        lines.append(f"  {name:<35} | {parts[0]} | {parts[1]} | {parts[2]} | {parts[3]}")

    # Local recalibration test: what if we recalibrate using only 2025 stats?
    lines.append(f"\n  Local recalibration test (rescale using 2025's own mean/std):")
    sub_2025 = cal[era_mask(cal, "2025")].copy()
    if len(sub_2025) > 10:
        m_mean = sub_2025["predicted"].mean()
        m_std = sub_2025["predicted"].std()
        mkt_mean = sub_2025["market_implied_margin"].mean()
        mkt_std = sub_2025["market_implied_margin"].std()
        sub_2025["local_calib"] = (sub_2025["predicted"] - m_mean) / m_std * mkt_std + mkt_mean
        sub_2025["local_edge"] = sub_2025["local_calib"] - sub_2025["market_implied_margin"]
        local_resid = sub_2025["actual"] - sub_2025["market_implied_margin"]

        r_global, _ = safe_corr(sub_2025["margin_edge"], local_resid)
        r_local, _ = safe_corr(sub_2025["local_edge"], local_resid)
        lines.append(f"    Global calibration: corr(edge, residual) = {r_global:+.4f}")
        lines.append(f"    Local calibration:  corr(edge, residual) = {r_local:+.4f}")
        lines.append(f"    (If local >> global, then calibration drift is a factor)")
    lines.append("")


def section_market_structure(cal, training_df, lines):
    """Section 8: Market Structure Changes."""
    lines.append("=" * 80)
    lines.append("8. MARKET STRUCTURE CHANGES")
    lines.append("=" * 80)

    lines.append(f"\n  {'Season':>7} | {'Games':>6} | {'H2H cov%':>8} | {'Avg books':>9} | "
                 f"{'Mean vig':>8} | {'Mean |ML|':>9} | {'Total cov%':>9}")
    lines.append("  " + "-" * 75)

    for season in sorted(cal["season"].unique()):
        sub = cal[cal["season"] == season]
        n_games = len(sub)
        h2h_cov = sub["consensus_h2h_home"].notna().mean() * 100

        # Num books from training data
        t_sub = training_df[training_df["game_pk"].isin(sub["game_pk"])]
        avg_books = t_sub["num_books"].mean() if "num_books" in t_sub.columns else np.nan

        # Vig = raw_home_prob + raw_away_prob - 1
        raw_h = sub["consensus_h2h_home"].apply(bt.american_to_implied_prob)
        raw_a = sub["consensus_h2h_away"].apply(bt.american_to_implied_prob)
        vig = (raw_h + raw_a - 1).mean()

        mean_ml = sub["consensus_h2h_home"].abs().mean()

        total_cov = 0
        if "consensus_total" in t_sub.columns:
            total_cov = t_sub["consensus_total"].notna().mean() * 100

        lines.append(f"  {int(season):>7} | {n_games:>6} | {h2h_cov:>7.1f}% | "
                     f"{avg_books:>9.1f} | {vig:>7.3f} | {mean_ml:>9.0f} | {total_cov:>8.1f}%")
    lines.append("")


def section_team_analysis(cal, lines):
    """Section 9: Team-Level Analysis."""
    lines.append("=" * 80)
    lines.append("9. TEAM-LEVEL ANALYSIS (2025, >= 1.5 runs)")
    lines.append("=" * 80)

    sub = cal[era_mask(cal, "2025")]
    _, bets = bet_stats_for_subset(sub, PRIMARY_THRESHOLD)

    if bets.empty:
        lines.append("  No 2025 bets.")
        lines.append("")
        return

    # For each bet, the "team we bet on" is home_team if side=HOME_ML, else away_team
    bets["bet_team"] = bets.apply(
        lambda r: r["home_team"] if r["side"] == "HOME_ML" else r["away_team"], axis=1
    )

    team_stats = []
    for team in sorted(bets["bet_team"].unique()):
        tb = bets[bets["bet_team"] == team]
        s = bt.compute_ml_stats(tb)
        s["team"] = team
        s["mean_edge"] = tb["margin_edge"].abs().mean()
        s["total_profit"] = tb["profit"].sum()
        team_stats.append(s)

    team_df = pd.DataFrame(team_stats).sort_values("profit", ascending=True)
    total_loss = bets["profit"].sum()

    lines.append(f"\n  Total 2025 P&L at >= {PRIMARY_THRESHOLD}: ${total_loss:+,.0f}")
    lines.append(f"\n  {'Team':<25} | {'Bets':>4} | {'W-L':>7} | {'ROI':>7} | "
                 f"{'P&L':>8} | {'% of loss':>9} | {'Avg Edge':>8}")
    lines.append("  " + "-" * 80)

    for _, row in team_df.iterrows():
        wl = f"{row['wins']}-{row['losses']}"
        pct_loss = row["profit"] / abs(total_loss) * 100 if total_loss != 0 else 0
        lines.append(f"  {row['team']:<25} | {row['n_bets']:>4} | {wl:>7} | "
                     f"{row['roi']:>+6.1f}% | ${row['profit']:>+7.0f} | "
                     f"{pct_loss:>+8.1f}% | {row['mean_edge']:>7.2f}")
    lines.append("")


def section_edge_distribution(cal, lines):
    """Section 10: Edge Distribution Comparison."""
    lines.append("=" * 80)
    lines.append("10. EDGE DISTRIBUTION COMPARISON")
    lines.append("=" * 80)

    lines.append(f"\n  {'Metric':<28} | {'2017-19':>10} | {'2021-24':>10} | "
                 f"{'2017-24':>10} | {'2025':>10}")
    lines.append("  " + "-" * 80)

    dist_metrics = [
        ("mean", lambda e: e.mean()),
        ("std", lambda e: e.std()),
        ("skew", lambda e: e.skew()),
        ("kurtosis", lambda e: e.kurtosis()),
        ("10th percentile", lambda e: e.quantile(0.10)),
        ("25th percentile", lambda e: e.quantile(0.25)),
        ("median", lambda e: e.quantile(0.50)),
        ("75th percentile", lambda e: e.quantile(0.75)),
        ("90th percentile", lambda e: e.quantile(0.90)),
        ("% |edge| >= 0.5", lambda e: (e.abs() >= 0.5).mean() * 100),
        ("% |edge| >= 1.0", lambda e: (e.abs() >= 1.0).mean() * 100),
        ("% |edge| >= 1.5", lambda e: (e.abs() >= 1.5).mean() * 100),
        ("% |edge| >= 2.0", lambda e: (e.abs() >= 2.0).mean() * 100),
    ]

    for name, fn in dist_metrics:
        parts = []
        for era in ["2017-2019", "2021-2024", "2017-2024", "2025"]:
            sub = cal[era_mask(cal, era)]
            val = fn(sub["margin_edge"])
            parts.append(f"{val:>10.3f}")
        lines.append(f"  {name:<28} | {parts[0]} | {parts[1]} | {parts[2]} | {parts[3]}")

    # KS test: 2025 vs prior
    ref_edges = cal[era_mask(cal, "2017-2024")]["margin_edge"].dropna()
    s2025_edges = cal[era_mask(cal, "2025")]["margin_edge"].dropna()
    if len(ref_edges) > 10 and len(s2025_edges) > 10:
        ks_stat, ks_p = ks_2samp(ref_edges, s2025_edges)
        lines.append(f"\n  KS test (2025 vs 2017-2024): stat={ks_stat:.4f}, p={ks_p:.4f}")

    # Are bigger edges MORE wrong in 2025?
    lines.append(f"\n  ROI by edge magnitude (2025 vs prior at threshold sweep):")
    lines.append(f"  {'Threshold':<12} | {'2017-2024 ROI':>13} | {'2025 ROI':>10} | {'Δ':>8}")
    lines.append("  " + "-" * 50)
    for thresh in THRESHOLD_SWEEP:
        s_prior, _ = bet_stats_for_subset(cal[era_mask(cal, "2017-2024")], thresh)
        s_2025, _ = bet_stats_for_subset(cal[era_mask(cal, "2025")], thresh)
        delta = s_2025["roi"] - s_prior["roi"] if s_2025["n_bets"] > 0 and s_prior["n_bets"] > 0 else np.nan
        delta_str = f"{delta:>+7.1f}pp" if not np.isnan(delta) else "    N/A"
        lines.append(f"  >= {thresh:<8} | {s_prior['roi']:>+12.1f}% | "
                     f"{s_2025['roi']:>+9.1f}% | {delta_str}")
    lines.append("")


# ═══════════════════════════════════════════════════════════════
# Executive Summary
# ═══════════════════════════════════════════════════════════════

def build_executive_summary(cal, training_df):
    """Scan for key anomalies and return bullet-point findings."""
    findings = []

    # 1. Monthly ROI < -25%
    cal_copy = cal.copy()
    cal_copy["month"] = pd.to_datetime(cal_copy["date"]).dt.month
    cal_copy["month_name"] = pd.to_datetime(cal_copy["date"]).dt.strftime("%b")
    for m in sorted(cal_copy[era_mask(cal_copy, "2025")]["month"].unique()):
        sub = cal_copy[era_mask(cal_copy, "2025") & (cal_copy["month"] == m)]
        s, _ = bet_stats_for_subset(sub, PRIMARY_THRESHOLD)
        if s["n_bets"] >= 10 and s["roi"] < -25:
            mn = sub["month_name"].iloc[0]
            findings.append(f"TEMPORAL: {mn} 2025 ROI = {s['roi']:+.1f}% ({s['n_bets']} bets)")

    # 2. Home/away divergence > 10pp
    for side_label, edge_cond in [("Home", True), ("Away", False)]:
        sub_2025 = cal[era_mask(cal, "2025")]
        sub_prior = cal[era_mask(cal, "2017-2024")]
        if side_label == "Home":
            s25, _ = bet_stats_for_subset(sub_2025[sub_2025["margin_edge"] > 0], PRIMARY_THRESHOLD)
            spr, _ = bet_stats_for_subset(sub_prior[sub_prior["margin_edge"] > 0], PRIMARY_THRESHOLD)
        else:
            s25, _ = bet_stats_for_subset(sub_2025[sub_2025["margin_edge"] < 0], PRIMARY_THRESHOLD)
            spr, _ = bet_stats_for_subset(sub_prior[sub_prior["margin_edge"] < 0], PRIMARY_THRESHOLD)
        if s25["n_bets"] > 0 and spr["n_bets"] > 0:
            diff = s25["roi"] - spr["roi"]
            if abs(diff) > 10:
                findings.append(f"SIDE BIAS: {side_label} bets 2025 ROI {s25['roi']:+.1f}% "
                                f"vs prior {spr['roi']:+.1f}% (Δ{diff:+.1f}pp)")

    # 3. Dog/fav divergence > 10pp
    for bet_type in ["Dog", "Favorite"]:
        sub_2025 = cal[era_mask(cal, "2025")]
        sub_prior = cal[era_mask(cal, "2017-2024")]
        _, bets25 = bet_stats_for_subset(sub_2025, PRIMARY_THRESHOLD)
        _, betspr = bet_stats_for_subset(sub_prior, PRIMARY_THRESHOLD)
        if not bets25.empty and not betspr.empty:
            if bet_type == "Dog":
                s25 = bt.compute_ml_stats(bets25[bets25["is_dog"]])
                spr = bt.compute_ml_stats(betspr[betspr["is_dog"]])
            else:
                s25 = bt.compute_ml_stats(bets25[~bets25["is_dog"]])
                spr = bt.compute_ml_stats(betspr[~betspr["is_dog"]])
            if s25["n_bets"] > 0 and spr["n_bets"] > 0:
                diff = s25["roi"] - spr["roi"]
                if abs(diff) > 10:
                    findings.append(f"DOG/FAV: {bet_type} bets 2025 ROI {s25['roi']:+.1f}% "
                                    f"vs prior {spr['roi']:+.1f}% (Δ{diff:+.1f}pp)")

    # 4. Feature KS p < 0.01 or correlation flip
    cal_copy["market_residual"] = cal_copy["actual"] - cal_copy["market_implied_margin"]
    ref_all = cal_copy[era_mask(cal_copy, "2017-2024")]
    s2025 = cal_copy[era_mask(cal_copy, "2025")]

    for feat in LASSO_STABLE_9:
        if feat not in cal_copy.columns:
            continue
        # KS test
        v_ref = ref_all[feat].dropna()
        v_2025 = s2025[feat].dropna()
        if len(v_ref) > 10 and len(v_2025) > 10:
            _, ks_p = ks_2samp(v_ref, v_2025)
            if ks_p < 0.01:
                findings.append(f"DISTRIBUTION: {feat} shifted (KS p={ks_p:.4f})")

        # Correlation flip
        r_prior, _ = safe_corr(ref_all[feat], ref_all["market_residual"])
        r_2025, _ = safe_corr(s2025[feat], s2025["market_residual"])
        if (not np.isnan(r_prior) and not np.isnan(r_2025) and
                abs(r_prior) > 0.01 and np.sign(r_prior) != np.sign(r_2025)):
            findings.append(f"CORRELATION FLIP: {feat} prior={r_prior:+.3f} → 2025={r_2025:+.3f}")

    # 5. Calibration: edge-residual correlation negative
    r_edge_2025, _ = safe_corr(
        s2025["margin_edge"],
        s2025["actual"] - s2025["market_implied_margin"]
    )
    if not np.isnan(r_edge_2025) and r_edge_2025 < 0:
        findings.append(f"CALIBRATION: corr(edge, residual) = {r_edge_2025:+.3f} in 2025 (NEGATIVE)")

    # 6. Teams contributing >30% of losses
    _, bets_2025 = bet_stats_for_subset(s2025, PRIMARY_THRESHOLD)
    if not bets_2025.empty:
        total_loss = bets_2025["profit"].sum()
        if total_loss < 0:
            bets_2025["bet_team"] = bets_2025.apply(
                lambda r: r["home_team"] if r["side"] == "HOME_ML" else r["away_team"], axis=1
            )
            for team in bets_2025["bet_team"].unique():
                tb = bets_2025[bets_2025["bet_team"] == team]
                team_loss = tb["profit"].sum()
                pct = team_loss / total_loss * 100
                if pct > 30:
                    findings.append(f"TEAM: {team} contributes {pct:.0f}% of 2025 losses "
                                    f"(${team_loss:+,.0f})")

    return findings


# ═══════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════

def main():
    log.info("=" * 60)
    log.info("2025 SEASON DIAGNOSTIC DEEP DIVE")
    log.info("=" * 60)

    # Load Lasso no-market OOF predictions
    lasso_path = MODELS_DIR / "mlb_oof_margin_lasso_nomarket_predictions.csv"
    if not lasso_path.exists():
        log.error(f"Lasso predictions not found: {lasso_path}")
        log.error("Run: python3 06c_ridge_lasso_experiment.py --no-market")
        return
    margin_oof = pd.read_csv(lasso_path)
    margin_oof["date"] = margin_oof["date"].astype(str).str[:10]
    log.info(f"Loaded Lasso OOF: {len(margin_oof)} games, "
             f"seasons {margin_oof['season'].min()}-{margin_oof['season'].max()}")

    # Load training data
    training_df = bt.load_training_data_for_odds()

    # Compute RMSE
    margin_rmse = root_mean_squared_error(margin_oof["actual"], margin_oof["predicted"])
    log.info(f"OOF margin RMSE: {margin_rmse:.2f}")

    # Match with odds and calibrate
    margin_matched = bt.match_with_odds(margin_oof, training_df)
    cal = bt.calibrate_predictions(margin_matched, margin_rmse)
    cal = cal.drop_duplicates(subset=["game_pk"], keep="first")
    log.info(f"Calibrated games: {len(cal)}")

    # Merge Lasso features from training data
    feature_cols = ["game_pk"] + LASSO_STABLE_9
    available = [c for c in feature_cols if c in training_df.columns]
    feat_merge = training_df[available].drop_duplicates(subset=["game_pk"], keep="first")
    existing = set(cal.columns)
    new_cols = [c for c in feat_merge.columns if c not in existing or c == "game_pk"]
    cal = cal.merge(feat_merge[new_cols], on="game_pk", how="left")

    # Verify baseline: 2025 ROI at >= 1.5
    s2025, _ = bet_stats_for_subset(cal[era_mask(cal, "2025")], PRIMARY_THRESHOLD)
    log.info(f"2025 baseline: {s2025['n_bets']} bets, ROI {s2025['roi']:+.1f}%")

    # Build executive summary first (needs all data)
    findings = build_executive_summary(cal, training_df)

    # Build report
    report = []
    report.append("MLB 2025 SEASON DIAGNOSTIC REPORT")
    report.append("=" * 80)
    report.append(f"Generated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')}")
    report.append(f"Model: Lasso no-market (walk-forward, 9 folds)")
    report.append(f"OOF RMSE: {margin_rmse:.2f}")
    report.append(f"Calibrated games: {len(cal)}")
    report.append(f"Primary threshold: {PRIMARY_THRESHOLD} runs")
    report.append("")

    # Era summary
    report.append("ERA SUMMARY:")
    for era in ["2017-2019", "2021-2024", "2017-2024", "2025"]:
        sub = cal[era_mask(cal, era)]
        s, _ = bet_stats_for_subset(sub, PRIMARY_THRESHOLD)
        report.append(format_stats_line(era, s))
    report.append("")

    # Executive summary
    report.append("=" * 80)
    report.append("EXECUTIVE SUMMARY — KEY ANOMALIES")
    report.append("=" * 80)
    if findings:
        for f in findings:
            report.append(f"  • {f}")
    else:
        report.append("  No major anomalies detected.")
    report.append("")

    # Run all sections
    section_mlb_environment(training_df, report)
    section_temporal(cal, report)
    section_side_bias(cal, report)
    section_dog_vs_fav(cal, report)
    section_sp_quality(cal, report)
    section_feature_distributions(cal, report)
    section_feature_edge_correlation(cal, report)
    section_calibration(cal, report)
    section_market_structure(cal, training_df, report)
    section_team_analysis(cal, report)
    section_edge_distribution(cal, report)

    # Integrity checks
    report.append("=" * 80)
    report.append("INTEGRITY CHECKS")
    report.append("=" * 80)

    # Home + away should sum to total
    sub_2025 = cal[era_mask(cal, "2025")]
    _, bets_all = bet_stats_for_subset(sub_2025, PRIMARY_THRESHOLD)
    if not bets_all.empty:
        _, bets_home = bet_stats_for_subset(sub_2025[sub_2025["margin_edge"] > 0], PRIMARY_THRESHOLD)
        _, bets_away = bet_stats_for_subset(sub_2025[sub_2025["margin_edge"] < 0], PRIMARY_THRESHOLD)
        n_home = len(bets_home) if not bets_home.empty else 0
        n_away = len(bets_away) if not bets_away.empty else 0
        report.append(f"  Home ({n_home}) + Away ({n_away}) = {n_home + n_away} "
                      f"vs Total ({len(bets_all)}) — {'OK' if n_home + n_away == len(bets_all) else 'MISMATCH'}")

        dog_bets = bets_all[bets_all["is_dog"]]
        fav_bets = bets_all[~bets_all["is_dog"]]
        n_dog = len(dog_bets)
        n_fav = len(fav_bets)
        report.append(f"  Dog ({n_dog}) + Fav ({n_fav}) = {n_dog + n_fav} "
                      f"vs Total ({len(bets_all)}) — {'OK' if n_dog + n_fav == len(bets_all) else 'MISMATCH'}")

    # Market residual mean should be ~0
    resid = cal["actual"] - cal["market_implied_margin"]
    report.append(f"  Market residual mean: {resid.mean():+.3f} (should be ~0)")
    report.append("")

    # Save report
    report_text = "\n".join(report)
    report_path = MODELS_DIR / "mlb_2025_diagnostic_report.txt"
    with open(report_path, "w") as f:
        f.write(report_text)
    log.info(f"\nSaved diagnostic report -> {report_path}")

    # Print executive summary to stdout
    print(f"\n{'='*70}")
    print("2025 DIAGNOSTIC — EXECUTIVE SUMMARY")
    print(f"{'='*70}")
    print(f"  Model: Lasso no-market | RMSE: {margin_rmse:.2f}")
    print(f"  2025 at >= {PRIMARY_THRESHOLD}: {s2025['n_bets']} bets, "
          f"ROI {s2025['roi']:+.1f}%, p={s2025['p_value']:.3f}")
    print()
    if findings:
        for f in findings:
            print(f"  • {f}")
    else:
        print("  No major anomalies detected.")
    print(f"\n  Full report: {report_path}")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()
