"""
11 — Segmented Backtest: Find Profitable Subsets in No-Market Model
====================================================================
Slices the no-market model's OOF predictions by situational segments
to identify subsets where the baseball signal concentrates.

Segments:
  1. Market Closeness — how uncertain is the market?
  2. SP Quality Differential — magnitude of SP gap
  3. Season Phase — month bins (early season = more uncertainty)
  4. Model Confidence — magnitude of raw prediction
  5. Feature Agreement — how many selected features agree on the same side

Cross-segments (2D):
  - Market Closeness x Model Confidence
  - SP Quality Gap x Season Phase

Safeguards:
  - Minimum 50 bets to report, 100 for threshold search
  - Bonferroni correction (~50 tests → p < 0.001 required)
  - Best threshold flagged as in-sample
  - Global calibration only (no per-segment recalibration)

Run: python3 11_segmented_backtest.py
"""

import sys
import json
import importlib
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.stats import norm
from config import MODELS_DIR, HISTORICAL_DIR, get_logger

log = get_logger("11_segmented_backtest")

# Import 10_backtest_mlb (digit prefix requires importlib)
bt = importlib.import_module("10_backtest_mlb")

# Segment parameters
MIN_BETS_REPORT = 50
MIN_BETS_THRESHOLD_SEARCH = 100
BONFERRONI_P = 0.001  # ~50 tests
PROD_THRESHOLD = 0.5  # runs
THRESHOLD_SWEEP = [0.0, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0]

# The 8 no-market margin features (all _diff convention: positive = home advantage)
NOMARKET_MARGIN_FEATURES = [
    "sp_season_ip_diff", "sp_k_pct_diff", "sp_k_bb_diff", "sp_xfip_diff",
    "team_run_diff_10_diff", "bb_rate_diff",
    "lineup_ops_vs_hand_diff", "sp_whiff_x_lineup_ops_diff",
]

# Features where LOWER values favor home (inverted: xFIP)
# xFIP: home_sp_xfip - away_sp_xfip → negative = home better
# All others: positive = home better
INVERTED_FEATURES = {"sp_xfip_diff"}


def load_nomarket_features():
    """Load selected features from the no-market model."""
    path = MODELS_DIR / "mlb_selected_features_nomarket.json"
    if path.exists():
        with open(path) as f:
            data = json.load(f)
        return data.get("margin_features", NOMARKET_MARGIN_FEATURES)
    return NOMARKET_MARGIN_FEATURES


def add_segment_columns(calibrated_df, training_df):
    """
    Merge feature columns from training data and compute segment bins.

    Returns calibrated_df with added segment columns.
    """
    df = calibrated_df.copy()

    # Merge feature columns from training data
    feature_cols = [
        "game_pk", "date",
        # SP stats for SP quality segment
        "sp_season_ip_diff", "sp_k_pct_diff", "sp_k_bb_diff", "sp_xfip_diff",
        # Batting/lineup for feature agreement
        "team_run_diff_10_diff", "bb_rate_diff",
        "lineup_ops_vs_hand_diff", "sp_whiff_x_lineup_ops_diff",
        # Also grab lineup_bb_k_ratio_diff (survived Boruta but not in the 8)
    ]
    available = [c for c in feature_cols if c in training_df.columns]
    merge_cols = training_df[available].drop_duplicates(subset=["game_pk"], keep="first")

    # Only merge columns not already in df
    existing = set(df.columns)
    new_cols = [c for c in merge_cols.columns if c not in existing or c == "game_pk"]
    df = df.merge(merge_cols[new_cols], on="game_pk", how="left", suffixes=("", "_feat"))

    # ── Segment 1: Market Closeness ──
    # |market_home_prob - 0.5| measures how uncertain the market is
    df["market_closeness"] = (df["market_home_prob"] - 0.5).abs()
    df["seg_market"] = pd.cut(
        df["market_closeness"],
        bins=[-0.001, 0.05, 0.10, 0.15, 1.0],
        labels=["Toss-up (<5%)", "Lean (5-10%)", "Clear fav (10-15%)", "Strong fav (15%+)"]
    )

    # ── Segment 2: SP Quality Differential ──
    # Use sp_season_ip_diff as primary (model's #1 feature)
    sp_col = "sp_season_ip_diff"
    if sp_col in df.columns:
        sp_abs = df[sp_col].abs()
        # Quartile bins (handle NaN)
        valid_mask = sp_abs.notna()
        if valid_mask.sum() > 100:
            q25, q50, q75 = sp_abs[valid_mask].quantile([0.25, 0.5, 0.75])
            df["seg_sp_gap"] = pd.cut(
                sp_abs,
                bins=[-0.001, q25, q50, q75, sp_abs.max() + 1],
                labels=["Q1 (small)", "Q2", "Q3", "Q4 (large)"]
            )
        else:
            df["seg_sp_gap"] = np.nan
    else:
        df["seg_sp_gap"] = np.nan

    # ── Segment 3: Season Phase ──
    df["month"] = pd.to_datetime(df["date"]).dt.month
    df["seg_season_phase"] = pd.cut(
        df["month"],
        bins=[0, 4, 6, 8, 13],
        labels=["Mar-Apr", "May-Jun", "Jul-Aug", "Sep+"]
    )

    # ── Segment 4: Model Confidence ──
    pred_abs = df["predicted"].abs()
    valid_pred = pred_abs.notna()
    if valid_pred.sum() > 100:
        pq25, pq50, pq75 = pred_abs[valid_pred].quantile([0.25, 0.5, 0.75])
        df["seg_confidence"] = pd.cut(
            pred_abs,
            bins=[-0.001, pq25, pq50, pq75, pred_abs.max() + 1],
            labels=["Q1 (low)", "Q2", "Q3", "Q4 (high)"]
        )
    else:
        df["seg_confidence"] = np.nan

    # ── Segment 5: Feature Agreement ──
    margin_feats = load_nomarket_features()
    available_feats = [f for f in margin_feats if f in df.columns]

    if available_feats:
        home_count = pd.DataFrame(index=df.index)
        total_count = pd.DataFrame(index=df.index)

        for feat in available_feats:
            if feat in INVERTED_FEATURES:
                # For xFIP: negative = home better
                home_count[feat] = (df[feat] < 0).astype(float)
            else:
                # For most features: positive = home better
                home_count[feat] = (df[feat] > 0).astype(float)
            total_count[feat] = df[feat].notna().astype(float)

        home_sum = home_count.sum(axis=1)
        total_sum = total_count.sum(axis=1)
        away_sum = total_sum - home_sum

        # Agreement = fraction of features agreeing on the majority side
        agree_count = pd.concat([home_sum, away_sum], axis=1).max(axis=1)
        df["feature_agreement"] = agree_count / total_sum.clip(lower=1)
        df["feature_agree_count"] = agree_count.astype(int)
        df["feature_total_count"] = total_sum.astype(int)

        df["seg_agreement"] = pd.cut(
            df["feature_agreement"],
            bins=[-0.001, 0.625, 0.75, 1.01],
            labels=["Mixed (<=5/8)", "Moderate (6/8)", "Strong (7-8/8)"]
        )
    else:
        df["feature_agreement"] = np.nan
        df["seg_agreement"] = np.nan

    return df


def run_segment_backtest(df, seg_col, segment_name):
    """
    Run backtest per bin of a segment column.

    Returns dict with per-bin stats at production threshold + best threshold sweep.
    """
    results = {}
    bins = df[seg_col].dropna().unique()

    for bin_label in sorted(bins, key=str):
        mask = df[seg_col] == bin_label
        subset = df[mask]

        if len(subset) < MIN_BETS_REPORT:
            results[str(bin_label)] = {
                "n_games": len(subset), "n_bets": 0,
                "note": f"< {MIN_BETS_REPORT} games, skipped"
            }
            continue

        # Production threshold stats
        bets = bt.simulate_ml_bets(subset, PROD_THRESHOLD)
        stats = bt.compute_ml_stats(bets)
        stats["n_games"] = len(subset)

        # Threshold sweep (in-sample, flagged)
        best_thresh = None
        best_roi = -999
        if len(subset) >= MIN_BETS_THRESHOLD_SEARCH:
            for thresh in THRESHOLD_SWEEP:
                t_bets = bt.simulate_ml_bets(subset, thresh)
                t_stats = bt.compute_ml_stats(t_bets)
                if t_stats["n_bets"] >= MIN_BETS_REPORT and t_stats["roi"] > best_roi:
                    best_roi = t_stats["roi"]
                    best_thresh = thresh

        stats["best_threshold"] = best_thresh
        stats["best_roi_insample"] = best_roi if best_thresh is not None else None
        results[str(bin_label)] = stats

    return results


def run_cross_segment(df, seg_col1, seg_col2, cross_name):
    """
    Run 2D cross-segment backtest.

    Returns dict of dicts: {(bin1, bin2): stats}.
    """
    results = {}
    bins1 = sorted(df[seg_col1].dropna().unique(), key=str)
    bins2 = sorted(df[seg_col2].dropna().unique(), key=str)

    for b1 in bins1:
        for b2 in bins2:
            mask = (df[seg_col1] == b1) & (df[seg_col2] == b2)
            subset = df[mask]
            key = f"{b1} x {b2}"

            if len(subset) < MIN_BETS_REPORT:
                results[key] = {
                    "n_games": len(subset), "n_bets": 0,
                    "note": f"< {MIN_BETS_REPORT} games"
                }
                continue

            bets = bt.simulate_ml_bets(subset, PROD_THRESHOLD)
            stats = bt.compute_ml_stats(bets)
            stats["n_games"] = len(subset)
            results[key] = stats

    return results


def build_segmented_report(df, segment_results, cross_results):
    """Build human-readable segmented backtest report."""
    lines = []
    lines.append("MLB SEGMENTED BACKTEST REPORT (No-Market Model)")
    lines.append("=" * 80)
    lines.append(f"Generated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')}")
    lines.append(f"Production threshold: {PROD_THRESHOLD} runs")
    lines.append(f"Bonferroni p-value: {BONFERRONI_P} (~50 tests)")
    lines.append(f"Min bets to report: {MIN_BETS_REPORT}")
    lines.append(f"Min bets for threshold search: {MIN_BETS_THRESHOLD_SEARCH}")
    lines.append(f"Total OOF games with H2H odds: {len(df)}")
    lines.append("")

    # Overall baseline
    all_bets = bt.simulate_ml_bets(df, PROD_THRESHOLD)
    all_stats = bt.compute_ml_stats(all_bets)
    lines.append(f"OVERALL BASELINE (>= {PROD_THRESHOLD} runs):")
    lines.append(f"  Bets: {all_stats['n_bets']}, W-L: {all_stats['wins']}-{all_stats['losses']}, "
                 f"Win%: {all_stats['win_pct']:.1f}%, ROI: {all_stats['roi']:+.1f}%, "
                 f"Dog%: {all_stats['dog_pct']:.0f}%, p: {all_stats['p_value']:.4f}")
    lines.append("")

    # Each segment
    for seg_name, results in segment_results.items():
        lines.append("=" * 80)
        lines.append(f"SEGMENT: {seg_name}")
        lines.append("=" * 80)
        lines.append(
            f"  {'Bin':<25} | {'Games':>6} | {'Bets':>5} | {'W-L':>10} | "
            f"{'Win%':>6} | {'ROI':>7} | {'Dog%':>5} | {'p-val':>7} | {'Best*':>7}"
        )
        lines.append("  " + "-" * 95)

        for bin_label, stats in results.items():
            if stats.get("note"):
                lines.append(f"  {bin_label:<25} | {stats['n_games']:>6} | {stats.get('note', '')}")
                continue

            wl = f"{stats['wins']}-{stats['losses']}"
            best_str = ""
            if stats.get("best_threshold") is not None:
                best_str = f"{stats['best_roi_insample']:+.1f}%"

            # Flag significant results
            sig = ""
            if stats["p_value"] < BONFERRONI_P and stats["roi"] > 0:
                sig = " ***"
            elif stats["p_value"] < 0.05 and stats["roi"] > 0:
                sig = " *"

            lines.append(
                f"  {bin_label:<25} | {stats['n_games']:>6} | {stats['n_bets']:>5} | "
                f"{wl:>10} | {stats['win_pct']:>5.1f}% | {stats['roi']:>+6.1f}% | "
                f"{stats['dog_pct']:>4.0f}% | {stats['p_value']:>7.4f} | "
                f"{best_str:>7}{sig}"
            )

        lines.append("")
        lines.append("  * p < 0.05 (nominal)   *** p < 0.001 (Bonferroni-significant)")
        lines.append("  Best* = best in-sample threshold ROI (DO NOT use for decisions)")
        lines.append("")

    # Cross-segments
    for cross_name, results in cross_results.items():
        lines.append("=" * 80)
        lines.append(f"CROSS-SEGMENT: {cross_name}")
        lines.append("=" * 80)
        lines.append(
            f"  {'Combination':<45} | {'Games':>6} | {'Bets':>5} | "
            f"{'Win%':>6} | {'ROI':>7} | {'p-val':>7}"
        )
        lines.append("  " + "-" * 90)

        # Sort by ROI descending for readability
        sorted_items = sorted(results.items(),
                              key=lambda x: x[1].get("roi", -999), reverse=True)

        for key, stats in sorted_items:
            if stats.get("note"):
                lines.append(f"  {key:<45} | {stats['n_games']:>6} | {stats.get('note', '')}")
                continue

            sig = ""
            if stats["p_value"] < BONFERRONI_P and stats["roi"] > 0:
                sig = " ***"
            elif stats["p_value"] < 0.05 and stats["roi"] > 0:
                sig = " *"

            wl = f"{stats['wins']}-{stats['losses']}"
            lines.append(
                f"  {key:<45} | {stats['n_games']:>6} | {stats['n_bets']:>5} | "
                f"{stats['win_pct']:>5.1f}% | {stats['roi']:>+6.1f}% | "
                f"{stats['p_value']:>7.4f}{sig}"
            )

        lines.append("")
        lines.append("  * p < 0.05 (nominal)   *** p < 0.001 (Bonferroni-significant)")
        lines.append("")

    # Integrity check
    lines.append("=" * 80)
    lines.append("INTEGRITY CHECKS")
    lines.append("=" * 80)

    for seg_name, results in segment_results.items():
        total_games = sum(s.get("n_games", 0) for s in results.values())
        total_bets = sum(s.get("n_bets", 0) for s in results.values()
                         if not s.get("note"))
        lines.append(f"  {seg_name}: {total_games} games across bins "
                     f"(expected ~{len(df)}), {total_bets} bets at prod threshold")

    lines.append(f"  Overall: {all_stats['n_bets']} bets at prod threshold")
    lines.append("")

    # Summary of interesting findings
    lines.append("=" * 80)
    lines.append("SUMMARY OF FINDINGS")
    lines.append("=" * 80)

    interesting = []
    for seg_name, results in segment_results.items():
        for bin_label, stats in results.items():
            if stats.get("note"):
                continue
            if stats["roi"] > 0 and stats["n_bets"] >= MIN_BETS_REPORT:
                interesting.append({
                    "segment": seg_name,
                    "bin": bin_label,
                    "roi": stats["roi"],
                    "p_value": stats["p_value"],
                    "n_bets": stats["n_bets"],
                    "bonferroni": stats["p_value"] < BONFERRONI_P,
                })

    for cross_name, results in cross_results.items():
        for key, stats in results.items():
            if stats.get("note"):
                continue
            if stats["roi"] > 0 and stats["n_bets"] >= MIN_BETS_REPORT:
                interesting.append({
                    "segment": cross_name,
                    "bin": key,
                    "roi": stats["roi"],
                    "p_value": stats["p_value"],
                    "n_bets": stats["n_bets"],
                    "bonferroni": stats["p_value"] < BONFERRONI_P,
                })

    if interesting:
        interesting.sort(key=lambda x: x["roi"], reverse=True)
        lines.append(f"Segments with positive ROI at {PROD_THRESHOLD}-run threshold:")
        lines.append("")
        for item in interesting:
            bf = "BONFERRONI" if item["bonferroni"] else "nominal" if item["p_value"] < 0.05 else "not sig"
            lines.append(
                f"  {item['segment']} / {item['bin']}: "
                f"ROI {item['roi']:+.1f}%, {item['n_bets']} bets, "
                f"p={item['p_value']:.4f} ({bf})"
            )
        lines.append("")

        bonf_count = sum(1 for i in interesting if i["bonferroni"])
        nom_count = sum(1 for i in interesting if i["p_value"] < 0.05 and not i["bonferroni"])
        lines.append(f"  Bonferroni-significant (p < {BONFERRONI_P}): {bonf_count}")
        lines.append(f"  Nominally significant (p < 0.05): {nom_count}")
        lines.append(f"  Expected by chance (~50 tests): ~2-3 at p<0.05, ~0 at Bonferroni")
    else:
        lines.append("  No segments with positive ROI found at production threshold.")

    lines.append("")

    # Notes
    lines.append("=" * 80)
    lines.append("NOTES")
    lines.append("=" * 80)
    lines.append("- All predictions are truly out-of-sample (walk-forward, per-fold Boruta)")
    lines.append("- Calibration is GLOBAL — no per-segment recalibration (that would be leakage)")
    lines.append("- 'Best*' threshold ROI is IN-SAMPLE — do not use for trading decisions")
    lines.append(f"- Bonferroni threshold: p < {BONFERRONI_P} for ~50 segment tests")
    lines.append("- Feature agreement uses sign of each _diff feature to count home/away support")
    lines.append("- For sp_xfip_diff: negative = home SP better (inverted convention)")
    lines.append("- Market closeness = |market_home_prob - 0.5| from de-vigged consensus H2H")

    return "\n".join(lines)


def main():
    log.info("=" * 60)
    log.info("MLB SEGMENTED BACKTEST (No-Market Model)")
    log.info("=" * 60)

    # Load data
    margin_oof, _ = bt.load_oof_predictions("_nomarket")
    training_df = bt.load_training_data_for_odds()

    if margin_oof is None:
        log.error("No margin OOF predictions. Run 06_train_mlb_model.py --no-market first.")
        sys.exit(1)

    # Compute actual RMSE for calibration
    from sklearn.metrics import root_mean_squared_error
    margin_rmse = root_mean_squared_error(margin_oof["actual"], margin_oof["predicted"])
    log.info(f"OOF margin RMSE: {margin_rmse:.2f}")

    # Match with odds and calibrate (global)
    margin_matched = bt.match_with_odds(margin_oof, training_df)
    calibrated = bt.calibrate_predictions(margin_matched, margin_rmse)
    log.info(f"Calibrated games with H2H odds: {len(calibrated)}")

    # Deduplicate calibrated df (walk-forward can produce duplicate game_pk)
    pre_dedup = len(calibrated)
    calibrated = calibrated.drop_duplicates(subset=["game_pk"], keep="first")
    if len(calibrated) < pre_dedup:
        log.info(f"Deduplicated calibrated: {pre_dedup} -> {len(calibrated)} games")

    # Add segment columns (merges features from training data)
    df = add_segment_columns(calibrated, training_df)
    log.info(f"Segments computed. Columns: {sorted([c for c in df.columns if c.startswith('seg_')])}")

    # Verify overall baseline matches standard backtest
    all_bets = bt.simulate_ml_bets(df, PROD_THRESHOLD)
    all_stats = bt.compute_ml_stats(all_bets)
    log.info(f"Overall baseline: {all_stats['n_bets']} bets, "
             f"ROI {all_stats['roi']:+.1f}%, p={all_stats['p_value']:.4f}")

    # ── Run segment backtests ──
    segment_results = {}

    segments = [
        ("Market Closeness (|prob - 0.5|)", "seg_market"),
        ("SP Quality Gap (|sp_season_ip_diff|)", "seg_sp_gap"),
        ("Season Phase (month)", "seg_season_phase"),
        ("Model Confidence (|prediction|)", "seg_confidence"),
        ("Feature Agreement (margin features)", "seg_agreement"),
    ]

    for seg_name, seg_col in segments:
        log.info(f"\nSegment: {seg_name}")
        non_null = df[seg_col].notna().sum()
        log.info(f"  Non-null: {non_null}/{len(df)}")

        results = run_segment_backtest(df, seg_col, seg_name)
        segment_results[seg_name] = results

        for bin_label, stats in sorted(results.items()):
            if stats.get("note"):
                log.info(f"  {bin_label}: {stats['note']}")
            else:
                log.info(f"  {bin_label}: {stats['n_bets']} bets, "
                         f"ROI {stats['roi']:+.1f}%, p={stats['p_value']:.4f}")

    # ── Run cross-segment backtests ──
    cross_results = {}

    log.info("\nCross-segment: Market Closeness x Model Confidence")
    cross_results["Market Closeness x Model Confidence"] = run_cross_segment(
        df, "seg_market", "seg_confidence", "Market Closeness x Model Confidence"
    )

    log.info("Cross-segment: SP Quality Gap x Season Phase")
    cross_results["SP Quality Gap x Season Phase"] = run_cross_segment(
        df, "seg_sp_gap", "seg_season_phase", "SP Quality Gap x Season Phase"
    )

    # ── Build and save report ──
    report = build_segmented_report(df, segment_results, cross_results)
    report_path = MODELS_DIR / "mlb_segmented_backtest_report_nomarket.txt"
    with open(report_path, "w") as f:
        f.write(report)
    log.info(f"\nSaved segmented backtest report -> {report_path}")

    # Print summary
    print(f"\n{'='*70}")
    print("SEGMENTED BACKTEST SUMMARY")
    print(f"{'='*70}")
    print(f"  OOF games with H2H: {len(df)}")
    print(f"  Overall ML ROI (>= {PROD_THRESHOLD} runs): {all_stats['roi']:+.1f}%")
    print(f"  Segments tested: {len(segment_results)}")
    print(f"  Cross-segments tested: {len(cross_results)}")

    # Count interesting findings
    pos_roi = []
    for seg_name, results in {**segment_results, **cross_results}.items():
        for bin_label, stats in results.items():
            if not stats.get("note") and stats.get("roi", -999) > 0 and stats.get("n_bets", 0) >= MIN_BETS_REPORT:
                pos_roi.append((seg_name, bin_label, stats["roi"], stats["p_value"]))

    if pos_roi:
        pos_roi.sort(key=lambda x: x[2], reverse=True)
        print(f"\n  Positive ROI segments ({len(pos_roi)}):")
        for seg, b, roi, p in pos_roi[:10]:
            sig = "***" if p < BONFERRONI_P else "*" if p < 0.05 else ""
            print(f"    {seg} / {b}: ROI {roi:+.1f}%, p={p:.4f} {sig}")
    else:
        print("\n  No segments with positive ROI at production threshold.")

    print(f"\n  Full report: {report_path}")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()
