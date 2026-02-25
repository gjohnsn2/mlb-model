"""
13 — Robustness Experiments: Dog Filters & Rolling Calibration
===============================================================
Tests two structural hypotheses from the 2025 diagnostic:

  Experiment 1: Dog Filters
    Does the model's edge live in identifying underpriced favorites,
    not in picking upsets? Test by restricting the bet universe.

  Experiment 2: Rolling Calibration
    Does global calibration (all years' mean/std) go stale? Test by
    recalibrating on a trailing window of N seasons instead.

Both use existing Lasso no-market OOF predictions — no retraining.

Inputs:
  - models/mlb_oof_margin_lasso_nomarket_predictions.csv
  - data/historical/training_data_mlb_v2.csv

Outputs:
  - models/mlb_robustness_report.txt
  - models/mlb_robustness_report.html

Run: python3 13_robustness_experiments.py    # ~30 sec, read-only
"""

import warnings
import importlib
import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn.metrics import root_mean_squared_error
from config import MODELS_DIR, HISTORICAL_DIR, get_logger

warnings.filterwarnings("ignore", category=RuntimeWarning, module="numpy")

log = get_logger("13_robustness")

bt = importlib.import_module("10_backtest_mlb")

THRESHOLD_SWEEP = [0.5, 1.0, 1.5, 2.0]
PRIMARY_THRESHOLD = 1.5

# ═══════════════════════════════════════════════════════════════
# Experiment 1: Dog Filters
# ═══════════════════════════════════════════════════════════════

DOG_FILTERS = {
    "All bets (baseline)":    lambda bets: bets,
    "Favorites only":         lambda bets: bets[~bets["is_dog"]],
    "Dogs only":              lambda bets: bets[bets["is_dog"]],
    "Dogs implied > 0.40":    lambda bets: bets[~bets["is_dog"] | (bets["_dog_prob"] > 0.40)],
    "Dogs implied > 0.35":    lambda bets: bets[~bets["is_dog"] | (bets["_dog_prob"] > 0.35)],
    "No longshots (> 0.30)":  lambda bets: bets[~bets["is_dog"] | (bets["_dog_prob"] > 0.30)],
}


def add_dog_prob(bets, cal):
    """Merge market_home_prob onto bets and compute dog implied prob."""
    bets = bets.merge(
        cal[["game_pk", "market_home_prob"]].drop_duplicates("game_pk"),
        on="game_pk", how="left"
    )
    # Dog implied prob = the probability of the side we're betting on
    # For HOME_ML dog: market_home_prob < 0.5, so dog_prob = market_home_prob
    # For AWAY_ML dog: market_home_prob >= 0.5, so dog_prob = 1 - market_home_prob
    bets["_dog_prob"] = bets.apply(
        lambda r: r["market_home_prob"] if r["side"] == "HOME_ML"
                  else 1 - r["market_home_prob"], axis=1
    )
    return bets


def run_dog_filter_experiment(cal):
    """Run Experiment 1: dog filter variants across thresholds and eras."""
    results = []

    eras = {
        "2017-2019": (2017, 2019),
        "2021-2024": (2021, 2024),
        "2017-2024": (2017, 2024),
        "2025":      (2025, 2025),
        "ALL":       (2017, 2025),
    }

    for thresh in THRESHOLD_SWEEP:
        # Generate all bets once
        all_bets = bt.simulate_ml_bets(cal, thresh)
        if all_bets.empty:
            continue
        all_bets = add_dog_prob(all_bets, cal)

        for filter_name, filter_fn in DOG_FILTERS.items():
            filtered = filter_fn(all_bets)
            if filtered.empty:
                continue

            for era_name, (lo, hi) in eras.items():
                era_bets = filtered[(filtered["season"] >= lo) & (filtered["season"] <= hi)]
                s = bt.compute_ml_stats(era_bets)
                s["threshold"] = thresh
                s["filter"] = filter_name
                s["era"] = era_name
                results.append(s)

    return pd.DataFrame(results)


# ═══════════════════════════════════════════════════════════════
# Experiment 2: Rolling Calibration
# ═══════════════════════════════════════════════════════════════

ROLLING_WINDOWS = [
    ("Global (baseline)", None),      # current behavior
    ("Trailing 2 seasons", 2),
    ("Trailing 3 seasons", 3),
    ("Trailing 4 seasons", 4),
    ("Expanding (all prior)", "expanding"),
]


def calibrate_rolling(matched_df, margin_rmse, window):
    """
    Recalibrate predictions using a rolling window of prior seasons.

    For each season, compute model mean/std and market mean/std from only
    the trailing N seasons of data, then rescale that season's predictions.

    window=None: global (all data, current behavior)
    window=int: trailing N seasons
    window="expanding": all prior seasons (no future leak)
    """
    df = matched_df.copy()

    # De-vig and compute market implied margin (same as bt.calibrate_predictions)
    h2h_mask = df["consensus_h2h_home"].notna() & df["consensus_h2h_away"].notna()
    df = df[h2h_mask].copy()

    raw_home = df["consensus_h2h_home"].apply(bt.american_to_implied_prob)
    raw_away = df["consensus_h2h_away"].apply(bt.american_to_implied_prob)
    total_vig = raw_home + raw_away
    valid = total_vig.notna() & (total_vig > 0)
    df.loc[valid, "market_home_prob"] = raw_home[valid] / total_vig[valid]
    df["market_implied_margin"] = margin_rmse * norm.ppf(
        df["market_home_prob"].clip(0.001, 0.999)
    )

    if window is None:
        # Global calibration (baseline)
        model_mean = df["predicted"].mean()
        model_std = df["predicted"].std()
        market_mean = df["market_implied_margin"].mean()
        market_std = df["market_implied_margin"].std()
        df["calibrated_pred"] = (
            (df["predicted"] - model_mean) / model_std * market_std + market_mean
        )
        df["margin_edge"] = df["calibrated_pred"] - df["market_implied_margin"]
        return df

    # Per-season rolling calibration
    seasons = sorted(df["season"].unique())
    cal_parts = []

    for season in seasons:
        season_mask = df["season"] == season
        season_df = df[season_mask].copy()

        if window == "expanding":
            # All seasons before this one
            prior_mask = df["season"] < season
        else:
            # Trailing N seasons
            prior_seasons = [s for s in seasons if s < season][-window:]
            prior_mask = df["season"].isin(prior_seasons)

        prior_df = df[prior_mask]

        if len(prior_df) < 100:
            # Not enough prior data — fall back to global for early seasons
            model_mean = df["predicted"].mean()
            model_std = df["predicted"].std()
            market_mean = df["market_implied_margin"].mean()
            market_std = df["market_implied_margin"].std()
        else:
            model_mean = prior_df["predicted"].mean()
            model_std = prior_df["predicted"].std()
            market_mean = prior_df["market_implied_margin"].mean()
            market_std = prior_df["market_implied_margin"].std()

        if model_std < 0.001:
            model_std = 0.001  # safety

        season_df["calibrated_pred"] = (
            (season_df["predicted"] - model_mean) / model_std * market_std + market_mean
        )
        season_df["margin_edge"] = season_df["calibrated_pred"] - season_df["market_implied_margin"]
        cal_parts.append(season_df)

    return pd.concat(cal_parts, ignore_index=True)


def run_rolling_calibration_experiment(matched_df, margin_rmse):
    """Run Experiment 2: rolling calibration variants."""
    results = []

    eras = {
        "2017-2019": (2017, 2019),
        "2021-2024": (2021, 2024),
        "2017-2024": (2017, 2024),
        "2025":      (2025, 2025),
        "ALL":       (2017, 2025),
    }

    for window_name, window in ROLLING_WINDOWS:
        cal = calibrate_rolling(matched_df, margin_rmse, window)
        cal = cal.drop_duplicates(subset=["game_pk"], keep="first")

        for thresh in THRESHOLD_SWEEP:
            for era_name, (lo, hi) in eras.items():
                era_cal = cal[(cal["season"] >= lo) & (cal["season"] <= hi)]
                bets = bt.simulate_ml_bets(era_cal, thresh)
                s = bt.compute_ml_stats(bets)
                s["threshold"] = thresh
                s["window"] = window_name
                s["era"] = era_name

                # Also compute edge stats
                s["edge_mean"] = era_cal["margin_edge"].mean()
                s["edge_std"] = era_cal["margin_edge"].std()
                s["home_pct"] = (era_cal["margin_edge"] > 0).mean() * 100

                results.append(s)

    return pd.DataFrame(results)


# ═══════════════════════════════════════════════════════════════
# Report
# ═══════════════════════════════════════════════════════════════

def build_text_report(dog_results, rolling_results):
    """Build plain-text report."""
    lines = []
    lines.append("MLB ROBUSTNESS EXPERIMENTS")
    lines.append("=" * 80)
    lines.append(f"Generated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')}")
    lines.append("")

    # ── Experiment 1 ──
    lines.append("=" * 80)
    lines.append("EXPERIMENT 1: DOG FILTERS")
    lines.append("Does the model's edge live in favorites, not upsets?")
    lines.append("=" * 80)

    for thresh in THRESHOLD_SWEEP:
        lines.append(f"\n  Threshold >= {thresh} runs:")
        lines.append(f"  {'Filter':<25} | {'2017-19':>14} | {'2021-24':>14} | "
                     f"{'2017-24':>14} | {'2025':>14} | {'ALL':>14}")
        lines.append("  " + "-" * 90)

        for filter_name in DOG_FILTERS.keys():
            parts = []
            for era in ["2017-2019", "2021-2024", "2017-2024", "2025", "ALL"]:
                row = dog_results[
                    (dog_results["threshold"] == thresh) &
                    (dog_results["filter"] == filter_name) &
                    (dog_results["era"] == era)
                ]
                if row.empty or row.iloc[0]["n_bets"] == 0:
                    parts.append(f"{'--':>14}")
                else:
                    r = row.iloc[0]
                    parts.append(f"{r['n_bets']:>4}b {r['roi']:>+5.1f}%")
                    # Pad to 14 chars
                    parts[-1] = f"{parts[-1]:>14}"
            lines.append(f"  {filter_name:<25} | {parts[0]} | {parts[1]} | "
                         f"{parts[2]} | {parts[3]} | {parts[4]}")

    lines.append("")

    # ── Experiment 2 ──
    lines.append("=" * 80)
    lines.append("EXPERIMENT 2: ROLLING CALIBRATION")
    lines.append("Does a trailing calibration window help?")
    lines.append("=" * 80)

    for thresh in THRESHOLD_SWEEP:
        lines.append(f"\n  Threshold >= {thresh} runs:")
        lines.append(f"  {'Calibration':<25} | {'2017-19':>14} | {'2021-24':>14} | "
                     f"{'2017-24':>14} | {'2025':>14} | {'ALL':>14}")
        lines.append("  " + "-" * 90)

        for window_name, _ in ROLLING_WINDOWS:
            parts = []
            for era in ["2017-2019", "2021-2024", "2017-2024", "2025", "ALL"]:
                row = rolling_results[
                    (rolling_results["threshold"] == thresh) &
                    (rolling_results["window"] == window_name) &
                    (rolling_results["era"] == era)
                ]
                if row.empty or row.iloc[0]["n_bets"] == 0:
                    parts.append(f"{'--':>14}")
                else:
                    r = row.iloc[0]
                    parts.append(f"{r['n_bets']:>4}b {r['roi']:>+5.1f}%")
                    parts[-1] = f"{parts[-1]:>14}"
            lines.append(f"  {window_name:<25} | {parts[0]} | {parts[1]} | "
                         f"{parts[2]} | {parts[3]} | {parts[4]}")

    # Edge distribution under rolling calibration
    lines.append(f"\n  Edge distribution by calibration method (all games):")
    lines.append(f"  {'Calibration':<25} | {'Era':>10} | {'Edge mean':>10} | "
                 f"{'Edge std':>10} | {'Home%':>7}")
    lines.append("  " + "-" * 72)
    for window_name, _ in ROLLING_WINDOWS:
        for era in ["2017-2024", "2025"]:
            row = rolling_results[
                (rolling_results["threshold"] == PRIMARY_THRESHOLD) &
                (rolling_results["window"] == window_name) &
                (rolling_results["era"] == era)
            ]
            if not row.empty:
                r = row.iloc[0]
                lines.append(f"  {window_name:<25} | {era:>10} | {r['edge_mean']:>+10.4f} | "
                             f"{r['edge_std']:>10.4f} | {r['home_pct']:>6.1f}%")

    lines.append("")
    return "\n".join(lines)


def build_html_report(dog_results, rolling_results):
    """Build HTML report."""

    def roi_class(val):
        if pd.isna(val):
            return "flat"
        return "pos" if val > 0 else "neg" if val < 0 else "flat"

    def roi_cell(row):
        if row is None or row["n_bets"] == 0:
            return '<td class="num flat">--</td>'
        css = roi_class(row["roi"])
        return (f'<td class="num {css}">{row["n_bets"]}b {row["roi"]:+.1f}%</td>')

    html = []
    html.append("""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>MLB Robustness Experiments</title>
<style>
  :root {
    --bg: #0d1117; --surface: #161b22; --border: #30363d;
    --text: #e6edf3; --muted: #8b949e; --accent: #58a6ff;
    --green: #3fb950; --red: #f85149; --yellow: #d29922;
  }
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body { background: var(--bg); color: var(--text); font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Helvetica, Arial, sans-serif; font-size: 14px; line-height: 1.5; padding: 24px; max-width: 1200px; margin: 0 auto; }
  h1 { font-size: 24px; font-weight: 600; margin-bottom: 4px; }
  h2 { font-size: 18px; font-weight: 600; margin: 32px 0 12px; padding-bottom: 8px; border-bottom: 1px solid var(--border); color: var(--accent); }
  h3 { font-size: 15px; font-weight: 600; margin: 20px 0 8px; color: var(--muted); }
  .subtitle { color: var(--muted); margin-bottom: 24px; }
  .card { background: var(--surface); border: 1px solid var(--border); border-radius: 8px; padding: 16px; margin-bottom: 16px; }
  table { width: 100%; border-collapse: collapse; font-size: 13px; margin: 8px 0; }
  th { text-align: left; padding: 6px 10px; background: rgba(88,166,255,0.08); border-bottom: 2px solid var(--border); font-weight: 600; color: var(--accent); white-space: nowrap; }
  td { padding: 5px 10px; border-bottom: 1px solid var(--border); white-space: nowrap; }
  tr:hover td { background: rgba(88,166,255,0.04); }
  .num { text-align: right; font-variant-numeric: tabular-nums; font-family: 'SF Mono', SFMono-Regular, Consolas, monospace; font-size: 12.5px; }
  .pos { color: var(--green); }
  .neg { color: var(--red); }
  .flat { color: var(--muted); }
  .note { font-size: 12px; color: var(--muted); margin-top: 8px; font-style: italic; }
  .highlight-row td { background: rgba(63,185,80,0.06); }
  .baseline-row td { font-weight: 600; }
  nav { background: var(--surface); border: 1px solid var(--border); border-radius: 8px; padding: 12px 16px; margin-bottom: 24px; }
  nav a { color: var(--accent); text-decoration: none; font-size: 13px; margin-right: 16px; }
  nav a:hover { text-decoration: underline; }
  .verdict { padding: 12px 16px; margin: 12px 0; border-left: 3px solid var(--accent); background: rgba(88,166,255,0.06); border-radius: 0 4px 4px 0; }
</style>
</head>
<body>
<h1>MLB Robustness Experiments</h1>
<p class="subtitle">Lasso no-market OOF predictions &middot; No retraining &middot; Testing structural hypotheses from 2025 diagnostic</p>
<nav>
  <a href="#exp1">Exp 1: Dog Filters</a>
  <a href="#exp2">Exp 2: Rolling Calibration</a>
</nav>
""")

    # ── Experiment 1 ──
    html.append('<h2 id="exp1">Experiment 1: Dog Filters</h2>')
    html.append('<p>Hypothesis: the model\'s edge is in identifying underpriced favorites, not picking upsets.</p>')

    eras = ["2017-2019", "2021-2024", "2017-2024", "2025", "ALL"]

    for thresh in THRESHOLD_SWEEP:
        html.append(f'<h3>Threshold &ge; {thresh} runs</h3>')
        html.append('<div class="card"><table>')
        html.append('<thead><tr><th>Filter</th>')
        for era in eras:
            html.append(f'<th class="num">{era}</th>')
        html.append('</tr></thead><tbody>')

        for filter_name in DOG_FILTERS.keys():
            is_baseline = "baseline" in filter_name
            row_class = ' class="baseline-row"' if is_baseline else ''
            html.append(f'<tr{row_class}><td>{filter_name}</td>')
            for era in eras:
                match = dog_results[
                    (dog_results["threshold"] == thresh) &
                    (dog_results["filter"] == filter_name) &
                    (dog_results["era"] == era)
                ]
                if match.empty or match.iloc[0]["n_bets"] == 0:
                    html.append('<td class="num flat">--</td>')
                else:
                    r = match.iloc[0]
                    css = roi_class(r["roi"])
                    html.append(f'<td class="num {css}">{r["n_bets"]}b {r["roi"]:+.1f}%</td>')
            html.append('</tr>')

        html.append('</tbody></table></div>')

    html.append("""<div class="verdict">
<strong>Reading guide:</strong> Compare each filter row against "All bets (baseline)".
If "Favorites only" maintains or improves ROI across eras while "Dogs only" is negative,
the model's edge is fundamentally about favorites. If dogs at higher implied probabilities
(&gt; 0.35, &gt; 0.40) hold up while longshots don't, a probability floor is warranted.
</div>""")

    # ── Experiment 2 ──
    html.append('<h2 id="exp2">Experiment 2: Rolling Calibration</h2>')
    html.append('<p>Hypothesis: global calibration (mean/std from all 17K+ games) goes stale. '
                'A trailing window recalibrates on recent market conditions.</p>')

    for thresh in THRESHOLD_SWEEP:
        html.append(f'<h3>Threshold &ge; {thresh} runs</h3>')
        html.append('<div class="card"><table>')
        html.append('<thead><tr><th>Calibration</th>')
        for era in eras:
            html.append(f'<th class="num">{era}</th>')
        html.append('</tr></thead><tbody>')

        for window_name, _ in ROLLING_WINDOWS:
            is_baseline = "baseline" in window_name
            row_class = ' class="baseline-row"' if is_baseline else ''
            html.append(f'<tr{row_class}><td>{window_name}</td>')
            for era in eras:
                match = rolling_results[
                    (rolling_results["threshold"] == thresh) &
                    (rolling_results["window"] == window_name) &
                    (rolling_results["era"] == era)
                ]
                if match.empty or match.iloc[0]["n_bets"] == 0:
                    html.append('<td class="num flat">--</td>')
                else:
                    r = match.iloc[0]
                    css = roi_class(r["roi"])
                    html.append(f'<td class="num {css}">{r["n_bets"]}b {r["roi"]:+.1f}%</td>')
            html.append('</tr>')

        html.append('</tbody></table></div>')

    # Edge distribution table
    html.append('<h3>Edge Distribution by Calibration Method</h3>')
    html.append('<div class="card"><table>')
    html.append('<thead><tr><th>Calibration</th><th>Era</th>'
                '<th class="num">Edge mean</th><th class="num">Edge std</th>'
                '<th class="num">Home %</th></tr></thead><tbody>')

    for window_name, _ in ROLLING_WINDOWS:
        for era in ["2017-2024", "2025"]:
            match = rolling_results[
                (rolling_results["threshold"] == PRIMARY_THRESHOLD) &
                (rolling_results["window"] == window_name) &
                (rolling_results["era"] == era)
            ]
            if not match.empty:
                r = match.iloc[0]
                era_bold = ' style="font-weight:600"' if era == "2025" else ''
                html.append(f'<tr><td>{window_name}</td><td{era_bold}>{era}</td>'
                            f'<td class="num">{r["edge_mean"]:+.4f}</td>'
                            f'<td class="num">{r["edge_std"]:.4f}</td>'
                            f'<td class="num">{r["home_pct"]:.1f}%</td></tr>')

    html.append('</tbody></table></div>')

    html.append("""<div class="verdict">
<strong>Reading guide:</strong> Compare each rolling window against "Global (baseline)".
If a trailing window improves 2025 ROI without degrading 2017-2024, calibration drift is real
and the trailing window should be adopted. Watch the edge mean for 2025 — global shows -0.158
(systematic away bias). A good rolling window should pull this closer to 0.
</div>""")

    html.append('<p style="text-align:center;color:var(--muted);margin-top:32px;font-size:12px">'
                'Generated from 13_robustness_experiments.py</p>')
    html.append('</body></html>')

    return "\n".join(html)


# ═══════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════

def main():
    log.info("=" * 60)
    log.info("ROBUSTNESS EXPERIMENTS")
    log.info("=" * 60)

    # Load data
    lasso_path = MODELS_DIR / "mlb_oof_margin_lasso_nomarket_predictions.csv"
    if not lasso_path.exists():
        log.error(f"Lasso predictions not found: {lasso_path}")
        return
    margin_oof = pd.read_csv(lasso_path)
    margin_oof["date"] = margin_oof["date"].astype(str).str[:10]
    log.info(f"Loaded Lasso OOF: {len(margin_oof)} games")

    training_df = bt.load_training_data_for_odds()

    margin_rmse = root_mean_squared_error(margin_oof["actual"], margin_oof["predicted"])
    log.info(f"OOF margin RMSE: {margin_rmse:.2f}")

    # Match with odds
    matched = bt.match_with_odds(margin_oof, training_df)

    # ── Experiment 1: Dog Filters (uses global calibration) ──
    log.info("\nExperiment 1: Dog Filters")
    cal_global = bt.calibrate_predictions(matched, margin_rmse)
    cal_global = cal_global.drop_duplicates(subset=["game_pk"], keep="first")
    dog_results = run_dog_filter_experiment(cal_global)
    log.info(f"  Generated {len(dog_results)} result rows")

    # Print key finding
    for era in ["2017-2024", "2025"]:
        for filter_name in ["All bets (baseline)", "Favorites only", "Dogs only"]:
            row = dog_results[
                (dog_results["threshold"] == PRIMARY_THRESHOLD) &
                (dog_results["filter"] == filter_name) &
                (dog_results["era"] == era)
            ]
            if not row.empty:
                r = row.iloc[0]
                log.info(f"  {filter_name:<25} {era}: {r['n_bets']} bets, ROI {r['roi']:+.1f}%")

    # ── Experiment 2: Rolling Calibration ──
    log.info("\nExperiment 2: Rolling Calibration")
    rolling_results = run_rolling_calibration_experiment(matched, margin_rmse)
    log.info(f"  Generated {len(rolling_results)} result rows")

    for window_name, _ in ROLLING_WINDOWS:
        for era in ["2017-2024", "2025"]:
            row = rolling_results[
                (rolling_results["threshold"] == PRIMARY_THRESHOLD) &
                (rolling_results["window"] == window_name) &
                (rolling_results["era"] == era)
            ]
            if not row.empty:
                r = row.iloc[0]
                log.info(f"  {window_name:<25} {era}: {r['n_bets']} bets, "
                         f"ROI {r['roi']:+.1f}%, edge_mean={r['edge_mean']:+.3f}")

    # ── Save reports ──
    text_report = build_text_report(dog_results, rolling_results)
    text_path = MODELS_DIR / "mlb_robustness_report.txt"
    with open(text_path, "w") as f:
        f.write(text_report)
    log.info(f"\nSaved text report -> {text_path}")

    html_report = build_html_report(dog_results, rolling_results)
    html_path = MODELS_DIR / "mlb_robustness_report.html"
    with open(html_path, "w") as f:
        f.write(html_report)
    log.info(f"Saved HTML report -> {html_path}")

    # ── Print summary ──
    print(f"\n{'='*70}")
    print("ROBUSTNESS EXPERIMENTS SUMMARY")
    print(f"{'='*70}")

    print(f"\n  Experiment 1: Dog Filters (threshold >= {PRIMARY_THRESHOLD})")
    for filter_name in DOG_FILTERS.keys():
        parts = []
        for era in ["2017-2024", "2025", "ALL"]:
            row = dog_results[
                (dog_results["threshold"] == PRIMARY_THRESHOLD) &
                (dog_results["filter"] == filter_name) &
                (dog_results["era"] == era)
            ]
            if not row.empty and row.iloc[0]["n_bets"] > 0:
                r = row.iloc[0]
                parts.append(f"{era}: {r['n_bets']}b {r['roi']:+.1f}%")
            else:
                parts.append(f"{era}: --")
        print(f"    {filter_name:<25}  {' | '.join(parts)}")

    print(f"\n  Experiment 2: Rolling Calibration (threshold >= {PRIMARY_THRESHOLD})")
    for window_name, _ in ROLLING_WINDOWS:
        parts = []
        for era in ["2017-2024", "2025", "ALL"]:
            row = rolling_results[
                (rolling_results["threshold"] == PRIMARY_THRESHOLD) &
                (rolling_results["window"] == window_name) &
                (rolling_results["era"] == era)
            ]
            if not row.empty and row.iloc[0]["n_bets"] > 0:
                r = row.iloc[0]
                parts.append(f"{era}: {r['n_bets']}b {r['roi']:+.1f}%")
            else:
                parts.append(f"{era}: --")
        print(f"    {window_name:<25}  {' | '.join(parts)}")

    print(f"\n  Reports: {html_path}")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()
