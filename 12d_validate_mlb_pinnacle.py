"""
12d — MLB Pinnacle Validation Analysis
========================================
Compares model ML performance against Pinnacle closing lines vs consensus.
Pinnacle is the sharpest book — beating Pinnacle is harder than beating
retail consensus. This validates whether the model's edge is real.

Methodology:
  1. Load OOF margin predictions (walk-forward, truly out-of-sample)
  2. Match Pinnacle H2H closing lines to games via (date, home_team) → game_pk
  3. Calibrate predictions against BOTH Pinnacle and consensus market lines
  4. Simulate ML bets using both line sources
  5. Compare performance side-by-side, by season

Inputs:
  - models/mlb_oof_margin_predictions.csv
  - data/historical/pinnacle_mlb_odds.csv
  - data/historical/training_data_mlb_v2.csv

Output:
  - models/mlb_pinnacle_validation_report.txt

Run: python3 12d_validate_mlb_pinnacle.py
"""

import numpy as np
import pandas as pd
from scipy.stats import norm
from config import MODELS_DIR, HISTORICAL_DIR, MLB_MARGIN_MODEL_RMSE, get_logger

log = get_logger("12d_pinnacle_validation")

UNIT_SIZE = 100
ML_MARGIN_THRESHOLDS = [0.0, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0]
ML_PRODUCTION_THRESHOLD = 0.5

# Team name mapping: Pinnacle (Odds API) → Training data (MLB Stats API)
PINNACLE_TO_TRAINING = {
    "Oakland Athletics": "Athletics",  # 2025+ after move
}


def american_to_implied_prob(odds):
    if pd.isna(odds) or odds == 0:
        return np.nan
    if odds < 0:
        return abs(odds) / (abs(odds) + 100)
    else:
        return 100 / (odds + 100)


def american_to_decimal(odds):
    if pd.isna(odds) or odds == 0:
        return np.nan
    if odds < 0:
        return 1 + 100 / abs(odds)
    else:
        return 1 + odds / 100


def load_data():
    """Load OOF predictions, Pinnacle odds, and training data."""
    # OOF margin predictions
    margin_path = MODELS_DIR / "mlb_oof_margin_predictions.csv"
    oof = pd.read_csv(margin_path)
    oof["date"] = oof["date"].astype(str).str[:10]
    log.info(f"OOF margin predictions: {len(oof)} games")

    # Pinnacle odds
    pinn_path = HISTORICAL_DIR / "pinnacle_mlb_odds.csv"
    pinn = pd.read_csv(pinn_path)
    log.info(f"Pinnacle odds: {len(pinn)} rows, "
             f"{pinn['pinnacle_h2h_home'].notna().sum()} with H2H")

    # Training data (for consensus odds + game_pk mapping)
    train_path = HISTORICAL_DIR / "training_data_mlb_v2.csv"
    train = pd.read_csv(train_path, usecols=[
        "game_pk", "date", "home_team", "away_team",
        "consensus_h2h_home", "consensus_h2h_away",
        "consensus_total", "consensus_spread", "num_books",
        "actual_margin", "actual_total",
    ])
    train["date"] = train["date"].astype(str).str[:10]
    log.info(f"Training data: {len(train)} games")

    return oof, pinn, train


def match_pinnacle_to_games(pinn, train):
    """
    Match Pinnacle closing lines to training data games.

    Strategy:
    1. Take latest fetch per Pinnacle game (closing line)
    2. Extract game_date from commence_time
    3. Match to training data by (date, home_team)
    4. For doubleheaders, match by ordering within same (date, teams)
    """
    # Filter to rows with Pinnacle H2H
    has_h2h = pinn[pinn["pinnacle_h2h_home"].notna()].copy()

    # Take latest fetch per game (closest to game time = closing line)
    has_h2h = has_h2h.sort_values("fetch_date").drop_duplicates(
        subset=["home_team", "away_team", "commence_time"], keep="last"
    )
    log.info(f"Unique Pinnacle games with H2H: {len(has_h2h)}")

    # Extract game date from commence_time
    has_h2h["game_date"] = pd.to_datetime(has_h2h["commence_time"]).dt.strftime("%Y-%m-%d")

    # Apply team name mapping
    has_h2h["match_home"] = has_h2h["home_team"].map(
        lambda x: PINNACLE_TO_TRAINING.get(x, x)
    )
    has_h2h["match_away"] = has_h2h["away_team"].map(
        lambda x: PINNACLE_TO_TRAINING.get(x, x)
    )

    # Sort by commence_time within each (date, teams) group for DH ordering
    has_h2h = has_h2h.sort_values(["game_date", "match_home", "match_away", "commence_time"])

    # Build training lookup: (date, home_team) → list of game_pks (ordered)
    train_sorted = train.sort_values(["date", "home_team", "game_pk"])
    train_lookup = {}
    for _, row in train_sorted.iterrows():
        key = (row["date"], row["home_team"])
        if key not in train_lookup:
            train_lookup[key] = []
        train_lookup[key].append(row["game_pk"])

    # Match Pinnacle to game_pk
    matched = []
    dh_counter = {}  # Track position within doubleheader

    for _, row in has_h2h.iterrows():
        key = (row["game_date"], row["match_home"])
        game_pks = train_lookup.get(key)
        if game_pks is None:
            continue

        # For DH, use positional matching
        dh_key = (row["game_date"], row["match_home"], row["match_away"])
        pos = dh_counter.get(dh_key, 0)
        dh_counter[dh_key] = pos + 1

        if pos < len(game_pks):
            matched.append({
                "game_pk": game_pks[pos],
                "pinnacle_h2h_home": row["pinnacle_h2h_home"],
                "pinnacle_h2h_away": row["pinnacle_h2h_away"],
                "pinnacle_total": row.get("pinnacle_total"),
                "pinnacle_total_over_price": row.get("pinnacle_total_over_price"),
                "pinnacle_total_under_price": row.get("pinnacle_total_under_price"),
            })

    matched_df = pd.DataFrame(matched)
    log.info(f"Matched to game_pk: {len(matched_df)}/{len(has_h2h)} "
             f"({len(matched_df)/len(has_h2h)*100:.1f}%)")

    return matched_df


def calibrate_and_edge(df, h2h_home_col, h2h_away_col, margin_rmse, label):
    """
    Calibrate model predictions and compute margin-space edges.
    Same methodology as 10_backtest_mlb.py.
    """
    mask = df[h2h_home_col].notna() & df[h2h_away_col].notna()
    has_odds = df[mask].copy()

    if len(has_odds) == 0:
        return has_odds

    # De-vig market probabilities
    raw_home = has_odds[h2h_home_col].apply(american_to_implied_prob)
    raw_away = has_odds[h2h_away_col].apply(american_to_implied_prob)
    total_vig = raw_home + raw_away
    valid = total_vig.notna() & (total_vig > 0)
    has_odds.loc[valid, "market_home_prob"] = raw_home[valid] / total_vig[valid]

    # Market prob → implied margin
    has_odds["market_implied_margin"] = margin_rmse * norm.ppf(
        has_odds["market_home_prob"].clip(0.001, 0.999)
    )

    # Rescale model predictions to match market distribution
    model_mean = has_odds["predicted"].mean()
    model_std = has_odds["predicted"].std()
    market_mean = has_odds["market_implied_margin"].mean()
    market_std = has_odds["market_implied_margin"].std()

    has_odds["calibrated_pred"] = (
        (has_odds["predicted"] - model_mean) / model_std * market_std + market_mean
    )

    # Edge in runs
    has_odds["margin_edge"] = has_odds["calibrated_pred"] - has_odds["market_implied_margin"]

    log.info(f"  {label}: {len(has_odds)} games, "
             f"model std {model_std:.3f} → market std {market_std:.3f}, "
             f"edge mean {has_odds['margin_edge'].mean():.4f}")

    return has_odds


def simulate_ml_bets(df, threshold, h2h_home_col, h2h_away_col):
    """Simulate ML betting at given threshold using specified H2H lines."""
    bets = []

    for _, row in df.iterrows():
        edge = row["margin_edge"]
        if pd.isna(edge) or abs(edge) < threshold:
            continue

        if edge > 0:
            odds_used = row[h2h_home_col]
            won = row["actual"] > 0
        else:
            odds_used = row[h2h_away_col]
            won = row["actual"] < 0

        dec = american_to_decimal(odds_used)
        if pd.isna(dec):
            continue

        is_dog = (edge > 0 and row["market_home_prob"] < 0.5) or \
                 (edge <= 0 and row["market_home_prob"] >= 0.5)

        push = (row["actual"] == 0)
        if push:
            profit = 0
        elif won:
            profit = round(UNIT_SIZE * (dec - 1))
        else:
            profit = -UNIT_SIZE

        bets.append({
            "season": row.get("season"),
            "won": won,
            "push": push,
            "profit": profit,
            "is_dog": is_dog,
            "odds_used": odds_used,
        })

    return pd.DataFrame(bets)


def compute_stats(bets_df):
    """Compute W-L, win%, ROI, P&L, dog%, p-value."""
    if bets_df.empty:
        return {"n_bets": 0, "wins": 0, "losses": 0, "win_pct": 0,
                "profit": 0, "roi": 0, "p_value": 1.0, "dog_pct": 0}

    non_push = bets_df[~bets_df["push"]]
    wins = int(non_push["won"].sum())
    losses = len(non_push) - wins
    n_bets = wins + losses
    win_pct = wins / n_bets * 100 if n_bets > 0 else 0

    profit = bets_df["profit"].sum()
    total_risked = n_bets * UNIT_SIZE
    roi = profit / total_risked * 100 if total_risked > 0 else 0

    dog_pct = bets_df["is_dog"].mean() * 100 if "is_dog" in bets_df.columns else 0

    p_value = 1.0
    if n_bets > 0:
        profits = bets_df["profit"].values
        profit_mean = profits.mean()
        profit_std = profits.std(ddof=1)
        if profit_std > 0:
            z = profit_mean * np.sqrt(len(profits)) / profit_std
            p_value = 1 - norm.cdf(z)

    return {
        "n_bets": n_bets, "wins": wins, "losses": losses,
        "win_pct": win_pct, "profit": profit, "roi": roi,
        "p_value": p_value, "dog_pct": dog_pct,
    }


def main():
    log.info("=" * 60)
    log.info("MLB PINNACLE VALIDATION ANALYSIS")
    log.info("=" * 60)

    oof, pinn, train = load_data()

    # ── Match Pinnacle to games ──
    pinnacle_matched = match_pinnacle_to_games(pinn, train)

    # ── Merge OOF predictions + consensus odds + Pinnacle odds ──
    # First merge OOF with training data (for consensus odds)
    train_cols = ["game_pk", "consensus_h2h_home", "consensus_h2h_away",
                  "consensus_total", "consensus_spread", "num_books",
                  "actual_total"]
    # Only include actual_total if available
    available = [c for c in train_cols if c in train.columns]
    merged = oof.merge(train[available], on="game_pk", how="left",
                       suffixes=("", "_train"))

    # Then merge Pinnacle odds
    merged = merged.merge(pinnacle_matched, on="game_pk", how="left")

    # Filter to games that have BOTH consensus AND Pinnacle H2H
    both_mask = (merged["consensus_h2h_home"].notna() &
                 merged["pinnacle_h2h_home"].notna())
    both = merged[both_mask].copy()
    log.info(f"\nGames with BOTH consensus + Pinnacle H2H: {len(both)}")

    # Also count games with only one source
    cons_only = merged["consensus_h2h_home"].notna() & merged["pinnacle_h2h_home"].isna()
    pinn_only = merged["consensus_h2h_home"].isna() & merged["pinnacle_h2h_home"].notna()
    log.info(f"Consensus only: {cons_only.sum()}, Pinnacle only: {pinn_only.sum()}")

    # Season distribution
    for season in sorted(both["season"].unique()):
        n = (both["season"] == season).sum()
        log.info(f"  {int(season)}: {n} games")

    # ── Compute margin RMSE from OOF ──
    from sklearn.metrics import root_mean_squared_error
    margin_rmse = root_mean_squared_error(oof["actual"], oof["predicted"])
    log.info(f"\nOOF margin RMSE: {margin_rmse:.2f}")

    # ── Calibrate against BOTH line sources (on the overlapping set) ──
    log.info("\nCalibrating predictions:")
    cons_cal = calibrate_and_edge(
        both, "consensus_h2h_home", "consensus_h2h_away", margin_rmse, "Consensus"
    )
    pinn_cal = calibrate_and_edge(
        both, "pinnacle_h2h_home", "pinnacle_h2h_away", margin_rmse, "Pinnacle"
    )

    # ── Compare implied margins ──
    log.info("\nMarket line comparison:")
    cons_margin_std = cons_cal["market_implied_margin"].std()
    pinn_margin_std = pinn_cal["market_implied_margin"].std()
    corr = cons_cal["market_implied_margin"].corr(pinn_cal["market_implied_margin"])
    diff = (cons_cal["market_implied_margin"] - pinn_cal["market_implied_margin"]).abs()
    log.info(f"  Consensus margin std:  {cons_margin_std:.3f}")
    log.info(f"  Pinnacle margin std:   {pinn_margin_std:.3f}")
    log.info(f"  Correlation:           {corr:.4f}")
    log.info(f"  Mean |diff|:           {diff.mean():.3f} runs")
    log.info(f"  Median |diff|:         {diff.median():.3f} runs")

    # ── ML backtest: Consensus vs Pinnacle at each threshold ──
    lines = []
    lines.append("MLB PINNACLE VALIDATION REPORT")
    lines.append("=" * 70)
    lines.append(f"Generated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')}")
    lines.append(f"OOF margin RMSE: {margin_rmse:.2f}")
    lines.append(f"Games with both consensus + Pinnacle H2H: {len(both)}")
    lines.append(f"Edge method: Calibrated margin-space (model rescaled to market std)")
    lines.append("")

    # Season coverage
    lines.append("Season Coverage:")
    for season in sorted(both["season"].unique()):
        n = (both["season"] == season).sum()
        lines.append(f"  {int(season)}: {n:,} games")
    lines.append("")

    # Market comparison
    lines.append("=" * 70)
    lines.append("MARKET LINE COMPARISON (consensus vs Pinnacle)")
    lines.append("=" * 70)
    lines.append(f"  Consensus implied margin std:  {cons_margin_std:.3f}")
    lines.append(f"  Pinnacle implied margin std:   {pinn_margin_std:.3f}")
    lines.append(f"  Correlation (cons vs pinn):     {corr:.4f}")
    lines.append(f"  Mean |margin diff|:             {diff.mean():.3f} runs")
    lines.append(f"  Median |margin diff|:           {diff.median():.3f} runs")
    lines.append("")

    # ── Side-by-side ML results ──
    lines.append("=" * 70)
    lines.append("MONEYLINE BETS: CONSENSUS vs PINNACLE (same game set)")
    lines.append("=" * 70)
    lines.append(f"{'Threshold':>10} | {'--- CONSENSUS ---':>28} | {'--- PINNACLE ---':>28}")
    lines.append(f"{'':>10} | {'Bets':>5}  {'W-L':>9}  {'Win%':>5} {'ROI':>7} {'p-val':>6} | "
                 f"{'Bets':>5}  {'W-L':>9}  {'Win%':>5} {'ROI':>7} {'p-val':>6}")
    lines.append("-" * 80)

    all_cons_results = {}
    all_pinn_results = {}

    for thresh in ML_MARGIN_THRESHOLDS:
        cons_bets = simulate_ml_bets(cons_cal, thresh,
                                     "consensus_h2h_home", "consensus_h2h_away")
        pinn_bets = simulate_ml_bets(pinn_cal, thresh,
                                     "pinnacle_h2h_home", "pinnacle_h2h_away")
        cs = compute_stats(cons_bets)
        ps = compute_stats(pinn_bets)
        all_cons_results[thresh] = (cs, cons_bets)
        all_pinn_results[thresh] = (ps, pinn_bets)

        marker = " <--" if thresh == ML_PRODUCTION_THRESHOLD else ""
        c_wl = f"{cs['wins']}-{cs['losses']}"
        p_wl = f"{ps['wins']}-{ps['losses']}"
        lines.append(
            f"  >= {thresh:>4.2f} | {cs['n_bets']:>5}  {c_wl:>9}  "
            f"{cs['win_pct']:>4.1f}% {cs['roi']:>+6.1f}% {cs['p_value']:>5.3f} | "
            f"{ps['n_bets']:>5}  {p_wl:>9}  "
            f"{ps['win_pct']:>4.1f}% {ps['roi']:>+6.1f}% {ps['p_value']:>5.3f}{marker}"
        )
    lines.append("")

    # ── Dog% comparison ──
    lines.append("=" * 70)
    lines.append("UNDERDOG % BY THRESHOLD")
    lines.append("=" * 70)
    lines.append(f"{'Threshold':>10} | {'Cons Dog%':>10} | {'Pinn Dog%':>10}")
    lines.append("-" * 40)
    for thresh in ML_MARGIN_THRESHOLDS:
        cs = all_cons_results[thresh][0]
        ps = all_pinn_results[thresh][0]
        lines.append(f"  >= {thresh:>4.2f} | {cs['dog_pct']:>9.0f}% | {ps['dog_pct']:>9.0f}%")
    lines.append("")

    # ── By-season comparison at production threshold ──
    lines.append("=" * 70)
    lines.append(f"ML BY SEASON (>= {ML_PRODUCTION_THRESHOLD} runs): CONSENSUS vs PINNACLE")
    lines.append("=" * 70)
    lines.append(f"{'Season':>7} | {'--- CONSENSUS ---':>26} | {'--- PINNACLE ---':>26}")
    lines.append(f"{'':>7} | {'Bets':>5}  {'W-L':>9}  {'Win%':>5} {'ROI':>7} | "
                 f"{'Bets':>5}  {'W-L':>9}  {'Win%':>5} {'ROI':>7}")
    lines.append("-" * 72)

    prod_cons_bets = all_cons_results[ML_PRODUCTION_THRESHOLD][1]
    prod_pinn_bets = all_pinn_results[ML_PRODUCTION_THRESHOLD][1]

    for season in sorted(both["season"].unique()):
        if season not in prod_cons_bets["season"].values:
            continue
        cb = prod_cons_bets[prod_cons_bets["season"] == season]
        pb = prod_pinn_bets[prod_pinn_bets["season"] == season]
        cs = compute_stats(cb)
        ps = compute_stats(pb)
        if cs["n_bets"] > 0 or ps["n_bets"] > 0:
            c_wl = f"{cs['wins']}-{cs['losses']}"
            p_wl = f"{ps['wins']}-{ps['losses']}"
            lines.append(
                f"  {int(season):>5} | {cs['n_bets']:>5}  {c_wl:>9}  "
                f"{cs['win_pct']:>4.1f}% {cs['roi']:>+6.1f}% | "
                f"{ps['n_bets']:>5}  {p_wl:>9}  "
                f"{ps['win_pct']:>4.1f}% {ps['roi']:>+6.1f}%"
            )
    lines.append("")

    # ── Pinnacle total validation ──
    pinn_total_mask = both["pinnacle_total"].notna() & both["consensus_total"].notna()
    pinn_total_games = both[pinn_total_mask].copy()
    if len(pinn_total_games) > 0:
        lines.append("=" * 70)
        lines.append("TOTAL LINE COMPARISON (consensus vs Pinnacle)")
        lines.append("=" * 70)
        total_corr = pinn_total_games["consensus_total"].corr(pinn_total_games["pinnacle_total"])
        total_diff = (pinn_total_games["consensus_total"] - pinn_total_games["pinnacle_total"]).abs()
        lines.append(f"  Games with both totals: {len(pinn_total_games)}")
        lines.append(f"  Consensus total mean:   {pinn_total_games['consensus_total'].mean():.2f}")
        lines.append(f"  Pinnacle total mean:    {pinn_total_games['pinnacle_total'].mean():.2f}")
        lines.append(f"  Correlation:            {total_corr:.4f}")
        lines.append(f"  Mean |diff|:            {total_diff.mean():.2f} runs")
        lines.append(f"  Median |diff|:          {total_diff.median():.2f} runs")

        # Who is more accurate?
        if "actual_total" in pinn_total_games.columns:
            at_mask = pinn_total_games["actual_total"].notna()
            if at_mask.any():
                atg = pinn_total_games[at_mask]
                cons_total_error = (atg["consensus_total"] - atg["actual_total"]).abs().mean()
                pinn_total_error = (atg["pinnacle_total"] - atg["actual_total"]).abs().mean()
                lines.append(f"  Consensus MAE vs actual: {cons_total_error:.2f}")
                lines.append(f"  Pinnacle MAE vs actual:  {pinn_total_error:.2f}")
        lines.append("")

    # ── Model diagnostics on overlapping set ──
    lines.append("=" * 70)
    lines.append("MODEL DIAGNOSTICS (overlapping game set)")
    lines.append("=" * 70)
    lines.append(f"Model pred std:           {both['predicted'].std():.3f}")
    lines.append(f"Cons market margin std:   {cons_margin_std:.3f}")
    lines.append(f"Pinn market margin std:   {pinn_margin_std:.3f}")
    lines.append(f"Actual margin std:        {both['actual'].std():.3f}")

    r_model = both["predicted"].corr(both["actual"])
    r_cons = cons_cal["market_implied_margin"].corr(cons_cal["actual"])
    r_pinn = pinn_cal["market_implied_margin"].corr(pinn_cal["actual"])
    lines.append(f"corr(model, actual):      {r_model:.3f}")
    lines.append(f"corr(consensus, actual):  {r_cons:.3f}")
    lines.append(f"corr(pinnacle, actual):   {r_pinn:.3f}")

    # Edge correlation with market residual
    cons_resid = cons_cal["actual"] - cons_cal["market_implied_margin"]
    pinn_resid = pinn_cal["actual"] - pinn_cal["market_implied_margin"]
    r_cons_edge = cons_cal["margin_edge"].corr(cons_resid)
    r_pinn_edge = pinn_cal["margin_edge"].corr(pinn_resid)
    lines.append(f"corr(cons_edge, cons_residual):  {r_cons_edge:.3f}")
    lines.append(f"corr(pinn_edge, pinn_residual):  {r_pinn_edge:.3f}")
    lines.append("")

    # ── Notes ──
    lines.append("=" * 70)
    lines.append("NOTES")
    lines.append("=" * 70)
    lines.append("- Only games with BOTH consensus AND Pinnacle H2H are compared")
    lines.append("- Edges are computed independently for each line source")
    lines.append("- Each calibration rescales model to its own market's std")
    lines.append("- ML bets use actual H2H odds from the respective book")
    lines.append("- Pinnacle data from Odds API 'eu' region (H2H + totals)")
    lines.append("- Consensus data from Odds API 'us' region (median of 15+ books)")
    lines.append(f"- Margin RMSE: {margin_rmse:.2f}")
    lines.append("- Pinnacle closing lines = latest available fetch per game")

    report = "\n".join(lines)

    # Save report
    report_path = MODELS_DIR / "mlb_pinnacle_validation_report.txt"
    with open(report_path, "w") as f:
        f.write(report)
    log.info(f"\nSaved report -> {report_path}")

    # Print summary
    print(f"\n{'='*70}")
    print("MLB PINNACLE VALIDATION SUMMARY")
    print(f"{'='*70}")
    print(f"  Games compared: {len(both)}")
    if ML_PRODUCTION_THRESHOLD in all_cons_results:
        cs = all_cons_results[ML_PRODUCTION_THRESHOLD][0]
        ps = all_pinn_results[ML_PRODUCTION_THRESHOLD][0]
        print(f"  Consensus (>= {ML_PRODUCTION_THRESHOLD}): "
              f"{cs['wins']}-{cs['losses']} ({cs['win_pct']:.1f}%), "
              f"ROI {cs['roi']:+.1f}%, p={cs['p_value']:.4f}")
        print(f"  Pinnacle (>= {ML_PRODUCTION_THRESHOLD}): "
              f"{ps['wins']}-{ps['losses']} ({ps['win_pct']:.1f}%), "
              f"ROI {ps['roi']:+.1f}%, p={ps['p_value']:.4f}")
    print(f"  Full report: {report_path}")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()
