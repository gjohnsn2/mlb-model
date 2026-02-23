"""
12 -- Compare Model Versions
===============================
Compares production model vs. walk-forward fold performance.
Useful for detecting overfitting or model degradation.

Ported from CBB pipeline.
"""

import sys
import json
import pandas as pd
import numpy as np
from config import MODELS_ROOT, HISTORICAL_DIR, get_logger

log = get_logger("12_compare")


def compare():
    """Compare production model to OOF walk-forward results."""
    # Load training metrics
    metrics_path = MODELS_ROOT / "training_metrics.json"
    if not metrics_path.exists():
        log.error(f"Training metrics not found: {metrics_path}")
        log.error("Run 06_train_model.py first")
        sys.exit(1)

    with open(metrics_path) as f:
        metrics = json.load(f)

    log.info("Model Comparison Report")
    log.info("=" * 60)
    log.info(f"\nMargin Model:")
    log.info(f"  Walk-forward RMSE: {metrics.get('margin_rmse', 'N/A'):.3f}")
    log.info(f"  Walk-forward MAE:  {metrics.get('margin_mae', 'N/A'):.3f}")
    log.info(f"  Features: {len(metrics.get('margin_features', []))}")

    log.info(f"\nTotal Model:")
    log.info(f"  Walk-forward RMSE: {metrics.get('total_rmse', 'N/A'):.3f}")
    log.info(f"  Walk-forward MAE:  {metrics.get('total_mae', 'N/A'):.3f}")
    log.info(f"  Features: {len(metrics.get('total_features', []))}")


if __name__ == "__main__":
    compare()
