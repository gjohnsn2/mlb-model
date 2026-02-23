"""
Weather Impact Module
=======================
Models the effect of weather conditions on MLB game outcomes.

Key findings from research:
  - Temperature: Every 10F increase adds ~0.3 runs to total. Ball carries
    further in warm air (lower air density). Biggest effect at 90F+ vs 60F-.
  - Wind: Out to CF at 15+ mph adds ~1 run. In from CF at 15+ mph subtracts ~1 run.
    Cross-winds have moderate effects on fly balls.
  - Humidity: Minimal direct effect (myth that humid air = further carry is wrong;
    humid air is actually less dense, but the effect is tiny).
  - Rain probability: Affects bullpen planning more than direct scoring.
  - Indoor stadiums: Remove all weather effects.

Usage:
  From feature_engine.py or 05_build_features.py, call these functions
  to compute weather features for each game.
"""

import numpy as np
from config import get_logger

log = get_logger("utils_weather")


def temperature_run_adjustment(temp_f):
    """
    Estimate the run scoring adjustment based on temperature.
    Baseline: 72F (typical indoor/neutral temperature).
    Returns: adjustment in expected runs (positive = more runs).

    Based on published research: ~0.03 runs per degree F above/below 72.
    """
    if temp_f is None or np.isnan(temp_f):
        return 0.0

    baseline = 72.0
    coefficient = 0.03  # runs per degree F
    return (temp_f - baseline) * coefficient


def wind_scoring_factor(wind_speed_mph, wind_direction_deg, park_orientation_deg=None):
    """
    Estimate wind's effect on scoring.

    wind_direction_deg: meteorological direction (0=N, 90=E, 180=S, 270=W)
    park_orientation_deg: direction from home plate to center field

    Returns: adjustment in expected runs.

    Simplified model:
      - Wind out to CF: +runs
      - Wind in from CF: -runs
      - Crosswind: ~neutral
    """
    if wind_speed_mph is None or np.isnan(wind_speed_mph):
        return 0.0

    if wind_speed_mph < 5:
        return 0.0  # Minimal effect below 5 mph

    # Without park orientation, return a simple speed-based factor
    # This is a placeholder; full implementation needs park_orientation_deg
    # for each venue to compute wind direction relative to the field
    return 0.0


def compute_wind_direction_factor(
    wind_speed_mph, wind_dir_deg, outfield_dir_deg
):
    """
    Compute a signed wind factor: positive = blowing out, negative = blowing in.

    outfield_dir_deg: compass direction from home plate to center field for this park.
    """
    if wind_speed_mph is None or np.isnan(wind_speed_mph) or wind_speed_mph < 5:
        return 0.0

    if wind_dir_deg is None or np.isnan(wind_dir_deg):
        return 0.0

    if outfield_dir_deg is None or np.isnan(outfield_dir_deg):
        return 0.0

    # Compute angle between wind direction and outfield direction
    # Wind FROM a direction = meteorological convention
    # We want wind BLOWING TOWARD outfield = blowing out
    wind_blowing_toward = (wind_dir_deg + 180) % 360
    angle_diff = abs(wind_blowing_toward - outfield_dir_deg)
    if angle_diff > 180:
        angle_diff = 360 - angle_diff

    # cos(0) = 1.0 (blowing straight out), cos(90) = 0 (crosswind), cos(180) = -1 (straight in)
    import math
    alignment = math.cos(math.radians(angle_diff))

    # Scale by wind speed: ~0.1 runs per mph at perfect alignment
    coefficient = 0.05  # runs per mph at perfect alignment
    return alignment * wind_speed_mph * coefficient


def is_weather_relevant(home_team):
    """Check if weather affects this game (outdoor parks only)."""
    from utils_park_factors import is_indoor
    return not is_indoor(home_team)
