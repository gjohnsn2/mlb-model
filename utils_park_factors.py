"""
Park Factors Utility Module
==============================
Park-specific adjustments for MLB venues. Park factors are one of the
most important and underweighted features in MLB betting models.

Key concepts:
  - Run factor > 1.0: hitter-friendly (Coors Field, Great American Ballpark)
  - Run factor < 1.0: pitcher-friendly (Oracle Park, Petco Park)
  - HR factor: how much the park inflates/deflates home runs specifically
  - Wind/temperature interact with park factors (covered in weather module)

Park factors are relatively stable year-to-year but should be refreshed
at least annually. Indoor parks have consistent factors regardless of weather.
"""

import pandas as pd
import numpy as np
from config import get_logger

log = get_logger("utils_parks")

# Static park factor reference (2024 data)
# run_factor: runs scored relative to league average (1.0 = neutral)
# hr_factor: home runs relative to league average
PARK_FACTORS_STATIC = {
    "Colorado Rockies":        {"run": 1.38, "hr": 1.36, "venue": "Coors Field", "indoor": False, "elevation_ft": 5280},
    "Boston Red Sox":          {"run": 1.12, "hr": 1.06, "venue": "Fenway Park", "indoor": False, "elevation_ft": 20},
    "Cincinnati Reds":         {"run": 1.10, "hr": 1.22, "venue": "Great American Ball Park", "indoor": False, "elevation_ft": 490},
    "Texas Rangers":           {"run": 1.08, "hr": 1.15, "venue": "Globe Life Field", "indoor": True, "elevation_ft": 530},
    "Arizona Diamondbacks":    {"run": 1.06, "hr": 1.08, "venue": "Chase Field", "indoor": True, "elevation_ft": 1082},
    "Chicago Cubs":            {"run": 1.05, "hr": 1.12, "venue": "Wrigley Field", "indoor": False, "elevation_ft": 590},
    "Philadelphia Phillies":   {"run": 1.04, "hr": 1.10, "venue": "Citizens Bank Park", "indoor": False, "elevation_ft": 20},
    "Toronto Blue Jays":       {"run": 1.04, "hr": 1.08, "venue": "Rogers Centre", "indoor": True, "elevation_ft": 250},
    "Minnesota Twins":         {"run": 1.03, "hr": 1.06, "venue": "Target Field", "indoor": False, "elevation_ft": 830},
    "Milwaukee Brewers":       {"run": 1.02, "hr": 1.08, "venue": "American Family Field", "indoor": True, "elevation_ft": 635},
    "New York Yankees":        {"run": 1.02, "hr": 1.12, "venue": "Yankee Stadium", "indoor": False, "elevation_ft": 15},
    "Atlanta Braves":          {"run": 1.01, "hr": 1.05, "venue": "Truist Park", "indoor": False, "elevation_ft": 1050},
    "Baltimore Orioles":       {"run": 1.00, "hr": 1.08, "venue": "Camden Yards", "indoor": False, "elevation_ft": 30},
    "Los Angeles Angels":      {"run": 1.00, "hr": 0.98, "venue": "Angel Stadium", "indoor": False, "elevation_ft": 157},
    "Houston Astros":          {"run": 0.99, "hr": 1.02, "venue": "Minute Maid Park", "indoor": True, "elevation_ft": 42},
    "Detroit Tigers":          {"run": 0.99, "hr": 0.92, "venue": "Comerica Park", "indoor": False, "elevation_ft": 585},
    "Washington Nationals":    {"run": 0.98, "hr": 1.02, "venue": "Nationals Park", "indoor": False, "elevation_ft": 20},
    "Cleveland Guardians":     {"run": 0.98, "hr": 0.94, "venue": "Progressive Field", "indoor": False, "elevation_ft": 653},
    "Kansas City Royals":      {"run": 0.97, "hr": 0.90, "venue": "Kauffman Stadium", "indoor": False, "elevation_ft": 820},
    "Pittsburgh Pirates":      {"run": 0.97, "hr": 0.85, "venue": "PNC Park", "indoor": False, "elevation_ft": 730},
    "Seattle Mariners":        {"run": 0.96, "hr": 0.92, "venue": "T-Mobile Park", "indoor": True, "elevation_ft": 10},
    "Chicago White Sox":       {"run": 0.96, "hr": 1.04, "venue": "Guaranteed Rate Field", "indoor": False, "elevation_ft": 595},
    "St. Louis Cardinals":     {"run": 0.95, "hr": 0.88, "venue": "Busch Stadium", "indoor": False, "elevation_ft": 455},
    "Los Angeles Dodgers":     {"run": 0.95, "hr": 0.92, "venue": "Dodger Stadium", "indoor": False, "elevation_ft": 515},
    "San Diego Padres":        {"run": 0.94, "hr": 0.86, "venue": "Petco Park", "indoor": False, "elevation_ft": 15},
    "New York Mets":           {"run": 0.93, "hr": 0.90, "venue": "Citi Field", "indoor": False, "elevation_ft": 10},
    "Tampa Bay Rays":          {"run": 0.93, "hr": 0.88, "venue": "Tropicana Field", "indoor": True, "elevation_ft": 5},
    "San Francisco Giants":    {"run": 0.92, "hr": 0.80, "venue": "Oracle Park", "indoor": False, "elevation_ft": 5},
    "Miami Marlins":           {"run": 0.91, "hr": 0.82, "venue": "loanDepot Park", "indoor": True, "elevation_ft": 10},
    "Oakland Athletics":       {"run": 0.90, "hr": 0.78, "venue": "Oakland Coliseum", "indoor": False, "elevation_ft": 5},
}


def get_park_run_factor(team):
    """Get run park factor for team's home venue."""
    info = PARK_FACTORS_STATIC.get(team, {})
    return info.get("run", 1.0)


def get_park_hr_factor(team):
    """Get HR park factor for team's home venue."""
    info = PARK_FACTORS_STATIC.get(team, {})
    return info.get("hr", 1.0)


def is_indoor(team):
    """Check if team plays in an indoor/retractable roof venue."""
    info = PARK_FACTORS_STATIC.get(team, {})
    return info.get("indoor", False)


def get_total_adjustment(home_team, base_total=8.5):
    """
    Compute park factor adjustment for total predictions.
    Returns the number of runs to add/subtract from the base total.
    Example: Coors Field (1.38) on a base of 8.5 -> +3.23 runs adjustment
    """
    factor = get_park_run_factor(home_team)
    return base_total * (factor - 1.0)
