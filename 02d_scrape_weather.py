"""
02d -- Scrape Weather Data
============================
Fetches weather forecasts for game-day conditions at each venue.
Weather is a meaningful feature for MLB totals and can affect ML pricing.

Key factors:
  - Temperature: warmer = more runs (ball carries further)
  - Wind: out to CF/RF = more HRs, in from CF = fewer
  - Humidity: higher humidity = ball doesn't carry as far (myth debunked,
    but lower air density at altitude + heat still matters)
  - Precipitation: rain delays affect bullpen usage
  - Indoor stadiums: weather features are N/A

Indoor stadiums (retractable roof or dome):
  - Arizona Diamondbacks (Chase Field - retractable)
  - Houston Astros (Minute Maid Park - retractable)
  - Miami Marlins (loanDepot Park - retractable)
  - Milwaukee Brewers (American Family Field - retractable)
  - Seattle Mariners (T-Mobile Park - retractable)
  - Tampa Bay Rays (Tropicana Field - dome)
  - Texas Rangers (Globe Life Field - retractable)
  - Toronto Blue Jays (Rogers Centre - retractable)

Outputs:
  data/raw/weather_YYYY-MM-DD.csv

Requires: WEATHER_API_KEY env var (OpenWeatherMap or Visual Crossing)
"""

import sys
import requests
import pandas as pd
from config import RAW_DIR, TODAY, WEATHER_API_KEY, get_logger

log = get_logger("02d_weather")

INDOOR_TEAMS = {
    "Arizona Diamondbacks", "Houston Astros", "Miami Marlins",
    "Milwaukee Brewers", "Seattle Mariners", "Tampa Bay Rays",
    "Texas Rangers", "Toronto Blue Jays",
}


def fetch_weather():
    """Fetch weather forecasts for today's game venues."""
    if WEATHER_API_KEY == "YOUR_WEATHER_API_KEY":
        log.warning("WEATHER_API_KEY not set -- skipping weather data")
        log.warning("Set WEATHER_API_KEY env var for weather features")
        return None

    log.info(f"Fetching weather for {TODAY}...")

    # Load today's schedule to get venues
    schedule_path = RAW_DIR / f"schedule_{TODAY}.csv"
    if not schedule_path.exists():
        log.warning(f"Schedule not found: {schedule_path}")
        log.warning("Run 03_fetch_schedule.py first")
        return None

    schedule = pd.read_csv(schedule_path)
    log.info(f"Found {len(schedule)} games on schedule")

    # For each game, fetch weather at the home team's venue
    # Implementation depends on chosen weather API
    # Placeholder: return empty DataFrame with expected schema
    weather_data = []
    for _, game in schedule.iterrows():
        home_team = game.get("home_team", "")

        if home_team in INDOOR_TEAMS:
            weather_data.append({
                "game_id": game.get("game_id", ""),
                "home_team": home_team,
                "indoor": True,
                "temperature": 72,  # Climate-controlled
                "wind_speed": 0,
                "wind_direction_factor": 0,
                "humidity": 50,
                "precipitation_prob": 0,
            })
        else:
            # TODO: Implement actual weather API call
            weather_data.append({
                "game_id": game.get("game_id", ""),
                "home_team": home_team,
                "indoor": False,
                "temperature": None,
                "wind_speed": None,
                "wind_direction_factor": None,
                "humidity": None,
                "precipitation_prob": None,
            })

    if weather_data:
        df = pd.DataFrame(weather_data)
        out_path = RAW_DIR / f"weather_{TODAY}.csv"
        df.to_csv(out_path, index=False)
        log.info(f"Saved weather data for {len(df)} games to {out_path}")
        return df

    return None


if __name__ == "__main__":
    fetch_weather()
