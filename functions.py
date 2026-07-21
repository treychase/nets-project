"""
NBA box score scraper.

Pulls player performance data (base + advanced stats, per game) from the
NBA Stats API via nba_api for a range of seasons, and combines everything
into a single CSV.

Usage:
    python nba_scraper.py --start-year 2000 --end-year 2025

    # or import and call directly
    from nba_scraper import scrape_nba_data
    df = scrape_nba_data(2000, 2025)

Requirements:
    pip install nba_api pandas
"""

import argparse
import time

import pandas as pd
from nba_api.stats.endpoints import leaguedashplayerstats

REQUEST_DELAY = 1.5  # seconds between requests, be polite to the NBA API


def season_string(start_year: int) -> str:
    """2000 -> '2000-01'"""
    return f"{start_year}-{str(start_year + 1)[-2:]}"


def fetch_performance(season: str) -> pd.DataFrame:
    """Base + advanced player stats for one season from the NBA Stats API."""
    base = leaguedashplayerstats.LeagueDashPlayerStats(
        season=season,
        measure_type_detailed_defense="Base",
        per_mode_detailed="PerGame",
    ).get_data_frames()[0]
    time.sleep(REQUEST_DELAY)

    adv = leaguedashplayerstats.LeagueDashPlayerStats(
        season=season,
        measure_type_detailed_defense="Advanced",
        per_mode_detailed="PerGame",
    ).get_data_frames()[0]
    time.sleep(REQUEST_DELAY)

    adv_cols = ["PLAYER_ID"] + [
        c
        for c in [
            "OFF_RATING",
            "DEF_RATING",
            "NET_RATING",
            "TS_PCT",
            "EFG_PCT",
            "USG_PCT",
            "AST_PCT",
            "REB_PCT",
            "PIE",
            "PACE",
        ]
        if c in adv.columns
    ]
    df = base.merge(adv[adv_cols], on="PLAYER_ID", how="left")
    df["SEASON"] = season
    return df


def scrape_nba_data(
    start_year: int,
    end_year: int,
    output_csv: str = "nba_data.csv",
) -> pd.DataFrame:
    """
    Scrape box score data for every season in the range and write one
    combined CSV.

    Args:
        start_year: first season's starting year, e.g. 2000 for 2000-01
        end_year: last season's starting year, e.g. 2024 for 2024-25
        output_csv: path for the combined output file

    Returns:
        The combined DataFrame, one row per player-season.
    """
    all_seasons = []

    for year in range(start_year, end_year + 1):
        season = season_string(year)
        print(f"Scraping {season}...")

        try:
            perf = fetch_performance(season)
        except Exception as e:
            print(f"  performance fetch failed for {season}: {e}")
            continue

        all_seasons.append(perf)
        print(f"  {len(perf)} players")

    if not all_seasons:
        raise RuntimeError("No data scraped for any season in the range.")

    combined = pd.concat(all_seasons, ignore_index=True)
    combined.to_csv(output_csv, index=False)
    print(f"\nSaved {len(combined)} rows to {output_csv}")
    return combined


def main():
    parser = argparse.ArgumentParser(description="Scrape NBA box score data.")
    parser.add_argument("--start-year", type=int, required=True,
                        help="First season's starting year (2000 for the 2000-01 season)")
    parser.add_argument("--end-year", type=int, required=True,
                        help="Last season's starting year (2024 for the 2024-25 season)")
    parser.add_argument("--output", type=str, default="nba_data.csv",
                        help="Output CSV path (default: nba_data.csv)")
    args = parser.parse_args()

    if args.end_year < args.start_year:
        parser.error("--end-year must be >= --start-year")

    scrape_nba_data(args.start_year, args.end_year, args.output)


if __name__ == "__main__":
    main()