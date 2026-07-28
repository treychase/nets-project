"""
Fits the flexible-lag Bayesian AR(3) model (stat_models.py) for true
shooting percentage, assist rate, rebound rate, and defensive rating on the
same player-season universe, projects every active player's next season on
all four, and combines them into one CSV ranked by expected year-over-year
improvement -- a "who's about to break out" view rather than four separate
single-stat leaderboards.

Usage:
    python build_projections.py
"""

import argparse
import os

import pandas as pd

from eda_box_scores import load_data
from stat_models import build_ar_frame, fit_ar_stat_model, project_next_season

DEFAULT_INPUT = "nba_data.csv"
DEFAULT_OUTPUT = "bayesian_ar_output/next_season_projections_all.csv"

# target column -> (label, higher values are better?)
STATS = {
    "TS_PCT": ("True Shooting %", True),
    "AST_PCT": ("Assist %", True),
    "REB_PCT": ("Rebound %", True),
    "DEF_RATING": ("Defensive Rating", False),  # points allowed per 100 possessions -- lower is better
}


def build_all_projections(
    input_csv: str = DEFAULT_INPUT,
    output_csv: str = DEFAULT_OUTPUT,
    **fit_kwargs,
) -> pd.DataFrame:
    model_source = load_data(input_csv)
    ar_df = build_ar_frame(model_source, target_cols=list(STATS))
    print(f"{len(ar_df)} player-seasons in the shared model universe "
          f"({ar_df['PLAYER_ID'].nunique()} players, min "
          f"{ar_df['SEASON_MIN_EST'].min():.0f} minutes played)")

    combined = None
    for target_col, (label, _) in STATS.items():
        print(f"\nFitting {label} ({target_col})...")
        fit = fit_ar_stat_model(ar_df, target_col, **fit_kwargs)
        proj = project_next_season(fit, ar_df)
        print(f"  {len(proj)} players projected")
        if combined is None:
            combined = proj
        else:
            combined = combined.merge(
                proj.drop(columns=["PLAYER_NAME", "TEAM", "PROJECTED_AGE"]),
                on="PLAYER_ID", how="outer",
            )

    z_cols = []
    for target_col, (label, higher_is_better) in STATS.items():
        delta = combined[f"PROJECTED_{target_col}"] - combined[f"CURRENT_{target_col}"]
        if not higher_is_better:
            delta = -delta
        combined[f"DELTA_{target_col}"] = delta
        z_col = f"Z_{target_col}"
        combined[z_col] = (delta - delta.mean()) / delta.std()
        z_cols.append(z_col)

    # Equal-weighted composite of standardized, direction-adjusted deltas --
    # "expected improvement" across shooting efficiency, playmaking,
    # rebounding, and defense, on a comparable scale despite the four raw
    # stats living on very different units.
    combined["BREAKOUT_SCORE"] = combined[z_cols].mean(axis=1)
    combined = combined.sort_values("BREAKOUT_SCORE", ascending=False).reset_index(drop=True)

    out_path = output_csv
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    combined.to_csv(out_path, index=False)
    print(f"\nWrote {len(combined)} player projections to {out_path}")
    return combined


def main():
    parser = argparse.ArgumentParser(
        description="Fit the AR(3) stat models and write combined next-season projections."
    )
    parser.add_argument("--input", default=DEFAULT_INPUT, help="Box score CSV (default: %(default)s)")
    parser.add_argument("--output", default=DEFAULT_OUTPUT, help="Output CSV path (default: %(default)s)")
    parser.add_argument("--draws", type=int, default=1000, help="Posterior draws per chain (default: %(default)s)")
    parser.add_argument("--tune", type=int, default=1000, help="Tuning steps per chain (default: %(default)s)")
    parser.add_argument("--chains", type=int, default=4, help="MCMC chains (default: %(default)s)")
    args = parser.parse_args()

    build_all_projections(
        args.input, args.output,
        draws=args.draws, tune=args.tune, chains=args.chains,
    )


if __name__ == "__main__":
    main()
