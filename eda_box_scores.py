"""
EDA for the box score + tracking dataset, with an eye toward fitting a
Bayesian autoregressive model that predicts next-season TS% from age and
current shooting percentages.

This does NOT fit the AR model. It checks the assumptions that model will
lean on, so the model spec is informed by the data rather than guessed:

  1. Missingness — which columns/seasons have gaps, and whether tracking
     columns (only present 2013-14+) will silently shrink your sample.
  2. Distributions — are TS%, FG%, FG3%, FT% roughly bounded/bell-shaped,
     or do small-sample-size players create heavy tails/outliers?
  3. Age relationship — is TS% vs age linear, or does it need a
     quadratic/spline term? (Age-performance curves are rarely linear.)
  4. Multicollinearity — FG%, FG3%, FT%, and TS% are mechanically related
     (TS% is a weighted function of the others), so a model using all of
     them as separate predictors risks unstable coefficients.
  5. Autocorrelation — how correlated is a player's TS% with their own
     prior-season TS%? This is the core AR(1) assumption and needs to be
     checked before assuming the AR structure is worth using at all.
  6. Between- vs within-player variance — how much of the total variance
     in TS% is "which player is this" vs "season-to-season noise"? This
     tells you whether you need player-level random effects (you almost
     certainly do) rather than pooling everyone into one regression.
  7. Sample size / attrition — how many player-seasons actually have a
     "next season" to predict, and is dropping out of the league related
     to performance (survivorship bias) rather than random?

Usage:
    python eda_box_scores.py --input nba_data.csv --output-dir eda_output

Requirements:
    pip install pandas numpy matplotlib seaborn scipy statsmodels
"""

import argparse
import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.graphics.tsaplots import plot_acf

sns.set_theme(style="whitegrid")

SHOOTING_COLS = ["FG_PCT", "FG3_PCT", "FT_PCT", "TS_PCT", "EFG_PCT"]
CORE_NUMERIC_COLS = SHOOTING_COLS + ["AGE", "USG_PCT", "MIN", "GP"]


def load_data(input_csv: str) -> pd.DataFrame:
    df = pd.read_csv(input_csv)
    if "SEASON" not in df.columns:
        raise KeyError("Expected a SEASON column from nba_scraper.py output.")
    # SEASON as a sortable start-year int makes lag construction straightforward
    df["SEASON_START_YEAR"] = df["SEASON"].str[:4].astype(int)

    # nba_scraper.py's box score output is per-game (FGA is attempts per
    # game, not a season total), so a raw "FGA >= 50" filter would wipe out
    # every row. Approximate season attempts as per-game FGA times games
    # played, so the low-volume filter actually reflects sample size.
    if "FGA" in df.columns and "GP" in df.columns:
        df["SEASON_FGA_EST"] = df["FGA"] * df["GP"]
    else:
        df["SEASON_FGA_EST"] = np.nan

    return df


def missingness_report(df: pd.DataFrame, output_dir: str) -> None:
    """Column-level and season-level missingness, since tracking columns
    only exist from 2013-14 onward and will look 'missing' before that."""
    print("\n=== MISSINGNESS ===")
    missing_pct = df.isna().mean().sort_values(ascending=False) * 100
    print(missing_pct[missing_pct > 0].round(1).to_string())

    # Season-level: does missingness cluster in early seasons (tracking gap)
    # or is it scattered (data quality issue worth investigating)?
    tracking_cols = [c for c in df.columns if c.isupper() and "_" in c and
                      any(c.startswith(p) for p in
                          ["DRIVES_", "DEFENSE_", "CATCHSHOOT_", "PASSING_",
                           "PULLUPSHOT_", "REBOUNDING_", "SPEEDDISTANCE_",
                           "ELBOWTOUCH_", "POSTTOUCH_", "PAINTTOUCH_"])]
    if tracking_cols:
        by_season = df.groupby("SEASON_START_YEAR")[tracking_cols[0]].apply(
            lambda x: x.isna().mean()
        )
        print(f"\nMissingness of a sample tracking column ({tracking_cols[0]}) by season:")
        print(by_season.round(2).to_string())

    plt.figure(figsize=(10, 6))
    sns.heatmap(df[CORE_NUMERIC_COLS].isna(), cbar=False, yticklabels=False)
    plt.title("Missingness — core shooting/age/usage columns")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "missingness_heatmap.png"), dpi=150)
    plt.close()


def distribution_report(df: pd.DataFrame, output_dir: str) -> None:
    """Distributions of shooting percentages and age, with a minutes-played
    filter noted so low-sample-size players (a few made threes skewing
    FG3_PCT to 1.0 or 0.0) don't get mistaken for real signal."""
    print("\n=== DISTRIBUTIONS ===")
    print(df[CORE_NUMERIC_COLS].describe().round(3).to_string())

    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    for ax, col in zip(axes.flat, SHOOTING_COLS + ["AGE"]):
        sns.histplot(df[col].dropna(), kde=True, ax=ax)
        ax.set_title(col)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "distributions.png"), dpi=150)
    plt.close()

    # Flag players with very few field goal attempts across the season —
    # their shooting percentages are noisy and will inflate variance/outliers
    # if not filtered before modeling.
    if "SEASON_FGA_EST" in df.columns:
        low_volume = (df["SEASON_FGA_EST"] < 50).mean() * 100
        print(f"\n{low_volume:.1f}% of player-seasons have an estimated under 50 "
              f"season FGA (their FG%/TS% are high-variance and worth a minutes/"
              f"attempts floor before modeling)")


def age_relationship_report(df: pd.DataFrame, output_dir: str) -> None:
    """Check whether TS% vs age looks linear or needs a quadratic/spline
    term — AR models with a flat linear age term will misfit both ends
    of the career curve if the true relationship is curved."""
    print("\n=== AGE RELATIONSHIP ===")
    valid = df.dropna(subset=["AGE", "TS_PCT"])
    if "SEASON_FGA_EST" in valid.columns:
        valid = valid[valid["SEASON_FGA_EST"] >= 50]

    linear_corr = valid[["AGE", "TS_PCT"]].corr().iloc[0, 1]
    print(f"Linear correlation between AGE and TS_PCT: {linear_corr:.3f}")

    binned = valid.groupby(pd.cut(valid["AGE"], bins=range(18, 42, 2)))["TS_PCT"].mean()
    print("\nMean TS% by 2-year age bin (look for a peak/curve, not a straight line):")
    print(binned.round(3).to_string())

    plt.figure(figsize=(8, 5))
    sns.regplot(data=valid, x="AGE", y="TS_PCT", lowess=True,
                scatter_kws={"alpha": 0.15, "s": 10})
    plt.title("TS% vs Age (LOWESS smooth — compare shape to a straight line)")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "ts_pct_vs_age.png"), dpi=150)
    plt.close()


def multicollinearity_report(df: pd.DataFrame, output_dir: str) -> None:
    """FG%, FG3%, FT%, and TS% are mechanically related — TS% is a
    weighted combination of the others. Using all of them as separate
    predictors in one model risks unstable, hard-to-interpret coefficients."""
    print("\n=== MULTICOLLINEARITY ===")
    corr = df[SHOOTING_COLS].corr()
    print(corr.round(2).to_string())

    plt.figure(figsize=(6, 5))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", vmin=-1, vmax=1)
    plt.title("Correlation among shooting percentage columns")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "shooting_corr_heatmap.png"), dpi=150)
    plt.close()

    valid = df[SHOOTING_COLS].dropna()
    vif_data = pd.DataFrame({
        "column": SHOOTING_COLS,
        "VIF": [variance_inflation_factor(valid.values, i)
                for i in range(len(SHOOTING_COLS))],
    })
    print("\nVariance Inflation Factors (VIF > ~5 signals a collinearity problem):")
    print(vif_data.round(2).to_string(index=False))


def autocorrelation_report(df: pd.DataFrame, output_dir: str) -> pd.DataFrame:
    """The core AR(1) check: how correlated is a player's TS% with their
    own prior-season TS%? Built by shifting each player's rows forward one
    season, matched on consecutive SEASON_START_YEAR so a gap year (injury,
    out of the league) doesn't get treated as a valid lag-1 pair."""
    print("\n=== AUTOCORRELATION (lag-1 TS%) ===")

    id_col = "PLAYER_ID" if "PLAYER_ID" in df.columns else "PLAYER_NAME"
    cols = [id_col, "SEASON_START_YEAR", "TS_PCT", "AGE"]
    if "SEASON_FGA_EST" in df.columns:
        cols.append("SEASON_FGA_EST")
    d = df[cols].dropna(subset=["TS_PCT"]).sort_values([id_col, "SEASON_START_YEAR"])

    if "SEASON_FGA_EST" in d.columns:
        d = d[d["SEASON_FGA_EST"] >= 50]

    if len(d) == 0:
        print("No player-seasons left after the volume filter — check that "
              "GP/FGA columns are populated as expected.")
        return pd.DataFrame(columns=cols + ["PREV_TS_PCT", "PREV_SEASON"])

    d["PREV_TS_PCT"] = d.groupby(id_col)["TS_PCT"].shift(1)
    d["PREV_SEASON"] = d.groupby(id_col)["SEASON_START_YEAR"].shift(1)
    # Only keep true consecutive seasons — drop pairs with a gap year
    lagged = d[d["SEASON_START_YEAR"] - d["PREV_SEASON"] == 1].dropna(subset=["PREV_TS_PCT"])

    lag1_corr = lagged[["TS_PCT", "PREV_TS_PCT"]].corr().iloc[0, 1]
    print(f"Lag-1 (consecutive-season) TS% correlation: {lag1_corr:.3f}")
    if len(d) > 0:
        print(f"Player-seasons with a valid consecutive prior season: {len(lagged)} "
              f"of {len(d)} total ({len(lagged) / len(d) * 100:.1f}%)")

    plt.figure(figsize=(6, 6))
    sns.regplot(data=lagged, x="PREV_TS_PCT", y="TS_PCT",
                scatter_kws={"alpha": 0.2, "s": 10})
    plt.title(f"TS% vs prior-season TS% (r = {lag1_corr:.2f})")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "lag1_ts_pct.png"), dpi=150)
    plt.close()

    return lagged


def variance_decomposition_report(df: pd.DataFrame) -> None:
    """Between-player vs within-player variance in TS%. A large
    between-player share means most of the spread in TS% is just "who is
    this player" — evidence you need player-level random effects (a
    hierarchical/multilevel structure) rather than one pooled regression."""
    print("\n=== VARIANCE DECOMPOSITION ===")
    id_col = "PLAYER_ID" if "PLAYER_ID" in df.columns else "PLAYER_NAME"
    d = df.dropna(subset=["TS_PCT"])
    if "SEASON_FGA_EST" in d.columns:
        d = d[d["SEASON_FGA_EST"] >= 50]

    player_means = d.groupby(id_col)["TS_PCT"].transform("mean")
    between_var = player_means.var()
    within_var = (d["TS_PCT"] - player_means).var()
    total_var = between_var + within_var

    print(f"Between-player variance share: {between_var / total_var * 100:.1f}%")
    print(f"Within-player (season-to-season) variance share: {within_var / total_var * 100:.1f}%")
    print("A large between-player share supports a hierarchical model with "
          "player-level intercepts rather than pooling everyone together.")


def sample_size_report(df: pd.DataFrame) -> None:
    """How many player-seasons actually have a usable prior season, and
    whether players who leave the league differ systematically from those
    who stay — attrition that's related to performance (not random) biases
    a naive AR fit toward players who kept playing."""
    print("\n=== SAMPLE SIZE / ATTRITION ===")
    id_col = "PLAYER_ID" if "PLAYER_ID" in df.columns else "PLAYER_NAME"
    d = df.dropna(subset=["TS_PCT"]).sort_values([id_col, "SEASON_START_YEAR"])

    d["HAS_NEXT_SEASON"] = d.groupby(id_col)["SEASON_START_YEAR"].shift(-1) - d["SEASON_START_YEAR"] == 1
    attrition_ts = d.groupby("HAS_NEXT_SEASON")["TS_PCT"].mean()
    print("Mean TS% for player-seasons WITH vs WITHOUT a following season "
          "(a gap here suggests non-random attrition):")
    print(attrition_ts.round(3).to_string())

    pct_with_next = d["HAS_NEXT_SEASON"].mean() * 100
    print(f"\n{pct_with_next:.1f}% of player-seasons have a usable next season "
          f"for the AR target.")


def run_eda(input_csv: str, output_dir: str) -> None:
    os.makedirs(output_dir, exist_ok=True)
    df = load_data(input_csv)

    print(f"Loaded {len(df)} player-season rows, {df['SEASON'].nunique()} seasons "
          f"({df['SEASON'].min()} to {df['SEASON'].max()})")

    missingness_report(df, output_dir)
    distribution_report(df, output_dir)
    age_relationship_report(df, output_dir)
    multicollinearity_report(df, output_dir)
    lagged = autocorrelation_report(df, output_dir)
    variance_decomposition_report(df)
    sample_size_report(df)

    lagged.to_csv(os.path.join(output_dir, "lagged_ts_pct_pairs.csv"), index=False)
    print(f"\nSaved plots and lagged pairs dataset to {output_dir}/")


def main():
    parser = argparse.ArgumentParser(
        description="EDA on box score data, focused on Bayesian AR modeling assumptions."
    )
    parser.add_argument("--input", type=str, default="nba_data.csv",
                        help="Input CSV from nba_scraper.py (default: nba_data.csv)")
    parser.add_argument("--output-dir", type=str, default="eda_output",
                        help="Directory for saved plots/CSVs (default: eda_output)")
    args, _ = parser.parse_known_args()

    run_eda(args.input, args.output_dir)


if __name__ == "__main__":
    main()
