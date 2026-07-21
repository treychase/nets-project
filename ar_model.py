"""
AR(1) model for next-season True Shooting % (TS%), for qualified players.

This follows directly from eda_box_scores.py's findings:

  - Multicollinearity: FG%, FG3%, FT%, EFG%, and TS% are mechanically
    related, so the model uses only TS% (the summary shooting-efficiency
    stat) as the autoregressive predictor, not all of them separately.
  - Age relationship: aging curves are rarely linear, so the model
    includes both a linear and quadratic age term rather than assuming a
    straight-line effect.
  - Between- vs within-player variance: a large share of the variance in
    TS% is "which player is this," so the model includes a per-player
    random intercept (partial pooling) instead of one pooled regression.
  - Sample size / attrition: only "qualified" player-seasons (>= 50
    estimated season field goal attempts, matching eda_box_scores.py's
    convention) with a genuine consecutive next season are used to fit
    the model, avoiding gap-year pairs and small-sample noise.

Model: NEXT_TS_PCT ~ TS_PCT + AGE_NEXT_C + AGE_NEXT_C^2, with a random
intercept per player. AGE_NEXT is the player's age during the season being
predicted (current AGE + 1), centered on the training mean for stability.

For prediction, every qualified player in the most recent season gets a
next-season TS% forecast. Players with prior AR history get their own
estimated random intercept (empirical-Bayes shrinkage toward the
population fit); players with no history (e.g. rookies) fall back to the
population-level fit alone.

Usage:
    python ar_model.py --input nba_data.csv --output-dir eda_output

Requirements:
    pip install pandas numpy statsmodels
"""

import argparse
import os

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

from eda_box_scores import load_data

MIN_SEASON_FGA = 50


def build_ar_dataset(df: pd.DataFrame):
    """Qualified player-seasons, each paired with next season's TS% when a
    genuine consecutive next season exists (gap years, e.g. injuries or
    time out of the league, are dropped rather than treated as lag-1)."""
    id_col = "PLAYER_ID" if "PLAYER_ID" in df.columns else "PLAYER_NAME"

    d = df.dropna(subset=["TS_PCT", "AGE"])
    if "SEASON_FGA_EST" in d.columns:
        d = d[d["SEASON_FGA_EST"] >= MIN_SEASON_FGA]
    d = d.sort_values([id_col, "SEASON_START_YEAR"]).reset_index(drop=True)

    d["NEXT_TS_PCT"] = d.groupby(id_col)["TS_PCT"].shift(-1)
    d["NEXT_SEASON"] = d.groupby(id_col)["SEASON_START_YEAR"].shift(-1)
    d["AGE_NEXT"] = d["AGE"] + 1

    train = d[d["NEXT_SEASON"] - d["SEASON_START_YEAR"] == 1].dropna(
        subset=["NEXT_TS_PCT"]
    ).copy()

    return d, train, id_col


def fit_ar_model(train: pd.DataFrame, id_col: str, age_center: float):
    train = train.copy()
    train["AGE_NEXT_C"] = train["AGE_NEXT"] - age_center
    train["AGE_NEXT_C2"] = train["AGE_NEXT_C"] ** 2

    model = smf.mixedlm(
        "NEXT_TS_PCT ~ TS_PCT + AGE_NEXT_C + AGE_NEXT_C2",
        data=train,
        groups=train[id_col],
    )
    return model.fit(reml=True)


def predict_next_season(df: pd.DataFrame, result, age_center: float, id_col: str) -> pd.DataFrame:
    """Predict next-season TS% for every qualified player in the most
    recent season present in the data (the season we don't have a "next"
    for yet — that's exactly what's being forecast)."""
    latest_year = df["SEASON_START_YEAR"].max()
    latest = df[df["SEASON_START_YEAR"] == latest_year].dropna(subset=["TS_PCT", "AGE"]).copy()
    if "SEASON_FGA_EST" in latest.columns:
        latest = latest[latest["SEASON_FGA_EST"] >= MIN_SEASON_FGA]

    latest["AGE_NEXT"] = latest["AGE"] + 1
    latest["AGE_NEXT_C"] = latest["AGE_NEXT"] - age_center
    latest["AGE_NEXT_C2"] = latest["AGE_NEXT_C"] ** 2

    fe = result.fe_params
    fixed_pred = (
        fe["Intercept"]
        + fe["TS_PCT"] * latest["TS_PCT"]
        + fe["AGE_NEXT_C"] * latest["AGE_NEXT_C"]
        + fe["AGE_NEXT_C2"] * latest["AGE_NEXT_C2"]
    )

    random_effects = result.random_effects
    latest["HAS_PRIOR_HISTORY"] = latest[id_col].isin(random_effects.keys())
    random_intercept = latest[id_col].map(
        lambda pid: random_effects[pid].iloc[0] if pid in random_effects else 0.0
    )

    latest["PRED_NEXT_TS_PCT"] = (fixed_pred + random_intercept).clip(0, 1)

    resid_sd = np.sqrt(result.scale)
    latest["PRED_LOW"] = (latest["PRED_NEXT_TS_PCT"] - 1.96 * resid_sd).clip(0, 1)
    latest["PRED_HIGH"] = (latest["PRED_NEXT_TS_PCT"] + 1.96 * resid_sd).clip(0, 1)

    cols = [id_col, "PLAYER_NAME", "SEASON", "AGE", "TS_PCT",
            "PRED_NEXT_TS_PCT", "PRED_LOW", "PRED_HIGH", "HAS_PRIOR_HISTORY"]
    cols = [c for c in dict.fromkeys(cols) if c in latest.columns]
    return latest[cols].sort_values("PRED_NEXT_TS_PCT", ascending=False).reset_index(drop=True)


def run_ar_model(input_csv: str, output_dir: str) -> pd.DataFrame:
    os.makedirs(output_dir, exist_ok=True)
    df = load_data(input_csv)

    full, train, id_col = build_ar_dataset(df)
    print(f"AR(1) training pairs (consecutive qualified seasons, >= {MIN_SEASON_FGA} "
          f"estimated season FGA): {len(train)}")
    if len(train) < 20:
        raise ValueError(
            "Fewer than 20 consecutive qualified-season pairs available — not enough "
            "data to fit a stable AR(1) model. Check SEASON_FGA_EST/GP populated as expected."
        )

    age_center = train["AGE_NEXT"].mean()
    result = fit_ar_model(train, id_col, age_center)
    print(result.summary())

    re_var = result.cov_re.iloc[0, 0]
    resid_var = result.scale
    print(f"\nPlayer random-intercept variance: {re_var:.5f}")
    print(f"Residual (season-to-season noise) variance: {resid_var:.5f}")
    print(f"Share of TS%-next variance attributable to player identity: "
          f"{re_var / (re_var + resid_var) * 100:.1f}%")

    preds = predict_next_season(df, result, age_center, id_col)
    n_new = (~preds["HAS_PRIOR_HISTORY"]).sum()
    print(f"\nPredicting next-season TS% for {len(preds)} qualified players "
          f"({n_new} with no prior AR history, using the population fit alone).")

    out_path = os.path.join(output_dir, "next_season_ts_pct_predictions.csv")
    preds.to_csv(out_path, index=False)
    print(f"Saved predictions to {out_path}")

    return preds


def main():
    parser = argparse.ArgumentParser(
        description="Fit an AR(1) mixed-effects model for next-season TS% "
                     "and predict it for all qualified players."
    )
    parser.add_argument("--input", type=str, default="nba_data.csv",
                        help="Input CSV from functions.py's scraper (default: nba_data.csv)")
    parser.add_argument("--output-dir", type=str, default="eda_output",
                        help="Directory for the saved predictions CSV (default: eda_output)")
    args, _ = parser.parse_known_args()

    run_ar_model(args.input, args.output_dir)


if __name__ == "__main__":
    main()
