"""
Generic flexible-lag Bayesian AR(3) model for projecting an NBA player's
next-season stat from age, usage rate, and up to three prior seasons of the
stat itself.

This is the same architecture as the pooled TS% model in project.ipynb
(cells 8-16) -- age + age^2, a covariate, three flexible AR lags (each only
used when that exact season is present, not just "the previous row"), and
an n_lags_available covariate -- factored out so it can be reused for
assists (AST_PCT), rebounds (REB_PCT), and defensive efficiency
(DEF_RATING) without re-deriving the model per stat.

Usage:
    from eda_box_scores import load_data
    from stat_models import build_ar_frame, fit_ar_stat_model, project_next_season

    model_source = load_data("nba_data.csv")
    ar_df = build_ar_frame(model_source, target_cols=["TS_PCT", "AST_PCT"])
    fit = fit_ar_stat_model(ar_df, "AST_PCT")
    projections = project_next_season(fit, ar_df)
"""

from __future__ import annotations

from dataclasses import dataclass, field

import arviz as az
import numpy as np
import pandas as pd
import pymc as pm

# Minutes played (per-game MIN * GP) floor shared by every stat below, so all
# four models are fit on the same player-season universe and their projected
# deltas are comparable. A pure attempts floor (as used for the TS% model)
# doesn't make sense for rebounding/defense, so this uses playing time
# instead -- broader than "shoots enough" but still "played enough to matter".
MIN_SEASON_MINUTES = 500

AR_LAGS = (1, 2, 3)


def build_ar_frame(
    model_source: pd.DataFrame,
    target_cols: list[str],
    covariate_cols: tuple[str, ...] = ("USG_PCT",),
    min_minutes: float = MIN_SEASON_MINUTES,
) -> pd.DataFrame:
    """One row per qualifying player-season, with flexible AR(3) lag columns
    for every column in target_cols plus a shared N_LAGS_AVAILABLE (season
    availability doesn't depend on which stat is being modeled)."""
    df = model_source.copy()
    if "SEASON_MIN_EST" not in df.columns:
        df["SEASON_MIN_EST"] = df["MIN"] * df["GP"]

    id_cols = ["PLAYER_ID", "PLAYER_NAME", "TEAM_ABBREVIATION", "SEASON_START_YEAR", "AGE", "SEASON_MIN_EST"]
    cols = list(dict.fromkeys(id_cols + list(covariate_cols) + list(target_cols)))

    ar_df = df[cols].dropna().sort_values(["PLAYER_ID", "SEASON_START_YEAR"])
    ar_df = ar_df[ar_df["SEASON_MIN_EST"] >= min_minutes].reset_index(drop=True)

    for k in AR_LAGS:
        season_lag = ar_df.groupby("PLAYER_ID")["SEASON_START_YEAR"].shift(k)
        ar_df[f"HAS_LAG_{k}"] = (ar_df["SEASON_START_YEAR"] - season_lag) == k
    ar_df["N_LAGS_AVAILABLE"] = ar_df[[f"HAS_LAG_{k}" for k in AR_LAGS]].sum(axis=1)

    for target_col in target_cols:
        for k in AR_LAGS:
            raw_prev = ar_df.groupby("PLAYER_ID")[target_col].shift(k)
            ar_df[f"{target_col}_PREV_{k}"] = raw_prev.where(ar_df[f"HAS_LAG_{k}"])

    return ar_df


@dataclass
class ARStatFit:
    """A fitted flexible-lag AR(3) model for one target stat, plus the
    standardization constants needed to project new player-seasons."""

    target_col: str
    covariate_cols: list[str]
    idata: object
    y_mean: float
    y_sd: float
    age_mean: float
    age_sd: float
    cov_mean: dict = field(default_factory=dict)
    cov_sd: dict = field(default_factory=dict)
    n_lags_mean: float = 0.0
    n_lags_sd: float = 1.0


def fit_ar_stat_model(
    ar_df: pd.DataFrame,
    target_col: str,
    covariate_cols: tuple[str, ...] = ("USG_PCT",),
    chains: int = 4,
    draws: int = 1000,
    tune: int = 1000,
    target_accept: float = 0.9,
    seed: int = 42,
    progressbar: bool = False,
    compute_loglik: bool = True,
) -> ARStatFit:
    """Fit the pooled flexible-AR(3) model for one target stat. Priors are
    scaled to the target's own standard deviation rather than a fixed
    constant, since DEF_RATING (~100-120) and the percentage stats (~0-1)
    live on very different scales."""
    y = ar_df[target_col].to_numpy(dtype=float)
    age = ar_df["AGE"].to_numpy(dtype=float)
    y_mean, y_sd = y.mean(), y.std()
    age_mean, age_sd = age.mean(), age.std()

    cov_mean, cov_sd, cov_z = {}, {}, {}
    for c in covariate_cols:
        raw = ar_df[c].to_numpy(dtype=float)
        m, s = raw.mean(), raw.std()
        cov_mean[c], cov_sd[c] = m, s
        cov_z[c] = (raw - m) / s

    def lagged_z(k: int) -> np.ndarray:
        raw = ar_df[f"{target_col}_PREV_{k}"].to_numpy(dtype=float)
        has = ar_df[f"HAS_LAG_{k}"].to_numpy()
        filled = np.where(has, raw, y_mean)
        return (filled - y_mean) / y_sd

    age_z = (age - age_mean) / age_sd
    prev_z = {k: lagged_z(k) for k in AR_LAGS}
    n_lags = ar_df["N_LAGS_AVAILABLE"].to_numpy(dtype=float)
    n_lags_mean, n_lags_sd = n_lags.mean(), n_lags.std()
    n_lags_z = (n_lags - n_lags_mean) / n_lags_sd

    with pm.Model():
        alpha = pm.Normal("alpha", mu=y_mean, sigma=3 * y_sd)
        phi = {k: pm.Normal(f"phi{k}", mu=0, sigma=1) for k in AR_LAGS}
        beta_age = pm.Normal("beta_age", mu=0, sigma=1)
        beta_age_sq = pm.Normal("beta_age_sq", mu=0, sigma=1)
        beta_cov = {c: pm.Normal(f"beta_{c.lower()}", mu=0, sigma=1) for c in covariate_cols}
        beta_n_lags = pm.Normal("beta_n_lags", mu=0, sigma=1)
        sigma = pm.HalfNormal("sigma", sigma=y_sd)

        mu = alpha + beta_age * age_z + beta_age_sq * age_z**2 + beta_n_lags * n_lags_z
        for k in AR_LAGS:
            mu = mu + phi[k] * prev_z[k]
        for c in covariate_cols:
            mu = mu + beta_cov[c] * cov_z[c]

        pm.Normal("obs", mu=mu, sigma=sigma, observed=y)
        idata = pm.sample(
            draws, tune=tune, chains=chains, target_accept=target_accept,
            random_seed=seed, progressbar=progressbar,
        )
        if compute_loglik:
            # Needed for az.loo (PSIS-LOO-CV) -- cheap relative to sampling itself,
            # since it only evaluates the observed-data log density at already-drawn
            # posterior samples rather than doing any new MCMC.
            pm.compute_log_likelihood(idata, progressbar=progressbar)

    return ARStatFit(
        target_col=target_col,
        covariate_cols=list(covariate_cols),
        idata=idata,
        y_mean=y_mean, y_sd=y_sd,
        age_mean=age_mean, age_sd=age_sd,
        cov_mean=cov_mean, cov_sd=cov_sd,
        n_lags_mean=n_lags_mean, n_lags_sd=n_lags_sd,
    )


def project_ar_stat(
    fit: ARStatFit,
    age_value: float,
    covariate_values: dict,
    prev_values: dict,
) -> np.ndarray:
    """Posterior samples of the projected stat value for one player-season.

    prev_values: {1: value_or_None, 2: value_or_None, 3: value_or_None},
    each the stat's value that many seasons back (None/absent if that lag
    isn't available -- imputed to the population mean, same as training,
    with n_lags_available capturing how much real history backs the
    projection).
    """
    post = fit.idata.posterior
    age_z_ = (age_value - fit.age_mean) / fit.age_sd

    n_avail = sum(v is not None for v in prev_values.values())
    n_lags_z_ = (n_avail - fit.n_lags_mean) / fit.n_lags_sd

    mu_samples = (
        post["alpha"]
        + post["beta_age"] * age_z_
        + post["beta_age_sq"] * age_z_**2
        + post["beta_n_lags"] * n_lags_z_
    )
    for k in AR_LAGS:
        v = prev_values.get(k)
        v_z = ((v if v is not None else fit.y_mean) - fit.y_mean) / fit.y_sd
        mu_samples = mu_samples + post[f"phi{k}"] * v_z
    for c in fit.covariate_cols:
        cv = (covariate_values[c] - fit.cov_mean[c]) / fit.cov_sd[c]
        mu_samples = mu_samples + post[f"beta_{c.lower()}"] * cv

    return mu_samples.values.flatten()


def project_next_season(fit: ARStatFit, ar_df: pd.DataFrame, hdi_prob: float = 0.94) -> pd.DataFrame:
    """Project every player active in the most recent scraped season on
    fit.target_col, mirroring project.ipynb's next-season dashboard cells."""
    target_col = fit.target_col
    df = ar_df.sort_values(["PLAYER_ID", "SEASON_START_YEAR"]).copy()
    df[f"{target_col}_LAG_1"] = df.groupby("PLAYER_ID")[target_col].shift(1)
    df[f"{target_col}_LAG_2"] = df.groupby("PLAYER_ID")[target_col].shift(2)
    df["SEASON_LAG_1"] = df.groupby("PLAYER_ID")["SEASON_START_YEAR"].shift(1)
    df["SEASON_LAG_2"] = df.groupby("PLAYER_ID")["SEASON_START_YEAR"].shift(2)

    latest_year = df["SEASON_START_YEAR"].max()
    next_rows = df[df["SEASON_START_YEAR"] == latest_year].copy()

    next_rows["PROJ_PREV_1"] = next_rows[target_col]
    has_lag_2 = (next_rows["SEASON_START_YEAR"] - next_rows["SEASON_LAG_1"]) == 1
    next_rows["PROJ_PREV_2"] = next_rows[f"{target_col}_LAG_1"].where(has_lag_2)
    has_lag_3 = (next_rows["SEASON_START_YEAR"] - next_rows["SEASON_LAG_2"]) == 2
    next_rows["PROJ_PREV_3"] = next_rows[f"{target_col}_LAG_2"].where(has_lag_3)

    next_rows["PROJECTED_AGE"] = next_rows["AGE"] + 1

    results = []
    for _, row in next_rows.iterrows():
        prev_values = {
            1: row["PROJ_PREV_1"],
            2: row["PROJ_PREV_2"] if pd.notna(row["PROJ_PREV_2"]) else None,
            3: row["PROJ_PREV_3"] if pd.notna(row["PROJ_PREV_3"]) else None,
        }
        covariate_values = {c: row[c] for c in fit.covariate_cols}
        samples = project_ar_stat(fit, row["PROJECTED_AGE"], covariate_values, prev_values)
        hdi = az.hdi(samples, hdi_prob=hdi_prob)
        results.append({
            "PLAYER_ID": row["PLAYER_ID"],
            "PLAYER_NAME": row["PLAYER_NAME"],
            "TEAM": row["TEAM_ABBREVIATION"],
            "PROJECTED_AGE": row["PROJECTED_AGE"],
            f"CURRENT_{target_col}": row[target_col],
            f"PROJECTED_{target_col}": samples.mean(),
            f"{target_col}_HDI_LOW": hdi[0],
            f"{target_col}_HDI_HIGH": hdi[1],
        })

    return pd.DataFrame(results)
