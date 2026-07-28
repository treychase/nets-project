"""
Cross-validation for the flexible-lag Bayesian AR(3) stat models
(stat_models.py / build_projections.py), covering TS_PCT, AST_PCT,
REB_PCT, and DEF_RATING.

Two complementary checks per stat:

1. **PSIS-LOO-CV** (`az.loo`) on the full-data fit -- this project's
   established Bayesian CV method (see docs/TECHNICAL.md, the pooled vs.
   hierarchical TS% comparison). Cheap: it reuses the already-drawn
   posterior via Pareto-smoothed importance sampling rather than refitting
   per held-out point, but that approximation can be unreliable, which is
   exactly what the Pareto-k diagnostic below is for.
2. **K-fold holdout validation** -- an actual refit-and-predict check
   (row-level random folds, no shuffling leakage since AR lag features are
   realized historical values, not model outputs) reporting genuine
   out-of-sample RMSE / MAE / R^2. This is the check that doesn't rely on
   PSIS-LOO's importance-sampling approximation at all.

Usage:
    python cross_validate.py
    python cross_validate.py --k 5 --stats TS_PCT AST_PCT
"""

import argparse
import os

import arviz as az
import numpy as np
import pandas as pd
import pymc as pm

from eda_box_scores import load_data
from stat_models import AR_LAGS, build_ar_frame, fit_ar_stat_model, project_ar_stat

ALL_STATS = ["TS_PCT", "AST_PCT", "REB_PCT", "DEF_RATING"]
OUTPUT_DIR = "bayesian_ar_output"
DEFAULT_INPUT = "nba_data.csv"


def _make_folds(n: int, k: int, seed: int) -> list[np.ndarray]:
    rng = np.random.default_rng(seed)
    idx = rng.permutation(n)
    return np.array_split(idx, k)


def loo_report(ar_df: pd.DataFrame, target_col: str, **fit_kwargs) -> dict:
    """PSIS-LOO-CV on a model fit to the full dataset."""
    fit = fit_ar_stat_model(ar_df, target_col, **fit_kwargs)
    loo = az.loo(fit.idata, pointwise=True)
    pareto_k = loo.pareto_k.values
    return {
        "elpd_loo": float(loo.elpd_loo),
        "elpd_loo_se": float(loo.se),
        "p_loo": float(loo.p_loo),
        "n_obs": len(pareto_k),
        "max_pareto_k": float(pareto_k.max()),
        "pct_pareto_k_over_0.7": float((pareto_k > 0.7).mean() * 100),
        "loo_reliable": bool((pareto_k <= 0.7).all()),
    }


def kfold_report(ar_df: pd.DataFrame, target_col: str, k: int = 5, seed: int = 42, **fit_kwargs) -> dict:
    """Refit on k-1 folds, predict the held-out fold, pooled across all k folds."""
    folds = _make_folds(len(ar_df), k, seed)
    y_true_all, y_pred_all = [], []

    for fold_idx in range(k):
        test_idx = folds[fold_idx]
        train_idx = np.concatenate([folds[i] for i in range(k) if i != fold_idx])
        train_df = ar_df.iloc[train_idx].reset_index(drop=True)
        test_df = ar_df.iloc[test_idx]

        fit = fit_ar_stat_model(train_df, target_col, compute_loglik=False, **fit_kwargs)

        for _, row in test_df.iterrows():
            prev_values = {
                lag: (row[f"{target_col}_PREV_{lag}"] if row[f"HAS_LAG_{lag}"] else None)
                for lag in AR_LAGS
            }
            covariate_values = {c: row[c] for c in fit.covariate_cols}
            samples = project_ar_stat(fit, row["AGE"], covariate_values, prev_values)
            y_pred_all.append(samples.mean())
            y_true_all.append(row[target_col])

    y_true = np.array(y_true_all)
    y_pred = np.array(y_pred_all)
    resid = y_true - y_pred
    ss_res = np.sum(resid**2)
    ss_tot = np.sum((y_true - y_true.mean()) ** 2)

    return {
        "k_folds": k,
        "n_obs": len(y_true),
        "rmse": float(np.sqrt(np.mean(resid**2))),
        "mae": float(np.mean(np.abs(resid))),
        "r2": float(1 - ss_res / ss_tot),
    }


def run_cross_validation(
    input_csv: str = DEFAULT_INPUT,
    stats: list[str] = ALL_STATS,
    k: int = 5,
    output_dir: str = OUTPUT_DIR,
    **fit_kwargs,
) -> pd.DataFrame:
    model_source = load_data(input_csv)
    ar_df = build_ar_frame(model_source, target_cols=stats)
    print(f"{len(ar_df)} player-seasons in the shared model universe\n")

    rows = []
    for target_col in stats:
        print(f"=== {target_col} ===")

        print("  Fitting full model + PSIS-LOO-CV...")
        loo = loo_report(ar_df, target_col, **fit_kwargs)
        reliability = "reliable" if loo["loo_reliable"] else "UNRELIABLE (Pareto k > 0.7 present)"
        print(
            f"    elpd_loo = {loo['elpd_loo']:.1f} (se {loo['elpd_loo_se']:.1f}), "
            f"p_loo = {loo['p_loo']:.1f}, max Pareto k = {loo['max_pareto_k']:.2f} -> {reliability}"
        )

        print(f"  Running {k}-fold holdout validation ({k} refits)...")
        kfold = kfold_report(ar_df, target_col, k=k, **fit_kwargs)
        print(
            f"    RMSE = {kfold['rmse']:.4f}, MAE = {kfold['mae']:.4f}, "
            f"R^2 = {kfold['r2']:.3f} (n={kfold['n_obs']})"
        )
        print()

        rows.append({"target_col": target_col, **loo, **kfold})

    result = pd.DataFrame(rows)
    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, "cross_validation.csv")
    result.to_csv(out_path, index=False)
    print(f"Wrote cross-validation summary to {out_path}")
    return result


def main():
    parser = argparse.ArgumentParser(description="Cross-validate the AR(3) stat models.")
    parser.add_argument("--input", default=DEFAULT_INPUT, help="Box score CSV (default: %(default)s)")
    parser.add_argument("--stats", nargs="+", default=ALL_STATS, help="Target columns to validate (default: all four)")
    parser.add_argument("--k", type=int, default=5, help="Number of K-fold splits (default: %(default)s)")
    parser.add_argument("--output-dir", default=OUTPUT_DIR, help="Output directory (default: %(default)s)")
    parser.add_argument("--draws", type=int, default=1000, help="Posterior draws per chain (default: %(default)s)")
    parser.add_argument("--tune", type=int, default=1000, help="Tuning steps per chain (default: %(default)s)")
    parser.add_argument("--chains", type=int, default=4, help="MCMC chains (default: %(default)s)")
    args = parser.parse_args()

    run_cross_validation(
        args.input, args.stats, args.k, args.output_dir,
        draws=args.draws, tune=args.tune, chains=args.chains,
    )


if __name__ == "__main__":
    main()
