# Technical documentation: Bayesian AR(3) true shooting % model

This document covers the data pipeline, model specification, and diagnostics behind `project.ipynb`. It assumes familiarity with Bayesian inference, NUTS/HMC sampling, and basic NBA advanced stats (TS%, USG%). For a plain-language summary of what this actually tells you, see `docs/writeup.html`.

## 1. Data

Source: `nba_api`'s `LeagueDashPlayerStats` endpoint (Base + Advanced measure types), scraped per season and combined by `functions.py::scrape_nba_data`. Current window: 2021-22 through 2025-26 (five completed seasons; a sixth season passed to the scraper returns zero rows since it hasn't started).

Cleaning steps (`project.ipynb`, pre-EDA cells):

- **Trade dedup.** Players traded mid-season get one row per team plus a combined "TOT" row from the API. Only the TOT row is kept when present, so each player has exactly one row per season regardless of how many teams they were on.
- **Attempts floor.** `SEASON_FGA_EST = FGA (per game) * GP` approximates season-total attempts, since the scraper's box score is per-game. Rows below `MIN_SEASON_FGA = 50` estimated attempts are dropped before modeling — below that, FG%/TS% are dominated by small-sample noise (a player who goes 3-for-4 has a "75%" that means nothing).

Model input columns: `PLAYER_ID`, `PLAYER_NAME`, `TEAM_ABBREVIATION`, `SEASON_START_YEAR`, `AGE`, `TS_PCT`, `USG_PCT`, `SEASON_FGA_EST`.

## 2. Feature construction: flexible lags

Earlier versions of this model required a full three-consecutive-season window before a player-season could be used at all (an implicit AR(3) with listwise deletion). That excludes anyone in their first, second, or third season, and anyone with a gap year — which, on the real 2021-26 data, was a large share of the league. The current version keeps every player-season instead:

```python
for k in (1, 2, 3):
    ar_df[f"PREV_TS_PCT_{k}"] = ar_df.groupby("PLAYER_ID")["TS_PCT"].shift(k)
    season_lag = ar_df.groupby("PLAYER_ID")["SEASON_START_YEAR"].shift(k)
    ar_df[f"HAS_LAG_{k}"] = (ar_df["SEASON_START_YEAR"] - season_lag) == k

ar_df["N_LAGS_AVAILABLE"] = ar_df[["HAS_LAG_1", "HAS_LAG_2", "HAS_LAG_3"]].sum(axis=1)
```

`HAS_LAG_k` checks the season gap directly rather than trusting `shift(k)` alone — `shift(k)` returns whatever row happens to be `k` rows back in a player's own history, which is only a true "k seasons ago" value if there's no gap (injury, G-League, out of the league) in between. A player with seasons `[2021, 2023, 2024]` has a real lag-1 for 2024 (2023 is right there) but no real lag-1 for 2023 (2022 doesn't exist) — checking the year delta catches this; blindly using `shift(1)` would not.

At sampling time, missing lags are imputed to the population mean TS% in standardized space:

```python
def lagged_z(k):
    raw = ar_model_df[f"PREV_TS_PCT_{k}"].to_numpy()
    has = ar_model_df[f"HAS_LAG_{k}"].to_numpy()
    filled = np.where(has, raw, ts_mean)
    return (filled - ts_mean) / ts_sd
```

A missing lag becomes exactly `0` on the standardized scale, i.e. it contributes nothing to `mu` regardless of what `phi_k` is fitted to. `N_LAGS_AVAILABLE` (0–3, itself standardized) is included as its own covariate so the model can still learn whether early-career and established players differ systematically, separately from what the individual lags say.

On the real 2021–26 data this produced **2,286 usable player-seasons**, split:

| Lags available | Player-seasons |
|---|---|
| 0 (first qualifying season in window) | 817 |
| 1 | 555 |
| 2 | 404 |
| 3 (full history) | 510 |

For comparison, the strict "3 consecutive seasons" version of this model used 540 rows total — the flexible version uses everyone who cleared the attempts floor, a >4x increase in usable data, at the cost of most rows carrying less autoregressive signal than before.

## 3. Model 1: pooled Bayesian Gaussian AR(3)

```
TS%ᵢ ~ Normal(μᵢ, σ)
μᵢ = α + φ1·prev1ᵢ + φ2·prev2ᵢ + φ3·prev3ᵢ + β_age·ageᵢ + β_age²·ageᵢ² + β_usg·usgᵢ + β_lags·n_lagsᵢ
```

All of `prev1/2/3`, `age`, `usg`, `n_lags` are standardized (z-scored) before fitting; `prev1/2/3` share one scale (the target `y`'s mean/sd) since they're the same underlying quantity at different offsets, rather than each getting its own.

Priors: `alpha ~ Normal(y.mean(), 0.2)`, all `phi`/`beta` coefficients `~ Normal(0, 1)`, `sigma ~ HalfNormal(0.2)`. Sampled with `pm.sample(2000, tune=2000, chains=4, target_accept=0.9)`.

**Real-data result:**

| param | mean | 94% HDI |
|---|---|---|
| alpha | 0.5703 | — |
| phi1 | 0.0282 | [0.0252, 0.0316] |
| phi2 | 0.0059 | [0.0013, 0.0103] |
| phi3 | 0.0131 | [0.0076, 0.0188] |
| beta_age | 0.0047 | [0.0020, 0.0071] |
| beta_age_sq | -0.0050 | [-0.0066, -0.0032] |
| beta_usg | 0.0003 | [-0.0018, 0.0023] |
| beta_n_lags | 0.0042 | [0.0018, 0.0064] |
| sigma | 0.0523 | [0.0508, 0.0537] |

0 divergent transitions, max r-hat 1.0008, min ESS (bulk) 6,569. Bayesian R² = 0.389 (± 0.008) — lower than the strict-history version's ~0.6, which is the direct tradeoff of including ~36% of rows with zero autoregressive signal.

Notable: `phi2 < phi3` — the two-seasons-back lag has a *smaller* fitted weight than the three-seasons-back lag, both HDIs excluding zero. This is a real, small, non-monotonic pattern in the fitted persistence structure, not a coding artifact — see `docs/writeup.html` §"a wrinkle in the pattern" for the plain-language version and the notebook's own caveat about not over-reading small deviations from monotonic decay. `beta_usg`'s HDI straddles zero — once age and AR history are accounted for, usage rate isn't pulling additional weight. `beta_n_lags` is small but reliably positive: more career history behind a projection is associated with a *slightly* higher predicted TS%, independent of age, which is broadly consistent with survivorship (marginal players wash out of the league faster than efficient ones).

## 4. Model 2: hierarchical (partial-pooling) extension

Adds a non-centered player-level intercept on top of the same structure:

```
TS%ᵢ,ₜ ~ Normal(αᵢ + φ1·prev1 + φ2·prev2 + φ3·prev3 + β_age·age + β_age²·age² + β_usg·usg + β_lags·n_lags, σ)
αᵢ = μ_α + τ_α · offsetᵢ,   offsetᵢ ~ Normal(0, 1)
```

Non-centered parameterization (`alpha_offset ~ Normal(0,1)` scaled by `tau_alpha`, rather than `alpha_player ~ Normal(mu_alpha, tau_alpha)` directly) avoids Neal's-funnel divergences. `target_accept=0.98` (raised from the pooled model's 0.9, and higher than the AR(1) hierarchical version's 0.95) because with flexible lags many players contribute only 1–2 observations, which sharpens the funnel geometry.

**Real-data result (population-level parameters):**

| param | mean | 94% HDI |
|---|---|---|
| mu_alpha | 0.5696 | [0.5652, 0.5739] |
| tau_alpha | 0.0481 | [0.0445, 0.0518] |
| phi1 | 0.0007 | [-0.0024, 0.0039] |
| phi2 | -0.0086 | [-0.0123, -0.0050] |
| phi3 | 0.0005 | [-0.0040, 0.0047] |
| beta_age | 0.0083 | [0.0044, 0.0123] |
| beta_age_sq | -0.0075 | [-0.0097, -0.0052] |
| beta_usg | -0.0015 | [-0.0045, 0.0015] |
| beta_n_lags | 0.0024 | [0.0003, 0.0045] |
| sigma | 0.0380 | [0.0365, 0.0394] |

0 divergent transitions, max r-hat 1.0032, min ESS (bulk) 1,599 (`tau_alpha`) — lower than the pooled model's but still acceptable. `tau_alpha` (0.048) is an order of magnitude larger than what the same design showed on the pre-flexible-lag AR(1) data (≈0.004–0.011 across earlier runs) — with per-player intercepts now absorbing much of what the AR terms explained in the pooled model (`phi1`/`phi2`/`phi3` collapse toward zero here), individual player identity is doing more of the work once the model is allowed 780 free intercepts against 2,286 observations.

### 4.1 Model comparison — and why the LOO ranking here isn't trustworthy

```
              rank  elpd_loo   p_loo   elpd_diff  weight    se       dse    warning
hierarchical  0     3821.09    629.37  0.00        0.92     44.98    0.00   True
pooled        1     3495.91    9.13    325.18      0.08     44.27    31.31  False
```

`az.compare` ranks the hierarchical model first by `elpd_loo`, but its LOO estimate carries `warning=True`. Investigating: **131 of 2,286 observations (5.7%) have Pareto-k > 0.7**, with a max of 1.13, versus **0 flagged observations for the pooled model**. Of those 131, **113 have `N_LAGS_AVAILABLE == 0`** — a player's only qualifying season in the entire 2021–26 window (commonly a veteran near retirement who logged one qualifying season before or after this window's edge, e.g. Carmelo Anthony, Dwight Howard, Trevor Ariza in the actual flagged set).

This is the textbook failure mode for PSIS-LOO on hierarchical models: leaving out a player's *only* observation removes essentially all information the model has about that player's intercept, which is precisely the "highly influential observation" scenario where the importance-sampling approximation breaks down (see Vehtari, Gelman & Gabry 2017). The practical takeaway: **do not read the hierarchical model's LOO win at face value here.** The pooled model's comparison is the one to trust, and it doesn't show hierarchical pooling earning its added complexity. If this needs to be resolved rather than just flagged, the options are (a) k-fold CV instead of PSIS-LOO, (b) refitting moment-matching for the flagged observations, or (c) restricting per-player intercepts to players with some minimum observation count and falling back to the population-level distribution otherwise — none of which are implemented here; this iteration documents the issue rather than engineering around it.

## 5. Next-season projection

For each player active in the most recent scraped season (`SEASON_START_YEAR == max`), the *next* season is projected using:

- `prev1` = that player's most recent season's own TS% (always available for anyone in this set)
- `prev2`/`prev3` = the 1- and 2-seasons-back TS%, each independently checked against the season gap the same way as training (so a gap year correctly drops that lag rather than pulling in a stale value)
- `age` = current age + 1
- `usg` = held at the most recent observed value (next season's actual role is unknown — this is a steady-state assumption, not a usage forecast)

`project_ts_pct_hier` draws from a player's own fitted `alpha_player` when they're in the training set; for anyone else it draws from `Normal(mu_alpha, tau_alpha)` per posterior sample, i.e. the population-level intercept distribution rather than reusing another player's fitted value.

Real-data result: **475 of 780 active players** (61%) get a projection; the rest lack even a single prior TS% observation in-window at their most recent-season row (this floor is inherent — an AR model needs at least one observed lag to condition on; a true zero-history rookie can't be projected this way regardless of flexible lags, since `prev1` requires the row itself to already have a TS% on record).

## 6. Extending to assists, rebounds, and defensive efficiency

Sections 2–3 above (flexible lags, pooled Gaussian AR(3)) are specific to TS% only insofar as `project.ipynb` hardcodes them for it. `stat_models.py` factors that architecture out into two reusable pieces:

- `build_ar_frame(model_source, target_cols, covariate_cols=("USG_PCT",), min_minutes=500)` — builds the flexible-lag AR(3) feature frame (Section 2) for one or more target columns at once, filtered on `SEASON_MIN_EST = MIN * GP >= 500` rather than the TS% model's attempts floor. A shot-attempts floor doesn't make sense for rebounding or defense (a rim-running big can rebound plenty without shooting much), so this uses a playing-time floor instead, applied uniformly so every stat is fit on the *same* player-season universe.
- `fit_ar_stat_model(ar_df, target_col, ...)` / `project_ar_stat(...)` / `project_next_season(...)` — the pooled model (Section 3) and its projection step, generalized to any target column. Priors scale off the target's own standard deviation (`alpha ~ Normal(y_mean, 3*y_sd)`, `sigma ~ HalfNormal(y_sd)`) instead of the TS%-specific fixed constants (`0.2`), since `DEF_RATING` lives on a ~100–120 scale while the percentage stats live on ~0–1.

`build_projections.py` fits this pooled model for four target columns — `TS_PCT`, `AST_PCT` (assist rate), `REB_PCT` (rebound rate), and `DEF_RATING` (points allowed per 100 possessions, where *lower* is better) — on the shared player-season universe, projects next season for every active player on each, and writes one combined CSV. Note this refits TS% with the simpler pooled model rather than reusing the hierarchical fit from Section 4; the hierarchical extension remains the deeper, TS%-specific analysis, while this pipeline optimizes for a consistent model across four stats over squeezing out the last bit of TS%-specific accuracy.

For each stat, `DELTA_<stat> = PROJECTED − CURRENT` (sign-flipped for `DEF_RATING`, so positive always means "expected to get better"), then z-scored across all projected players. `BREAKOUT_SCORE` is the mean of the four z-scores — a composite "expected to improve the most, across the board" ranking, comparable across stats despite their different units and variances.

On the real 2021–26 data (`SEASON_MIN_EST >= 500`): 1,856 player-seasons, 639 players, 379 of whom are active in the most recent season and get a next-season projection on all four stats. Each pooled model samples in ~10–12 seconds (`draws=1000, tune=1000, chains=4, target_accept=0.9`) — the full four-stat pipeline runs in under two minutes.

## 7. Dashboard

`bayesian_ar_output/next_season_projections_all.csv` (`PLAYER_ID`, `PLAYER_NAME`, `TEAM`, `PROJECTED_AGE`, then `CURRENT_<stat>`/`PROJECTED_<stat>`/`<stat>_HDI_LOW`/`<stat>_HDI_HIGH`/`DELTA_<stat>`/`Z_<stat>` for each of the four stats, plus `BREAKOUT_SCORE`) is the interchange format between `build_projections.py` and the dashboard. `dashboard.py` is a standalone module — it only depends on `pandas`, not `pymc`/`arviz` — so viewing or rebuilding the dashboard doesn't require re-running any model:

```
python dashboard.py              # rebuild bayesian_ar_output/dashboard.html from the CSV and open it
python dashboard.py --no-open     # rebuild only
python dashboard.py --csv path/to/other.csv --output path/to/other.html
```

The HTML is a single self-contained file (inline CSS/JS, no external requests), sorted by `BREAKOUT_SCORE` descending by default — who's projected to improve the most across shooting, playmaking, rebounding, and defense together. Every column header is clickable to re-sort by a specific stat instead (sorting is stable, so re-sorting by Team groups players together while preserving their relative order within each team). The search box filters by player name or team substring; the FGA/game slider is there to separate a real small-sample-size breakout candidate from one that's mostly regression-to-the-mean noise on very little playing time.

## 8. Reproducing this

```
pip install -r requirements.txt   # or use .devcontainer
jupyter nbconvert --to notebook --execute --inplace project.ipynb
python build_projections.py       # if you only need to refresh the 4-stat CSV, not the full notebook
python dashboard.py
```

Sampling the TS% models takes roughly 20 seconds (pooled) + 3 minutes (hierarchical) on a single machine with 4 chains; each of the four `build_projections.py` models adds another ~10–12 seconds, and the projection loops only evaluate already-drawn posterior samples, not new MCMC, so they add negligible time.
