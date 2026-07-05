"""Isolated bench: numeric-imputation default mean vs median.

Lever: ``PreprocessingBackendConfig.imputer_strategy`` (default "mean").
Production path: the default flows through ``create_polarsds_pipeline`` ->
``Blueprint.impute(method=...)`` (training/pipeline/__init__.py:653), the
shared preprocessing pipeline every model consumes in the suite path.

Question: on skewed / heavy-tailed numeric features with missing values
(the realistic tabular case), does mean- or median-imputation give the
honest holdout winner downstream? Median is the robust location estimate;
mean is pulled by the long tail that NaN-bearing real columns usually have.

Methodology: 5 synthetic scenarios x 3 seeds. Each builds a train + an
honest holdout split, drops NaN into skewed numeric columns (MCAR + a
mild MNAR pattern), runs the REAL pipeline for each strategy, fits a
downstream model, and scores the honest holdout. Majority-of-cells win
decides; a flip needs a clear median majority.

Run: python -m mlframe.training._benchmarks.bench_imputer_strategy_default
"""

from __future__ import annotations

import numpy as np
import polars as pl
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import roc_auc_score, r2_score

from mlframe.training._preprocessing_configs import PreprocessingBackendConfig
from mlframe.training.pipeline import create_polarsds_pipeline


def _missing_mask(rng, X, frac_mcar=0.12):
    """MCAR holes + a mild MNAR pattern (high-value cells more likely missing)."""
    n, p = X.shape
    mask = rng.random((n, p)) < frac_mcar
    # MNAR: top-decile cells of each column get extra missingness.
    for j in range(p):
        hi = X[:, j] > np.quantile(X[:, j], 0.9)
        mask[:, j] |= hi & (rng.random(n) < 0.4)
    return mask


def _make_scenario(name, rng, n=3000, p=8):
    """Return (X, y, task) with skewed numeric features + signal."""
    if name == "lognormal_binary":
        X = rng.lognormal(0.0, 1.0, size=(n, p))
        beta = rng.normal(0, 1, p)
        logit = (np.log1p(X) - np.log1p(X).mean(0)) @ beta
        y = (logit + rng.normal(0, 1, n) > 0).astype(int)
        task = "clf"
    elif name == "pareto_binary":
        X = rng.pareto(2.5, size=(n, p)) + 1.0
        beta = rng.normal(0, 1, p)
        logit = (np.log(X) - np.log(X).mean(0)) @ beta
        y = (logit + rng.normal(0, 1.2, n) > 0).astype(int)
        task = "clf"
    elif name == "exp_skew_regr":
        X = rng.exponential(2.0, size=(n, p))
        beta = rng.normal(0, 1, p)
        y = X @ beta + rng.normal(0, 2.0, n)
        task = "regr"
    elif name == "mixed_outlier_binary":
        X = rng.normal(0, 1, size=(n, p))
        # inject heavy multiplicative outliers into half the columns
        for j in range(0, p, 2):
            spike = rng.random(n) < 0.05
            X[spike, j] *= rng.uniform(20, 60, spike.sum())
        beta = rng.normal(0, 1, p)
        y = (X @ beta + rng.normal(0, 1, n) > 0).astype(int)
        task = "clf"
    elif name == "gamma_regr":
        X = rng.gamma(1.5, 2.0, size=(n, p))
        beta = rng.normal(0, 1, p)
        y = np.log1p(X) @ beta + rng.normal(0, 1.0, n)
        task = "regr"
    else:
        raise ValueError(name)
    return X.astype(np.float64), y, task


def _run_cell(name, strategy, seed):
    rng = np.random.default_rng(seed)
    X, y, task = _make_scenario(name, rng)
    n, p = X.shape
    miss = _missing_mask(rng, X)
    cut = int(n * 0.7)
    cols = [f"f{j}" for j in range(p)]

    def _frame(rows):
        # Build with polars NULL (not float NaN): Blueprint.impute fills nulls.
        return pl.DataFrame({c: [None if miss[i, j] else float(X[i, j]) for i in rows] for j, c in enumerate(cols)})

    cfg = PreprocessingBackendConfig(
        imputer_strategy=strategy,
        scaler_name="standard",
        categorical_encoding=None,
        prefer_polarsds=True,
    )
    tr_df, ho_df = _frame(range(cut)), _frame(range(cut, n))
    pipe = create_polarsds_pipeline(tr_df, cfg, verbose=0)
    Xtr = pipe.transform(tr_df).to_numpy()
    Xho = pipe.transform(ho_df).to_numpy()
    # Guard against any residual non-finite (e.g. all-null column).
    Xtr = np.nan_to_num(Xtr)
    Xho = np.nan_to_num(Xho)
    ytr, yho = y[:cut], y[cut:]

    if task == "clf":
        m = LogisticRegression(max_iter=500, C=1.0)
        m.fit(Xtr, ytr)
        return roc_auc_score(yho, m.predict_proba(Xho)[:, 1])
    m = Ridge(alpha=1.0)
    m.fit(Xtr, ytr)
    return r2_score(yho, m.predict(Xho))


def main():
    scenarios = [
        "lognormal_binary",
        "pareto_binary",
        "exp_skew_regr",
        "mixed_outlier_binary",
        "gamma_regr",
    ]
    seeds = [11, 23, 47]
    median_wins = mean_wins = ties = 0
    print(f"{'scenario':<22}{'seed':>5}{'mean':>10}{'median':>10}{'winner':>9}")
    for name in scenarios:
        for s in seeds:
            v_mean = _run_cell(name, "mean", s)
            v_median = _run_cell(name, "median", s)
            d = v_median - v_mean
            if abs(d) < 1e-4:
                w, ties = "tie", ties + 1
            elif d > 0:
                w, median_wins = "median", median_wins + 1
            else:
                w, mean_wins = "mean", mean_wins + 1
            print(f"{name:<22}{s:>5}{v_mean:>10.4f}{v_median:>10.4f}{w:>9}")
    total = median_wins + mean_wins + ties
    print(f"\nmedian_wins={median_wins}  mean_wins={mean_wins}  ties={ties}  (of {total})")
    verdict = "FLIP->median" if median_wins > mean_wins + ties else "KEEP mean"
    print(f"VERDICT: {verdict}")


if __name__ == "__main__":
    main()
