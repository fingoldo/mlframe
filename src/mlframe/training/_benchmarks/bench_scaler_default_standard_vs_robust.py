"""Isolated bench: numeric-scaling default standard vs robust.

Lever: ``PreprocessingBackendConfig.scaler_name`` (default "standard").
Production path: the default flows through ``create_polarsds_pipeline`` ->
``_apply_safe_scaler`` (training/pipeline/__init__.py:703), the shared
preprocessing pipeline every scale-sensitive model consumes in the suite path.

Question: on feature matrices that carry heavy outliers / heavy tails (the
realistic raw-tabular case), does StandardScaler (center by mean, scale by
std -- BOTH statistics are dragged by the tail) or RobustScaler (center by
median, scale by IQR -- breakdown-resistant) give the honest-holdout winner
downstream for SCALE-SENSITIVE models (linear / SVM)? Tree models ignore any
monotone per-column scaling, so the scaler default only moves the needle for
linear/kernel/neural learners -- those are what this bench scores.

A StandardScaler column with a 5% multiplicative-spike contamination has its
std inflated by the spikes, so the informative bulk gets squashed into a tiny
range and the regulariser (L2) then over-penalises the real signal; RobustScaler's
IQR ignores the spikes, preserving the bulk's dynamic range.

Methodology: 5 synthetic scenarios x >=5 seeds. Each builds a train + an honest
holdout split, runs the REAL pipeline for each scaler, fits a downstream
scale-sensitive model (LogisticRegression / Ridge), and scores the honest
holdout. Majority-of-cells win decides; a flip needs a clear robust majority
WITHOUT a material regression on the clean (no-outlier) scenario.

VERDICT (qual-20, 2026-06-15): KEEP standard. Across 5 scenarios x 5 seeds the
robust scaler shows NO scenario-majority win and only ~1e-3-magnitude deltas
either way (robust 8 / standard 12 / tie 5 of 25). Root cause: once every column
is individually normalised, a single global L2 regulariser (LogisticRegression /
Ridge) absorbs the std-vs-IQR scale-family difference -- the honest-holdout AUC/R2
is near-invariant to the choice. The spike contamination sits in the data BOTH
scalers see; robust's IQR only marginally re-shapes the per-column dynamic range.
The clean Gaussian + lognormal scenarios actively (if barely) favour standard.
A flip needs a scenario-majority robust win WITHOUT a clean-data regression; this
bench produces neither, so ``scaler_name="standard"`` stays the most-accurate
default. Re-run this bench to re-test on different hardware / data.

Run: python -m mlframe.training._benchmarks.bench_scaler_default_standard_vs_robust
"""

from __future__ import annotations

import sys

# Block native cupy import under contention (segfaults on pip-CUDA hosts w/o CUDA_PATH).
sys.modules.setdefault("cupy", None)

import numpy as np
import polars as pl
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import roc_auc_score, r2_score

from mlframe.training._preprocessing_configs import PreprocessingBackendConfig
from mlframe.training.pipeline import create_polarsds_pipeline


def _make_scenario(name, rng, n=3000, p=8):
    """Return (X, y, task) with an outlier / heavy-tail structure where present."""
    if name == "spike_contam_binary":
        # Heterogeneous per-column scales (so per-column scaling MATTERS under one
        # global L2 regulariser) + 5% additive spikes that inflate std but not IQR.
        col_scale = rng.uniform(0.3, 8.0, p)
        X = rng.normal(0, 1, size=(n, p)) * col_scale
        beta = rng.normal(0, 1, p) / col_scale
        logit = X @ beta
        y = (logit + rng.normal(0, 1, n) > 0).astype(int)
        for j in range(0, p, 2):
            spike = rng.random(n) < 0.05
            X[spike, j] += rng.uniform(20, 80, spike.sum()) * col_scale[j] * rng.choice([-1, 1], spike.sum())
        task = "clf"
    elif name == "cauchy_tail_binary":
        # Heavy-tailed (Cauchy-ish) feature noise added to a Gaussian signal.
        Z = rng.normal(0, 1, size=(n, p))
        beta = rng.normal(0, 1, p)
        y = (Z @ beta + rng.normal(0, 1, n) > 0).astype(int)
        X = Z + rng.standard_cauchy(size=(n, p)) * 0.5
        task = "clf"
    elif name == "spike_contam_regr":
        col_scale = rng.uniform(0.3, 8.0, p)
        X = rng.normal(0, 1, size=(n, p)) * col_scale
        beta = rng.normal(0, 1, p) / col_scale
        y = X @ beta + rng.normal(0, 1.0, n)
        for j in range(0, p, 2):
            spike = rng.random(n) < 0.05
            X[spike, j] += rng.uniform(20, 80, spike.sum()) * col_scale[j] * rng.choice([-1, 1], spike.sum())
        task = "regr"
    elif name == "lognormal_regr":
        # Strongly right-skewed positive features (mean >> median).
        X = rng.lognormal(0.0, 1.3, size=(n, p))
        beta = rng.normal(0, 1, p)
        y = (np.log(X) - np.log(X).mean(0)) @ beta + rng.normal(0, 1.0, n)
        task = "regr"
    elif name == "clean_gaussian_binary":
        # NO outliers -- the regression-guard scenario: robust must not lose materially.
        X = rng.normal(0, 1, size=(n, p))
        beta = rng.normal(0, 1, p)
        y = (X @ beta + rng.normal(0, 1, n) > 0).astype(int)
        task = "clf"
    else:
        raise ValueError(name)
    return X.astype(np.float64), y, task


def _run_cell(name, scaler_name, seed):
    rng = np.random.default_rng(seed)
    X, y, task = _make_scenario(name, rng)
    n, p = X.shape
    cut = int(n * 0.7)
    cols = [f"f{j}" for j in range(p)]

    def _frame(rows):
        return pl.DataFrame({c: [float(X[i, j]) for i in rows] for j, c in enumerate(cols)})

    cfg = PreprocessingBackendConfig(
        imputer_strategy="median",
        scaler_name=scaler_name,
        categorical_encoding=None,
        prefer_polarsds=True,
    )
    tr_df, ho_df = _frame(range(cut)), _frame(range(cut, n))
    pipe = create_polarsds_pipeline(tr_df, cfg, verbose=0)
    Xtr = np.nan_to_num(pipe.transform(tr_df).to_numpy())
    Xho = np.nan_to_num(pipe.transform(ho_df).to_numpy())
    ytr, yho = y[:cut], y[cut:]

    if task == "clf":
        m = LogisticRegression(max_iter=1000, C=1.0)
        m.fit(Xtr, ytr)
        return roc_auc_score(yho, m.predict_proba(Xho)[:, 1])
    m = Ridge(alpha=1.0)
    m.fit(Xtr, ytr)
    return r2_score(yho, m.predict(Xho))


def main():
    scenarios = [
        "spike_contam_binary",
        "cauchy_tail_binary",
        "spike_contam_regr",
        "lognormal_regr",
        "clean_gaussian_binary",
    ]
    seeds = [11, 23, 47, 71, 97]
    robust_wins = standard_wins = ties = 0
    per_scenario: dict[str, list[int]] = {}
    print(f"{'scenario':<24}{'seed':>5}{'standard':>11}{'robust':>11}{'winner':>10}")
    for name in scenarios:
        r = s = 0
        for seed in seeds:
            v_std = _run_cell(name, "standard", seed)
            v_rob = _run_cell(name, "robust", seed)
            d = v_rob - v_std
            if abs(d) < 1e-4:
                w, ties = "tie", ties + 1
            elif d > 0:
                w, robust_wins, r = "robust", robust_wins + 1, r + 1
            else:
                w, standard_wins, s = "standard", standard_wins + 1, s + 1
            print(f"{name:<24}{seed:>5}{v_std:>11.4f}{v_rob:>11.4f}{w:>10}")
        per_scenario[name] = [r, s]
    total = robust_wins + standard_wins + ties
    print(f"\nrobust_wins={robust_wins}  standard_wins={standard_wins}  ties={ties}  (of {total})")
    print("per-scenario [robust, standard]:")
    for k, v in per_scenario.items():
        print(f"  {k:<24} {v}")
    verdict = "FLIP->robust" if robust_wins > standard_wins + ties else "KEEP standard"
    print(f"VERDICT: {verdict}")


if __name__ == "__main__":
    main()
