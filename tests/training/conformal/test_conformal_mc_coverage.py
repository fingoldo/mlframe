"""Monte-Carlo empirical-coverage guarantees for the conformal quantile kernels.

These lock in the FINITE-SAMPLE correctness of the nonconformity-quantile rank
that every conformal estimator in the package rests on. Existing tests check
single-shot biz_value wins; here we average empirical coverage over many trials
and assert it meets the nominal ``1 - alpha`` at a SMALL calibration size, where
the ``ceil((n+1)(1-alpha))`` finite-sample correction actually matters.

An off-by-one in the rank (dropping the ``+1`` correction) drops coverage from
~0.90 to ~0.87 at ``n_cal=30`` -- below nominal -- so this test fails on that
regression. Validated against the buggy variant during authoring.
"""
from __future__ import annotations

import numpy as np

from mlframe.training.composite.conformal import (
    _signed_finite_sample_quantile,
    conformal_quantile,
    weighted_conformal_quantile,
)


def _abs_resid_coverage(qfn, *, n_cal: int, trials: int, alpha: float, weighted: bool) -> float:
    """Mean empirical coverage of a symmetric abs-residual band over MC trials."""
    rng = np.random.default_rng(0)
    covs = []
    for _ in range(trials):
        cal = rng.standard_normal(n_cal)
        q = qfn(cal, np.ones(n_cal), alpha) if weighted else qfn(cal, alpha)
        test = rng.standard_normal(2000)
        covs.append(float(np.mean(np.abs(test) <= q)))
    return float(np.mean(covs))


def test_conformal_quantile_mc_coverage_small_n() -> None:
    """abs-residual band hits nominal 1-alpha at small n (off-by-one drops it)."""
    cov = _abs_resid_coverage(conformal_quantile, n_cal=30, trials=2000, alpha=0.1, weighted=False)
    # Correct rank measured 0.904; off-by-one measured 0.871. Floor splits them.
    assert cov >= 0.895, f"conformal_quantile under-covered at n_cal=30: {cov:.4f}"
    assert cov <= 0.94, f"conformal_quantile grossly over-covered: {cov:.4f}"


def test_weighted_uniform_reduces_to_unweighted_coverage() -> None:
    """Uniform weights + the test-point self-weight atom reproduce the band."""
    cov = _abs_resid_coverage(weighted_conformal_quantile, n_cal=200, trials=400, alpha=0.1, weighted=True)
    assert cov >= 0.895, f"weighted(uniform) under-covered: {cov:.4f}"
    assert cov <= 0.92, f"weighted(uniform) over-covered: {cov:.4f}"


def test_weighted_conformal_restores_coverage_under_shift() -> None:
    """Tibshirani weighting restores 1-alpha under covariate shift where the
    unweighted band under-covers; the normalisation/self-weight must be right."""
    rng = np.random.default_rng(1)
    alpha, mu = 0.1, 1.0
    covs_w, covs_u = [], []
    for _ in range(400):
        xc = rng.standard_normal(300)
        rc = rng.standard_normal(300) * (0.5 + np.abs(xc))
        w = np.exp(mu * xc - mu * mu / 2)  # dP_test/dP_cal for N(mu,1) vs N(0,1)
        qw = weighted_conformal_quantile(rc, w, alpha)
        qu = conformal_quantile(rc, alpha)
        xt = rng.standard_normal(3000) + mu
        rt = rng.standard_normal(3000) * (0.5 + np.abs(xt))
        covs_w.append(float(np.mean(np.abs(rt) <= qw)))
        covs_u.append(float(np.mean(np.abs(rt) <= qu)))
    mean_w, mean_u = float(np.mean(covs_w)), float(np.mean(covs_u))
    assert mean_w >= 0.89, f"weighted under-covered under shift: {mean_w:.4f}"
    assert mean_u < mean_w - 0.03, (
        f"unweighted should under-cover under shift (got {mean_u:.4f} vs weighted {mean_w:.4f})"
    )


def test_signed_cqr_quantile_mc_coverage() -> None:
    """Signed (CQR-style) score quantile delivers nominal coverage on a symmetric
    base band -- the signed rank must not take abs()."""
    rng = np.random.default_rng(2)
    alpha, c = 0.1, 1.0
    covs = []
    for _ in range(400):
        y = rng.standard_normal(200)
        scores = np.maximum(-c - y, y - c)
        q = _signed_finite_sample_quantile(scores, alpha)
        yt = rng.standard_normal(2000)
        covs.append(float(np.mean((yt >= -c - q) & (yt <= c + q))))
    cov = float(np.mean(covs))
    assert cov >= 0.895, f"signed CQR quantile under-covered: {cov:.4f}"
    assert cov <= 0.93, f"signed CQR quantile over-covered: {cov:.4f}"
