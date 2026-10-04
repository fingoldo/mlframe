"""rank_ecdf / gaussian_copula residual fits must honour sample_weight instead of silently ignoring it."""

from __future__ import annotations

import numpy as np

from mlframe.training.composite.transforms._gaussian_copula import _gaussian_copula_residual_fit
from mlframe.training.composite.transforms._rank_ecdf import _ecdf_knots, _rank_ecdf_residual_fit


def _two_cluster(n: int = 400):
    """Half the rows near 0 (weight 1) and half near 10 (weight 99): the weighted median must sit in the heavy cluster."""
    rng = np.random.default_rng(3)
    x = np.concatenate([rng.normal(0.0, 0.1, n), rng.normal(10.0, 0.1, n)])
    w = np.concatenate([np.ones(n), np.full(n, 99.0)])
    return x, w


def test_ecdf_knots_weighted_cdf_follows_the_weights() -> None:
    """With 99% of the mass on the upper cluster the ECDF at the lower cluster's top is ~0.01, not 0.5."""
    x, w = _two_cluster()
    knots, u = _ecdf_knots(x, w)
    assert np.interp(5.0, knots, u) < 0.05
    knots_u, u_u = _ecdf_knots(x)
    assert abs(np.interp(5.0, knots_u, u_u) - 0.5) < 0.01


def test_ecdf_knots_without_weights_is_unchanged() -> None:
    """All-ones weights reproduce the unweighted plotting-position CDF exactly."""
    x, _ = _two_cluster()
    k0, u0 = _ecdf_knots(x)
    k1, u1 = _ecdf_knots(x, np.ones_like(x))
    assert np.array_equal(k0, k1)
    assert np.allclose(u0, u1, rtol=0, atol=1e-12)


def test_ecdf_knots_drop_zero_weight_rows() -> None:
    """Rows with zero weight carry no mass: the knots equal those of the subset with positive weight."""
    x = np.array([1.0, 2.0, 3.0, 4.0, 100.0])
    w = np.array([1.0, 1.0, 1.0, 1.0, 0.0])
    k, u = _ecdf_knots(x, w)
    k_ref, u_ref = _ecdf_knots(x[:4])
    assert np.array_equal(k, k_ref) and np.allclose(u, u_ref)


def test_rank_ecdf_fit_uses_sample_weight() -> None:
    """The stored y ECDF changes when sample_weight shifts the mass between the two clusters."""
    x, w = _two_cluster()
    plain = _rank_ecdf_residual_fit(x, x)
    weighted = _rank_ecdf_residual_fit(x, x, sample_weight=w)
    assert np.interp(5.0, weighted["y_knots"], weighted["y_cdf"]) < 0.05 < np.interp(5.0, plain["y_knots"], plain["y_cdf"])


def test_gaussian_copula_fit_uses_sample_weight() -> None:
    """Weighting concentrates the normal-scores regression on the heavy rows: the slope moves versus the unweighted fit."""
    rng = np.random.default_rng(5)
    n = 600
    base = rng.normal(size=n)
    y = np.where(base > 0, base, 0.1 * base) + rng.normal(scale=0.2, size=n)
    w = np.where(base > 0, 50.0, 1.0)
    plain = _gaussian_copula_residual_fit(y, base)
    weighted = _gaussian_copula_residual_fit(y, base, sample_weight=w)
    assert weighted["y_cdf"][len(weighted["y_cdf"]) // 4] != plain["y_cdf"][len(plain["y_cdf"]) // 4]
    assert abs(weighted["alpha"] - plain["alpha"]) > 0.05
