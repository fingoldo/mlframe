"""Contract / unit tests for ``CompositeDistributionEstimator``.

CRPS finiteness + lower-is-better (perfect vs noisy predictor), monotone CDF,
sampling shape, non-crossing of the underlying dense quantile matrix, and the
unfitted / bad-arg error paths. Uses a deterministic pinball stub so the suite is
fast and does not require LightGBM for the contract checks.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.exceptions import NotFittedError

from mlframe.training.composite.distributional import (
    CompositeDistributionEstimator,
    _DEFAULT_DENSE_QUANTILES,
)


class _StubQuantileInner(BaseEstimator, RegressorMixin):
    """Pinball stub: predicts ``mean(T) + sigma * z(alpha)`` (ascending in alpha).

    Emits an alpha-monotone offset around the train mean of ``T`` so the heads
    form a valid ascending quantile fan -- mimics a real pinball learner with no
    training cost. ``sigma`` controls the width of the implied distribution.
    """

    def __init__(self, alpha: float = 0.5, sigma: float = 1.0) -> None:
        self.alpha = alpha
        self.sigma = sigma

    def fit(self, X, y, sample_weight=None):
        self._mean_ = float(np.mean(np.asarray(y, dtype=np.float64)))
        self.n_features_in_ = X.shape[1] if hasattr(X, "shape") else len(X.columns)
        return self

    def predict(self, X):
        from scipy.stats import norm

        n = X.shape[0] if hasattr(X, "shape") else len(X)
        offset = self.sigma * float(norm.ppf(float(self.alpha)))
        return np.full(n, self._mean_ + offset, dtype=np.float64)


def _make_xy(n: int = 300, seed: int = 0):
    rng = np.random.default_rng(seed)
    base = rng.normal(0.0, 1.0, n)
    x1 = rng.normal(0.0, 1.0, n)
    y = 2.0 * base + 0.5 * x1 + rng.normal(0.0, 1.0, n)
    return pd.DataFrame({"base": base, "x1": x1}), y


def _fit(quantiles=None, sigma=1.0):
    X, y = _make_xy()
    est = CompositeDistributionEstimator(
        base_estimator=_StubQuantileInner(sigma=sigma),
        transform_name="linear_residual",
        base_column="base",
        quantiles=quantiles,
    )
    est.fit(X, y)
    return est, X, y


# ----------------------------------------------------------------------
# Fit / config
# ----------------------------------------------------------------------
def test_default_dense_grid_is_19_levels():
    assert len(_DEFAULT_DENSE_QUANTILES) == 19
    assert _DEFAULT_DENSE_QUANTILES[0] == pytest.approx(0.05)
    assert _DEFAULT_DENSE_QUANTILES[-1] == pytest.approx(0.95)


def test_fit_uses_dense_grid_by_default():
    est, _, _ = _fit()
    assert est.quantiles_.shape[0] == 19


def test_fit_requires_base_estimator():
    with pytest.raises(ValueError, match="base_estimator must not be None"):
        CompositeDistributionEstimator(base_estimator=None, base_column="base").fit(*_make_xy())


# ----------------------------------------------------------------------
# Non-crossing
# ----------------------------------------------------------------------
def test_underlying_quantile_matrix_non_crossing():
    est, X, _ = _fit()
    qmat = est.predict_quantile(X)
    diffs = np.diff(qmat, axis=1)
    assert np.all(diffs >= -1e-9), "quantiles must be non-crossing per row"


# ----------------------------------------------------------------------
# CDF monotone
# ----------------------------------------------------------------------
def test_predict_cdf_shape_and_range():
    est, X, _ = _fit()
    grid = np.linspace(-6.0, 6.0, 25)
    cdf = est.predict_cdf(X, grid)
    assert cdf.shape == (X.shape[0], grid.shape[0])
    assert np.all(cdf >= 0.0) and np.all(cdf <= 1.0)


def test_predict_cdf_monotone_non_decreasing():
    est, X, _ = _fit()
    grid = np.linspace(-6.0, 6.0, 40)
    cdf = est.predict_cdf(X, grid)
    diffs = np.diff(cdf, axis=1)
    assert np.all(diffs >= -1e-12), "CDF must be non-decreasing along y_grid"


# ----------------------------------------------------------------------
# Sampling
# ----------------------------------------------------------------------
def test_sample_shape():
    est, X, _ = _fit()
    draws = est.sample(X, n=50, random_state=0)
    assert draws.shape == (X.shape[0], 50)
    assert np.all(np.isfinite(draws))


def test_sample_within_quantile_envelope():
    est, X, _ = _fit()
    qmat = est.predict_quantile(X)
    lo, hi = qmat.min(axis=1), qmat.max(axis=1)
    draws = est.sample(X, n=100, random_state=1)
    # interp clamps to the [lowest, highest] fitted quantile per row.
    assert np.all(draws >= lo[:, None] - 1e-9)
    assert np.all(draws <= hi[:, None] + 1e-9)


def test_sample_matches_per_row_np_interp_reference():
    # The njit shared-x interpolation kernel must reproduce the per-row
    # np.interp PIT inversion it replaced, to FP round-off (~1e-15). Rebuild
    # the exact reference from the same seeded uniforms.
    est, X, _ = _fit()
    qmat = est.predict_quantile(X)
    levels = est.quantiles_
    n = 40
    rng = np.random.default_rng(7)
    u = rng.random((qmat.shape[0], n))
    ref = np.empty_like(u)
    for i in range(qmat.shape[0]):
        ref[i, :] = np.interp(u[i, :], levels, qmat[i, :])
    from mlframe.training.composite.distributional import _interp_rows_shared_x

    got = _interp_rows_shared_x(
        u,
        np.ascontiguousarray(levels, dtype=np.float64),
        np.ascontiguousarray(qmat, dtype=np.float64),
        np.empty_like(u),
    )
    assert np.max(np.abs(got - ref)) < 1e-12


def test_sample_rejects_non_positive_n():
    est, X, _ = _fit()
    with pytest.raises(ValueError, match="n must be > 0"):
        est.sample(X, n=0)


# ----------------------------------------------------------------------
# CRPS
# ----------------------------------------------------------------------
def test_crps_finite_and_scalar():
    est, X, y = _fit()
    val = est.crps(X, y)
    assert np.isfinite(val) and val >= 0.0


def test_crps_per_row_shape():
    est, X, y = _fit()
    per_row = est.crps(X, y, reduce="none")
    assert per_row.shape == (X.shape[0],)
    assert np.all(np.isfinite(per_row))


def test_crps_rejects_length_mismatch():
    est, X, y = _fit()
    with pytest.raises(ValueError, match="!= n_samples"):
        est.crps(X, y[:-3])


def test_crps_rejects_bad_reduce():
    est, X, y = _fit()
    with pytest.raises(ValueError, match="reduce must be"):
        est.crps(X, y, reduce="sum")


def test_crps_lower_for_perfect_than_noisy_predictor():
    """A predictor whose distribution is centred on y (perfect) scores a lower
    mean CRPS than one offset away from y (noisy). CRPS is strictly proper, so
    miscentring strictly increases the score."""
    X, y = _make_xy()
    good = CompositeDistributionEstimator(
        base_estimator=_StubQuantileInner(sigma=1.0),
        base_column="base",
    ).fit(X, y)
    crps_good = good.crps(X, y)
    # Noisy: shift the targets so the predicted distribution is systematically
    # off-centre relative to the evaluation targets.
    crps_bad = good.crps(X, y + 3.0)
    assert crps_bad > crps_good * 1.5, (crps_good, crps_bad)


# ----------------------------------------------------------------------
# Unfitted guard
# ----------------------------------------------------------------------
def test_unfitted_methods_raise():
    est = CompositeDistributionEstimator(
        base_estimator=_StubQuantileInner(),
        base_column="base",
    )
    X, _ = _make_xy()
    with pytest.raises(NotFittedError):
        est.predict_quantile(X)
    with pytest.raises(NotFittedError):
        est.predict_cdf(X, [0.0])
    with pytest.raises(NotFittedError):
        est.sample(X, n=5)
    with pytest.raises(NotFittedError):
        est.crps(X, np.zeros(X.shape[0]))
