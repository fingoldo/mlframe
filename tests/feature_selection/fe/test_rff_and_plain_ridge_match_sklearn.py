"""``PlainRidge`` and ``RffSampler`` give what scikit-learn's ``Ridge`` and ``RBFSampler`` give, so the gradient-interaction seeder keeps its numbers."""

from __future__ import annotations

import numpy as np
import pytest

from mlframe.feature_selection.filters._linear_probes import PlainRidge, RffSampler


def _reg(n, p, seed):
    """Columns of different scales, one constant, and a linear target with noise."""
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, p)) * rng.uniform(0.5, 4.0, size=p) + rng.normal(size=p)
    X[:, -1] = 3.0
    return X, X[:, 0] - 0.5 * X[:, 1] + 0.3 * rng.standard_normal(n)


@pytest.mark.parametrize("alpha", [0.05, 1.0, 40.0])
def test_plain_ridge_matches_sklearn_ridge(alpha):
    """Coefficients, intercept and predictions equal ``Ridge(alpha)``."""
    from sklearn.linear_model import Ridge

    X, y = _reg(700, 9, 8)
    ref = Ridge(alpha=alpha).fit(X, y)
    got = PlainRidge(alpha).fit(X, y)
    np.testing.assert_allclose(got.coef_, ref.coef_, rtol=1e-7, atol=1e-9)
    assert got.intercept_ == pytest.approx(float(ref.intercept_), rel=1e-7, abs=1e-9)
    np.testing.assert_allclose(got.predict(X[:50]), ref.predict(X[:50]), rtol=1e-7, atol=1e-7)


@pytest.mark.parametrize("dtype", [np.float64, np.float32])
def test_rff_sampler_equals_sklearn_rbf_sampler(dtype):
    """Same draws and same map as ``RBFSampler`` for the same seed, in both dtypes."""
    from sklearn.kernel_approximation import RBFSampler

    X = np.random.default_rng(0).standard_normal((200, 6)).astype(dtype)
    ref = RBFSampler(gamma=0.7, n_components=64, random_state=5).fit(X)
    got = RffSampler(gamma=0.7, n_components=64, random_state=5).fit(X)
    np.testing.assert_array_equal(got.random_weights_, ref.random_weights_)
    np.testing.assert_array_equal(got.random_offset_, ref.random_offset_)
    tol = 1e-6 if dtype == np.float32 else 1e-12
    np.testing.assert_allclose(got.transform(X), ref.transform(X), rtol=tol, atol=tol)


def test_the_seeder_surrogate_equals_the_sklearn_one():
    """``_fit_rff_ridge`` / ``_oof_r2`` of the seeder agree with the sklearn pipeline they replaced on a smooth target."""
    from sklearn.kernel_approximation import RBFSampler
    from sklearn.linear_model import Ridge
    from sklearn.model_selection import KFold, cross_val_predict

    from mlframe.feature_selection.filters import _gradient_interaction_seeder as seeder
    from mlframe.metrics.core import fast_r2_score

    rng = np.random.default_rng(4)
    Xs = rng.standard_normal((500, 5))
    y = np.sin(Xs[:, 0]) + Xs[:, 1] * Xs[:, 2] + 0.1 * rng.standard_normal(500)
    rbf, ridge = seeder._fit_rff_ridge(Xs, y, 80, 0.5, 1.0, 3)
    ref_rbf = RBFSampler(gamma=0.5, n_components=80, random_state=3)
    ref_ridge = Ridge(alpha=1.0).fit(ref_rbf.fit_transform(Xs), y)
    np.testing.assert_allclose(ridge.predict(rbf.transform(Xs)), ref_ridge.predict(ref_rbf.transform(Xs)), rtol=1e-6, atol=1e-6)
    ref_pred = cross_val_predict(Ridge(alpha=1.0), ref_rbf.transform(Xs), y, cv=KFold(3, shuffle=True, random_state=3))
    assert seeder._oof_r2(Xs, y, 80, 0.5, 1.0, 3) == pytest.approx(float(fast_r2_score(y, ref_pred)), abs=1e-6)
