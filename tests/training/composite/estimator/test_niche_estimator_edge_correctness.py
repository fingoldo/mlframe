"""Edge-correctness regression tests for niche composite estimators.

Covers two bugs found in the niche-estimator class sweep:

1. ``CompositeSimplexEstimator`` on a degenerate (all-zero) composition row:
   ``_close`` divided by a zero sum -> NaN log-ratio coordinates that silently
   corrupted the inner regressor's target. Now the all-zero row maps to the
   uniform composition so the log-ratio map stays finite.
2. ``ranking._rank01`` claimed average (tie-shared) ranks but assigned distinct
   positional ranks, fabricating a within-group order on tied relevance values.
"""

import numpy as np
from sklearn.linear_model import LinearRegression

from mlframe.training.composite.simplex import (
    CompositeSimplexEstimator,
    _close,
    multiplicative_zero_replacement,
)
from mlframe.training.composite.ranking import _rank01


def test_close_all_zero_row_maps_to_uniform_not_nan():
    """An all-zero (degenerate) composition row must close to uniform, not NaN."""
    y = np.array([[0.0, 0.0, 0.0], [0.3, 0.3, 0.4]])
    out = _close(y)
    assert np.isfinite(out).all(), "all-zero row produced non-finite closure (pre-fix bug)"
    np.testing.assert_allclose(out[0], [1 / 3, 1 / 3, 1 / 3])
    np.testing.assert_allclose(out[1].sum(), 1.0)


def test_simplex_fit_predict_finite_with_degenerate_row():
    """Fit/predict must stay finite when a training composition row is all-zero."""
    rng = np.random.default_rng(0)
    X = rng.normal(size=(30, 3))
    y = rng.random((30, 3))
    y /= y.sum(1, keepdims=True)
    y[0] = 0.0  # degenerate simplex row
    m = CompositeSimplexEstimator(LinearRegression(), transform="ilr")
    m.fit(X, y)  # pre-fix: raised "Input y contains NaN" / fit on NaN target
    p = m.predict(X)
    assert np.isfinite(p).all()
    np.testing.assert_allclose(p.sum(axis=1), 1.0, atol=1e-9)
    assert (p > 0).all()


def test_zero_replacement_finite_after_uniform_close():
    """End-to-end: close+zero-replace of an all-zero row yields a finite log input."""
    y = np.array([[0.0, 0.0, 0.0]])
    rep = multiplicative_zero_replacement(_close(y))
    assert np.isfinite(np.log(rep)).all()


def test_rank01_ties_shared_average():
    """Tied values must share the mean rank (zero ordering residual), not distinct ranks."""
    a = np.array([5.0, 5.0, 5.0, 1.0])
    r = _rank01(a)
    # The three tied 5.0 share rank mean(1,2,3)=2.0; the lone 1.0 gets rank 0.
    np.testing.assert_allclose(r, [2.0, 2.0, 2.0, 0.0])


def test_rank01_no_ties_unchanged():
    """With distinct values the average rank equals the positional rank."""
    a = np.array([3.0, 1.0, 2.0])
    np.testing.assert_allclose(_rank01(a), [2.0, 0.0, 1.0])


def test_rank01_equal_inputs_give_zero_residual():
    """rank(y)-rank(base) is zero when y and base induce the same (tied) order."""
    y = np.array([7.0, 7.0, 7.0])
    base = np.array([2.0, 2.0, 2.0])
    np.testing.assert_allclose(_rank01(y) - _rank01(base), 0.0)
