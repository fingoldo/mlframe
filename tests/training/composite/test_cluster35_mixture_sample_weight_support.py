"""Audit 2026-11-20 finding 35-F29: the gated mixture decides sample_weight support from the signature, not from a caught TypeError.

Before the fix any TypeError from fit made the branch refit without weights, so an unrelated bug silently dropped the weights.
"""

import logging

import numpy as np
import pytest
from sklearn.base import BaseEstimator, RegressorMixin

from sklearn.linear_model import LogisticRegression

from mlframe.training.composite.gated_regression_mixture import GatedRegressionMixture


def _labels(X):
    """Subpopulation labels: the low/high split on the first feature."""
    return (X[:, 0] > 0).astype(int)


def _mixture(regressor):
    """A mixture with a logistic gate and the same regressor on both branches."""
    return GatedRegressionMixture(
        gate_classifier=LogisticRegression(), low_regressor=regressor, high_regressor=regressor, n_splits=2,
    )


class _Weighted(BaseEstimator, RegressorMixin):
    """A regressor that accepts sample_weight and records whether it got weights."""

    def fit(self, X, y, sample_weight=None):
        """Record whether weights were passed."""
        self.got_weights_ = sample_weight is not None
        self.mean_ = float(np.mean(y))
        return self

    def predict(self, X):
        """Predict the training mean."""
        return np.full(len(X), self.mean_)


class _Unweighted(BaseEstimator, RegressorMixin):
    """A regressor whose fit has no sample_weight parameter."""

    def fit(self, X, y):
        """Fit the mean."""
        self.mean_ = float(np.mean(y))
        return self

    def predict(self, X):
        """Predict the training mean."""
        return np.full(len(X), self.mean_)


class _BrokenFit(BaseEstimator, RegressorMixin):
    """A regressor whose fit raises an unrelated TypeError."""

    def fit(self, X, y, sample_weight=None):
        """Raise a TypeError unrelated to weights."""
        raise TypeError("unrelated bug")

    def predict(self, X):
        """Unused."""
        return np.zeros(len(X))


def _data(n=120, seed=0):
    """Features, a target, and a gate probability per row."""
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, 2))
    y = X[:, 0] + 0.1 * rng.normal(size=n)
    return X, y


def test_weighted_regressor_receives_weights():
    """A regressor that takes sample_weight is fitted with the branch weights."""
    X, y = _data()
    m = _mixture(_Weighted())
    m.fit(X, y, _labels(X))
    assert all(bm.got_weights_ for bm in m.branch_models_.values())


def test_unweighted_regressor_is_fitted_with_a_warning(caplog):
    """A regressor without sample_weight is fitted unweighted and the fact is logged at WARNING."""
    X, y = _data()
    m = _mixture(_Unweighted())
    with caplog.at_level(logging.WARNING):
        m.fit(X, y, _labels(X))
    assert m.branch_models_
    assert any("does not accept sample_weight" in r.getMessage() for r in caplog.records)


def test_unrelated_type_error_propagates():
    """A TypeError from fit is raised, not swallowed into an unweighted refit."""
    X, y = _data()
    m = _mixture(_BrokenFit())
    with pytest.raises(TypeError, match="unrelated bug"):
        m.fit(X, y, _labels(X))
