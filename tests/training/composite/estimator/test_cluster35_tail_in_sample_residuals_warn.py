"""Audit 2026-11-20 finding 35-F33: the tail is fitted on in-sample residuals only with a WARNING.

When residual_X is not given, the threshold and GPD come from in-sample residuals, which are too small for a boosted body. The fit now says so.
"""

import logging

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin

from mlframe.training.composite.extremes import TailCompositeEstimator


class _ZeroBase(BaseEstimator, RegressorMixin):
    """A regressor predicting zero, so the composite residual equals y."""

    def fit(self, X, y):
        """Nothing to fit."""
        return self

    def predict(self, X):
        """Zeros for every row."""
        return np.zeros(len(X))


def _data(n=500, seed=0):
    """A frame with one feature and heavy-tailed targets."""
    rng = np.random.default_rng(seed)
    X = pd.DataFrame({"f0": np.linspace(0, 1, n), "base": np.zeros(n)})
    y = rng.standard_t(3, size=n)
    return X, y


def test_in_sample_tail_fit_logs_a_warning(caplog):
    """No residual_X: the fit logs that the tail uses in-sample residuals."""
    X, y = _data()
    est = TailCompositeEstimator(base_estimator=_ZeroBase(), transform_name="diff", base_column="base")
    with caplog.at_level(logging.WARNING):
        est.fit(X, y)
    assert any("IN-SAMPLE residuals" in r.getMessage() for r in caplog.records)


def test_held_out_residuals_do_not_warn(caplog):
    """With residual_X and residual_y given there is no in-sample warning."""
    X, y = _data()
    est = TailCompositeEstimator(base_estimator=_ZeroBase(), transform_name="diff", base_column="base")
    with caplog.at_level(logging.WARNING):
        est.fit(X, y, residual_X=X, residual_y=y)
    assert not any("IN-SAMPLE residuals" in r.getMessage() for r in caplog.records)
