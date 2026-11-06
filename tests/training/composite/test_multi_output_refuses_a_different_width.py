"""``CompositeMultiOutputEstimator.predict`` refuses input whose width differs from the fitted width.

Columns of a narrower input were passed to each per-column pipeline unaligned, so the predictions were wrong without an error (audit API3-15).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LinearRegression

from mlframe.training.composite.multi_output import CompositeMultiOutputEstimator


def _frame():
    """Three features, the first two named as the per-column base columns, and a two-output target."""
    rng = np.random.default_rng(0)
    X = pd.DataFrame(rng.normal(size=(120, 3)) + 5.0, columns=["a", "b", "c"])
    Y = np.column_stack([X["a"] * 2.0 + X["c"], X["b"] * 1.5 - X["c"]]) + rng.normal(scale=0.1, size=(120, 2))
    return X, Y


def test_predict_on_fewer_columns_than_fitted_raises_value_error():
    """Fitted on three features, predicting on two raises ValueError instead of returning predictions."""
    X, Y = _frame()
    est = CompositeMultiOutputEstimator(base_estimator=LinearRegression(), base_columns_map={0: "a", 1: "b"}).fit(X, Y)
    with pytest.raises(ValueError):
        est.predict(X[["a", "b"]])


def test_predict_on_the_fitted_width_still_works():
    """With the fitted width the prediction matrix has one column per output."""
    X, Y = _frame()
    est = CompositeMultiOutputEstimator(base_estimator=LinearRegression(), base_columns_map={0: "a", 1: "b"}).fit(X, Y)
    assert est.predict(X).shape == (120, 2)
