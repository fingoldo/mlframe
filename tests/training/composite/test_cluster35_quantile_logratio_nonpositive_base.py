"""Audit 2026-11-20 finding 35-F11: predict_quantile must route base <= 0 rows to the fallback, as predict does, not raise for the batch."""

import numpy as np
import pandas as pd

from mlframe.training.composite.estimator._estimator import CompositeTargetEstimator
from mlframe.training.composite.transforms.linear import _logratio_fit


class _ConstT:
    """Inner model predicting a constant T for every row."""

    def predict(self, X, **kwargs):
        """Return T = 0.0 for every row."""
        return np.zeros(len(X))

    def predict_quantile(self, X, alpha=0.5, **kwargs):
        """Return the constant T quantile for every row."""
        return np.zeros(len(X))


def test_predict_quantile_routes_nonpositive_base_rows_to_fallback():
    """A batch with one base <= 0 returns a finite value for the valid row and the fallback for the other, without raising."""
    y_train = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0])
    base_train = np.linspace(1.0, 2.0, 10)
    wrapped = CompositeTargetEstimator.from_fitted_inner(
        fitted_inner=_ConstT(), transform_name="logratio", base_column="b",
        transform_fitted_params=_logratio_fit(y_train, base_train), y_train=y_train, base_train=base_train,
    )
    X = pd.DataFrame({"b": [2.0, 0.0]})
    out = np.asarray(wrapped.predict_quantile(X, alpha=0.5)).reshape(-1)
    assert np.isfinite(out[0])
    assert out[1] == np.median(y_train)
