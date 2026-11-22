"""Audit 2026-11-20 finding 35-F25: from_fitted_inner keeps a base fit range that discovery already stamped.

The comment in the code promised it; the code overwrote the range with the one computed from ``base_train``, which moved the soft-shrink
thresholds silently.
"""

import numpy as np

from mlframe.training.composite.estimator import CompositeTargetEstimator
from mlframe.training.composite.estimator._soft_shrink import BASE_FIT_RANGE_KEY


class _Inner:
    """A constant predictor standing in for the fitted inner estimator."""

    def predict(self, X):
        """Return zeros for each row."""
        return np.zeros(len(X))


def test_stamped_base_fit_range_is_kept():
    """A spec carrying a range keeps it; base_train with a different range does not replace it."""
    stamped = {"lo": -1.0, "hi": 1.0, "iqr": 2.0}
    params = {"alpha": 0.0, "beta": 1.0, BASE_FIT_RANGE_KEY: stamped}
    base_train = np.linspace(100.0, 200.0, 50)
    y = np.linspace(0.0, 1.0, 50)
    inst = CompositeTargetEstimator.from_fitted_inner(
        fitted_inner=_Inner(), transform_name="linear_residual", base_column="b",
        transform_fitted_params=params, y_train=y, base_train=base_train,
    )
    assert inst.fitted_params_[BASE_FIT_RANGE_KEY] == stamped
