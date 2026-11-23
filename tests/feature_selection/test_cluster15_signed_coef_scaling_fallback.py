"""Cluster 15 regression: the signed coefficient scales by the data stds when no train frame is given, as the legacy path does (audit 2026-11-20, finding 15-F15).

Before the fix a direct call without train_data left the coefficients unscaled, while the legacy path scaled them by the data. The new
optional ``data`` argument is the fallback; a given train_data still wins.
"""

import numpy as np
import pandas as pd

from mlframe.feature_selection.wrappers._helpers_importance_agg import get_signed_linear_coef


class _Linear:
    """Linear model stub exposing a fixed coef_."""

    def __init__(self, coef):
        """Store the coefficient row."""
        self.coef_ = np.asarray([coef], dtype=float)


def _frame():
    """A two-column numeric frame with distinct stds."""
    return pd.DataFrame({"a": [0.0, 2.0, 4.0], "b": [0.0, 1.0, 2.0]})


def test_data_fallback_scales_like_the_legacy_path():
    """With only data supplied, the coefficients are scaled by its column stds (15-F15)."""
    X = _frame()
    out = get_signed_linear_coef(_Linear([1.0, 1.0]), ["a", "b"], data=X)
    expected = np.array([1.0, 1.0]) * np.nanstd(X.values, axis=0)
    assert np.allclose([out["a"], out["b"]], expected)


def test_train_data_still_takes_precedence():
    """Control: an explicit train frame is used for the scaling (15-F15)."""
    X = _frame()
    train = pd.DataFrame({"a": [0.0, 10.0], "b": [0.0, 10.0]})
    out = get_signed_linear_coef(_Linear([1.0, 1.0]), ["a", "b"], train_data=train, data=X)
    expected = np.array([1.0, 1.0]) * np.nanstd(train.values, axis=0)
    assert np.allclose([out["a"], out["b"]], expected)
