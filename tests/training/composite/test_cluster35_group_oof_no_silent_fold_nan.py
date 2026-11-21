"""Audit 2026-11-20 finding 35-F13: a failing OOF fold must raise, not NaN-fill its rows and let the aggregate silently go constant."""

import numpy as np
import pandas as pd
import pytest

from mlframe.training.composite.group_aggregate_macro import _oof_multi_target_predictions


class _Failing:
    """Estimator whose fit always fails, standing in for a single-output model handed multi-output targets."""

    def fit(self, X, y):
        """Refuse to fit."""
        raise ValueError("single-output estimator cannot take 2 targets")

    def predict(self, X):
        """Unused."""
        raise AssertionError("predict must not be reached")


def test_failing_fold_raises_instead_of_nan_filling():
    """The fold error propagates; no NaN-filled OOF matrix is returned."""
    X = pd.DataFrame({"a": np.arange(40.0)})
    Y = np.column_stack([np.arange(40.0), np.arange(40.0) * 2])
    with pytest.raises(ValueError, match="single-output"):
        _oof_multi_target_predictions(lambda: _Failing(), X, Y, n_splits=4, random_state=0)
