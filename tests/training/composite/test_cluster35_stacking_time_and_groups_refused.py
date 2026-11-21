"""Audit 2026-11-20 finding 35-F7: time_aware with groups must be refused, not silently replaced by GroupKFold (which drops time order)."""

import numpy as np
import pytest

from mlframe.training.composite.ensemble.feature_stacking import composite_oof_predictions


class _Mean:
    """Predicts the training mean; the folds are what this test is about."""

    def fit(self, X, y, **kwargs):
        """Store the training mean."""
        self.m_ = float(np.mean(y))
        return self

    def predict(self, X):
        """Return the stored mean for every row."""
        return np.full(len(X), self.m_)


def test_time_aware_with_groups_is_refused():
    """The combination raises a ValueError naming both options."""
    X = np.arange(40.0).reshape(-1, 1)
    y = np.arange(40.0)
    groups = np.repeat(np.arange(8), 5)
    with pytest.raises(ValueError, match="time_aware and groups"):
        composite_oof_predictions(lambda: _Mean(), X, y, n_splits=4, time_aware=True, groups=groups)
