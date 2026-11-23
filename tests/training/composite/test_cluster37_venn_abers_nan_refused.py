"""Audit 2026-11-20 finding 37-F3: a NaN positive-class score on the calibration set is refused, not turned into a grid point."""

import numpy as np
import pytest

from mlframe.training.composite.venn_abers import calibrate_venn_abers


class _Stub:
    """A binary classifier whose predict_proba returns NaN for one row."""

    classes_ = np.array([0, 1])
    estimator_ = object()

    def predict_proba(self, X):
        """Return a fixed probability per row, NaN on the second row."""
        p = np.full(len(X), 0.6)
        p[1] = np.nan
        return np.column_stack([1 - p, p])


def test_nan_calibration_score_is_refused():
    """calibrate_venn_abers raises ValueError naming the non-finite scores."""
    est = _Stub()
    with pytest.raises(ValueError, match="non-finite positive-class score"):
        calibrate_venn_abers(est, np.zeros((5, 1)), np.array([0, 1, 0, 1, 0]))
