"""Audit 2026-11-20 finding 29-F19: an all-NaN loss is a ValueError with a reason, not a bare AssertionError.

The bare assert said nothing and disappeared under python -O. The test feeds NaN predictions and checks the ValueError names the cause.
"""

import numpy as np
import pytest

from mlframe.votenrank.geometric_weight_blend import geometric_weight_blend


def test_all_nan_predictions_raise_a_named_value_error():
    """Every restart's loss is NaN, so the blend refuses with a ValueError that names the predictions."""
    preds = [np.full(6, np.nan), np.full(6, np.nan)]
    y = np.array([0.0, 1.0, 0.0, 1.0, 0.0, 1.0])
    with pytest.raises(ValueError, match="non-finite loss"):
        geometric_weight_blend(preds, y, lambda t, p: float(np.mean((t - p) ** 2)), n_restarts=2)
