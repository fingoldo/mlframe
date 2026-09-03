"""CALIBRATION-9: estimate_calibration_quality_binned must raise a clear ValueError on a non-positive
nbins instead of letting nbins=0 reach the njit bin_predictions kernel, where bin_size = s // nbins
raises an opaque ZeroDivisionError with no context about which argument was wrong.
"""

from __future__ import annotations

import numpy as np
import pytest

from mlframe.calibration.quality import estimate_calibration_quality_binned


def test_nbins_zero_raises_clear_value_error_not_zero_division_error():
    """nbins=0 must raise ValueError naming nbins, not an opaque ZeroDivisionError from the njit kernel."""
    rng = np.random.default_rng(0)
    y_true = rng.integers(0, 2, 50).astype(np.float64)
    y_pred = rng.uniform(0, 1, 50)
    with pytest.raises(ValueError, match="nbins"):
        estimate_calibration_quality_binned(y_true, y_pred, nbins=0)


def test_nbins_negative_raises_clear_value_error():
    """A negative nbins must also raise the clear ValueError, not reach the kernel."""
    rng = np.random.default_rng(1)
    y_true = rng.integers(0, 2, 50).astype(np.float64)
    y_pred = rng.uniform(0, 1, 50)
    with pytest.raises(ValueError, match="nbins"):
        estimate_calibration_quality_binned(y_true, y_pred, nbins=-5)


def test_nbins_positive_still_works():
    """A normal positive nbins is unaffected by the new guard."""
    rng = np.random.default_rng(2)
    y_true = rng.integers(0, 2, 200).astype(np.float64)
    y_pred = rng.uniform(0, 1, 200)
    pockets_predicted, pockets_true, data, metrics = estimate_calibration_quality_binned(y_true, y_pred, nbins=10)
    assert pockets_predicted.shape == (10,)
    assert pockets_true.shape == (10,)
    assert data.shape == (10, 4)
    assert metrics
