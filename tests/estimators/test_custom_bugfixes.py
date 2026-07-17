"""Regression tests for confirmed bugs in mlframe.estimators.custom."""

import numpy as np
import pytest

from mlframe.estimators.custom import (
    create_dummy_lagged_predictions,
    soft_winsorize,
)


def test_create_dummy_lagged_predictions_negative_lag_uses_np_nan():
    # DEP1: ``np.NaN`` was removed in numpy 2 -> AttributeError on the lag<=0 branch.
    """Create dummy lagged predictions negative lag uses np nan."""
    out = create_dummy_lagged_predictions(np.array([1.0, 2.0, 3.0, 4.0]), strategy="constant_lag", lag=-1)
    assert isinstance(out, np.ndarray)


def test_create_dummy_lagged_predictions_every_accepted_strategy_returns_ndarray():
    # TYPE2: only "constant_lag" is implemented; it must return an ndarray (no UnboundLocalError).
    """Create dummy lagged predictions every accepted strategy returns ndarray."""
    out = create_dummy_lagged_predictions(np.array([1.0, 2.0, 3.0]), strategy="constant_lag", lag=1)
    assert isinstance(out, np.ndarray)


def test_create_dummy_lagged_predictions_rejects_adaptive_lag():
    # TYPE2: "adaptive_lag" had no body -> UnboundLocalError. Now an explicit, honest error.
    """Create dummy lagged predictions rejects adaptive lag."""
    with pytest.raises(ValueError):
        create_dummy_lagged_predictions(np.array([1.0, 2.0, 3.0]), strategy="adaptive_lag")


def test_create_dummy_lagged_predictions_empty_raises():
    # EDGE21: median([]) is NaN; empty input must raise a clear error rather than emit all-NaN.
    """Create dummy lagged predictions empty raises."""
    with pytest.raises(ValueError):
        create_dummy_lagged_predictions(np.array([], dtype=float), strategy="constant_lag", lag=1)


def test_soft_winsorize_max_equals_threshold_raises():
    # EDGE20: data max == abs_upper_threshold -> zero upper span. The old ``<0`` guard let this through
    # silently; the corrected ``<=0`` guard reports it (a zero span is a divide-by-zero hazard, not a no-op).
    # Lower side is kept valid (abs_lower_threshold=2.0 > data min 1.0) to isolate the upper-span behaviour.
    """Soft winsorize max equals threshold raises."""
    data = np.array([1.0, 2.0, 3.0, 10.0], dtype="float64")
    with pytest.raises(ValueError):
        soft_winsorize(data, abs_lower_threshold=2.0, rel_lower_limit=0.2, abs_upper_threshold=10.0, rel_upper_limit=5.0, distribution="linear")
