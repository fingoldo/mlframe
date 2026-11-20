"""A sample_weight with the wrong length is refused, not read past its buffer (audit 2026-11-20, finding 17-F4)."""
import numpy as np
import pytest

from mlframe.metrics.regression._regression_metrics import fast_mean_absolute_error


def test_short_sample_weight_is_refused():
    """N-1 weights for N rows used to be indexed out of bounds by the unchecked kernel."""
    y = np.arange(1.0, 6.0)
    p = y + 0.5
    with pytest.raises(ValueError, match="sample_weight"):
        fast_mean_absolute_error(y, p, sample_weight=np.ones(4))


def test_matching_sample_weight_still_works():
    """A weight per row keeps the weighted MAE."""
    y = np.array([1.0, 2.0, 3.0])
    p = np.array([2.0, 2.0, 3.0])
    assert abs(fast_mean_absolute_error(y, p, sample_weight=np.ones(3)) - 1.0 / 3.0) < 1e-9
