"""rolling_moving_average keeps fractional means for integer input (audit 2026-11-20, finding 24-F4)."""

import numpy as np
import pytest

from mlframe.feature_engineering.numerical import rolling_moving_average


@pytest.mark.parametrize("compensated", [True, False])
def test_integer_input_is_not_truncated_to_integer_means(compensated):
    """The mean of [1, 2] is 1.5; an integer output buffer used to store 1."""
    arr = np.array([1, 2, 4, 8], dtype=np.int64)
    got = rolling_moving_average(arr, n=2, compensated=compensated)
    np.testing.assert_allclose(got, [1.5, 3.0, 6.0])


def test_float_input_keeps_its_dtype():
    """Floating input is returned in its own precision."""
    arr = np.arange(6, dtype=np.float32)
    assert rolling_moving_average(arr, n=3).dtype == np.float32
