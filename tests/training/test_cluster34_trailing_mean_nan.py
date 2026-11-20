"""The trailing-mean kernel skips NaN entries instead of carrying a NaN into every later row (audit 2026-11-20, finding 34-F1)."""

import numpy as np

from mlframe.training.composite.discovery._base_engineering import _causal_rolling_mean_impl


def test_nan_in_the_window_does_not_poison_later_rows():
    """Row 3's window holds (NaN, 3): the mean is 3.0, and row 4 is then unaffected."""
    y = np.array([1.0, np.nan, 3.0, 4.0, 5.0])
    out = _causal_rolling_mean_impl(y, 2)
    assert out[2] == 1.0
    assert out[3] == 3.0
    assert out[4] == 3.5


def test_nan_free_series_matches_the_plain_trailing_mean():
    """Without NaN the output is the ordinary trailing mean."""
    y = np.arange(1.0, 9.0)
    out = _causal_rolling_mean_impl(y, 3)
    np.testing.assert_allclose(out[3:], [np.mean(y[i - 3 : i]) for i in range(3, 8)])
