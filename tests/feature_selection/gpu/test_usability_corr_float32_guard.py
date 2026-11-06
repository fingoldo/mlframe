"""The float32 usability-correlation default keeps the double kernel for inputs float32 cannot represent.

An offset of 1e8 next to a spread of 0.1 leaves float32 with a step of 8, so a linear pair reads as less than perfectly correlated and a
near-constant column reads as constant. Such inputs must not take the float32 path.
"""

from __future__ import annotations

import numpy as np

from mlframe.feature_selection.filters import batch_pair_usability_corr_gpu as bp


def test_offset_heavy_columns_are_not_float32_safe():
    """A column near 1e8 with a spread of 0.1 is flagged."""
    n = 1000
    y = np.arange(n, dtype=np.float64)
    x0 = 1e8 + 0.1 * np.arange(n)
    assert bp._float32_unsafe(y, x0.reshape(-1, 1)) is True


def test_well_scaled_columns_stay_on_float32():
    """Ordinary standard-normal columns keep the float32 path."""
    rng = np.random.default_rng(0)
    y = rng.normal(size=500)
    X = rng.normal(size=(500, 3))
    assert bp._float32_unsafe(y, X) is False


def test_non_finite_values_take_the_double_path():
    """A NaN or inf in any input routes the call to the double kernel."""
    rng = np.random.default_rng(1)
    y = rng.normal(size=100)
    X = rng.normal(size=(100, 2))
    X[3, 1] = np.nan
    assert bp._float32_unsafe(y, X) is True
