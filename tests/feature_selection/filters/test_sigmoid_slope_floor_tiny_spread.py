"""The sigmoid slope floors the interquantile range at 1e-12 (``max``), it does not pad it (``+``)."""

from __future__ import annotations

import numpy as np

from mlframe.feature_selection.filters import bases


def test_sigmoid_slope_uses_max_floor_on_a_near_constant_column() -> None:
    """With a range far below 1e-12 an additive pad would give a slope ~10% off the floored one; the host fit and the device cells use the floor."""
    x = np.linspace(0.0, 1.0, 101) * 1e-14
    _z, params = bases._sigmoid_fit(x)
    assert params["slope"] == 4.0 / (1e-12 / 9.0)


def test_sigmoid_slope_follows_the_range_on_a_regular_column() -> None:
    """Above the floor the slope is exactly ``36 / iqr`` with no pad skewing it."""
    x = np.linspace(-1.0, 1.0, 1001)
    _z, params = bases._sigmoid_fit(x)
    iqr = float(np.quantile(x, 0.9) - np.quantile(x, 0.1))
    assert params["slope"] == 4.0 / (iqr / 9.0)
