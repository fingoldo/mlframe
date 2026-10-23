"""The residual rescue ends cleanly when it has admitted every column of the pool."""

from __future__ import annotations

import numpy as np

from mlframe.feature_selection.filters._residual_rescue_fe import MIN_ROWS, residual_rescue


def test_pool_exhausted_before_max_admit_does_not_raise_and_returns_the_admitted_column():
    """One pool column that explains the target is admitted; the next round has no live candidate and the call returns instead of failing on an empty linear block."""
    rng = np.random.default_rng(0)
    n = max(4000, int(MIN_ROWS) * 2)
    x = rng.standard_normal(n)
    other = rng.standard_normal(n)
    y = 2.0 * x + 0.1 * rng.standard_normal(n)
    admitted = residual_rescue({"other": other}, {"x": x}, y, linear_channel=True, max_admit=4)
    assert [name for name, _t, _rel in admitted] == ["x"]
