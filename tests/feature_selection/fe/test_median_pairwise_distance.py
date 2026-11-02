"""The bisection median of pairwise distances equals the quickselect median bit for bit, for every shape of column."""

from __future__ import annotations

import numpy as np
import pytest

from mlframe.feature_selection.filters._orthogonal_hsic_fe import _median_pairwise_distance, _median_pairwise_distance_select


def _columns():
    """Columns with odd / even pair counts, heavy ties, constants, tiny sizes, both dtypes."""
    rng = np.random.default_rng(11)
    out = []
    for n in (2, 3, 4, 5, 7, 10, 33, 100, 257, 1000):
        out.append(rng.standard_normal(n))
        out.append(np.round(rng.standard_normal(n), 1))
        out.append(rng.integers(0, 4, n).astype(float))
        out.append(np.full(n, 2.5))
        out.append(np.exp(rng.standard_normal(n) * 3))
    return out


@pytest.mark.parametrize("dtype", [np.float64, np.float32])
def test_equals_the_quickselect_median(dtype):
    """Same float, not merely close, on every column."""
    for col in _columns():
        a = np.ascontiguousarray(col.astype(dtype))
        assert _median_pairwise_distance(a) == _median_pairwise_distance_select(a), (a.size, dtype)


def test_a_non_finite_column_takes_the_quickselect_path():
    """NaN / inf columns are handed to the old routine unchanged."""
    a = np.array([0.0, 1.0, np.nan, 3.0, 4.0])
    x, y = _median_pairwise_distance(a), _median_pairwise_distance_select(a)
    assert (np.isnan(x) and np.isnan(y)) or x == y
