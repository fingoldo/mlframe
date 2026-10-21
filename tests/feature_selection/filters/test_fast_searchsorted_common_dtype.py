"""``searchsorted_right`` compares in the common dtype of the edges and the column, exactly as ``np.searchsorted`` does.

Narrowing float64 edges to the float32 of the column moved a value that lies on an edge into the neighbouring bin (one row of a 250 000-row column), so the host binner and the device binner
placed it differently.
"""

import numpy as np
import pytest

from mlframe.feature_selection.filters._fast_host_ops import searchsorted_right

_N = 250_000  # above the parallel threshold, where the narrowing happened


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_float64_edges_are_not_rounded_to_a_float32_column(seed):
    """Values equal to float32 roundings of the edges land in the same bin as with numpy."""
    rng = np.random.default_rng(seed)
    x = rng.standard_normal(_N).astype(np.float32)
    edges = np.sort(rng.standard_normal(9))  # float64
    x[:9] = edges.astype(np.float32)  # on (the float32 rounding of) every edge
    assert np.array_equal(searchsorted_right(edges, x), np.searchsorted(edges, x, side="right"))


def test_same_dtype_inputs_and_small_columns_are_unchanged():
    """Equal dtypes and the small-column fallback agree with numpy."""
    rng = np.random.default_rng(3)
    for dtype in (np.float32, np.float64):
        x = rng.standard_normal(_N).astype(dtype)
        edges = np.sort(rng.standard_normal(9)).astype(dtype)
        assert np.array_equal(searchsorted_right(edges, x), np.searchsorted(edges, x, side="right"))
    small = rng.standard_normal(1000)
    assert np.array_equal(searchsorted_right(np.array([-0.5, 0.5]), small), np.searchsorted([-0.5, 0.5], small, side="right"))
