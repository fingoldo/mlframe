"""Edge inputs of the multi-core host primitives: 2-D columns and unsigned 64-bit codes."""

import numpy as np

from mlframe.feature_selection.filters import _fast_host_ops as F


def test_searchsorted_right_of_a_large_2d_float_array_matches_numpy():
    """A 2-D float array above the parallel threshold falls back to numpy instead of failing to compile the 1-D kernel."""
    rng = np.random.default_rng(0)
    x = rng.random((F._PARALLEL_MIN_N // 100 + 10, 100))
    edges = np.linspace(0.0, 1.0, 6)
    np.testing.assert_array_equal(F.searchsorted_right(edges, x), np.searchsorted(edges, x, side="right"))


def test_count_distinct_int_handles_uint64_codes_above_the_int64_range():
    """Codes near 2**64 with a small span are counted, not rejected with an overflow."""
    a = np.array([2**64 - 1, 2**64 - 2, 2**64 - 1], dtype=np.uint64)
    assert F.count_distinct_int(a) == 2


def test_count_distinct_int_signed_small_dtype_does_not_wrap():
    """An int8 column spanning -100..100 is shifted in a wider type, so its three values stay three."""
    assert F.count_distinct_int(np.array([-100, 100, 0, 100], dtype=np.int8)) == 3
