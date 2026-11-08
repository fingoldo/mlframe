"""``unique_inverse_dense`` returns exactly what ``np.unique(values, return_inverse=True)`` returns, on the counting path and on the fallback path."""

from __future__ import annotations

import numpy as np
import pytest

from mlframe.feature_selection.filters._internals import unique_inverse_dense

_DTYPES = [np.int8, np.int16, np.int32, np.int64, np.uint8, np.uint16, np.uint32, np.uint64]
_RANGES = [(0, 5), (-40, 90), (1000, 1050), (0, 20000)]
# types that cannot hold a range are left out of the grid instead of being skipped after the data is built
_FITTING = [(d, lo, hi) for d in _DTYPES for lo, hi in _RANGES if np.iinfo(d).min <= lo and hi - 1 <= np.iinfo(d).max]


@pytest.mark.parametrize("dtype,low,high", _FITTING)
def test_same_as_np_unique(dtype, low, high):
    """Dense, negative-valued, offset and wide integer vectors of every integer type (types that cannot hold the range are skipped)."""
    rng = np.random.default_rng(low + high)
    values = rng.integers(low, high, 6000).astype(dtype)
    uniq, inverse = unique_inverse_dense(values)
    ref_u, ref_i = np.unique(values, return_inverse=True)
    assert uniq.dtype == ref_u.dtype
    assert np.array_equal(uniq, ref_u)
    assert np.array_equal(inverse, ref_i.reshape(-1))


@pytest.mark.parametrize("values", [np.array([], dtype=np.int64), np.array([7], dtype=np.int64), np.array([2**40, 3, 2**40, -5], dtype=np.int64), np.array([1.5, 1.5, 2.0]), np.array(["b", "a", "b"], dtype=object)])
def test_empty_single_wide_and_non_integer_inputs_take_np_unique(values):
    """Empty and one-element vectors, a range far above the rows, floats and strings all give np.unique's answer."""
    uniq, inverse = unique_inverse_dense(values)
    ref_u, ref_i = np.unique(values, return_inverse=True)
    assert uniq.tolist() == ref_u.tolist()
    assert inverse.tolist() == ref_i.reshape(-1).tolist()
