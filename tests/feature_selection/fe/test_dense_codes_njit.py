"""The one-pass integer renumbering equals the count-table and ``np.unique`` formulations on every integer dtype, with gaps, negatives and large values."""

import numpy as np
import pytest

from mlframe.feature_selection.filters._added_information_gate import _dense_codes


def _reference(a):
    """The ``np.unique`` inverse and its size."""
    _, inv = np.unique(np.asarray(a), return_inverse=True)
    inv = inv.reshape(-1).astype(np.int64)
    return inv, (int(inv.max()) + 1 if inv.size else 0)


@pytest.mark.parametrize("dtype", [np.int8, np.uint8, np.int16, np.int32, np.int64, np.uint16])
def test_small_integer_codes(dtype):
    """Occupied values with gaps renumber in increasing order."""
    r = np.random.default_rng(0)
    a = r.choice(np.array([0, 3, 4, 9, 40, 100], dtype=dtype), size=5000)
    got, k = _dense_codes(a)
    want, kw = _reference(a)
    assert k == kw and np.array_equal(got, want)


def test_negative_and_large_values_use_unique():
    """A negative value or one above 2**22 takes the sort path with the same result; 2-D input keeps its flat order."""
    for a in (np.array([-3, 5, 5, 7]), np.array([1, 2**23, 2**23, 3]), np.arange(12).reshape(3, 4) % 5, np.array([], dtype=np.int64), np.array([1.5, 2.5, 1.5])):
        got, k = _dense_codes(a)
        want, kw = _reference(a)
        assert k == kw and np.array_equal(got, want)
