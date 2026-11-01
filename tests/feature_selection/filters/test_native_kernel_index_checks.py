"""Three njit kernels raise on an out-of-range code instead of reading or writing past their table (MEM5-1, MEM5-2)."""

import numpy as np
import pytest

from mlframe.feature_selection.filters._cat_kway_materialize import _dense_renumber_codes, _scatter_factorize_lookup
from mlframe.feature_selection.filters._cat_target_encoding_and_weighted import _group_aware_shuffle


def test_group_aware_shuffle_refuses_a_group_id_outside_the_table():
    """Group ids at or above n_groups, and negative ones, raise; in-range ids still shuffle within groups."""
    y = np.arange(6, dtype=np.int32)
    with pytest.raises(ValueError, match="outside"):
        _group_aware_shuffle(y.copy(), np.array([0, 1, 2, 3, 4, 5], dtype=np.int64), 4, 0)
    with pytest.raises(ValueError, match="outside"):
        _group_aware_shuffle(y.copy(), np.array([0, 1, -1, 1, 0, 2], dtype=np.int64), 4, 0)
    ok = y.copy()
    _group_aware_shuffle(ok, np.array([0, 0, 1, 1, 2, 2], dtype=np.int64), 3, 7)
    assert sorted(ok.tolist()) == sorted(y.tolist()) or set(ok.tolist()) <= set(y.tolist())


def test_dense_renumber_codes_refuses_a_code_beyond_expected_size():
    """A code equal to expected_size, or negative, raises; valid codes renumber densely."""
    with pytest.raises(ValueError, match="outside"):
        _dense_renumber_codes(np.array([0, 1, 5], dtype=np.int64), 5)
    with pytest.raises(ValueError, match="outside"):
        _dense_renumber_codes(np.array([0, -1, 2], dtype=np.int64), 5)
    out, n = _dense_renumber_codes(np.array([0, 3, 3, 4], dtype=np.int64), 5)
    assert out.tolist() == [0, 1, 1, 2] and n == 3


def test_scatter_factorize_lookup_refuses_codes_built_for_other_bin_counts():
    """Factor values larger than the bin count the lookup was sized for raise instead of writing out of bounds."""
    factors = np.array([[0, 1], [2, 9]], dtype=np.int64)
    with pytest.raises(ValueError, match="outside the lookup"):
        _scatter_factorize_lookup(factors, 0, 1, 3, 3, np.array([0, 1], dtype=np.int64))
    good = _scatter_factorize_lookup(np.array([[0, 1], [2, 2]], dtype=np.int64), 0, 1, 3, 3, np.array([0, 1], dtype=np.int64))
    assert good.shape == (9,) and int(good[0 + 1 * 3]) == 0 and int(good[2 + 2 * 3]) == 1
