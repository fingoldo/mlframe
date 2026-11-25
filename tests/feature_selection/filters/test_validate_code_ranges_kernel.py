"""``_validate_code_ranges`` finds the first used column with a code outside ``[0, nbins)`` in one parallel pass, for every integer type, layout and row count (block boundaries included)."""

import numpy as np
import pytest

from mlframe.feature_selection.filters import _permutation_null as pn


def _data(n, k=6, dtype=np.int8, order="C", seed=0):
    """A valid code matrix: column ``j`` holds codes in ``[0, j + 3)``."""
    rng = np.random.default_rng(seed)
    cols = [rng.integers(0, j + 3, n) for j in range(k)]
    return np.array(np.column_stack(cols), dtype=dtype, order=order), np.arange(3, 3 + k, dtype=np.int64)


@pytest.mark.parametrize("n", [1, 2, 7, 2047, 2048, 2049, 4097, 100003])
@pytest.mark.parametrize("dtype", [np.int8, np.uint8, np.int16, np.int32, np.int64])
@pytest.mark.parametrize("order", ["C", "F"])
def test_valid_codes_pass_and_the_offending_column_is_named(n, dtype, order):
    """No error for valid codes (also with columns listed twice and out of order); a code equal to nbins or below zero in one used column raises and names that column and its range."""
    data, nbins = _data(n, dtype=dtype, order=order)
    pn._validate_code_ranges(data, nbins, np.array([5, 0, 3, 3, 1]))
    bad = data.copy(order=order)
    bad[n // 2, 4] = nbins[4]
    with pytest.raises(ValueError, match=rf"column 4 holds codes in \[\d+, {int(nbins[4])}\] outside \[0, {int(nbins[4])}\)"):
        pn._validate_code_ranges(bad, nbins, np.array([0, 4, 5]))
    pn._validate_code_ranges(bad, nbins, np.array([0, 1, 5]))  # the bad column is not used: no error
    if np.dtype(dtype).kind == "i":
        neg = data.copy(order=order)
        neg[0, 2] = -1
        with pytest.raises(ValueError, match=r"column 2 holds codes in \[-1,"):
            pn._validate_code_ranges(neg, nbins, np.array([2]))


def test_the_first_bad_column_in_index_order_is_reported():
    """Two bad columns: the lower index is named."""
    data, nbins = _data(5000)
    data[10, 5] = nbins[5]
    data[11, 1] = nbins[1] + 2
    with pytest.raises(ValueError, match=r"column 1 holds"):
        pn._validate_code_ranges(data, nbins, np.array([5, 1, 0]))


def test_an_empty_matrix_is_accepted():
    """No rows: nothing to check, and a code outside the range cannot exist."""
    assert pn._validate_code_ranges(np.empty((0, 3), dtype=np.int8), np.array([3, 3, 3]), np.array([0, 1])) is None
