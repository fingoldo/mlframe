"""``_distinct_exceeds`` answers exactly what ``np.unique(y).size > cap`` answers."""
import numpy as np
import pytest

from mlframe.feature_selection.filters._adaptive_nbins import _bin_y_for_mi, _distinct_exceeds


@pytest.mark.parametrize("dtype", [np.int8, np.int32, np.int64, np.uint8, np.bool_])
@pytest.mark.parametrize("n_distinct", [1, 2, 3, 63, 64, 65, 200])
@pytest.mark.parametrize("offset", [0, -5, 10**9])
def test_matches_np_unique(dtype, n_distinct, offset):
    """Around the cap, with negative and far-offset values (the far offset takes the sorting branch for a wide range)."""
    if dtype is np.bool_ and n_distinct > 2:
        pytest.skip("a bool vector holds at most two values")
    if np.dtype(dtype).itemsize < 4 and (n_distinct > 120 or offset != 0):
        pytest.skip("value does not fit the narrow dtype")
    rng = np.random.default_rng(n_distinct)
    y = (rng.integers(0, n_distinct, 1000) + (offset if dtype not in (np.bool_, np.uint8, np.int8) else 0)).astype(dtype)
    y[:n_distinct] = np.arange(n_distinct).astype(dtype) + (offset if dtype not in (np.bool_, np.uint8, np.int8) else 0)
    assert _distinct_exceeds(y, 64) == (np.unique(y).size > 64)


def test_wide_range_and_empty_inputs():
    """A range beyond the bincount limit falls back to the sort; an empty vector never exceeds."""
    wide = np.array([0, 10**12, 5, 10**12, 7], dtype=np.int64)
    assert _distinct_exceeds(wide, 3) and not _distinct_exceeds(wide, 4)
    assert not _distinct_exceeds(np.empty(0, dtype=np.int64), 64)


def test_bin_y_for_mi_unchanged_on_class_codes_and_wide_integers():
    """Class-code targets pass through; a wide-cardinality integer target is quantile-binned as before."""
    codes = np.random.default_rng(0).integers(0, 5, 500)
    y_b, k = _bin_y_for_mi(codes)
    assert k == int(codes.max()) + 1 and np.array_equal(y_b, codes)
    wide = np.random.default_rng(1).integers(0, 10**6, 5000)
    _, k_wide = _bin_y_for_mi(wide)
    assert 2 <= k_wide <= 10
