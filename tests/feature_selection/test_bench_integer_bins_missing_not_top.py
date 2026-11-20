"""Missing values are not binned into the top rank bin (audit 2026-11-20, finding 11-F1)."""
import numpy as np

from mlframe.feature_selection._benchmarks.fs_hybrid._arms import _integer_bins


def test_nan_rows_do_not_land_in_the_top_bin():
    """With more unique values than bins, the NaN rows used to take the top bin; they must take bin 0 and the finite rows keep their ranks."""
    values = np.arange(40, dtype=np.float64)
    values[:5] = np.nan
    codes = _integer_bins(values, n_bins=4)
    assert np.all(codes[:5] == 0)
    assert codes[-1] == 3
    assert codes[5:].min() >= 0
