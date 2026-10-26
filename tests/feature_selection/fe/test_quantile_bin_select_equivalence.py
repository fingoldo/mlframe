"""The selection-based equi-frequency binner returns exactly the bins of the stable-argsort reference, on tie-free and tied data alike."""
import numpy as np
import pytest

from mlframe.feature_selection.filters._numba_utils import _quantile_bin_njit, _quantile_bin_sorted_njit


def _column(kind, n, rng):
    """A test column of the named shape."""
    if kind == "normal":
        return rng.standard_normal(n)
    if kind == "ties_few":
        return np.round(rng.standard_normal(n), 2)
    if kind == "ties_many":
        return np.round(rng.standard_normal(n), 0)
    if kind == "int5":
        return rng.integers(0, 5, n).astype(float)
    if kind == "constant":
        return np.full(n, 2.0)
    if kind == "sorted":
        return np.sort(rng.standard_normal(n))
    if kind == "infinite":
        x = rng.standard_normal(n)
        x[::7] = np.inf
        x[3::11] = -np.inf
        return x
    if kind == "tie_on_boundary":
        x = np.sort(rng.standard_normal(n))
        x[n // 2 - 1] = x[n // 2]
        return rng.permutation(x)
    return rng.standard_normal(n).astype(np.float32)


@pytest.mark.parametrize("kind", ["normal", "ties_few", "ties_many", "int5", "constant", "sorted", "infinite", "tie_on_boundary", "float32"])
@pytest.mark.parametrize("n", [50, 97, 1000, 20001])
@pytest.mark.parametrize("n_bins", [2, 10, 12, 20])
def test_selected_bins_equal_the_sorted_bins(kind, n, n_bins):
    """Same int32 bin vector for every shape of column, including a tie across a bin boundary and NaN-free infinities."""
    x = np.ascontiguousarray(_column(kind, n, np.random.default_rng(n + n_bins)))
    assert np.array_equal(_quantile_bin_njit(x, n_bins), _quantile_bin_sorted_njit(x, n_bins))


def test_nan_column_takes_the_reference_path():
    """A column holding NaN is binned by the reference (NaN sorts last), not by the selection."""
    x = np.random.default_rng(0).standard_normal(500)
    x[::13] = np.nan
    assert np.array_equal(_quantile_bin_njit(x, 10), _quantile_bin_sorted_njit(x, 10))
