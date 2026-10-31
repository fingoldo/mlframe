"""``discretize_2d_quantile_batch`` on a frame where only some columns hold a NaN equals the per-column nanpercentile + searchsorted reference."""

import numpy as np
import pytest

from mlframe.feature_selection.filters.discretization import discretize_2d_quantile_batch


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_mixed_nan_and_clean_columns_match_reference(dtype):
    """Clean columns take the sort kernels, NaN columns take nanpercentile; every column equals its own reference."""
    rng = np.random.default_rng(3)
    arr = rng.normal(size=(3000, 12)).astype(dtype)
    arr[:, 4] = np.round(arr[:, 4], 1)
    arr[rng.random(3000) < 0.1, 2] = np.nan
    arr[rng.random(3000) < 0.3, 9] = np.nan
    got = discretize_2d_quantile_batch(arr, n_bins=8)
    q = np.linspace(0, 100, 9)
    for j in range(arr.shape[1]):
        edges = np.nanpercentile(arr[:, j], q)
        np.testing.assert_array_equal(got[:, j], np.searchsorted(edges[1:-1], arr[:, j], side="right"))


def test_all_nan_columns_and_no_nan_columns():
    """Every column with a NaN and a frame with none both stay reference-equal."""
    rng = np.random.default_rng(4)
    arr = rng.normal(size=(500, 3)).astype(np.float32)
    ref_clean = discretize_2d_quantile_batch(arr, n_bins=5, assume_finite=True)
    np.testing.assert_array_equal(discretize_2d_quantile_batch(arr, n_bins=5), ref_clean)
    arr[0, :] = np.nan
    got = discretize_2d_quantile_batch(arr, n_bins=5)
    q = np.linspace(0, 100, 6)
    for j in range(3):
        np.testing.assert_array_equal(got[:, j], np.searchsorted(np.nanpercentile(arr[:, j], q)[1:-1], arr[:, j], side="right"))
