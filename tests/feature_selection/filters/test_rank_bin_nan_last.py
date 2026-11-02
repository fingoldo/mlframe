"""CPU rank binning orders NaN last like numpy and cupy, so the GPU twin's byte-identity contract holds with NaN (tracker 2026-11-02 wave 12, PAR12D-2)."""

import numpy as np

from mlframe.feature_selection.filters._numba_utils import _quantile_bin_njit


def _numpy_reference(x, n_bins):
    """Equi-frequency bins from a stable numpy sort (NaN last)."""
    n = x.shape[0]
    order = np.argsort(x, kind="stable")
    out = np.empty(n, dtype=np.int32)
    base, rem = divmod(n, n_bins)
    pos = 0
    for b in range(n_bins):
        size = base + (1 if b < rem else 0)
        out[order[pos : pos + size]] = b
        pos += size
    return out


def test_nan_rows_land_in_the_top_bin_like_numpy():
    """x = arange(100) with NaN at rows 3 and 50: the NaN rows got bin 0 on the CPU and shifted every boundary by two."""
    x = np.arange(100.0)
    x[3] = x[50] = np.nan
    np.testing.assert_array_equal(_quantile_bin_njit(x, 10), _numpy_reference(x, 10))


def test_random_nan_and_tied_columns_match_numpy():
    """Tie-heavy columns with scattered NaN agree row for row."""
    rng = np.random.default_rng(0)
    for _ in range(10):
        x = rng.integers(0, 6, 400).astype(float)
        x[rng.choice(400, 5, replace=False)] = np.nan
        np.testing.assert_array_equal(_quantile_bin_njit(x, 8), _numpy_reference(x, 8))
