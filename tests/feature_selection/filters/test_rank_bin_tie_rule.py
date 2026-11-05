"""A run of equal values is one group for the rank binners: it takes the bin of its first sorted position and is never split.

The earlier rule placed equal values across bins by row order, so a constant column with row-ordered labels was split into
bins and the MI of that split read the row order as signal (ln 2 on the reproducer). The CPU fast path, the CPU sorted
reference and the GPU rank binners must agree on every tied column.
"""

from __future__ import annotations

import numpy as np
import pytest

from mlframe.feature_selection.filters._numba_utils import _quantile_bin_njit, _quantile_bin_sorted_njit


def _tied_columns(seed: int = 0):
    """Five tie-heavy columns: continuous with a point mass at zero, integer-valued, all-zero, and integer-valued with scattered NaN."""
    rng = np.random.default_rng(seed)
    n = 800
    cols = [
        rng.normal(size=n),
        rng.integers(0, 4, size=n).astype(float),
        np.where(rng.random(n) < 0.5, 0.0, rng.normal(size=n)),
        np.zeros(n),
    ]
    with_nan = rng.integers(0, 3, size=n).astype(float)
    with_nan[rng.random(n) < 0.1] = np.nan
    cols.append(with_nan)
    return cols


def test_constant_column_with_row_ordered_labels_stays_in_one_bin():
    """A constant column has one value, so every row lands in bin 0 whatever the labels' row order."""
    x = np.zeros(200)
    assert np.unique(_quantile_bin_sorted_njit(x, 5)).tolist() == [0]
    assert np.unique(_quantile_bin_njit(x, 5)).tolist() == [0]


@pytest.mark.parametrize("col", range(5))
def test_fast_path_matches_sorted_reference_on_tied_columns(col):
    """The fast njit path and the sorted reference assign identical codes on every tied column, NaN included."""
    x = _tied_columns()[col]
    np.testing.assert_array_equal(_quantile_bin_njit(x.copy(), 6), _quantile_bin_sorted_njit(x.copy(), 6))


@pytest.mark.parametrize("col", range(5))
def test_no_tie_group_straddles_two_bins(col):
    """Every distinct (non-NaN) value maps to exactly one bin."""
    x = _tied_columns()[col]
    codes = _quantile_bin_sorted_njit(x.copy(), 6)
    values = np.unique(x[~np.isnan(x)])
    assert values.size > 0, "the column has no non-NaN value"
    for v in values:
        assert np.unique(codes[x == v]).size == 1


def test_gpu_rank_binners_match_the_cpu_reference_on_tied_columns():
    """Single-column and batched GPU rank codes equal the CPU sorted reference on every tied column."""
    cp = pytest.importorskip("cupy")
    from mlframe.feature_selection.filters._gpu_resident_rank_bin import rank_bin_codes_batch_gpu_resident, rank_bin_codes_gpu_resident

    cols = _tied_columns()
    X = np.stack(cols, axis=1)
    cpu = np.stack([_quantile_bin_sorted_njit(c.copy(), 6) for c in cols], axis=1)
    Xg = cp.asarray(X)
    batch = cp.asnumpy(rank_bin_codes_batch_gpu_resident(Xg, 6))
    single = np.stack([cp.asnumpy(rank_bin_codes_gpu_resident(Xg[:, j], 6)) for j in range(X.shape[1])], axis=1)
    np.testing.assert_array_equal(batch, cpu)
    np.testing.assert_array_equal(single, cpu)
