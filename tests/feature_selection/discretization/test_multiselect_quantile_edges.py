"""The multiselect-based 2-D quantile-edge kernel must stay bit-identical to ``np.percentile`` and to the
per-kth ``np.partition`` kernel it replaced, across sizes, dtypes and tie structures."""

from __future__ import annotations

import numpy as np
import pytest

from mlframe.feature_selection.filters.discretization._kernels import (
    _quantile_edges_2d_njit,
    _quantile_edges_2d_njit_partition,
)
from mlframe.feature_selection.filters.discretization._multiselect import multiselect_inplace


def _kths(n: int, quantiles: np.ndarray) -> np.ndarray:
    lo = np.floor((quantiles / 100.0) * (n - 1)).astype(np.int64)
    ks = set()
    for l in lo.tolist():
        if l >= n - 1:
            ks.add(n - 1)
        else:
            ks.update((int(l), int(l) + 1))
    return np.array(sorted(ks), dtype=np.int64)


def _column(kind: str, n: int, rng) -> np.ndarray:
    if kind == "normal":
        return rng.standard_normal(n)
    if kind == "ties":
        return rng.integers(0, 3, n).astype(np.float64)
    if kind == "constant":
        return np.full(n, 2.5)
    if kind == "sorted":
        return np.sort(rng.standard_normal(n))
    if kind == "reversed":
        return np.sort(rng.standard_normal(n))[::-1].copy()
    if kind == "heavy_tail":
        return rng.standard_cauchy(n)
    raise ValueError(kind)


@pytest.mark.parametrize("kind", ["normal", "ties", "constant", "sorted", "reversed", "heavy_tail"])
@pytest.mark.parametrize("n", [2, 3, 25, 31, 450, 5000])
def test_multiselect_places_exact_order_statistics(kind, n):
    rng = np.random.default_rng(n)
    col = _column(kind, n, rng)
    for n_bins in (2, 10, 37):
        kths = _kths(n, np.linspace(0, 100, n_bins + 1))
        work = col.copy()
        multiselect_inplace(work, kths)
        np.testing.assert_array_equal(work[kths], np.sort(col)[kths])
        np.testing.assert_array_equal(np.sort(work), np.sort(col))


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("n", [5, 450, 3000])
def test_quantile_edges_kernel_bit_identical_to_percentile_and_partition_kernel(dtype, n):
    rng = np.random.default_rng(7)
    cols = [_column(k, n, rng) for k in ("normal", "ties", "constant", "sorted", "reversed", "heavy_tail")]
    arr = np.ascontiguousarray(np.column_stack(cols * 5).astype(dtype))
    for n_bins in (3, 10, 20):
        q = np.linspace(0, 100, n_bins + 1)
        kths = _kths(n, q)
        new = np.empty((q.shape[0], arr.shape[1]))
        old = np.empty_like(new)
        _quantile_edges_2d_njit(arr, q, kths, new)
        _quantile_edges_2d_njit_partition(arr, q, kths, old)
        np.testing.assert_array_equal(new, old)
        np.testing.assert_array_equal(new, np.percentile(arr, q, axis=0))


@pytest.mark.parametrize("parallel", [False, True])
@pytest.mark.parametrize("n_bins", [2, 10, 32, 33])
def test_count_le_codes_identical_to_binary_search_incl_nan_and_ties(parallel, n_bins):
    from mlframe.feature_selection.filters.discretization._kernels import (
        _count_le_2d_njit,
        _count_le_2d_njit_parallel,
        _searchsorted_2d_right_njit,
    )

    rng = np.random.default_rng(n_bins)
    n = 700
    arr = np.column_stack([_column(k, n, rng) for k in ("normal", "ties", "constant", "heavy_tail")] * 4).astype(np.float32)
    arr[rng.random(arr.shape) < 0.01] = np.nan
    edges = np.nanpercentile(arr, np.linspace(0, 100, n_bins + 1), axis=0)
    inner = np.ascontiguousarray(edges[1:-1])
    ref = np.empty(arr.shape, dtype=np.int16)
    got = np.empty_like(ref)
    _searchsorted_2d_right_njit(inner, arr, ref)
    (_count_le_2d_njit_parallel if parallel else _count_le_2d_njit)(inner, arr, got)
    np.testing.assert_array_equal(got, ref)
    for j in range(arr.shape[1]):
        np.testing.assert_array_equal(ref[:, j], np.searchsorted(inner[:, j], arr[:, j], side="right"))


@pytest.mark.parametrize("parallel", [False, True])
def test_discretize_2d_batch_matches_per_column_1d_path(parallel):
    from mlframe.feature_selection.filters.discretization import discretize_2d_quantile_batch, discretize_array

    rng = np.random.default_rng(3)
    arr = np.column_stack([_column(k, 450, rng) for k in ("normal", "ties", "constant", "sorted", "heavy_tail")] * 3).astype(np.float32)
    for n_bins in (4, 10, 40):
        got = discretize_2d_quantile_batch(arr, n_bins=n_bins, dtype=np.int16, parallel=parallel, assume_finite=True)
        for j in range(arr.shape[1]):
            np.testing.assert_array_equal(got[:, j], discretize_array(arr=arr[:, j], n_bins=n_bins, method="quantile", dtype=np.int16))
