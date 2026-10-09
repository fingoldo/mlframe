"""The usability pair-combo MI table is the same number whether the candidate chunk is built column-major or row-major."""

from __future__ import annotations

import numpy as np
import pytest

cp = pytest.importorskip("cupy")

from mlframe.feature_selection.filters import _usability_pool_resident as pool
from mlframe.feature_selection.filters import _usability_pool_resident_ktc as ktc
from mlframe.feature_selection.filters._fe_batched_mi import binned_mm_mi_from_values_gpu
from mlframe.feature_selection.filters._fe_batched_mi_mm_cm import binned_mm_mi_from_cm_gpu
from mlframe.feature_selection.filters._gpu_resident_select import _radix_select_interior_edges


def _table(monkeypatch, cm, n, npairs, ncombos):
    """The resident table computed with the chosen layout."""
    monkeypatch.setenv("MLFRAME_FE_POOL_CM", "1" if cm else "0")
    args = ktc._make_pooltable_inputs({"n_rows": n, "npairs": npairs, "n_combos": ncombos})
    return pool.score_pair_combos_table_resident(*args)


@pytest.mark.parametrize("n,npairs,ncombos", [(5_000, 2, 96), (20_000, 3, 289), (3_001, 1, 578)])
def test_table_is_identical_in_both_layouts(monkeypatch, n, npairs, ncombos):
    """Same values, same edges, same integer counts: the tables are equal bit for bit (including the -1 sentinel rows)."""
    row = _table(monkeypatch, False, n, npairs, ncombos)
    col = _table(monkeypatch, True, n, npairs, ncombos)
    assert row is not None and col is not None and row.shape == col.shape
    np.testing.assert_array_equal(col, row)


@pytest.mark.parametrize("dtype", [np.float64, np.float32])
def test_cm_mm_kernel_matches_the_row_major_one(dtype):
    """Directly on a block with ties, constants and a low-cardinality column."""
    rng = np.random.default_rng(0)
    n, K = 30_000, 24
    X = rng.normal(size=(n, K)).astype(dtype)
    X[:, 0] = 1.0
    X[:, 1] = rng.integers(0, 3, size=n)
    X[:, 2] = np.round(X[:, 2], 1)
    y = rng.integers(0, 6, size=n).astype(np.int64)
    d_y = cp.asarray(y)
    d_x = cp.asarray(X)
    edges = _radix_select_interior_edges(d_x, 10)
    hy = float(-(np.bincount(y) / n * np.log(np.bincount(y) / n)).sum())
    row = binned_mm_mi_from_values_gpu(d_x, edges, d_y, 10, 6, hy, 6, codes_trusted=True)
    d_cm = cp.ascontiguousarray(d_x.T)
    edges_cm = _radix_select_interior_edges(d_cm, 10, data_is_cm=True)
    np.testing.assert_array_equal(cp.asnumpy(edges_cm), cp.asnumpy(edges))
    col = binned_mm_mi_from_cm_gpu(d_cm, edges_cm, d_y, 10, 6, hy, 6, codes_trusted=True)
    np.testing.assert_array_equal(col, row)


def test_oversized_histogram_returns_none():
    """A target with too many classes for the shared tile declines like the row-major function."""
    d = cp.asarray(np.random.default_rng(1).normal(size=(4, 1000)))
    e = cp.asarray(np.sort(np.random.default_rng(2).normal(size=(9, 4)), axis=0))
    assert binned_mm_mi_from_cm_gpu(d, e, cp.zeros(1000, dtype=cp.int64), 10, 10_000, 0.0, 1) is None
