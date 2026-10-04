"""Launch-geometry and index-width guards for the GPU kernels (grid limits, 64-bit indices, host-side chunking).

Pure-Python tests drive the sizing helpers with injected sizes (no gigabyte allocations); the ``gpu`` tests launch the real kernels on tiny shapes
or on shapes that only cross a launch limit (not a memory limit) and compare against NumPy.
"""
from __future__ import annotations

import numpy as np
import pytest

from mlframe.feature_selection.filters._gpu_pairs import pair_chunk_bounds
from mlframe.feature_selection.filters._gpu_resident_select_kernels import tiled_transpose_blocks
from mlframe.feature_selection.filters.discretization._discretization_cuda import _searchsorted_launch_blocks
from mlframe.feature_selection.filters.gpu import _gpu_batched_bytes_per_perm
from mlframe.feature_selection.filters._gpu_batched import cap_batch_size
from mlframe.metrics._gpu_metrics import column_slabs


def _cuda_ok() -> bool:
    """True when cupy sees a device with at least ~300 MB free."""
    try:
        import cupy as cp

        free, _total = cp.cuda.runtime.memGetInfo()
        return bool(free > 300 * 1024**2)
    except Exception:
        return False


def test_searchsorted_launch_blocks_covers_all_cells():
    """The block count times threads covers every cell, including a grid whose cell count exceeds 2**31 (the 32-bit wrap point)."""
    threads = 256
    for n_rows, n_cols in [(1, 1), (255, 1), (1000, 1000), (2_200_000, 1000)]:
        blocks = _searchsorted_launch_blocks(n_rows, n_cols, threads)
        assert blocks * threads >= n_rows * n_cols
        assert (blocks - 1) * threads < n_rows * n_cols
    assert _searchsorted_launch_blocks(2_200_000, 1000, threads) * threads > 2**31


def test_searchsorted_launch_blocks_refuses_uncoverable_grid():
    """A grid that cannot cover every cell raises instead of leaving the uninitialised output buffer partly unwritten."""
    with pytest.raises(ValueError):
        _searchsorted_launch_blocks(2**31, 256, 1)


def test_pair_chunk_bounds_partitions_without_loss():
    """Pair chunks of at most the limit cover every pair exactly once and in order, including a count above 65535."""
    for n_pairs, limit in [(0, 5), (1, 5), (5, 5), (11, 5), (79_800, 65_535)]:
        bounds = pair_chunk_bounds(n_pairs, limit)
        assert all(0 < stop - start <= limit for start, stop in bounds)
        assert [i for start, stop in bounds for i in range(start, stop)] == list(range(n_pairs))
    assert len(pair_chunk_bounds(79_800, 65_535)) == 2


def test_cap_batch_size_respects_grid_y_and_host_budget():
    """The permutation batch never exceeds 65535 (gridDim.y), the host staging budget, or half of free VRAM."""
    assert cap_batch_size(10**9, free_bytes=10**15, n=10) == 65_535
    assert cap_batch_size(64, free_bytes=10**15, n=10) == 64
    big_n = 50_000_000
    assert cap_batch_size(64, free_bytes=10**15, n=big_n) == max(1, (256 * 1024 * 1024) // (4 * big_n))
    assert cap_batch_size(64, free_bytes=3 * _gpu_batched_bytes_per_perm(1000) * 2, n=1000) == 3
    assert cap_batch_size(64, free_bytes=0, n=1000) == 1


def test_tiled_transpose_blocks_linearises_tiles():
    """The tile count is ceil(n/32)*ceil(K/32) on one grid axis, so a row count past the 2-D grid limit still fits."""
    assert tiled_transpose_blocks(32, 32) == 1
    assert tiled_transpose_blocks(33, 1) == 2
    assert tiled_transpose_blocks(2_200_000, 5) == ((2_200_000 + 31) // 32) * 1
    assert tiled_transpose_blocks(2_200_000, 5) > 65_535
    with pytest.raises(ValueError):
        tiled_transpose_blocks(2**31 * 32, 64)


def test_column_slabs_cover_columns_in_order():
    """Column slabs hold at most the limit and tile the column range exactly."""
    for n_cols, limit in [(1, 3), (3, 3), (8, 3), (65_541, 65_535)]:
        slabs = column_slabs(n_cols, limit)
        assert all(0 < stop - start <= limit for start, stop in slabs)
        assert [c for start, stop in slabs for c in range(start, stop)] == list(range(n_cols))
    assert column_slabs(0, 3) == []


@pytest.mark.parametrize("n_rows,n_cols", [(40, 7), (3, 1), (257, 5)])
@pytest.mark.gpu
@pytest.mark.skipif(not _cuda_ok(), reason="no CUDA device with free VRAM")
def test_quantile_rawkernel_matches_numpy_searchsorted(n_rows, n_cols):
    """The fused 64-bit searchsorted kernel reproduces per-column ``np.searchsorted(side='right')`` on small shapes."""
    import cupy as cp

    from mlframe.feature_selection.filters.discretization._discretization_cuda import _discretize_quantile_rawkernel

    rng = np.random.default_rng(3)
    arr = rng.normal(size=(n_rows, n_cols))
    cuts = np.sort(rng.normal(size=(n_cols, 4)), axis=1)
    got = cp.asnumpy(_discretize_quantile_rawkernel(cp.asarray(arr), cp.asarray(cuts), 5, np.int16))
    expected = np.column_stack([np.searchsorted(cuts[j], arr[:, j], side="right") for j in range(n_cols)])
    np.testing.assert_array_equal(got, expected)


@pytest.mark.parametrize("dtype", [np.float32, np.float64, np.int8, np.int16])
@pytest.mark.gpu
@pytest.mark.skipif(not _cuda_ok(), reason="no CUDA device with free VRAM")
def test_transpose_past_old_grid_limit_uses_the_kernel_not_the_fallback(dtype, monkeypatch):
    """With more than 2,097,120 rows the tiled transpose still launches (no silent fall back to the slow strided copy) and matches ``.T``."""
    import cupy as cp

    from mlframe.feature_selection.filters import _gpu_resident_select_kernels as k

    n, cols = 2_200_000, 3
    rng = np.random.default_rng(0)
    host = rng.integers(-100, 100, size=(n, cols)).astype(dtype)
    d = cp.asarray(host)
    real = cp.ascontiguousarray

    def _spy(*args, **kwargs):
        """Fail the test if the strided-copy fallback runs."""
        raise AssertionError("tiled transpose fell back to cp.ascontiguousarray")

    monkeypatch.setattr(cp, "ascontiguousarray", _spy)
    out = k.transpose_codes_to_cm(d) if np.dtype(dtype).kind == "i" else k._transpose_to_cm(d)
    monkeypatch.setattr(cp, "ascontiguousarray", real)
    np.testing.assert_array_equal(cp.asnumpy(out), host.T)
    if dtype == np.float32:
        np.testing.assert_array_equal(cp.asnumpy(k._transpose_cm_to_rm(out)), host)


@pytest.mark.gpu
@pytest.mark.skipif(not _cuda_ok(), reason="no CUDA device with free VRAM")
def test_rmse_scores_with_more_columns_than_grid_y():
    """RMSE over 65541 prediction columns (past gridDim.y) runs in column slabs and matches NumPy."""
    import cupy as cp

    from mlframe.metrics._gpu_metrics import _is_numba_cuda_available, gpu_multiple_rmse_scores

    if not _is_numba_cuda_available():
        pytest.skip("numba.cuda unavailable")
    rng = np.random.default_rng(1)
    n, m = 64, 65_541
    actual = rng.normal(size=n)
    pred = rng.normal(size=(n, m))
    got = cp.asnumpy(gpu_multiple_rmse_scores(actual, pred))
    expected = np.sqrt(np.mean((actual[:, None] - pred) ** 2, axis=0))
    np.testing.assert_allclose(got, expected, rtol=1e-9)
