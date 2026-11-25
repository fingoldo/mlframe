"""The row-chunked pair MI widens its codes one row chunk at a time (PLAN-10: the whole-matrix int32 copy was 4.5 GB for 1000000 x 1130 int8 codes, the jump that stopped the first 1M x 1000 run)."""

from __future__ import annotations

import numpy as np
import pytest

import mlframe.feature_selection.filters.batch_pair_mi_gpu as bpmg
import mlframe.feature_selection.filters._batch_pair_mi_cuda_kernels as bpmk


def _inputs(n=6000, cols=12, nb=5, ny=3, seed=0):
    """Small code matrix, the target, the cardinalities and every column pair."""
    rng = np.random.default_rng(seed)
    data = rng.integers(0, nb, size=(n, cols)).astype(np.int8)
    y = rng.integers(0, ny, size=n).astype(np.int32)
    nbins = np.full(cols, nb, dtype=np.int64)
    freqs_y = np.bincount(y, minlength=ny).astype(np.float64) / n
    a, b = np.triu_indices(cols, k=1)
    return data, nbins, y, freqs_y, a.astype(np.int64), b.astype(np.int64)


@pytest.mark.skipif(not bpmg._CUDA_AVAIL, reason="numba.cuda not available on this host")
def test_int8_codes_give_the_int32_result_and_are_never_widened_whole(monkeypatch):
    """Row chunks forced small: int8 input equals int32 input equals the CPU kernel, and no int32 copy of the whole matrix is made."""
    data, nbins, y, freqs_y, a, b = _inputs()
    monkeypatch.setattr(bpmk, "_choose_row_chunk_rows", lambda *args, **kw: 1000)
    widened = []
    real = np.ascontiguousarray

    def spy(arr, dtype=None):
        """Record every int32 widening of a 2-D block as large as the whole matrix."""
        if dtype is not None and np.dtype(dtype) == np.int32 and getattr(arr, "ndim", 0) == 2 and arr.shape[0] >= data.shape[0]:
            widened.append(arr.shape)
        return real(arr, dtype=dtype) if dtype is not None else real(arr)

    monkeypatch.setattr(bpmk.np, "ascontiguousarray", spy)
    got8 = bpmg.batch_pair_mi_cuda_row_chunked(data, a, b, nbins, y, freqs_y)
    monkeypatch.undo()
    assert not widened, f"the whole code matrix was widened: {widened}"
    got32 = bpmg.batch_pair_mi_cuda_row_chunked(data.astype(np.int32), a, b, nbins, y, freqs_y)
    ref = bpmg.batch_pair_mi_njit_prange(data, a, b, nbins, y, freqs_y)
    np.testing.assert_allclose(got8, got32, atol=1e-12, rtol=0)
    np.testing.assert_allclose(got8, ref, atol=1e-9, rtol=1e-9)
