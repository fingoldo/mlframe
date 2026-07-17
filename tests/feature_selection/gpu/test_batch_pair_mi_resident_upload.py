"""RESIDENT UPLOAD (2026-07-12): ``batch_pair_mi_cuda``/``batch_pair_mi_cupy`` must upload the
fit-constant ``factors_data``/``classes_y``/``freqs_y``/``nbins`` ONCE across repeated calls with the
SAME content (successive pair-chunks of one greedy round), instead of a fresh ``to_device``/``cp.asarray``
every call. Proves the ``resident_operand`` adoption fix in ``_batch_pair_mi_cuda_kernels.py`` /
``batch_pair_mi_gpu.py`` engages and stays bit-identical to the pre-fix raw-upload path.
"""

from __future__ import annotations

import itertools

import numpy as np
import pytest

from mlframe.feature_selection.filters.batch_pair_mi_gpu import (
    _CUDA_AVAIL,
    _CUPY_AVAIL,
    batch_pair_mi_cuda,
    batch_pair_mi_cupy,
    batch_pair_mi_njit_prange,
)
from mlframe.feature_selection.filters._fe_resident_operands import clear_fe_resident_operands


@pytest.fixture(autouse=True)
def _clear_resident_cache():
    """Clear resident cache."""
    clear_fe_resident_operands()
    yield
    clear_fe_resident_operands()


def _build_pair_inputs(n_samples, nbins_per_col, n_classes_y, seed):
    """Build pair inputs."""
    rng = np.random.default_rng(seed)
    cols = [rng.integers(0, nb, size=n_samples) for nb in nbins_per_col]
    data = np.column_stack(cols).astype(np.int32)
    nbins = np.asarray(nbins_per_col, dtype=np.int32)
    y = rng.integers(0, n_classes_y, size=n_samples).astype(np.int32)
    freqs_y = np.bincount(y, minlength=n_classes_y).astype(np.float64) / n_samples
    pairs = list(itertools.combinations(range(len(nbins_per_col)), 2))
    pair_a = np.array([p[0] for p in pairs], dtype=np.int64)
    pair_b = np.array([p[1] for p in pairs], dtype=np.int64)
    return data, nbins, y, freqs_y, pair_a, pair_b


@pytest.mark.skipif(not _CUDA_AVAIL, reason="numba.cuda not available on this host")
def test_batch_pair_mi_cuda_uploads_factors_data_once_across_calls():
    """Two calls with the SAME (data, nbins, classes_y, freqs_y) but DIFFERENT pair chunks (mirrors two
    pair-subchunks of one greedy round) must upload factors_data via cp.asarray only ONCE."""
    import cupy as cp

    data, nbins, y, freqs_y, pair_a, pair_b = _build_pair_inputs(2000, [5, 5, 5, 5, 5, 5], 4, seed=3)
    half = len(pair_a) // 2
    assert half > 0

    upload_calls = {"n": 0}
    orig_asarray = cp.asarray

    def _counting_asarray(arr, *a, **kw):
        """Counting asarray."""
        if getattr(arr, "shape", None) == data.shape:
            upload_calls["n"] += 1
        return orig_asarray(arr, *a, **kw)

    cp.asarray = _counting_asarray
    try:
        mi1 = batch_pair_mi_cuda(data, pair_a[:half], pair_b[:half], nbins, y, freqs_y)
        mi2 = batch_pair_mi_cuda(data, pair_a[half:], pair_b[half:], nbins, y, freqs_y)
    finally:
        cp.asarray = orig_asarray

    assert upload_calls["n"] == 1, f"factors_data-shaped cp.asarray called {upload_calls['n']} times across 2 calls (expected 1)"

    mi_cpu = batch_pair_mi_njit_prange(data, pair_a, pair_b, nbins, y, freqs_y)
    np.testing.assert_allclose(np.concatenate([mi1, mi2]), mi_cpu, atol=1e-9, rtol=1e-9)


@pytest.mark.skipif(not _CUPY_AVAIL, reason="cupy not available on this host")
def test_batch_pair_mi_cupy_uploads_factors_data_once_across_calls():
    """Batch pair mi cupy uploads factors data once across calls."""
    import cupy as cp

    data, nbins, y, freqs_y, pair_a, pair_b = _build_pair_inputs(2000, [5, 5, 5, 5, 5, 5], 4, seed=4)
    half = len(pair_a) // 2
    assert half > 0

    upload_calls = {"n": 0}
    orig_asarray = cp.asarray

    def _counting_asarray(arr, *a, **kw):
        """Counting asarray."""
        if getattr(arr, "shape", None) == data.shape:
            upload_calls["n"] += 1
        return orig_asarray(arr, *a, **kw)

    cp.asarray = _counting_asarray
    try:
        mi1 = batch_pair_mi_cupy(data, pair_a[:half], pair_b[:half], nbins, y, freqs_y)
        mi2 = batch_pair_mi_cupy(data, pair_a[half:], pair_b[half:], nbins, y, freqs_y)
    finally:
        cp.asarray = orig_asarray

    assert upload_calls["n"] == 1, f"factors_data-shaped cp.asarray called {upload_calls['n']} times across 2 calls (expected 1)"

    mi_cpu = batch_pair_mi_njit_prange(data, pair_a, pair_b, nbins, y, freqs_y)
    np.testing.assert_allclose(np.concatenate([mi1, mi2]), mi_cpu, atol=1e-9, rtol=1e-9)


@pytest.mark.skipif(not _CUDA_AVAIL, reason="numba.cuda not available on this host")
def test_batch_pair_mi_cuda_and_cupy_share_one_resident_upload():
    """Cross-backend dedup: resident_operand keys on CONTENT, not role/backend label -- a cuda call
    followed by a cupy call on the SAME factors_data content must hit the SAME cached device buffer."""
    from mlframe.feature_selection.filters._fe_resident_operands import _FE_RESIDENT_OPERANDS

    data, nbins, y, freqs_y, pair_a, pair_b = _build_pair_inputs(1500, [4, 4, 4, 4], 3, seed=5)

    n_before = len(_FE_RESIDENT_OPERANDS)
    batch_pair_mi_cuda(data, pair_a, pair_b, nbins, y, freqs_y)
    n_after_cuda = len(_FE_RESIDENT_OPERANDS)
    batch_pair_mi_cupy(data, pair_a, pair_b, nbins, y, freqs_y)
    n_after_cupy = len(_FE_RESIDENT_OPERANDS)

    assert n_after_cuda > n_before
    assert n_after_cupy == n_after_cuda, "cupy call added new cache entries instead of hitting the cuda call's resident uploads"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s", "--tb=short"])
