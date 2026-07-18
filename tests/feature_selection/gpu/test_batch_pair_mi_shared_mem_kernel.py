"""Regression tests for the shared-memory-staged histogram kernel in ``batch_pair_mi_cuda_row_chunked``.

Root cause (2026-07-10 wellbore 100k-row ncu profiling, admin-elevated to get past ``ERR_NVGPUCTRPERM``):
the row-chunked kernel's plain global-atomics design achieved 99.22% occupancy but only 2.53% compute
throughput -- ncu's Warp State Statistics attributed 93.5% of the 703.7-cycle average stall between
issued instructions to "LG Throttle" (the L1 instruction queue for local/global memory operations
staying full), with ncu's own guidance reading "avoid redundant global memory accesses... combine
multiple lower-width memory operations into fewer wider memory operations". Each thread issued 2 GLOBAL
atomic adds PER ROW processed (up to chunk_rows per block -- 79,237 at the profiled shape), a volume the
LG queue could not keep up with.

The fix (:func:`_cuda_hist_kernel_shared_factory`) mirrors what :func:`_cuda_kernel_factory`'s original
non-chunked kernel already does: stage counts in a per-block dynamic SHARED-memory buffer (int32, fast,
on-chip) and flush to the persistent global int64 accumulator once per histogram cell, cutting global
atomic traffic from O(chunk_rows) to O(max_joint*n_classes_y) per block. Gated by
:func:`_hist_kernel_shared_fits_budget` on the device's actual per-block shared-memory budget; falls
back to the plain global-atomics kernel when the histogram is too large to stage (never over-allocates
shared memory, never a correctness risk). Measured 1.34x faster end-to-end at the production shape
(n_samples=79,237, one pair-subchunk of 5,427 pairs, max_joint=441, n_classes_y=20: 2.157s -> 1.613s),
bit-identical to both the CPU reference and the global-only kernel.
"""

from __future__ import annotations

import itertools

import numpy as np
import pytest

import mlframe.feature_selection.filters.batch_pair_mi_gpu as bpmg


def _build_pair_inputs(n_samples, n_cols, nbins_val, n_classes_y, seed=0):
    """Build pair inputs."""
    rng = np.random.default_rng(seed)
    cols = [rng.integers(0, nbins_val, size=n_samples) for _ in range(n_cols)]
    data = np.column_stack(cols).astype(np.int32)
    nbins = np.full(n_cols, nbins_val, dtype=np.int32)
    y = rng.integers(0, n_classes_y, size=n_samples).astype(np.int32)
    freqs_y = np.bincount(y, minlength=n_classes_y).astype(np.float64) / n_samples
    pairs = list(itertools.combinations(range(n_cols), 2))
    pair_a = np.array([p[0] for p in pairs], dtype=np.int64)
    pair_b = np.array([p[1] for p in pairs], dtype=np.int64)
    return data, nbins, y, freqs_y, pair_a, pair_b


def test_hist_kernel_shared_fits_budget_at_production_shape():
    """The production shape from the wellbore profiling run (max_joint=441, n_classes_y=20) must fit
    the shared-memory budget -- this is the shape the fix targets."""
    n = bpmg._hist_kernel_shared_fits_budget(max_joint=441, n_classes_y=20)
    assert n > 0
    assert n == (441 * 20 + 441) * 4


def test_hist_kernel_shared_fits_budget_rejects_oversized_histogram():
    """A pathologically large max_joint*n_classes_y combination must be rejected (0 bytes), not
    silently truncated or over-allocated."""
    n = bpmg._hist_kernel_shared_fits_budget(max_joint=4000, n_classes_y=30)
    assert n == 0


@pytest.mark.skipif(not bpmg._CUDA_AVAIL, reason="numba.cuda not available on this host")
def test_shared_kernel_matches_reference_at_production_like_shape():
    """Shared kernel matches reference at production like shape."""
    data, nbins, classes_y, freqs_y, pair_a, pair_b = _build_pair_inputs(
        n_samples=8000,
        n_cols=100,
        nbins_val=21,
        n_classes_y=20,
        seed=3,
    )
    assert bpmg._hist_kernel_shared_fits_budget(21 * 21, 20) > 0, "sanity: gate must fire at this shape"

    mi_ref = bpmg.batch_pair_mi_njit_prange(data, pair_a, pair_b, nbins, classes_y, freqs_y)
    mi_gpu = bpmg.batch_pair_mi_cuda_row_chunked(data, pair_a, pair_b, nbins, classes_y, freqs_y)
    np.testing.assert_allclose(mi_gpu, mi_ref, rtol=1e-6, atol=1e-9)


@pytest.mark.skipif(not bpmg._CUDA_AVAIL, reason="numba.cuda not available on this host")
def test_shared_kernel_matches_reference_across_pair_and_row_subchunk_boundaries():
    """Force BOTH pair-subchunking and row-chunking so the shared-memory kernel is re-launched (and
    re-zeroed) many times per accumulator -- confirms no state leaks across chunk boundaries."""
    data, nbins, classes_y, freqs_y, pair_a, pair_b = _build_pair_inputs(
        n_samples=4000,
        n_cols=60,
        nbins_val=17,
        n_classes_y=12,
        seed=9,
    )
    mi_ref = bpmg.batch_pair_mi_njit_prange(data, pair_a, pair_b, nbins, classes_y, freqs_y)

    orig_pair = bpmg._choose_pair_subchunk_rows
    orig_row = bpmg._choose_row_chunk_rows
    bpmg._choose_pair_subchunk_rows = lambda *a, **kw: 40
    bpmg._choose_row_chunk_rows = lambda *a, **kw: 500
    try:
        mi_gpu = bpmg.batch_pair_mi_cuda_row_chunked(data, pair_a, pair_b, nbins, classes_y, freqs_y)
    finally:
        bpmg._choose_pair_subchunk_rows = orig_pair
        bpmg._choose_row_chunk_rows = orig_row

    np.testing.assert_allclose(mi_gpu, mi_ref, rtol=1e-6, atol=1e-9)


@pytest.mark.skipif(not bpmg._CUDA_AVAIL, reason="numba.cuda not available on this host")
def test_falls_back_to_global_kernel_when_histogram_too_large_for_shared_mem():
    """A max_joint x n_classes_y combination too large for shared memory must still produce a correct
    result via the global-atomics fallback kernel, not raise or silently corrupt."""
    data, nbins, classes_y, freqs_y, pair_a, pair_b = _build_pair_inputs(
        n_samples=3000,
        n_cols=30,
        nbins_val=63,
        n_classes_y=30,
        seed=1,
    )
    assert bpmg._hist_kernel_shared_fits_budget(63 * 63, 30) == 0, "sanity: gate must reject this shape"

    mi_ref = bpmg.batch_pair_mi_njit_prange(data, pair_a, pair_b, nbins, classes_y, freqs_y)
    mi_gpu = bpmg.batch_pair_mi_cuda_row_chunked(data, pair_a, pair_b, nbins, classes_y, freqs_y)
    np.testing.assert_allclose(mi_gpu, mi_ref, rtol=1e-6, atol=1e-9)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short", "-x", "--no-cov"])
