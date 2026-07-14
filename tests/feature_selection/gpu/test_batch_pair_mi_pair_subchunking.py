"""Regression tests for pair-dimension chunking in ``batch_pair_mi_cuda_row_chunked``.

Root cause (2026-07-10 wellbore 100k-row profiling round 2): the row-chunked kernel chunked ROWS but
sized its histogram accumulator by the FULL ``n_pairs`` passed to one call
(``n_pairs * max_joint * n_classes_y * 8`` bytes) -- at a real production shape (86,736 pairs,
max_joint=441, n_classes_y=20) that accumulator alone needs ~6GB, bigger than the entire 4GB card,
REGARDLESS of how small the row-chunk was. cProfile showed the resulting allocation (probably silently
WDDM-oversubscribed, the exact hazard the whole VRAM-guard chain exists to avoid) thrashing: 165
``to_device`` calls averaging 4.6s each (an isolated microbench on the same host/shape shows ~1-30ms),
772s of a 1633s wall -- the single largest hotspot in the run.

The fix adds a SECOND, independent chunking dimension: pair-subchunks (``_choose_pair_subchunk_rows``)
as the outer loop, row-chunks as the inner loop, each with its own dedicated VRAM budget fraction so
neither dimension's allocation can starve or exceed the other. These tests pin: the accumulator is
never sized beyond a bounded pair-subchunk (not the full n_pairs), correctness is preserved across the
pair-subchunk boundary (results for pairs split across different subchunks must match a single-chunk
reference), and the fix is measured against the exact failing production shape.
"""
from __future__ import annotations

import itertools

import numpy as np
import pytest

import mlframe.feature_selection.filters.batch_pair_mi_gpu as bpmg
import mlframe.feature_selection.filters._batch_pair_mi_cuda_kernels as bpmk


def _build_pair_inputs(n_samples, n_cols, nbins_val, n_classes_y, seed=0):
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


def test_choose_pair_subchunk_rows_bounds_accumulator_to_budget():
    """The chosen pair-subchunk size must keep the accumulator within its dedicated VRAM budget
    fraction, never scaling with the FULL n_pairs regardless of how many pairs are ultimately passed."""
    max_joint, n_classes_y = 441, 20
    free_bytes = int(1.87 * 1024**3)  # the exact free_vram observed in the failing production log
    per_pair_bytes = max_joint * n_classes_y * 8 + max_joint * 8

    n = bpmg._choose_pair_subchunk_rows(max_joint, n_classes_y, free_bytes)
    accumulator_bytes = n * per_pair_bytes
    assert accumulator_bytes <= free_bytes * 0.2 * 1.01, (
        f"accumulator for the chosen pair-subchunk size ({accumulator_bytes} bytes) must stay within "
        f"its dedicated budget fraction, not grow with an unbounded total pair count"
    )
    # The bug this fixes: for the REAL failing n_pairs=86736, the OLD design's accumulator would be
    # 86736 * per_pair_bytes -- verify that is indeed far larger than the budget (confirms this is a
    # real, not hypothetical, over-allocation the fix prevents).
    old_unbounded_accumulator = 86736 * per_pair_bytes
    assert old_unbounded_accumulator > free_bytes, (
        "sanity check: the pre-fix unbounded accumulator size must exceed total free VRAM at the "
        "real failing shape, confirming this test targets a genuine (not hypothetical) over-allocation"
    )


def test_choose_pair_subchunk_rows_clamped_to_at_least_one():
    """Even a pathologically large max_joint/n_classes_y must still yield >=1 (never 0, which would
    infinite-loop the outer pair-subchunk range())."""
    n = bpmg._choose_pair_subchunk_rows(max_joint=1024, n_classes_y=16, free_bytes=1024)
    assert n >= 1


@pytest.mark.skipif(not bpmg._CUDA_AVAIL or not bpmk._CUPY_AVAIL, reason="numba.cuda/cupy not available")
def test_pair_subchunked_finalize_avoids_full_accumulator_readback(monkeypatch):
    """Regression: device-side finalize must avoid copy_to_host on the histogram accumulator."""
    data, nbins, classes_y, freqs_y, pair_a, pair_b = _build_pair_inputs(
        n_samples=2000, n_cols=30, nbins_val=6, n_classes_y=4, seed=5,
    )
    monkeypatch.setattr(bpmk, "_choose_pair_subchunk_rows", lambda *a, **kw: 50)
    import numba.cuda.cudadrv.devicearray as _devicearray
    calls = {"n": 0}
    orig_copy_to_host = _devicearray.DeviceNDArray.copy_to_host

    def _spy_copy_to_host(self, *a, **kw):
        if getattr(self, "ndim", 0) >= 2 and getattr(self, "dtype", None) is not None and "int64" in str(self.dtype):
            calls["n"] += 1
        return orig_copy_to_host(self, *a, **kw)

    monkeypatch.setattr(_devicearray.DeviceNDArray, "copy_to_host", _spy_copy_to_host)
    mi_gpu = bpmg.batch_pair_mi_cuda_row_chunked(data, pair_a, pair_b, nbins, classes_y, freqs_y)
    assert calls["n"] == 0
    mi_cpu = bpmg.batch_pair_mi_njit_prange(data, pair_a, pair_b, nbins, classes_y, freqs_y)
    np.testing.assert_allclose(mi_gpu, mi_cpu, atol=1e-9, rtol=1e-9)


@pytest.mark.skipif(not bpmg._CUDA_AVAIL, reason="numba.cuda not available on this host")
def test_near_zero_free_vram_bails_to_cpu_instead_of_71_million_launches(monkeypatch):
    """Regression test: a real 1M-row wellbore run hit ``free_vram=0.00GB`` (a cupy pool cap + other
    resident allocations left almost nothing free) at the exact production shape from the accumulator
    bug above (89,676 pairs). Both chunk-size choosers independently clamp to their SAFE-but-degenerate
    floor in that case (row_chunk_rows=1000, pair_subchunk_rows=1) -- individually correct (never
    over-allocates), but their PRODUCT was ~71 MILLION kernel launches: 796 row-chunks x 89,676
    pair-subchunks, each paying real upload+dispatch overhead, which would have taken many HOURS for
    work the CPU kernel finishes in seconds. This must raise (not silently grind) so the caller's
    existing fallback (``dispatch_batch_pair_mi``'s ``_try_cuda_row_chunked`` -> CPU njit) takes over
    immediately."""
    data, nbins, classes_y, freqs_y, pair_a, pair_b = _build_pair_inputs(
        n_samples=5000, n_cols=50, nbins_val=21, n_classes_y=20, seed=1,
    )
    monkeypatch.setattr(bpmk, "_choose_row_chunk_rows", lambda *a, **kw: 1000)
    monkeypatch.setattr(bpmk, "_choose_pair_subchunk_rows", lambda *a, **kw: 1)

    with pytest.raises(RuntimeError, match="too fragmented to be worthwhile"):
        bpmg.batch_pair_mi_cuda_row_chunked(data, pair_a, pair_b, nbins, classes_y, freqs_y)


@pytest.mark.skipif(not bpmg._CUDA_AVAIL, reason="numba.cuda not available on this host")
def test_near_zero_free_vram_dispatch_falls_back_to_cpu_end_to_end(monkeypatch):
    """End-to-end: when the launch-count guard trips, ``dispatch_batch_pair_mi`` must still return a
    correct result via the CPU njit fallback, not propagate the RuntimeError to the caller."""
    data, nbins, classes_y, freqs_y, pair_a, pair_b = _build_pair_inputs(
        n_samples=5000, n_cols=50, nbins_val=21, n_classes_y=20, seed=1,
    )
    mi_ref = bpmg.batch_pair_mi_njit_prange(data, pair_a, pair_b, nbins, classes_y, freqs_y)

    monkeypatch.setattr(bpmk, "_choose_row_chunk_rows", lambda *a, **kw: 1000)
    monkeypatch.setattr(bpmk, "_choose_pair_subchunk_rows", lambda *a, **kw: 1)
    monkeypatch.setattr(bpmg, "_gpu_upload_fits", lambda *a, **kw: False)

    mi, backend = bpmg.dispatch_batch_pair_mi(data, pair_a, pair_b, nbins, classes_y, freqs_y, force_backend="cuda")

    assert backend == "njit"
    np.testing.assert_allclose(mi, mi_ref, rtol=1e-6, atol=1e-9)


@pytest.mark.skipif(not bpmg._CUDA_AVAIL, reason="numba.cuda not available on this host")
def test_row_chunked_matches_reference_when_pair_subchunking_is_forced():
    """Force pair-subchunking to actually fire (tiny pair_subchunk_rows) and confirm the result still
    matches a single-shot reference -- i.e. splitting pairs across accumulator boundaries doesn't lose
    or corrupt any pair's MI."""
    data, nbins, classes_y, freqs_y, pair_a, pair_b = _build_pair_inputs(
        n_samples=2000, n_cols=10, nbins_val=5, n_classes_y=4, seed=7,
    )
    mi_ref = bpmg.batch_pair_mi_njit_prange(data, pair_a, pair_b, nbins, classes_y, freqs_y)

    orig_pair_fn = bpmg._choose_pair_subchunk_rows
    orig_row_fn = bpmg._choose_row_chunk_rows
    bpmg._choose_pair_subchunk_rows = lambda *a, **kw: 3  # forces many tiny pair-subchunks
    bpmg._choose_row_chunk_rows = lambda *a, **kw: 500  # also forces multiple row-chunks
    try:
        mi_chunked = bpmg.batch_pair_mi_cuda_row_chunked(data, pair_a, pair_b, nbins, classes_y, freqs_y)
    finally:
        bpmg._choose_pair_subchunk_rows = orig_pair_fn
        bpmg._choose_row_chunk_rows = orig_row_fn

    np.testing.assert_allclose(
        mi_chunked, mi_ref, rtol=1e-6, atol=1e-9,
        err_msg="pair-subchunked + row-chunked result must match the CPU reference even when both "
                "dimensions are forced to chunk aggressively",
    )


@pytest.mark.skipif(not bpmg._CUDA_AVAIL, reason="numba.cuda not available on this host")
def test_real_hardware_production_shape_completes_without_oom_and_matches_cpu():
    """End-to-end real-hardware validation at the exact production shape that caused the 772s hotspot
    (n_samples~79k, n_pairs~86736 via n_cols=417, n_classes_y=20, max_joint~441 via nbins=21). Must
    complete (no OOM/thrash) and match the CPU reference. Scaled n_samples down slightly to keep the
    test fast while preserving the pair/class/joint shape that triggered the bug."""
    data, nbins, classes_y, freqs_y, pair_a, pair_b = _build_pair_inputs(
        n_samples=8000, n_cols=417, nbins_val=21, n_classes_y=20, seed=3,
    )
    mi_gpu = bpmg.batch_pair_mi_cuda_row_chunked(data, pair_a, pair_b, nbins, classes_y, freqs_y)
    mi_cpu = bpmg.batch_pair_mi_njit_prange(data, pair_a, pair_b, nbins, classes_y, freqs_y)
    assert np.all(np.isfinite(mi_gpu))
    np.testing.assert_allclose(mi_gpu, mi_cpu, rtol=1e-6, atol=1e-9)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short", "-x", "--no-cov"])
