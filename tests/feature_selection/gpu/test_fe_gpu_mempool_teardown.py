"""Regression: the GPU FE step must free unused cupy pool blocks at teardown so repeated fits stay bounded.

Without the teardown (``_mrmr_fe_step/_step_core._free_gpu_fe_mempool``, called at the end of ``_run_fe_step``),
the cupy default pool RETAINS every block it cached during each GPU FE step; across repeated fits (CV folds,
repeated ``MRMR().fit``) the retained pool sits at its high-water mark near the device cap and the allocator
thrashes cudaMalloc/sync -- measured pre-fix on a 4 GB GTX 1050 Ti: consecutive 100k f32 STRICT fits degrade
11.2s -> 31.8s -> 32.3s. The fix frees the pool's UNUSED blocks at each FE-step teardown (post-compute, never
mid-pipeline; live resident buffers, which keep a reference, are untouched), holding the footprint flat (~2.9 GB)
and the wall at ~11.2s.

The pre/post difference is the absolute plateau + wall, both device/contention-sensitive (the pool reaches its
high-water mark on the FIRST fit, so a growth-rate or wall-ratio assertion does not discriminate), and a raw
``free_all_blocks`` spy cannot tell this teardown apart from the batch executor's own cleanup. So this targets
the dedicated helper directly: it issues a pool free under the GPU flags, frees real retained blocks, and is
inert (no pool touch) when neither GPU flag is set. Removing the teardown call or mis-gating it fails this."""

from __future__ import annotations

import pytest

cp = pytest.importorskip("cupy")

from mlframe.feature_selection.filters._mrmr_fe_step._step_core import _free_gpu_fe_mempool


def test_helper_inert_without_gpu_flags(monkeypatch):
    monkeypatch.delenv("MLFRAME_FE_GPU_STRICT", raising=False)
    monkeypatch.delenv("MLFRAME_CMI_GPU", raising=False)
    assert _free_gpu_fe_mempool() is False


def test_helper_issues_free_under_strict(monkeypatch):
    monkeypatch.setenv("MLFRAME_FE_GPU_STRICT", "1")
    monkeypatch.delenv("MLFRAME_CMI_GPU", raising=False)
    assert _free_gpu_fe_mempool() is True


def test_helper_issues_free_under_cmi_gpu(monkeypatch):
    monkeypatch.delenv("MLFRAME_FE_GPU_STRICT", raising=False)
    monkeypatch.setenv("MLFRAME_CMI_GPU", "1")
    assert _free_gpu_fe_mempool() is True


def _need_cuda() -> bool:
    try:
        from pyutilz.core.pythonlib import is_cuda_available

        return is_cuda_available()
    except Exception:
        return False


@pytest.mark.gpu
@pytest.mark.skipif(not _need_cuda(), reason="no CUDA")
def test_helper_actually_releases_retained_blocks(monkeypatch):
    monkeypatch.setenv("MLFRAME_FE_GPU_STRICT", "1")
    # Isolate from sibling-test pool state. The helper reclaims whatever ``cupy.get_default_memory_pool()``
    # returns, so point that (and ``cp.empty``'s allocator) at a FRESH pool this test fully owns: a prior MRMR
    # fit leaves live module-level resident buffers that pin partly-used arenas in the SHARED pool, making an
    # absolute ``free_bytes() == 0`` there false-fail. On a clean, fully-owned pool the absolute contract is
    # exact and pollution-immune. Restore the original allocator + default-pool getter on teardown so a sibling's
    # live buffers are never stranded. A removed / mis-gated teardown call still fails (free_bytes stays > 0).
    orig_pool = cp.get_default_memory_pool()
    fresh = cp.cuda.MemoryPool()
    monkeypatch.setattr(cp, "get_default_memory_pool", lambda: fresh)
    cp.cuda.set_allocator(fresh.malloc)
    try:
        # allocate then drop -> the block is retained free by the pool (no live ref), not returned to the device.
        a = cp.empty(4 * 1024 * 1024, dtype=cp.float64)  # 32 MiB
        del a
        assert fresh.free_bytes() > 0, "expected a retained free block before teardown"
        assert _free_gpu_fe_mempool() is True
        assert fresh.free_bytes() == 0, "teardown must return the pool's unused blocks to the device"
    finally:
        cp.cuda.set_allocator(orig_pool.malloc)
        fresh.free_all_blocks()
