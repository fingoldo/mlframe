"""Sensors for the GPU / resource findings of the 2026-11-20 MRMR audit wave (C-F01..C-F05)."""

import logging
from types import SimpleNamespace

import numpy as np
import pytest

from mlframe.utils.log_throttle import reset_throttle_counts


def test_edge_pool_workers_see_the_callers_allocator():
    """C-F01: a worker of ``map_on_edge_pool`` runs under the allocator (and device) of the calling thread, not the thread-local default of a fresh thread."""
    cp = pytest.importorskip("cupy")
    try:
        _ = cp.cuda.Device().id
    except Exception:
        pytest.skip("no usable CUDA device")
    from mlframe.feature_selection.filters._edge_pool import map_on_edge_pool

    def custom_allocator(size):
        """A recognisable allocator: the default pool's."""
        return cp.get_default_memory_pool().malloc(size)

    seen = []

    def probe(_item):
        """The allocator and device this worker thread runs under."""
        seen.append((cp.cuda.get_allocator(), int(cp.cuda.Device().id)))
        return 0

    with cp.cuda.using_allocator(custom_allocator):
        map_on_edge_pool(probe, list(range(8)), min_items=1)
    assert seen and all(a is custom_allocator for a, _ in seen)


def test_stage_vram_check_is_conservative_for_an_unreadable_device(monkeypatch):
    """C-F04: a device whose memory cannot be read does not count as roomy (the cascades run one by one); a host without cupy has nothing to protect."""
    pytest.importorskip("cupy")
    import cupy as cp

    from mlframe.feature_selection.filters._mrmr_fit_impl import _fe_stage_concurrency as sc

    def boom():
        """A driver error while reading the memory pool."""
        raise RuntimeError("driver error")

    monkeypatch.setattr(cp, "get_default_memory_pool", boom)
    assert sc._stages_fit_device((1.0, 1.0, 1.0), 50000) is False


def test_stage_failure_is_not_replaced_by_the_cleanup_error(monkeypatch):
    """C-F05: the stage's own exception reaches the caller when the post-stage synchronize also fails (a sticky device error)."""
    from contextlib import nullcontext

    from mlframe.feature_selection.filters import _fe_device_scheduler as sched
    from mlframe.feature_selection.filters._mrmr_fit_impl import _fe_stage_concurrency as sc

    slot = SimpleNamespace(device_id=0)
    monkeypatch.setattr(sched, "plan_stage_slots", lambda *a, **k: [slot])
    monkeypatch.setattr(sched, "stage_device_slot", lambda *a, **k: nullcontext())
    monkeypatch.setattr(sc, "_synchronize_device", lambda device_id: (_ for _ in ()).throw(RuntimeError("sticky CUDA error")))
    monkeypatch.setattr(sc, "_stages_fit_device", lambda *a, **k: True)

    def failing_stage(_est):
        """A stage whose family raises."""
        raise ValueError("bad recipe in the family")

    est = SimpleNamespace()
    with pytest.raises(ValueError, match="bad recipe"):
        sc.run_stages_concurrently(est, [failing_stage], weights=(1.0,), vram_weights=(1.0,), n_rows=1000)


def test_null_max_batched_path_declines_a_table_over_the_shared_limit(caplog):
    """C-F02: 200 classes x 64 bins is 51200 bytes of shared memory per permutation, over the 48 KB default: the batched path returns ``None`` with a logged reason instead of failing the launch."""
    cp = pytest.importorskip("cupy")
    try:
        _ = cp.cuda.Device().id
    except Exception:
        pytest.skip("no usable CUDA device")
    from mlframe.feature_selection.filters import _null_max_perm_gpu as g

    reset_throttle_counts()
    x = cp.zeros((2, 64), dtype=cp.float32)
    interior = cp.zeros((63, 2), dtype=cp.float32)
    y = cp.zeros(64, dtype=cp.int32)
    perms = np.zeros((2, 64), dtype=np.int64)
    with caplog.at_level(logging.WARNING):
        assert g.null_max_mi_perms_gpu(cp, x, interior, y, perms, 64, 200) is None
    assert any("over the" in r.getMessage() for r in caplog.records)


def test_label_bits_cache_keeps_two_keys_and_clears():
    """C-F03: two alternating keys both stay cached (a single slot repacked on every call) and the fit-end release empties the cache."""
    from mlframe.feature_selection.filters import _null_max_perm_gpu as g

    g.clear_label_bits_cache()
    g._store_label_bits("a", 0, np.zeros((1, 1)))
    g._store_label_bits("b", 0, np.zeros((1, 1)))
    assert g._cached_label_bits("a", 0) is not None and g._cached_label_bits("b", 0) is not None
    g._store_label_bits("c", 0, np.zeros((1, 1)))
    assert g._cached_label_bits("a", 0) is None and g._cached_label_bits("c", 0) is not None
    from mlframe.feature_selection.filters import _fit_scoped_release as rel

    rel.release_fit_scoped_caches()
    assert g._cached_label_bits("c", 0) is None
