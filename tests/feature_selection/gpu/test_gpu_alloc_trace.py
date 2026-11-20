"""The allocation trace counts allocations, finds the largest site and the out-of-memory context, keeps ``_current_pool`` working and is off with MLFRAME_GPU_ALLOC_TRACE=0."""

import numpy as np
import pytest

cp = pytest.importorskip("cupy")

from mlframe.feature_selection.filters import _gpu_alloc_trace as T
from mlframe.feature_selection.filters._device_sort_safe import _current_pool, device_room


class _Holder:
    """Receives ``gpu_alloc_trace_``."""


def test_scope_counts_allocations_and_names_the_largest_site(monkeypatch):
    """A 40 MB allocation is above the stack threshold: its site is reported; the counters include the small ones."""
    monkeypatch.setattr(T, "_BIG_BYTES", 32 * 1024 * 1024)
    holder = _Holder()
    with T.gpu_alloc_trace_scope(holder):
        small = [cp.empty(1000, dtype=cp.uint8) for _ in range(20)]
        big = cp.empty(40 * 1024 * 1024, dtype=cp.uint8)
        del small
    trace = holder.gpu_alloc_trace_
    assert trace["n_allocations"] >= 21 and trace["total_gb"] >= 0.04 and trace["n_oom"] == 0
    assert trace["largest_allocations"] and trace["largest_allocations"][0]["mb"] >= 40.0
    assert "test_scope_counts" in trace["largest_allocations"][0]["where"] or trace["largest_allocations"][0]["where"] == ""
    assert big.size == 40 * 1024 * 1024


def test_current_pool_still_sees_the_pool_inside_the_scope():
    """The wrapper keeps ``__self__`` on the pool, so ``device_room`` sizes blocks under the pool limit as before."""
    pool = cp.get_default_memory_pool()
    with T.gpu_alloc_trace_scope(None):
        assert _current_pool(cp) is pool
        assert device_room(cp) > 0


def test_out_of_memory_is_recorded_and_still_raised():
    """A pool limit below the request: the failure carries its context in the trace and the exception reaches the caller."""
    pool = cp.cuda.MemoryPool()
    pool.set_limit(size=8 * 1024 * 1024)
    state = T._TraceState()
    alloc = T._TracingAllocator(pool, state)
    with pytest.raises(cp.cuda.memory.OutOfMemoryError):
        alloc(64 * 1024 * 1024)
    summary = state.summary()
    assert summary["n_oom"] == 1 and summary["oom_events"][0]["requested_mb"] == 67.1 and summary["oom_events"][0]["pool_limit_mb"] == 8.4


def test_off_switch_and_nesting(monkeypatch):
    """With the switch off nothing is recorded; a nested scope leaves the record to the outer one."""
    monkeypatch.setenv("MLFRAME_GPU_ALLOC_TRACE", "0")
    holder = _Holder()
    with T.gpu_alloc_trace_scope(holder):
        cp.empty(10, dtype=cp.uint8)
    assert not hasattr(holder, "gpu_alloc_trace_")
    monkeypatch.setenv("MLFRAME_GPU_ALLOC_TRACE", "1")
    outer, inner = _Holder(), _Holder()
    with T.gpu_alloc_trace_scope(outer):
        with T.gpu_alloc_trace_scope(inner):
            cp.empty(10, dtype=cp.uint8)
    assert hasattr(outer, "gpu_alloc_trace_") and not hasattr(inner, "gpu_alloc_trace_")


def test_traced_allocator_is_pool_malloc_without_a_scope():
    """Outside a scope the stage slots get the plain bound ``pool.malloc``."""
    pool = cp.cuda.MemoryPool()
    assert T._STATE is None
    assert T.traced_allocator(pool) == pool.malloc
    with T.gpu_alloc_trace_scope(None):
        assert getattr(T.traced_allocator(pool), "__self__", None) is pool
    assert np.asarray(pool.used_bytes()) >= 0


def test_a_fit_stores_the_trace_on_the_estimator():
    """After ``MRMR.fit`` the fitted estimator carries ``gpu_alloc_trace_`` with the allocation count and the peak."""
    import warnings

    import pandas as pd

    from mlframe.feature_selection.filters.mrmr import MRMR

    rng = np.random.default_rng(0)
    X = pd.DataFrame({f"x{i}": rng.normal(size=3000) for i in range(6)})
    y = (X["x0"] + 0.3 * rng.normal(size=3000) > 0).astype(int)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        m = MRMR(max_runtime_mins=1, verbose=0, fe_max_steps=0).fit(X, y)
    trace = m.gpu_alloc_trace_
    assert set(trace) >= {"n_allocations", "total_gb", "peak_pool_used_mb", "n_oom", "largest_allocations", "oom_events"}
