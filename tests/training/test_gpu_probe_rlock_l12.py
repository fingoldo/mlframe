"""Regression: _GPU_PROBE_LOCK must be reentrant.

`_cb_gpu_usable` acquires `_GPU_PROBE_LOCK`, then calls `_cached_gpu_info`
which tries to acquire the same lock. With a plain `threading.Lock` this
deadlocks on the very first probe of a process (when `_GPU_INFO_PROBED`
is still False so `_cached_gpu_info` cannot short-circuit on its lock-free
fast path). With an `RLock` the same thread can re-enter.

The fuzz cell `c0097_867cf5d3-cb_hgb_lgb_mlp_xgb-pl_utf8-n1000` reliably
surfaced the deadlock when CatBoost was the first GPU consumer.
"""
from __future__ import annotations

import threading

from mlframe.training import _cb_pool


def test_gpu_probe_lock_is_reentrant() -> None:
    """The lock guarding GPU probes must be an RLock so _cb_gpu_usable can
    call _cached_gpu_info without self-deadlock."""
    assert isinstance(_cb_pool._GPU_PROBE_LOCK, type(threading.RLock()))


def test_reentrant_acquire_does_not_deadlock() -> None:
    """A second acquire on the same thread must return immediately."""
    lock = _cb_pool._GPU_PROBE_LOCK
    acquired_outer = lock.acquire(timeout=1.0)
    try:
        assert acquired_outer
        acquired_inner = lock.acquire(timeout=1.0)
        try:
            assert acquired_inner, "second acquire deadlocked — _GPU_PROBE_LOCK is not reentrant"
        finally:
            if acquired_inner:
                lock.release()
    finally:
        if acquired_outer:
            lock.release()


def test_cb_gpu_usable_then_cached_gpu_info_no_hang() -> None:
    """End-to-end: simulate the deadlock chain by forcing the un-probed
    state, then call _cb_gpu_usable in a worker thread with a wall-clock
    timeout. With a plain Lock this would hang forever; with an RLock it
    returns in microseconds.

    Skip when running under pytest-xdist: the CatBoost native GPU probe
    can SIGSEGV the worker process on hosts where CUDA initialization
    races with sibling-worker probes (observed 2026-05-24 big machine,
    'worker gw0 crashed'). The reentrant-lock contract is already
    proven behaviourally by test_gpu_probe_lock_is_reentrant +
    test_reentrant_acquire_does_not_deadlock above; this end-to-end
    test adds the native-probe round-trip which isn't worth a process
    crash on parallel runners.
    """
    import os
    if os.environ.get("PYTEST_XDIST_WORKER"):
        import pytest
        pytest.skip(
            "Native CatBoost GPU probe can SIGSEGV the xdist worker; "
            "reentrant-lock contract is proven by the lock-only tests."
        )
    # Reset the probe state so we exercise the lock-holding path on this thread.
    _cb_pool._GPU_INFO_PROBED = False
    _cb_pool._GPU_INFO_CACHE = None
    _cb_pool._CB_GPU_USABLE_CACHE = None

    result = {"value": None, "error": None}

    def _worker() -> None:
        try:
            result["value"] = _cb_pool._cb_gpu_usable()
        except Exception as exc:  # pragma: no cover - defensive
            result["error"] = repr(exc)

    t = threading.Thread(target=_worker, daemon=True)
    t.start()
    t.join(timeout=30.0)
    assert not t.is_alive(), "_cb_gpu_usable hung — reentrant lock missing"
    assert result["error"] is None, f"_cb_gpu_usable raised: {result['error']}"
    assert isinstance(result["value"], bool)
