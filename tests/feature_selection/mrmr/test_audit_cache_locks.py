"""Concurrency + cache-key regression tests for the mrmr audit fix wave (see audits/mrmr_audit_2026-07-25/).

Pins that the resident-cache clear() helpers hold the same lock their populate/iterate path takes (so a
teardown clear racing a concurrent fit cannot raise "dictionary changed size during iteration"), and that
the FE-GPU gate memo folds the strict/opt-out env state into its key (no stale verdict after a flip).
"""

from __future__ import annotations

import threading

import pytest


def _hammer_clear_vs_locked_iteration(cache, lock, clear_fn, fill):
    """Race a locked iteration of ``cache`` (mimicking the real getter) against ``clear_fn``.

    Fills the cache, then spins one thread iterating under ``lock`` (with a tiny yield mid-iteration so a
    lock-less clear would mutate the dict during iteration) and one thread repeatedly clearing + refilling.
    Returns the first exception observed in the iterator thread, or None.
    """
    errors: list[BaseException] = []
    stop = threading.Event()

    def _iterate():
        """Iterate the cache under the lock until stopped, capturing any mutation-during-iteration error."""
        while not stop.is_set():
            try:
                with lock:
                    for _ in cache:
                        # Force the GIL to hand off mid-iteration; a lock-less clear would strike here.
                        pass
            except BaseException as exc:
                errors.append(exc)
                return

    def _clear_and_refill():
        """Hammer clear_fn (the function under test) against the locked iterator, refilling under the lock."""
        # Refill under the lock (mirroring the real getter's populate path) so the ONLY unlocked mutation
        # under test is clear_fn itself: pre-fix it mutates during the locked iteration, post-fix it waits.
        for _ in range(2000):
            clear_fn()
            with lock:
                fill()

    fill()
    it = threading.Thread(target=_iterate)
    it.start()
    try:
        _clear_and_refill()
    finally:
        stop.set()
        it.join(timeout=10)
    return errors[0] if errors else None


def test_cmi_xc_resident_cache_clear_is_lock_guarded():
    """clear_cmi_xc_resident_cache must hold _FACTORS_DEVICE_LOCK so a teardown clear cannot race the resident-factors iterator (dictionary-changed-size-during-iteration)."""
    cc = pytest.importorskip("mlframe.feature_selection.filters.info_theory._cmi_cuda")

    def _fill():
        """Repopulate the resident factors cache with a full batch of dummy entries."""
        for i in range(64):
            cc._FACTORS_DEVICE_CACHE[i] = i

    err = _hammer_clear_vs_locked_iteration(cc._FACTORS_DEVICE_CACHE, cc._FACTORS_DEVICE_LOCK, cc.clear_cmi_xc_resident_cache, _fill)
    cc.clear_cmi_xc_resident_cache()
    assert err is None, f"clear raced the locked iterator: {err!r}"


def test_mah_y_binning_cache_clear_is_lock_guarded():
    """clear_mah_y_binning_cache must hold _Y_BINNING_LOCK so a teardown clear cannot race the y-binning getter's locked iteration."""
    mah = pytest.importorskip("mlframe.feature_selection.filters._mah")

    def _fill():
        """Repopulate the y-binning cache with a full batch of dummy entries."""
        for i in range(64):
            mah._Y_BINNING_CACHE[(i, i)] = i

    err = _hammer_clear_vs_locked_iteration(mah._Y_BINNING_CACHE, mah._Y_BINNING_LOCK, mah.clear_mah_y_binning_cache, _fill)
    mah.clear_mah_y_binning_cache()
    assert err is None, f"clear raced the locked iterator: {err!r}"


def test_fe_gpu_gate_cache_invalidates_on_strict_env_flip(monkeypatch):
    """The FE-GPU discretize gate memo must fold the strict-mode env into its key, so flipping MLFRAME_FE_GPU_STRICT mid-process returns the new verdict, not a stale cached one."""
    pc = pytest.importorskip("mlframe.feature_selection.filters._feature_engineering_pairs._pairs_core")

    calls: list[str] = []

    def _fake_uncached(n_rows, n_cands):
        """Stand-in gate computation that records the strict-mode env it saw and derives its verdict from it."""
        strict = __import__("os").environ.get("MLFRAME_FE_GPU_STRICT", "")
        calls.append(strict)
        return strict == "1"

    monkeypatch.setattr(pc, "_fe_gpu_discretize_enabled_uncached", _fake_uncached)
    with pc._GPU_GATE_CACHE_LOCK:
        pc._GPU_GATE_CACHE.clear()

    monkeypatch.delenv("MLFRAME_FE_GPU_STRICT", raising=False)
    assert pc._fe_gpu_discretize_enabled(1000, 10) is False
    assert pc._fe_gpu_discretize_enabled(1000, 10) is False  # served from cache, no recompute

    monkeypatch.setenv("MLFRAME_FE_GPU_STRICT", "1")
    assert pc._fe_gpu_discretize_enabled(1000, 10) is True, "strict-mode flip must invalidate the memo, not return the stale False"
    assert calls.count("1") == 1, "the flipped verdict must be recomputed exactly once, then cached under the new env key"


def test_fe_gpu_gate_cache_survives_concurrent_eviction():
    """Concurrent _fe_gpu_discretize_enabled callers evicting the bounded LRU under _GPU_GATE_CACHE_LOCK must never corrupt the dict or raise."""
    pc = pytest.importorskip("mlframe.feature_selection.filters._feature_engineering_pairs._pairs_core")

    with pc._GPU_GATE_CACHE_LOCK:
        pc._GPU_GATE_CACHE.clear()

    errors: list[BaseException] = []

    def _worker(base):
        """Drive many distinct gate keys through the memo so the bounded LRU evicts under concurrency."""
        try:
            for i in range(4000):
                pc._fe_gpu_discretize_enabled(base + i, (i % 7) + 1)
        except BaseException as exc:
            errors.append(exc)

    threads = [threading.Thread(target=_worker, args=(t * 100000,)) for t in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=30)
    assert not errors, f"concurrent gate-cache eviction corrupted the LRU: {errors[0]!r}"
    assert len(pc._GPU_GATE_CACHE) <= pc._GPU_GATE_CACHE_MAX
