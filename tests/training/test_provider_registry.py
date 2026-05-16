"""
Tests for the phase-B provider registry + lifecycle (round-3 R2-2,
chaos C3, C5, C18, C22; perf F1).

Coverage:
  * ``acquire_provider`` context manager: load on first acquire, drop
    on last release (when not in LRU strong-keep tier).
  * Reference counting: A.acquire / B.acquire / A.release / B.release
    -> load 1x, unload 1x only on final release. (round-3 chaos C22)
  * Concurrent acquire (TOCTOU race): two threads on same signature
    load the provider exactly once. (round-3 R2-2)
  * LRU strong-keep tier: keep_n_providers=2 holds two recent
    providers across release-zero, evicts oldest on third.
  * shutdown_all() releases everything; provider_status() reflects.
  * prewarm() schedules background load via Future, exception
    propagates on wait_prewarm. (round-3 chaos C3)

The tests use a ``DummyProvider`` mock so they don't touch HF / network /
GPU. The HF provider has its own integration test in
``test_hf_provider.py`` (heavy, optionally skipped).
"""

from __future__ import annotations

import threading
import time
from typing import List

import numpy as np
import pytest

from mlframe.training.feature_handling import (
    CacheConfig,
    EmbeddingProvider,
    acquire_provider,
    prewarm,
    provider_status,
    shutdown_all,
    wait_prewarm,
)


# =====================================================================
# Mock provider that records calls
# =====================================================================


class _DummyProvider:
    """Minimal Frozen-protocol-shaped mock. Records acquire/release
    calls for assertion."""

    def __init__(self, signature: str, dim: int = 4, fail_on_acquire: bool = False):
        self._signature = signature
        self._dim = dim
        self.acquire_calls = 0
        self.release_calls = 0
        self.fail_on_acquire = fail_on_acquire

    @property
    def signature(self) -> str:
        return self._signature

    @property
    def embedding_dim(self) -> int:
        return self._dim

    def fit(self, train_texts):
        return self

    def transform(self, texts):
        return np.zeros((len(texts), self._dim), dtype=np.float32)

    def acquire(self) -> None:
        if self.fail_on_acquire:
            raise RuntimeError("synthetic acquire failure")
        self.acquire_calls += 1

    def release(self) -> None:
        self.release_calls += 1


@pytest.fixture(autouse=True)
def _shutdown_after_each_test():
    """Each test starts with a clean registry. shutdown_all() at the
    end purges any provider state leaked by failures so a single
    error doesn't poison subsequent tests."""
    shutdown_all()
    yield
    shutdown_all()


@pytest.fixture
def cache_cfg():
    """CacheConfig with a small explicit keep_n_providers so tests are
    deterministic regardless of GPU availability."""
    return CacheConfig(persistence="off", keep_n_providers=2)


# =====================================================================
# 1. acquire/release lifecycle
# =====================================================================


class TestAcquireReleaseLifecycle:
    def test_first_acquire_loads_provider(self, cache_cfg):
        p = _DummyProvider("sig-A")
        assert p.acquire_calls == 0
        with acquire_provider(p, cache_cfg) as got:
            assert got is p
            assert p.acquire_calls == 1
            assert p.release_calls == 0
        # Released after exit -- but in LRU tier so stays loaded.
        # release_calls bumps only when LRU evicts.
        # Since keep_n=2 and we only loaded one, it stays.
        assert p.release_calls == 0

    def test_release_when_not_in_lru_after_final_refcount_drop(self, cache_cfg):
        # keep_n=0 -> immediate release after refcount==0.
        # We can't set keep_n=0 (validator), so verify with keep_n=1
        # and a second provider that evicts the first.
        p1 = _DummyProvider("sig-1")
        p2 = _DummyProvider("sig-2")
        cfg = CacheConfig(persistence="off", keep_n_providers=1)
        with acquire_provider(p1, cfg):
            pass
        # p1 in LRU tier (keep_n=1); release not called yet.
        assert p1.release_calls == 0
        with acquire_provider(p2, cfg):
            pass
        # p2 promoted, p1 evicted from LRU -> released.
        assert p1.release_calls == 1
        assert p2.release_calls == 0

    def test_refcount_two_acquires_one_load(self, cache_cfg):
        """A.acquire / B.acquire / A.release / B.release ->
        load 1x, NOT unloaded mid-use (round-3 chaos C22)."""
        p = _DummyProvider("sig-rc")
        with acquire_provider(p, cache_cfg) as a:
            with acquire_provider(p, cache_cfg) as b:
                assert a is b
                assert p.acquire_calls == 1
        assert p.acquire_calls == 1
        assert p.release_calls == 0  # in LRU tier


# =====================================================================
# 2. Concurrent acquire (TOCTOU race) -- round-3 R2-2
# =====================================================================


class TestConcurrentAcquire:
    def test_two_threads_one_load(self, cache_cfg):
        p = _DummyProvider("sig-concurrent")
        original_acquire = p.acquire
        n_workers = 4

        # Deterministic race-widening: a threading.Barrier inside the acquire forces all workers
        # to reach the load critical section simultaneously, so the TOCTOU window is opened
        # explicitly rather than depending on time.sleep(0.05) and CPU scheduling. The lock
        # serialising acquire_provider MUST still produce a single load -- that's the contract
        # under test.
        barrier = threading.Barrier(n_workers, timeout=5.0)

        def _race_acquire():
            try:
                barrier.wait()
            except threading.BrokenBarrierError:
                pass
            original_acquire()

        p.acquire = _race_acquire  # type: ignore

        results: List[bool] = []
        results_lock = threading.Lock()

        def worker():
            with acquire_provider(p, cache_cfg) as got:
                with results_lock:
                    results.append(got is p)

        threads = [threading.Thread(target=worker) for _ in range(n_workers)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert results == [True] * n_workers
        # Despite n_workers concurrent acquires, the provider was loaded ONCE.
        assert p.acquire_calls == 1


# =====================================================================
# 3. LRU strong-keep tier
# =====================================================================


class TestLRUStrongKeep:
    def test_two_providers_within_keep_n(self):
        cfg = CacheConfig(persistence="off", keep_n_providers=2)
        p1 = _DummyProvider("sig-a")
        p2 = _DummyProvider("sig-b")
        with acquire_provider(p1, cfg):
            pass
        with acquire_provider(p2, cfg):
            pass
        # Both still loaded (keep_n=2, exactly two providers).
        assert p1.release_calls == 0
        assert p2.release_calls == 0

    def test_third_provider_evicts_oldest(self):
        cfg = CacheConfig(persistence="off", keep_n_providers=2)
        p1 = _DummyProvider("sig-a")
        p2 = _DummyProvider("sig-b")
        p3 = _DummyProvider("sig-c")
        for p in (p1, p2):
            with acquire_provider(p, cfg):
                pass
        # Now p1, p2 in LRU. Acquire p3 -> evict p1.
        with acquire_provider(p3, cfg):
            pass
        assert p1.release_calls == 1
        assert p2.release_calls == 0
        assert p3.release_calls == 0


# =====================================================================
# 4. shutdown_all + provider_status
# =====================================================================


class TestShutdown:
    def test_shutdown_releases_all_in_lru(self, cache_cfg):
        providers = [_DummyProvider(f"sig-{i}") for i in range(3)]
        # Acquire each in sequence
        for p in providers:
            with acquire_provider(p, cache_cfg):
                pass
        # First provider is evicted (keep_n=2), released. Last 2 in LRU.
        assert providers[0].release_calls == 1
        # Now shutdown
        shutdown_all()
        for p in providers[1:]:
            assert p.release_calls == 1, f"{p.signature} should be released on shutdown"

    def test_provider_status_after_acquire(self, cache_cfg):
        p = _DummyProvider("sig-status")
        with acquire_provider(p, cache_cfg):
            status = provider_status()
            assert "sig-status" in status
            assert status["sig-status"]["loaded"] is True
            assert status["sig-status"]["refcount"] == 1


# =====================================================================
# 5. Pre-warm via Future -- round-3 chaos C3
# =====================================================================


class TestPrewarm:
    def test_prewarm_loads_in_background(self):
        p = _DummyProvider("sig-prewarm-ok")
        fut = prewarm(p)
        # Wait for completion
        fut.result(timeout=5.0)
        assert p.acquire_calls == 1

    def test_prewarm_exception_surfaces_via_wait(self):
        """Round-3 chaos C3: previously naked threading.Thread swallowed
        exceptions to stderr; main thread never knew. Future.result()
        propagates."""
        p = _DummyProvider("sig-prewarm-fail", fail_on_acquire=True)
        prewarm(p)
        with pytest.raises(RuntimeError, match="synthetic acquire failure"):
            wait_prewarm(p, timeout=5.0)


# =====================================================================
# 6. Naked acquire/release banned -- only context manager API
# =====================================================================


class TestContextManagerOnly:
    """Round-3 chaos C22 -- public API is acquire_provider() context
    manager, NOT the provider's own .acquire() / .release(). The
    provider methods are still public on the protocol (for the
    registry to call), but consumers should not use them directly.

    This test pins the structural contract: ``acquire_provider``
    returns a context manager, and the registry tracks lifecycle
    without naked manipulation.
    """

    def test_acquire_provider_is_context_manager(self, cache_cfg):
        p = _DummyProvider("sig-cm")
        cm = acquire_provider(p, cache_cfg)
        # Should support __enter__ / __exit__
        assert hasattr(cm, "__enter__")
        assert hasattr(cm, "__exit__")
        with cm as got:
            assert got is p

    def test_release_called_on_exception_in_with_block(self, cache_cfg):
        cfg = CacheConfig(persistence="off", keep_n_providers=1)
        p1 = _DummyProvider("sig-exc")
        p2 = _DummyProvider("sig-exc-2")  # to force eviction
        try:
            with acquire_provider(p1, cfg):
                raise ValueError("user error")
        except ValueError:
            pass
        # p1 still in LRU tier despite exception -- exception in
        # the with-body doesn't break refcount semantic.
        # Force eviction:
        with acquire_provider(p2, cfg):
            pass
        assert p1.release_calls == 1
