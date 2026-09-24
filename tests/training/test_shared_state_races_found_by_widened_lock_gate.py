"""Races in module-level shared state that the lock meta-tests found once they looked past ``*_CACHE`` names.

Each test drives the real function from many threads with a tiny interpreter switch interval, so the interleaving
that breaks the unlocked code happens within a few thousand calls, or reproduces the one interleaving directly.
"""

from __future__ import annotations

import sys
import threading
from collections import OrderedDict

import numpy as np
import pandas as pd
import pytest


def _hammer(work, n_threads: int = 16) -> list:
    """Run ``work(k)`` on ``n_threads`` threads at once with a tiny switch interval; return the exceptions raised."""
    errors: list = []
    barrier = threading.Barrier(n_threads)

    def run(k: int) -> None:
        """Wait for every thread, then run the work and record any exception."""
        barrier.wait()
        try:
            work(k)
        except Exception as exc:  # the assertion below reports it
            errors.append(exc)

    old = sys.getswitchinterval()
    sys.setswitchinterval(1e-6)
    try:
        threads = [threading.Thread(target=run, args=(k,)) for k in range(n_threads)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
    finally:
        sys.setswitchinterval(old)
    return errors


class _FakePool:
    """Answers num_row/num_col like a quantised CatBoost Pool."""

    def __init__(self, rows: int) -> None:
        """Size the fake pool."""
        self.rows = rows

    def num_row(self) -> int:
        """Rows."""
        return self.rows

    def num_col(self) -> int:
        """Columns."""
        return 10


def test_catboost_pool_budget_survives_concurrent_admits(monkeypatch):
    """Concurrent admits iterate and evict the size ledger; unlocked, that raised or left the ledger out of step with the cache."""
    from mlframe.training.cb import _cb_pool_budget as budget

    monkeypatch.setenv("MLFRAME_CB_POOL_CACHE_MAX_BYTES", str(5_000))
    budget.reset_cache_bytes("race")
    cache: dict = {}

    def work(k: int) -> None:
        """Admit many pools; admit_pool stores an admitted one itself, under the lock its eviction walk holds."""
        for i in range(400):
            budget.admit_pool(cache, "race", (k, i), _FakePool(50 + (i % 7)))

    try:
        assert _hammer(work) == []
        assert budget.cache_bytes("race") <= 5_000
    finally:
        budget.reset_cache_bytes("race")


def test_ranknet_pair_cache_eviction_tolerates_a_concurrent_evictor(monkeypatch):
    """Two threads could pick the same oldest key and the second ``pop`` raised KeyError mid-loss."""
    torch = pytest.importorskip("torch")
    from mlframe.training.neural import _ranker_losses as rl

    class _RacedDict(dict):
        """Another thread evicts the chosen key just before this thread's pop."""

        def pop(self, key, *default):
            """Drop the key as the other thread would, then run the real pop."""
            super().pop(key, None)
            return super().pop(key, *default)

    monkeypatch.setattr(rl, "_ranknet_pair_cache", _RacedDict())
    monkeypatch.setattr(rl, "_RANKNET_PAIR_CACHE_SIZE", 4)  # evict on nearly every call
    g = torch.Generator().manual_seed(0)
    for _ in range(50):
        rel = torch.randint(0, 4, (8,), generator=g).float()
        loss = rl.ranknet_pairwise_loss(torch.randn(8, generator=g), rel)
        assert torch.isfinite(loss)


def test_dtype_pairs_memo_hit_survives_the_weakref_evictor_firing_mid_lookup(monkeypatch):
    """The GC evictor can drop the key between ``get`` and ``move_to_end``; the hit path raised KeyError there."""
    from mlframe.training.core import _phase_train_one_target as mod

    class _EvictingOnGet(OrderedDict):
        """Behaves like the evictor running right after a successful lookup."""

        def get(self, key, default=None):
            """Return the value, then drop the key as the weakref callback would."""
            value = super().get(key, default)
            if value is not None:
                self.pop(key, None)
            return value

    df = pd.DataFrame({"a": np.arange(3), "b": np.ones(3)})
    memo = _EvictingOnGet()
    monkeypatch.setattr(mod, "_DTYPE_PAIRS_MEMO", memo)
    first = mod._canonical_dtype_pairs(df)
    assert mod._canonical_dtype_pairs(df) == first


def test_hash_memo_survives_concurrent_lookups():
    """Callers hash outside the operands lock, so threads reorder and evict the memo at once; unlocked, OrderedDict raised."""
    from mlframe.feature_selection.filters import _fe_resident_operands as ops

    ops.clear_hash_memo()
    arrays = [np.full(64, k, dtype=np.float32) for k in range(3 * ops._HASH_MEMO_MAX_ENTRIES)]

    def work(k: int) -> None:
        """Hash an overlapping rotation of arrays so hits, inserts and evictions interleave."""
        for i in range(600):
            a = arrays[(i * 7 + k) % len(arrays)]
            assert ops._content_hash_memoized(a) == ops._content_hash(a)

    try:
        assert _hammer(work) == []
    finally:
        ops.clear_hash_memo()
