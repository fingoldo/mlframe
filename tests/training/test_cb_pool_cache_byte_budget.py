"""The CatBoost Pool caches must be bounded in bytes, not only in entries.

`_CB_POOL_CACHE` (train) and `_CB_VAL_POOL_CACHE` (val) each retained up to 16 quantised `Pool` objects
under an entry cap alone. A Pool owns the whole dataset at roughly one byte per (row, feature) cell, so on
the 7M-row frames `_predict_guards` cites for its 50-70s rebuild, 500 features put a single Pool near
3.5 GB and the cap at ~56 GB per cache -- in a module-level dict that outlives every fit, with the two caps
independent of each other.

These tests drive the budget helper with stand-in Pools that report a shape, so no CatBoost build is needed
to exercise the arithmetic that decides what is retained.
"""

from __future__ import annotations

import pytest

from mlframe.training.cb import _cb_pool_budget as budget


class _FakePool:
    """A Pool stand-in that reports only the shape the size estimate reads."""

    def __init__(self, rows: int, cols: int):
        """Hold the shape this Pool claims to have."""
        self._rows = rows
        self._cols = cols

    def num_row(self):
        """Row count, as CatBoost's Pool reports it."""
        return self._rows

    def num_col(self):
        """Column count, as CatBoost's Pool reports it."""
        return self._cols


@pytest.fixture(autouse=True)
def _clean_bookkeeping():
    """Each test starts and ends with no recorded sizes."""
    budget.reset_cache_bytes()
    yield
    budget.reset_cache_bytes()


def test_the_size_estimate_is_one_byte_per_cell():
    """Rows times columns is the order of magnitude a quantised Pool actually occupies."""
    assert budget.estimate_pool_bytes(_FakePool(7_000_000, 500)) == 3_500_000_000


def test_a_pool_without_a_shape_is_not_counted_rather_than_refused():
    """An object that does not report its shape must stay cacheable, not be silently dropped."""
    assert budget.estimate_pool_bytes(object()) == 0


def test_sixteen_large_pools_do_not_accumulate(monkeypatch):
    """The failure the finding describes: the entry cap alone let 16 x 3.5 GB reach ~56 GB."""
    monkeypatch.setenv("MLFRAME_CB_POOL_CACHE_MAX_BYTES", str(8 * 1024**3))
    cache: dict = {}
    for i in range(16):
        pool = _FakePool(7_000_000, 500)
        if budget.admit_pool(cache, "train", i, pool):
            cache[i] = pool
    assert budget.cache_bytes("train") <= 8 * 1024**3, f"the cache holds {budget.cache_bytes('train') / 1024**3:.1f} GiB"
    assert len(cache) == 2, f"8 GiB should hold exactly two 3.5 GB Pools, got {len(cache)}"


def test_the_oldest_entry_is_the_one_evicted(monkeypatch):
    """Eviction order is FIFO, matching the entry-cap loop it sits alongside."""
    monkeypatch.setenv("MLFRAME_CB_POOL_CACHE_MAX_BYTES", str(1000))
    cache: dict = {}
    for key in ("a", "b", "c"):
        pool = _FakePool(20, 20)  # 400 bytes each
        if budget.admit_pool(cache, "train", key, pool):
            cache[key] = pool
    assert list(cache) == ["b", "c"], f"expected the oldest to go first, cache holds {list(cache)}"


def test_a_pool_larger_than_the_whole_budget_is_not_cached(monkeypatch):
    """Caching it would evict everything and then be evicted itself on the next insert."""
    monkeypatch.setenv("MLFRAME_CB_POOL_CACHE_MAX_BYTES", str(1024))
    cache: dict = {"keep": _FakePool(10, 10)}
    budget.admit_pool(cache, "train", "keep", cache["keep"])
    assert budget.admit_pool(cache, "train", "huge", _FakePool(1000, 1000)) is False
    assert "keep" in cache, "an oversized entry that was refused should not have evicted the existing one"


def test_the_two_caches_are_budgeted_separately():
    """Train and val Pools are held in different dicts and must not consume each other's headroom."""
    train: dict = {}
    val: dict = {}
    budget.admit_pool(train, "train", 1, _FakePool(100, 100))
    budget.admit_pool(val, "val", 1, _FakePool(200, 200))
    assert budget.cache_bytes("train") == 10_000
    assert budget.cache_bytes("val") == 40_000


def test_an_entry_dropped_from_the_cache_stops_counting():
    """Sizes are pruned against the live dict, so an invalidated entry does not hold phantom budget."""
    cache: dict = {}
    pool = _FakePool(100, 100)
    budget.admit_pool(cache, "train", "k", pool)
    cache["k"] = pool
    assert budget.cache_bytes("train") == 10_000
    cache.pop("k")
    budget.admit_pool(cache, "train", "k2", _FakePool(10, 10))
    assert budget.cache_bytes("train") == 100, "the removed entry was still counted against the budget"


def test_a_malformed_env_override_falls_back_to_the_default(monkeypatch):
    """A typo in the env var must not disable the budget."""
    monkeypatch.setenv("MLFRAME_CB_POOL_CACHE_MAX_BYTES", "eight gigabytes")
    assert budget.cb_pool_cache_max_bytes() == budget._CB_POOL_CACHE_MAX_BYTES_DEFAULT
