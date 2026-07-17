"""CPX28 identity + restart regression tests for DiscoveryCache.

The CPX28 optimization batches the ``.lru`` flush (in-memory ledger, flushed on eviction + close)
and keeps an incremental byte accumulator instead of globbing + getsize-ing every ``*.pkl`` on each
``set``. These tests pin that the optimization is behaviourally identical to the disk-scan / flush-
every-op baseline (same get results, same surviving keys after eviction, same final on-disk size) and
that the in-memory state survives a process-style reopen of the same directory.

The OLD baseline lives in ``mlframe.training.composite._old_cache_store_cpx28_baseline`` (a committed
``git show HEAD:`` snapshot) so the A/B compares two real artifacts, not a from-memory rewrite.
"""

from __future__ import annotations

import os

import pytest

from mlframe.training.composite.cache_store import (
    DiscoveryCache as NEW,
    _discovery_cache_bytes_total,
)

try:
    from mlframe.training.composite._old_cache_store_cpx28_baseline import DiscoveryCache as OLD

    _HAVE_OLD = True
except Exception:  # pragma: no cover - baseline snapshot absent
    _HAVE_OLD = False


def _drive(cls, d, *, max_entries, max_size_mb, ops):
    """Run a scripted op sequence; return (get_results, surviving_keys, total_bytes)."""
    c = cls(d, max_entries=max_entries, max_size_mb=max_size_mb)
    get_results = []
    for op in ops:
        kind = op[0]
        if kind == "set":
            c.set(op[1], op[2])
        elif kind == "get":
            v = c.get(op[1], default="__MISS__")
            get_results.append((op[1], v if v == "__MISS__" else tuple(sorted(v.items()))))
        elif kind == "inval":
            c.invalidate(op[1])
    close = getattr(c, "close", None)
    if close is not None:
        close()
    surviving = {os.path.splitext(f)[0] for f in os.listdir(d) if f.endswith(".pkl")}
    total = _discovery_cache_bytes_total(c)
    return get_results, surviving, total


def _make_ops(n):
    """Builds a set/get/re-get/invalidate operation sequence exercising LRU-order perturbation and eviction."""
    ops = []
    for i in range(n):
        k = f"{i:08x}"
        ops.append(("set", k, {"v": i, "w": i * 2}))
        ops.append(("get", k, None))
    # Re-get some old keys to perturb LRU ordering, invalidate a few.
    for i in range(0, n, 7):
        ops.append(("get", f"{i:08x}", None))
    for i in range(0, n, 13):
        ops.append(("inval", f"{i:08x}", None))
    return ops


@pytest.mark.skipif(not _HAVE_OLD, reason="CPX28 baseline snapshot not present")
@pytest.mark.parametrize("max_entries,max_size_mb", [(20, None), (None, 0.002), (15, 0.01)])
def test_cpx28_identity_old_vs_new(tmp_path, max_entries, max_size_mb):
    """The current DiscoveryCache implementation replays the same op sequence identically to the pre-CPX28 baseline snapshot."""
    ops = _make_ops(60)
    old_dir = tmp_path / "old"
    new_dir = tmp_path / "new"
    old_dir.mkdir()
    new_dir.mkdir()
    old_res = _drive(OLD, str(old_dir), max_entries=max_entries, max_size_mb=max_size_mb, ops=ops)
    new_res = _drive(NEW, str(new_dir), max_entries=max_entries, max_size_mb=max_size_mb, ops=ops)
    assert new_res[0] == old_res[0], "get results diverged"
    assert new_res[1] == old_res[1], "surviving key set diverged after eviction"
    assert new_res[2] == old_res[2], "final on-disk byte total diverged"


def test_cpx28_lru_survives_reopen(tmp_path):
    """The in-memory ledger must be flushed on close so a reopen reconstructs access order."""
    d = str(tmp_path)
    c1 = NEW(d, max_entries=1000, max_size_mb=None)
    for i in range(5):
        c1.set(f"{i:08x}", {"v": i})
    # Bump access on key 0 so it is the most-recently-used, then close (flush).
    c1.get("00000000")
    c1.close()

    # Reopen: the prior run's flushed ledger must be read back from disk.
    c2 = NEW(d, max_entries=1000, max_size_mb=None)
    lru = c2._ensure_lru()
    assert set(lru) == {f"{i:08x}" for i in range(5)}, "ledger keys lost across reopen"
    assert lru["00000000"] == max(lru.values()), "MRU timestamp not preserved across reopen"


def test_cpx28_size_accumulator_rebuilt_on_reopen(tmp_path):
    """The byte accumulator must be re-derived from disk on a fresh open (restart correctness)."""
    d = str(tmp_path)
    c1 = NEW(d, max_entries=1000, max_size_mb=None)
    for i in range(8):
        c1.set(f"{i:08x}", {"v": i, "pad": list(range(i))})
    c1.close()
    on_disk = _discovery_cache_bytes_total(c1)

    c2 = NEW(d, max_entries=1000, max_size_mb=None)
    c2._ensure_sizes()  # force the lazy rebuild
    assert c2._total_bytes == on_disk, "rebuilt accumulator disagrees with on-disk footprint"


def test_cpx28_eviction_survives_reopen(tmp_path):
    """Eviction after a reopen must use the rebuilt accumulator + ledger and stay under the cap."""
    d = str(tmp_path)
    c1 = NEW(d, max_entries=10, max_size_mb=None)
    for i in range(10):
        c1.set(f"{i:08x}", {"v": i})
    c1.close()
    assert len({f for f in os.listdir(d) if f.endswith(".pkl")}) == 10

    c2 = NEW(d, max_entries=10, max_size_mb=None)
    for i in range(10, 15):
        c2.set(f"{i:08x}", {"v": i})
    c2.close()
    surviving = {os.path.splitext(f)[0] for f in os.listdir(d) if f.endswith(".pkl")}
    assert len(surviving) == 10, "eviction did not hold the cap after reopen"
    # The 5 newest keys must survive; the 5 oldest-accessed must have evicted.
    assert {f"{i:08x}" for i in range(10, 15)} <= surviving
