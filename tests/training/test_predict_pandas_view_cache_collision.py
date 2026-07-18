"""TC10 regression: the predict-path pandas-view cache must be collision-safe.

``_ensure_pandas_view`` caches polars->pandas conversions keyed by ``id(df)``.
CPython recycles ``id()`` once an object is GC'd, so a transient polars frame
freed mid-predict can hand its address to an unrelated later frame; keying on
``id`` alone then serves the freed frame's STALE pandas view for different data
-> silently wrong predictions. The fix folds a weakref to the source frame into
the entry and recomputes when the stored ref no longer resolves to the live
object. These tests pin both the collision-safety and the still-working
same-object cache hit.
"""

from __future__ import annotations

import gc

import polars as pl

import mlframe.training.core.predict as P


def _deep_copy_view(df, *args, **kwargs):
    """Returns an independently-owned pandas copy (no shared Arrow buffers), isolating cache-key collisions from view-sharing."""
    # Independent (non-zero-copy) view per call so the test isolates the CACHE
    # key collision from polars' shared-Arrow-buffer reuse on a freed frame.
    return df.to_pandas().copy(deep=True)


def test_tc10_recycled_id_does_not_serve_stale_view(monkeypatch) -> None:
    """A new frame reusing a freed frame's id must NOT get the freed frame's view.

    CPython recycles ``id()`` once an object is GC'd, so a cache keyed on ``id``
    alone can serve a freed frame's stale view for different data. Rather than
    spin on chance id-recycling (which is platform/allocator-dependent and never
    materialised within a bounded loop on some CPython builds), we deterministically
    inject the exact collision: a cache entry stored under the LIVE frame's id whose
    weakref resolves to a DIFFERENT object (mismatched live ref) or to nothing
    (dead ref). Both must be treated as a miss and recompute the correct view.
    """
    import weakref

    monkeypatch.setattr(P, "get_pandas_view_of_polars_df", _deep_copy_view)

    # Case 1: mismatched LIVE ref. ``other`` stands in for the freed frame whose id
    # got recycled to ``b``; its stale view carries 111. The stored weakref resolves
    # to ``other`` (not ``b``), so the lookup must miss and recompute ``b``'s 222.
    cache: dict = {}
    b = pl.DataFrame({"a": [222, 222, 222]})
    other = pl.DataFrame({"a": [111, 111, 111]})
    stale_view = other.to_pandas().copy(deep=True)
    cache[id(b)] = (weakref.ref(other), stale_view)
    got = P._ensure_pandas_view(b, cache)["a"].to_list()[0]
    assert got == 222, f"stale view served on mismatched-live-ref collision: got {got}"

    # Case 2: DEAD ref. The source frame the entry belonged to has been GC'd, so its
    # weakref no longer resolves; the recycled id must still recompute the live data.
    tmp = pl.DataFrame({"a": [111, 111, 111]})
    dead_ref = weakref.ref(tmp)
    del tmp
    gc.collect()
    cache2: dict = {}
    c = pl.DataFrame({"a": [333, 333, 333]})
    cache2[id(c)] = (dead_ref, stale_view)
    got2 = P._ensure_pandas_view(c, cache2)["a"].to_list()[0]
    assert got2 == 333, f"stale view served on dead-ref collision: got {got2}"


def test_tc10_same_live_frame_is_cache_hit(monkeypatch) -> None:
    """The cache must still serve ONE conversion for repeated calls on the SAME live frame."""
    calls = {"n": 0}

    def _counting_view(df, *args, **kwargs):
        """Counts calls to the pandas-view converter, to prove repeated calls on the same live frame hit the cache."""
        calls["n"] += 1
        return df.to_pandas()

    monkeypatch.setattr(P, "get_pandas_view_of_polars_df", _counting_view)

    cache: dict = {}
    df = pl.DataFrame({"a": [1, 2, 3]})
    v1 = P._ensure_pandas_view(df, cache)
    v2 = P._ensure_pandas_view(df, cache)
    assert v1 is v2
    assert calls["n"] == 1, "same live frame must convert exactly once"
