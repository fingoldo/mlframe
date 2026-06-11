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
    # Independent (non-zero-copy) view per call so the test isolates the CACHE
    # key collision from polars' shared-Arrow-buffer reuse on a freed frame.
    return df.to_pandas().copy(deep=True)


def test_tc10_recycled_id_does_not_serve_stale_view(monkeypatch) -> None:
    """A new frame reusing a freed frame's id must NOT get the freed frame's view."""
    monkeypatch.setattr(P, "get_pandas_view_of_polars_df", _deep_copy_view)

    reuses = 0
    for _ in range(20000):
        cache: dict = {}
        a = pl.DataFrame({"a": [111, 111, 111]})
        ia = id(a)
        P._ensure_pandas_view(a, cache)  # populates cache[id(a)]
        del a
        gc.collect()
        b = pl.DataFrame({"a": [222, 222, 222]})
        if id(b) == ia:
            reuses += 1
            got = P._ensure_pandas_view(b, cache)["a"].to_list()[0]
            assert got == 222, f"stale view: b carries 222 but cache served {got}"
            if reuses >= 5:
                break
        del b

    assert reuses > 0, "test did not exercise an id()-reuse collision; tighten the loop"


def test_tc10_same_live_frame_is_cache_hit(monkeypatch) -> None:
    """The cache must still serve ONE conversion for repeated calls on the SAME live frame."""
    calls = {"n": 0}

    def _counting_view(df, *args, **kwargs):
        calls["n"] += 1
        return df.to_pandas()

    monkeypatch.setattr(P, "get_pandas_view_of_polars_df", _counting_view)

    cache: dict = {}
    df = pl.DataFrame({"a": [1, 2, 3]})
    v1 = P._ensure_pandas_view(df, cache)
    v2 = P._ensure_pandas_view(df, cache)
    assert v1 is v2
    assert calls["n"] == 1, "same live frame must convert exactly once"
