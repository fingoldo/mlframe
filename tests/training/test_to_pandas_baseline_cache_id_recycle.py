"""Regression: the polars->pandas baseline view cache must not serve a stale view when id() is recycled onto a new same-shape frame."""
import polars as pl

from mlframe.training.baselines import _dummy_baseline_compute as m


def test_recycled_id_same_schema_does_not_return_stale_view():
    """A recycled id on a same-schema frame does not reuse the old view."""
    m._TO_PANDAS_BASELINE_CACHE = None
    a = pl.DataFrame({"c": ["x", "y", "z"]})
    va = m._to_pandas_for_baseline(a)
    key = id(a)
    # Simulate GC + id recycling: a different frame with identical columns/shape under the same cache key.
    b = pl.DataFrame({"c": ["p", "q", "r"]})
    entry = m._TO_PANDAS_BASELINE_CACHE.pop(key)
    m._TO_PANDAS_BASELINE_CACHE[id(b)] = entry
    vb = m._to_pandas_for_baseline(b)
    assert list(vb["c"]) == ["p", "q", "r"]
    assert list(va["c"]) == ["x", "y", "z"]
