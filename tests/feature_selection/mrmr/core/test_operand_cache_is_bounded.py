"""The pair-ranking operand cache has a ceiling, and stores what its consumers actually read.

``_cached_operand`` memoized a full-length float column per operand index and never evicted. The usability PRESCAN is capped, but the
gate-failure path asks for an operand for every pair with a positive pair MI, so the cache could end up holding one column per operand in the
pool: at 5000 operands and a million rows that is tens of gigabytes standing behind a memoization whose stated justification was CPU time, in
a package whose own discipline is that frames reach 100+ GB.

Least-recently-used eviction keeps the win, because a small pool of raw operands recurs across a much larger pool of pairs, so the working set
is small even when the pool is not.
"""

from __future__ import annotations

from collections import OrderedDict

import numpy as np
import pytest

from mlframe.feature_selection.filters._fe_usability_signal import _crit_np_dtype


def _bounded_cache(limit: int):
    """The cache shape ``_cached_operand`` uses: an LRU keyed by operand index, storing at the consumers' dtype."""
    cache: OrderedDict = OrderedDict()
    built: list = []

    def cached_operand(idx, n_rows=64):
        """Memoize one operand column, evicting the least recently used once the ceiling is reached."""
        hit = cache.get(idx)
        if hit is not None:
            cache.move_to_end(idx)
            return hit
        built.append(idx)
        val = np.asarray(np.full(n_rows, float(idx)), dtype=_crit_np_dtype())
        cache[idx] = val
        if len(cache) > limit:
            cache.popitem(last=False)
        return val

    return cache, built, cached_operand


def test_the_cache_never_grows_past_its_ceiling():
    """Touching far more operands than the ceiling must not grow the cache past it."""
    limit = 8
    cache, _built, cached_operand = _bounded_cache(limit)
    for idx in range(200):
        cached_operand(idx)
    assert len(cache) == limit, f"the cache holds {len(cache)} entries for a ceiling of {limit}"


def test_a_recurring_working_set_still_hits():
    """The memoization exists because a small operand pool recurs across many pairs; that case must not start missing."""
    cache, built, cached_operand = _bounded_cache(8)
    for _round in range(20):
        for idx in range(4):
            cached_operand(idx)
    assert built == [0, 1, 2, 3], f"a working set well inside the ceiling was rebuilt: {built}"
    assert len(cache) == 4


def test_the_least_recently_used_entry_is_the_one_evicted():
    """Eviction order is LRU, so the entry a caller keeps touching survives."""
    cache, _built, cached_operand = _bounded_cache(3)
    for idx in (0, 1, 2):
        cached_operand(idx)
    cached_operand(0)  # 0 is now the most recently used, 1 the least
    cached_operand(3)
    assert 1 not in cache, f"evicted the wrong entry: {sorted(cache)}"
    assert set(cache) == {0, 2, 3}


def test_cached_columns_are_stored_at_the_dtype_the_consumers_read():
    """Both consumers cast to the usability dtype, so storing it is bit-identical and smaller."""
    cache, _built, cached_operand = _bounded_cache(4)
    col = cached_operand(0, n_rows=100)
    assert col.dtype == _crit_np_dtype()
    assert cache[0].nbytes == 100 * np.dtype(_crit_np_dtype()).itemsize


@pytest.mark.parametrize("limit", [1, 2, 16])
def test_the_ceiling_holds_at_any_size(limit):
    """Whatever the configured bound, the cache respects it."""
    cache, _built, cached_operand = _bounded_cache(limit)
    for idx in range(limit * 5):
        cached_operand(idx)
    assert len(cache) == limit
