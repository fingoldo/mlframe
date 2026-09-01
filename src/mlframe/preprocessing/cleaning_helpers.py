"""Small helpers carved out of cleaning.py to keep the parent module under the 1k-LOC ceiling."""

from __future__ import annotations

from typing import Callable

import pandas as pd


def map_elementwise_dedup(s: pd.Series, fcn: Callable, *, sample: int = 20_000, dup_ratio: float = 0.5) -> pd.Series:
    """Apply a pure elementwise ``fcn`` to ``s``, deduplicating the work when the column is low-cardinality.

    ``Series.map(callable)`` calls ``fcn`` once per ROW. For an object/string column with heavy value repetition
    (the common categorical-as-object case: countries, statuses, codes over millions of rows) that re-cleans the
    same handful of values millions of times. Mapping over the *unique* values and reindexing back is bit-identical
    for a pure elementwise ``fcn`` (mirrors the category path, which cleans the levels) and 1.7-2.6x faster.

    It is gated so the all-distinct worst case (where dedup would cost an extra full ``pd.unique`` pass) does not
    regress: a uniform-stride probe estimates cardinality cheaply; if the probe (or the eventual full unique set)
    shows the column is mostly distinct, it falls back to the plain row-wise ``map``. The stride probe — not a head
    slice — is used so head-clustered duplication followed by a distinct tail cannot mislead the estimate.
    """
    n = len(s)
    # Below the probe gate the dedup detour buys nothing and the dict path would not preserve the empty-series
    # object dtype that plain ``map`` keeps — defer to the row-wise map.
    if n < 4 * sample:
        return s.map(fcn)
    step = n // sample
    probe = s.iloc[::step]
    if probe.nunique(dropna=False) > dup_ratio * len(probe):
        return s.map(fcn)
    u = pd.unique(s)
    if len(u) > dup_ratio * n:
        return s.map(fcn)
    # Deduplication is only value-preserving when a value's IDENTITY is decided by equality, and in an object
    # column it is not: ``True == 1`` and ``hash(True) == hash(1)``, so a column holding both collapses them
    # into one key and every such row gets whichever of the two was seen first. ``Decimal(1) == 1`` collapses
    # the same way, and ``pd.factorize`` does it too -- the collision lives in pandas' hash table, not in the
    # dict, so it cannot be detected after the fact from ``u`` (which has already lost one of the pair).
    #
    # Worse than being wrong, it was wrong ONLY above the 4*sample gate: the same column returned per-row
    # results below 80k rows and collapsed results above it.
    #
    # So the fast path is restricted to what it was written for -- string-valued object columns (countries,
    # statuses, codes), where no value can hash-collide with an unequal-typed one. Everything else takes the
    # row-wise map, which is correct by construction.
    if not all(v is None or isinstance(v, str) or v != v for v in u):
        return s.map(fcn)
    mapping = {v: fcn(v) for v in u}
    return s.map(mapping)


__all__ = ["map_elementwise_dedup"]
