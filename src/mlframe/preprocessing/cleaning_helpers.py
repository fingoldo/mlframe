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
    mapping = {v: fcn(v) for v in u}
    return s.map(mapping)


__all__ = ["map_elementwise_dedup"]
