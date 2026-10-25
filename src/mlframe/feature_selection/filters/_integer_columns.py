"""One definition of an integer column for the integer-structure FE sweeps (pairwise modular, integer lattice), and one list of them per frame.

An integer column is an integer dtype, a float column whose values are all whole numbers, or a float column measured with a little noise: its values cluster at integers
(mean distance to the nearest integer at most ``NEAR_INT_MEAN_DIST``; a continuous column spreads evenly, mean 0.25) and take at least two distinct rounded values. The sweeps work
on the ROUNDED values (``to_int64``), never on truncated ones.

The eligibility test reads every column once, which costs seconds on a million-row frame, and both sweeps run in the same fit. ``eligible_integer_columns`` therefore
remembers its answer per frame (a small LRU keyed by the frame identity, shape, column names and a fingerprint of 64 evenly spaced rows), so the second sweep gets the list
for free. The decision for a noisy column is made on at most ``NEAR_INT_DECISION_ROWS`` evenly spaced rows: it classifies the column, it does not select among candidates.
"""

from __future__ import annotations

import logging
from collections import OrderedDict
from typing import Any, Optional, Sequence, cast

import numpy as np

logger = logging.getLogger(__name__)

NEAR_INT_MEAN_DIST = 0.12
NEAR_INT_MIN_ROWS = 50
NEAR_INT_DECISION_ROWS = 20_000
_EXACT_PROBE_ROWS = 2_000  # a float column with a fractional value among its first rows is not exactly integer; the full pass runs only for the survivors
_CACHE_SIZE = 4

__all__ = ["NEAR_INT_MEAN_DIST", "eligible_integer_columns", "is_integer_col", "is_integer_like_col", "to_int64"]


def is_integer_col(x: np.ndarray) -> bool:
    """True iff x is integer-typed OR exactly-integer-valued float (no fractional part), finite."""
    a = np.asarray(x)
    if np.issubdtype(a.dtype, np.integer):
        return True
    if not np.issubdtype(a.dtype, np.floating):
        return False
    head = a[:_EXACT_PROBE_ROWS]
    head = head[np.isfinite(head)]
    if head.size and not np.all(head == np.floor(head)):
        return False  # decided on the first rows; the common non-integer float column costs a probe, not a full pass
    finite = a[np.isfinite(a)]
    return bool(finite.size > 0 and np.all(finite == np.floor(finite)))


def is_integer_like_col(x: np.ndarray) -> bool:
    """True iff ``is_integer_col`` or the float values sit within noise of integers (see the module docstring) and take at least two distinct rounded values."""
    if is_integer_col(x):
        return True
    a = np.asarray(x)
    if not np.issubdtype(a.dtype, np.floating):
        return False
    if a.shape[0] > NEAR_INT_DECISION_ROWS:
        a = a[:: int(np.ceil(a.shape[0] / NEAR_INT_DECISION_ROWS))]
    finite = a[np.isfinite(a)]
    if finite.size < NEAR_INT_MIN_ROWS:
        return False
    rounded = np.rint(finite)
    return bool(float(np.mean(np.abs(finite - rounded))) <= NEAR_INT_MEAN_DIST and rounded.max() > rounded.min())


def to_int64(x: Any) -> np.ndarray:
    """The column as int64 codes: floats are ROUNDED to the nearest integer (a noisy integer column), not truncated; exact integers are unchanged."""
    a = np.asarray(x)
    if np.issubdtype(a.dtype, np.floating):
        with np.errstate(invalid="ignore"):  # NaN / inf rows become INT64_MIN; the replay functions mask them to NaN afterwards
            return cast(np.ndarray, np.rint(a).astype(np.int64))
    return cast(np.ndarray, a.astype(np.int64))


_cache: "OrderedDict[tuple, tuple[bytes, dict]]" = OrderedDict()


def _fingerprint(X: Any, columns: Sequence[str]) -> bytes:
    """A cheap fingerprint of the frame's values: 64 evenly spaced rows of every column, hashed."""
    n = len(X)
    idx = np.linspace(0, n - 1, num=min(n, 64)).astype(np.int64) if n else np.zeros(0, dtype=np.int64)
    h = 0
    for c in columns:
        h = hash((h, np.asarray(X[c])[idx].tobytes()))
    return h.to_bytes(8, "little", signed=True)


def eligible_integer_columns(X: Any, cols: Optional[Sequence[str]] = None) -> list[str]:
    """The columns of ``X`` (restricted to ``cols`` when given, order kept) that are integer columns by ``is_integer_like_col``.

    The answer is remembered per frame, so the second integer-structure sweep of a fit reads it instead of scanning the columns again.
    """
    columns = list(X.columns)
    wanted = columns if cols is None else [c for c in cols if c in X.columns]
    key = (id(X), len(X), tuple(columns))
    fp = _fingerprint(X, columns)
    entry = _cache.get(key)
    if entry is None or entry[0] != fp:
        entry = (fp, {})
        _cache[key] = entry
        while len(_cache) > _CACHE_SIZE:
            _cache.popitem(last=False)
    else:
        _cache.move_to_end(key)
    known = entry[1]
    for c in wanted:
        if c not in known:
            known[c] = is_integer_like_col(np.asarray(X[c]))
    return [c for c in wanted if known[c]]
