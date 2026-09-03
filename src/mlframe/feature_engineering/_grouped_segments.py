"""``iter_group_segments``: the shared group-sort/segment primitive both ``grouped.py`` and
``grouped_rank.py`` build on. Carved into its own dependency-free module (monolith split,
CLAUDE.md "sibling re-export" convention) specifically so those two siblings can each depend on
THIS module without depending on each other -- ``grouped.py`` re-exports ``per_group_rank``/
``per_group_sliding_window`` from ``grouped_rank.py``, so a ``grouped_rank -> grouped`` import
back would create an import cycle.
"""

from __future__ import annotations

from typing import Tuple, cast

import numpy as np

try:
    from numba import njit

    HAS_NUMBA = True
except ImportError:
    HAS_NUMBA = False

    def njit(*args, **kwargs):  # no-op fallback so the module imports without numba
        """No-op decorator stand-in for ``numba.njit`` when numba is unavailable, so the module still imports and runs (slower, pure Python)."""

        def wrap(fn):
            """Identity decorator applied when ``njit`` is called with arguments (e.g. ``@njit(cache=True)``)."""
            return fn

        if args and callable(args[0]):
            return args[0]
        return wrap


__all__ = ["iter_group_segments", "HAS_NUMBA", "njit"]


@njit(cache=True)
def _stable_counting_segments_int(g, gmin, span):
    """Stable counting sort of integer group ids in O(n + span).

    Returns ``(sort_idx, starts, ends)`` identical to
    ``np.argsort(g, kind="stable")`` + boundary detection: rows are ordered by
    ``(group_id, original_index)``, so within-group original order is preserved.
    Only valid for integer keys with a bounded ``span`` (the caller gates on RAM).
    """
    n = g.shape[0]
    counts = np.zeros(span + 1, dtype=np.int64)
    for i in range(n):
        counts[g[i] - gmin] += 1
    offsets = np.empty(span + 1, dtype=np.int64)
    acc = 0
    nonempty = 0
    for b in range(span + 1):
        offsets[b] = acc
        if counts[b] > 0:
            nonempty += 1
        acc += counts[b]
    sort_idx = np.empty(n, dtype=np.intp)
    cursor = offsets.copy()
    for i in range(n):
        b = g[i] - gmin
        sort_idx[cursor[b]] = i
        cursor[b] += 1
    starts = np.empty(nonempty, dtype=np.intp)
    ends = np.empty(nonempty, dtype=np.intp)
    k = 0
    for b in range(span + 1):
        if counts[b] > 0:
            starts[k] = offsets[b]
            ends[k] = offsets[b] + counts[b]
            k += 1
    return sort_idx, starts, ends


def iter_group_segments(
    group_ids: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Sort + segment by group_id.

    Returns ``(sort_idx, starts, ends)`` such that for each group
    ``i``, the rows in the sorted order are ``sort_idx[starts[i]:ends[i]]``.
    Within each group the ORIGINAL row order is preserved (stable sort).

    O(n log n) time, O(n) memory. Standalone (callers that need the
    raw indices for custom iteration use this directly without
    paying for the ``per_group_apply`` callback machinery).
    """
    g = np.ascontiguousarray(group_ids)
    n = g.size
    if n == 0:
        return (
            np.empty(0, dtype=np.intp),
            np.empty(0, dtype=np.intp),
            np.empty(0, dtype=np.intp),
        )
    # Integer keys with a bounded value span use an O(n) stable counting sort instead of the
    # O(n log n) ``argsort(kind="stable")``; this is the shared bottleneck of every per-group helper
    # here (47-100x on the segmentation step @10M, see _benchmarks/bench_group_sort.py). The output is
    # bit-identical (rows ordered by (group_id, original_index)). Gated on ``span <= 4n + 1M`` so the
    # ``span+1`` counts array stays RAM-safe; sparse / huge-span / non-integer keys keep the argsort path.
    if HAS_NUMBA and np.issubdtype(g.dtype, np.integer) and n > 1:
        gmin = int(g.min())
        span = int(g.max()) - gmin
        if 0 <= span <= 4 * n + 1_000_000:
            return cast(Tuple[np.ndarray, np.ndarray, np.ndarray], _stable_counting_segments_int(g, gmin, span))

    sort_idx = np.argsort(g, kind="stable")
    g_sorted = g[sort_idx]
    bnd = np.where(g_sorted[1:] != g_sorted[:-1])[0] + 1
    starts = np.concatenate(([0], bnd)).astype(np.intp)
    ends = np.concatenate((bnd, [n])).astype(np.intp)
    return sort_idx, starts, ends
