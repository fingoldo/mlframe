"""Multi-core host primitives for the large-column binning the FE stages repeat per column.

``np.searchsorted`` of a 1M-row column against a handful of bin edges is a single-threaded binary search per element (~24 ms); the FE stages do it for
every group / source column. The parallel kernel below does the same comparison-only search across all cores (~5 ms) and returns identical integers: for any
finite edges it is exactly ``np.searchsorted(edges, x, side="right")``, with numpy's NaN rule (a NaN sorts after every edge, so it lands in the last slot).
"""

from __future__ import annotations

import numpy as np
from numba import njit, prange

from mlframe._numba_parallel_guard import parallel_kernel_entry

# Below this many elements numpy's own call is cheaper than waking the thread pool.
_PARALLEL_MIN_N = 200_000


@njit(parallel=True, nogil=True, cache=True)
def _searchsorted_right_par(edges: np.ndarray, x: np.ndarray, out: np.ndarray) -> None:
    """``out[i] = number of edges <= x[i]`` (a NaN takes the last slot, as numpy places it after every edge)."""
    ne = edges.shape[0]
    for i in prange(x.shape[0]):
        v = x[i]
        if v != v:
            out[i] = ne
            continue
        lo = 0
        hi = ne
        while lo < hi:
            mid = (lo + hi) >> 1
            if edges[mid] <= v:
                lo = mid + 1
            else:
                hi = mid
        out[i] = lo


def searchsorted_right(edges: np.ndarray, x: np.ndarray) -> np.ndarray:
    """``np.searchsorted(edges, x, side="right")`` for ascending 1-D ``edges`` and a 1-D float column, multi-core for large columns."""
    x = np.asarray(x)
    if x.size < _PARALLEL_MIN_N or x.dtype.kind != "f":
        return np.searchsorted(edges, x, side="right")
    e = np.ascontiguousarray(edges, dtype=np.float64)
    xc = np.ascontiguousarray(x, dtype=np.float64)
    out = np.empty(xc.shape[0], dtype=np.int64)
    with parallel_kernel_entry():
        _searchsorted_right_par(e, xc, out)
    return out


def count_distinct_int(a: np.ndarray) -> int:
    """Number of distinct values of an integer vector: an O(n) occupancy count for a small value range, ``np.unique`` otherwise (identical result)."""
    a = np.asarray(a).ravel()
    if a.size == 0:
        return 0
    if a.dtype.kind in "iub":
        lo = int(a.min())
        span = int(a.max()) - lo + 1
        if span <= (1 << 22):
            return int(np.count_nonzero(np.bincount((a.astype(np.int64, copy=False) - lo), minlength=span)))
    return int(np.unique(a).size)


class ContentMemo:
    """Small thread-safe LRU of derived arrays keyed by the CONTENT of the input column (+ a caller tag). Fit-constant columns (the target above all) are
    re-derived by many stages with a fresh copy each time; the result is cached on a content hash and returned as a private copy so a caller can mutate it."""

    def __init__(self, max_entries: int = 8) -> None:
        """Create an empty memo holding at most ``max_entries`` results."""
        import threading
        from collections import OrderedDict

        self._data: "OrderedDict[tuple, np.ndarray]" = OrderedDict()
        self._lock = threading.Lock()
        self._max = int(max_entries)

    def get_or_compute(self, tag: str, arr: np.ndarray, compute) -> np.ndarray:
        """``compute(arr)`` memoised on ``(tag, dtype, shape, content hash of arr)``."""
        from ._fe_resident_operands import _content_hash

        a = np.ascontiguousarray(np.asarray(arr))
        key = (tag, a.dtype.str, a.shape, int(_content_hash(a)))
        with self._lock:
            hit = self._data.get(key)
            if hit is not None:
                self._data.move_to_end(key)
                return hit.copy()
        out = np.asarray(compute(a))
        with self._lock:
            self._data[key] = out.copy()
            while len(self._data) > self._max:
                self._data.popitem(last=False)  # evict-ok: memo; a miss recomputes the value
        return out
