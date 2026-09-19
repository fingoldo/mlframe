"""Extremality measured against a FIT-TIME reference instead of against whatever rows happen to be present.

``_compute_extremality_matrix`` ranks each column WITHIN the frame it is handed. That makes the feature depend
on the batch: the same row scored ``[0.808, 0.793, 0.653]`` inside a 50k-row split and ``[0.0, 0.0, 0.0]`` scored
alone, because a single row is its own median. Train, val and test were each ranked against themselves, and a
production request scoring one row got a third answer again -- a train/serve skew in a default-on feature.

Fixing the definition also removes the dominant cost. Ranking within the batch is a full argsort per column over
every row (1.9 minutes on a 2.18M x ~90 frame in one production log); scoring against a stored reference is a
binary search per value, and the reference itself is built once, from a bounded sample.

The reference is the sorted finite values of each column at fit time. ``percentile = searchsorted / (n + 1)``
mirrors the within-batch convention closely enough that the two agree to O(1/n) on the fitting data itself,
and the extremality is the same ``|percentile - 0.5| * 2``.
"""

from __future__ import annotations

import logging
import os
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from numba import njit

logger = logging.getLogger(__name__)

# Cap on the rows kept per column. The reference only has to describe the DISTRIBUTION, and a uniform stride
# over the sorted values preserves every quantile to within one sample step -- keeping all 2M rows would cost
# memory and buy nothing measurable. Sampling the SORTED array (not the raw rows) keeps it deterministic.
DEFAULT_MAX_REFERENCE_ROWS = 100_000

# Below this many cells the thread-pool start-up outweighs the per-column work; score serially.
_THREADED_MIN_CELLS = 200_000


@njit(cache=True, nogil=True)
def _extremality_column_vs_unique_reference_njit(
    col: np.ndarray, uniq: np.ndarray, count_below: np.ndarray, count_at: np.ndarray, denom: float, out_col: np.ndarray
) -> None:
    """One column's ``|p - 0.5| * 2`` against a tie-compressed reference; bit-identical to the v1 mid-rank.

    ``uniq`` holds the distinct reference values, ``count_below[u]`` how many reference values are strictly below
    ``uniq[u]`` and ``count_at[u]`` how many equal it. One binary search over the distinct values yields both the
    strict and the inclusive count that v1 found with two searches over the full reference, and on tie-heavy columns
    (ratios, counts, mostly-zero columns) the search space shrinks from 100k values to a handful. ``col`` and
    ``out_col`` are contiguous, unlike v1's strided walk down a C-ordered matrix. ``nogil`` lets the caller score
    columns on Python threads without numba's threading layer, which crashed alongside CatBoost's.
    """
    m = uniq.shape[0]
    for i in range(col.shape[0]):
        v = col[i]
        if np.isnan(v):
            continue
        lo = 0
        hi = m
        while lo < hi:
            mid = (lo + hi) // 2
            if uniq[mid] < v:
                lo = mid + 1
            else:
                hi = mid
        if lo < m:
            below = count_below[lo]
            upper = below + count_at[lo] if uniq[lo] == v else below
        else:
            below = count_below[m - 1] + count_at[m - 1]
            upper = below
        frac = ((below + upper) * 0.5 + 0.5) / denom
        out_col[i] = abs(frac - 0.5) * 2.0


def _compress_reference(ref: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Distinct values of a sorted reference with the count strictly below and the count equal to each."""
    uniq, count_at = np.unique(ref, return_counts=True)
    count_below = np.zeros(uniq.size, dtype=np.int64)
    if uniq.size > 1:
        count_below[1:] = np.cumsum(count_at)[:-1]
    return uniq, count_below, count_at.astype(np.int64)


def fit_extremality_reference(
    X: pd.DataFrame,
    columns: Optional[Sequence[str]] = None,
    *,
    max_reference_rows: int = DEFAULT_MAX_REFERENCE_ROWS,
) -> Dict[str, np.ndarray]:
    """Sorted per-column reference values, to be reused for every later frame.

    Columns with no finite value get an empty array and score NaN later, matching the within-batch behaviour
    for an all-NaN column.
    """
    cols = list(columns) if columns is not None else list(X.select_dtypes(include=[np.number]).columns)
    sorted_refs: List[np.ndarray] = [_EMPTY] * len(cols)

    def _fit(j: int) -> None:
        vals = np.asarray(X[cols[j]].to_numpy(dtype=np.float64, na_value=np.nan))
        vals = vals[np.isfinite(vals)]
        vals.sort()
        if vals.size > max_reference_rows:
            # Uniform stride over the SORTED values: keeps the quantile grid even, and is reproducible.
            idx = np.linspace(0, vals.size - 1, max_reference_rows).astype(np.int64)
            vals = vals[idx]
        sorted_refs[j] = vals

    # Per-column sorts release the GIL, so threads cut the 3.3s serial fit on 500k x 85.
    _map_columns(_fit, len(cols), len(X))
    return dict(zip(cols, sorted_refs))


def _map_columns(fn, n_cols: int, n_rows: int) -> None:
    """Run ``fn(j)`` for every column, on threads when the frame is big enough to pay for them."""
    n_threads = min(n_cols, os.cpu_count() or 1)
    if n_threads > 1 and n_rows * n_cols >= _THREADED_MIN_CELLS:
        with ThreadPoolExecutor(max_workers=n_threads) as pool:
            list(pool.map(fn, range(n_cols)))
    else:
        for j in range(n_cols):
            fn(j)


def extremality_matrix_from_reference(
    X: pd.DataFrame, reference: Dict[str, np.ndarray], columns: Optional[Sequence[str]] = None
) -> Tuple[np.ndarray, List[str]]:
    """``(n_rows, n_cols)`` extremality matrix scored against ``reference``; NaN where the source value was NaN.

    Columns absent from the reference are scored NaN rather than silently re-ranked within the batch -- a
    column the fit never saw has no reference distribution, and inventing one would restore the skew.
    """
    cols = list(columns) if columns is not None else list(X.select_dtypes(include=[np.number]).columns)
    n_rows, n_cols = len(X), len(cols)
    out = np.full((n_rows, n_cols), np.nan, dtype=np.float64)
    if n_cols == 0 or n_rows == 0:
        return out, cols

    _missing = [c for c in cols if c not in reference]
    if _missing:
        logger.info(
            "extremality: %d column(s) have no fit-time reference and score NaN (%s). Re-ranking them within "
            "this frame would make the score depend on which rows are present.",
            len(_missing), ", ".join(_missing[:10]) + (", ..." if len(_missing) > 10 else ""),
        )

    # v2: tie-compressed reference, contiguous columns, columns scored on threads (nogil kernel). The v1 kernel
    # (``_benchmarks/bench_extremality_reference.py::_extremality_vs_reference_njit_v1``, kept) walked a C-ordered matrix column by column with two binary searches
    # per value; it was 8.7s of a 10.6s call on 500k x 85 and 23.9s of a production preprocessing phase.
    # Each thread reads its own column straight from the frame: no C-ordered float64 copy of the whole frame and no
    # Fortran re-layout of it (0.66s of the remaining 4.3s when the matrix was built first).
    out_f = np.full((n_rows, n_cols), np.nan, dtype=np.float64, order="F")

    def _score(j: int) -> None:
        ref = reference.get(cols[j], _EMPTY)
        if ref.size == 0:
            return
        uniq, count_below, count_at = _compress_reference(ref)
        col = np.ascontiguousarray(X[cols[j]].to_numpy(dtype=np.float64, na_value=np.nan))
        _extremality_column_vs_unique_reference_njit(col, uniq, count_below, count_at, float(ref.size) + 1.0, out_f[:, j])

    _map_columns(_score, n_cols, n_rows)
    out[...] = out_f
    return out, cols


_EMPTY = np.empty(0, dtype=np.float64)

__all__ = [
    "DEFAULT_MAX_REFERENCE_ROWS",
    "extremality_matrix_from_reference",
    "fit_extremality_reference",
]
