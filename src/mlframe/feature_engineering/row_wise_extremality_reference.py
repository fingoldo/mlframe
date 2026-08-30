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
from typing import Dict, Optional, Sequence

import numpy as np
import pandas as pd
from numba import njit

logger = logging.getLogger(__name__)

# Cap on the rows kept per column. The reference only has to describe the DISTRIBUTION, and a uniform stride
# over the sorted values preserves every quantile to within one sample step -- keeping all 2M rows would cost
# memory and buy nothing measurable. Sampling the SORTED array (not the raw rows) keeps it deterministic.
DEFAULT_MAX_REFERENCE_ROWS = 100_000


@njit(cache=True, parallel=False)
def _extremality_vs_reference_njit(values: np.ndarray, ref_flat: np.ndarray, ref_starts: np.ndarray, ref_lens: np.ndarray, out: np.ndarray) -> None:
    """Fill ``out`` with ``|p - 0.5| * 2`` where ``p`` is each value's position in its column's reference.

    Serial for the same reason the within-batch kernel is: a ``parallel=True`` twin caused a Windows access
    violation when numba's threading layer ran alongside CatBoost's own during the preprocessing step.
    """
    n_rows, n_cols = values.shape
    for j in range(n_cols):
        start = ref_starts[j]
        length = ref_lens[j]
        if length == 0:
            continue
        denom = length + 1.0
        for i in range(n_rows):
            v = values[i, j]
            if np.isnan(v):
                continue
            lo = 0
            hi = length
            while lo < hi:
                mid = (lo + hi) // 2
                if ref_flat[start + mid] < v:
                    lo = mid + 1
                else:
                    hi = mid
            # ``lo`` values sit strictly below v; +0.5 centres the value inside its own tie block so a value
            # equal to the reference median scores 0 rather than half a step off it.
            frac = (lo + 0.5) / denom
            out[i, j] = abs(frac - 0.5) * 2.0


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
    reference: Dict[str, np.ndarray] = {}
    for col in cols:
        vals = np.asarray(X[col].to_numpy(dtype=np.float64))
        vals = vals[np.isfinite(vals)]
        vals.sort()
        if vals.size > max_reference_rows:
            # Uniform stride over the SORTED values: keeps the quantile grid even, and is reproducible.
            idx = np.linspace(0, vals.size - 1, max_reference_rows).astype(np.int64)
            vals = vals[idx]
        reference[col] = vals
    return reference


def extremality_matrix_from_reference(X: pd.DataFrame, reference: Dict[str, np.ndarray], columns: Optional[Sequence[str]] = None):
    """``(n_rows, n_cols)`` extremality matrix scored against ``reference``; NaN where the source value was NaN.

    Columns absent from the reference are scored NaN rather than silently re-ranked within the batch -- a
    column the fit never saw has no reference distribution, and inventing one would restore the skew.
    """
    cols = list(columns) if columns is not None else list(X.select_dtypes(include=[np.number]).columns)
    values = np.ascontiguousarray(X[cols].to_numpy(dtype=np.float64))
    n_rows, n_cols = values.shape
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

    lens = np.array([int(reference.get(c, _EMPTY).size) for c in cols], dtype=np.int64)
    starts = np.zeros(n_cols, dtype=np.int64)
    if n_cols > 1:
        starts[1:] = np.cumsum(lens)[:-1]
    flat = np.concatenate([reference.get(c, _EMPTY) for c in cols]) if lens.sum() else _EMPTY
    _extremality_vs_reference_njit(values, np.ascontiguousarray(flat, dtype=np.float64), starts, lens, out)
    return out, cols


_EMPTY = np.empty(0, dtype=np.float64)

__all__ = [
    "DEFAULT_MAX_REFERENCE_ROWS",
    "extremality_matrix_from_reference",
    "fit_extremality_reference",
]
