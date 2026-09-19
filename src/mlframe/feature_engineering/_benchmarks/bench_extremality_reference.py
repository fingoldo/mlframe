"""Extremality-vs-reference scorer: v1 kernel (kept for comparison) against the production v2 path.

v1 walked a C-ordered matrix column by column with two binary searches per value over the full reference, serially.
v2 (``row_wise_extremality_reference.extremality_matrix_from_reference``) compresses each reference to distinct values,
scores contiguous columns with a nogil kernel on threads, and is bit-identical. Measured on 500k x 85 float32 with 30%
NaN: fit 3.3s -> 0.74s, score 10.6s -> 2.9s.

Run: python -m mlframe.feature_engineering._benchmarks.bench_extremality_reference
"""

from __future__ import annotations

import time

import numpy as np
import pandas as pd
from numba import njit


@njit(cache=True, parallel=False)
def _extremality_vs_reference_njit_v1(values: np.ndarray, ref_flat: np.ndarray, ref_starts: np.ndarray, ref_lens: np.ndarray, out: np.ndarray) -> None:
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
            # MID-RANK across the tie block: ``lo`` counts values strictly below v, ``hi`` counts values at or
            # below it, and the percentile is the midpoint. Using ``lo`` alone gives every row of a constant or
            # heavily-tied column a percentile of ~0, i.e. maximal extremality -- which collapsed the feature to a
            # constant and got it dropped by the zero-variance pre-screen on a production frame full of ratios,
            # counts and mostly-zero columns. The within-batch ranking spreads ties over the whole range instead;
            # neither is more "correct" per row, but the midpoint is the only one that puts the MODE at the median
            # where it belongs, and it is what makes a lone row score the same as it does in a batch.
            lo = 0
            hi = length
            while lo < hi:
                mid = (lo + hi) // 2
                if ref_flat[start + mid] < v:
                    lo = mid + 1
                else:
                    hi = mid
            upper = lo
            hi2 = length
            while upper < hi2:
                mid = (upper + hi2) // 2
                if ref_flat[start + mid] <= v:
                    upper = mid + 1
                else:
                    hi2 = mid
            frac = ((lo + upper) * 0.5 + 0.5) / denom
            out[i, j] = abs(frac - 0.5) * 2.0


def run_v1(frame: pd.DataFrame, reference: dict) -> np.ndarray:
    """Score ``frame`` against ``reference`` with the v1 kernel (same layout the v1 production path built)."""
    cols = list(reference)
    values = np.ascontiguousarray(frame[cols].to_numpy(dtype=np.float64))
    lens = np.array([reference[c].size for c in cols], dtype=np.int64)
    starts = np.zeros(len(cols), dtype=np.int64)
    starts[1:] = np.cumsum(lens)[:-1]
    flat = np.concatenate([reference[c] for c in cols]) if lens.sum() else np.empty(0)
    out = np.full(values.shape, np.nan)
    _extremality_vs_reference_njit_v1(values, np.ascontiguousarray(flat, dtype=np.float64), starts, lens, out)
    return out


def main() -> None:
    """Time v1 vs v2 and check they agree."""
    from mlframe.feature_engineering.row_wise_extremality_reference import extremality_matrix_from_reference, fit_extremality_reference

    rng = np.random.default_rng(0)
    x = rng.lognormal(size=(500_000, 85)).astype(np.float32)
    x[rng.random(x.shape) < 0.3] = np.nan
    frame = pd.DataFrame(x, columns=[f"c{i}" for i in range(85)])
    reference = fit_extremality_reference(frame)
    run_v1(frame.iloc[:100], reference)
    extremality_matrix_from_reference(frame.iloc[:100], reference)
    t = time.perf_counter()
    a = run_v1(frame, reference)
    t1 = time.perf_counter() - t
    t = time.perf_counter()
    b, _ = extremality_matrix_from_reference(frame, reference)
    t2 = time.perf_counter() - t
    print(f"v1 {t1:.2f}s  v2 {t2:.2f}s  identical={np.array_equal(a, b, equal_nan=True)}")


if __name__ == "__main__":
    main()
