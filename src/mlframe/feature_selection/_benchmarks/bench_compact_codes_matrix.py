"""Micro-benchmark for the COMPACT CODES STORAGE downcast + the categorical cardinality cap.

Measures, on a representative MRMR codes-matrix shape (default 200k rows x 400 cols):
  * bytes of the int32 codes matrix vs the compact int8 downcast (the memory win),
  * wall time of the range-checked downcast (one min/max pass + astype),
  * wall time of cap_categorical_cardinality on a high-cardinality categorical block.

Run:  python -m mlframe.feature_selection._benchmarks.bench_compact_codes_matrix
"""
from __future__ import annotations

import time

import numpy as np

from mlframe.feature_selection.filters.discretization import cap_categorical_cardinality


def _bench(n_rows: int = 200_000, n_cols: int = 400, n_bins: int = 10, cap: int = 127, reps: int = 5) -> None:
    rng = np.random.default_rng(0)
    # Numeric-style codes 0..n_bins-1 (the common compact case) as int32 (the pre-downcast storage).
    data = rng.integers(0, n_bins, size=(n_rows, n_cols)).astype(np.int32)

    def downcast(mat):
        dmin, dmax = int(mat.min()), int(mat.max())
        if -128 <= dmin and dmax <= 127:
            return mat.astype(np.int8)
        if -32768 <= dmin and dmax <= 32767:
            return mat.astype(np.int16)
        return mat

    t = []
    for _ in range(reps):
        s = time.perf_counter()
        narrow = downcast(data)
        t.append(time.perf_counter() - s)
    assert np.array_equal(narrow.astype(np.int64), data.astype(np.int64))

    # High-cardinality categorical block for the cap timing (300 distinct codes -> folds to cap).
    hc = rng.integers(0, 300, size=(n_rows, 40)).astype(np.float64)
    tc = []
    for _ in range(reps):
        s = time.perf_counter()
        _ = cap_categorical_cardinality(hc, cap)
        tc.append(time.perf_counter() - s)

    int32_mb = data.nbytes / 2**20
    int8_mb = narrow.nbytes / 2**20
    print(f"shape={n_rows}x{n_cols}  n_bins={n_bins}  cap={cap}")
    print(f"  codes matrix bytes : int32={int32_mb:8.1f} MB  compact int8={int8_mb:8.1f} MB  " f"({int32_mb / max(int8_mb, 1e-9):.1f}x smaller)")
    print(f"  downcast wall      : min={min(t) * 1e3:7.2f} ms  median={np.median(t) * 1e3:7.2f} ms  " f"({data.nbytes / max(min(t), 1e-9) / 2**30:.1f} GB/s)")
    print(f"  cap (40 hc cols)   : min={min(tc) * 1e3:7.2f} ms  median={np.median(tc) * 1e3:7.2f} ms")


if __name__ == "__main__":
    _bench()
