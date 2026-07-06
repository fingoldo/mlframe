"""iter44: measure the per-call NaN-scan fraction of discretize_2d_quantile_batch, and verify
the assume_finite fast path is bit-identical to the default (NaN-free buffer).

The main FE-chunk caller (_pairs_core.py:1115) scrubs the buffer with np.nan_to_num(copy=False)
on the line immediately before the call, so the buffer is GUARANTEED NaN-free -- yet
discretize_2d_quantile_batch still runs ``np.isnan(arr2d).any()`` (a full O(n*k) pass + a full
bool-array allocation) on every call. ``assume_finite=True`` skips that scan; bit-identical by
construction (the scan returns False on a scrubbed buffer, so the same edges-njit branch runs).

Run:
  PYTHONPATH=<wt>/src MLFRAME_SKIP_NUMBA_PREWARM=1 CUDA_VISIBLE_DEVICES="" python bench_iter44_assume_finite.py
"""
from __future__ import annotations

import time

import numpy as np

from mlframe.feature_selection.filters.discretization import discretize_2d_quantile_batch


def _bench(fn, iters):
    fn()  # warm
    best = 1e9
    for _ in range(5):
        t = time.perf_counter()
        for _ in range(iters):
            fn()
        best = min(best, (time.perf_counter() - t) / iters)
    return best * 1e6


def main():
    rng = np.random.default_rng(0)
    nb = 10
    for n_rows, n_cols in [(2500, 60), (2500, 200), (2500, 600), (2500, 2000)]:
        arr = np.ascontiguousarray(rng.standard_normal((n_rows, n_cols)).astype(np.float32))
        ref = discretize_2d_quantile_batch(arr, n_bins=nb, dtype=np.int8, parallel=True)
        fast = discretize_2d_quantile_batch(arr, n_bins=nb, dtype=np.int8, parallel=True, assume_finite=True)
        assert np.array_equal(ref, fast), f"NOT bit-identical at {n_rows}x{n_cols}"  # nosec B101 - internal invariant check in src/mlframe/feature_selection/_benchmarks, not reachable with untrusted input
        iters = max(20, int(3e6 / arr.size))
        t_def = _bench(lambda: discretize_2d_quantile_batch(arr, n_bins=nb, dtype=np.int8, parallel=True), iters)
        t_fast = _bench(lambda: discretize_2d_quantile_batch(arr, n_bins=nb, dtype=np.int8, parallel=True, assume_finite=True), iters)
        print(f"{n_rows}x{n_cols:5d} f32 nb={nb}: default={t_def:9.2f}us  assume_finite={t_fast:9.2f}us  saved={t_def-t_fast:7.2f}us ({100*(t_def-t_fast)/t_def:4.1f}%)")
    print("BIT-IDENTICAL across all shapes: OK")


if __name__ == "__main__":
    main()
