"""iter44: isolate the per-call NaN-gate cost in discretize_2d_quantile_batch.

discretize_2d_quantile_batch runs ``np.isnan(arr2d).any()`` on EVERY call (1373x in the
scene 2500-row MRMR fit, ~56s of wall attributed to the function). ``np.isnan(x).any()``
materialises a full (n_rows x n_cols) bool array then reduces it -- O(n*k) work + a full
bool allocation, with NO short-circuit. A fused njit ``_any_nan_2d`` short-circuits on the
first NaN and allocates nothing. The FE buffer is post-nan_to_num (NaN-free) in the common
case, so the scan runs to completion either way -- the win is the eliminated bool alloc +
the contiguous single-pass C loop vs numpy's isnan(materialise)+any(reduce) two-pass.

Run:
  PYTHONPATH=<wt>/src MLFRAME_SKIP_NUMBA_PREWARM=1 CUDA_VISIBLE_DEVICES="" python bench_iter44_nan_gate.py
"""
from __future__ import annotations

import time

import numpy as np
from numba import njit, prange


@njit(nogil=True, cache=True)
def _any_nan_2d(arr2d) -> bool:
    n_rows = arr2d.shape[0]
    n_cols = arr2d.shape[1]
    for j in range(n_cols):
        for r in range(n_rows):
            v = arr2d[r, j]
            if v != v:
                return True
    return False


@njit(parallel=True, nogil=True, cache=True)
def _any_nan_2d_par(arr2d) -> bool:
    n_rows = arr2d.shape[0]
    n_cols = arr2d.shape[1]
    flags = np.zeros(n_cols, dtype=np.bool_)
    for j in prange(n_cols):
        f = False
        for r in range(n_rows):
            v = arr2d[r, j]
            if v != v:
                f = True
                break
        flags[j] = f
    return flags.any()


def _bench(fn, arr, iters):
    fn(arr)  # warm
    best = 1e9
    for _ in range(5):
        t = time.perf_counter()
        for _ in range(iters):
            fn(arr)
        best = min(best, (time.perf_counter() - t) / iters)
    return best * 1e6  # us


def main():
    rng = np.random.default_rng(0)
    for n_rows, n_cols in [(2500, 300), (2500, 1000), (2500, 4000), (2500, 8000)]:
        arr = rng.standard_normal((n_rows, n_cols)).astype(np.float32)
        arr = np.ascontiguousarray(arr)
        iters = max(3, int(2e7 / arr.size))
        t_np = _bench(lambda a: np.isnan(a).any(), arr, iters)
        t_nb = _bench(lambda a: _any_nan_2d(a), arr, iters)
        t_pa = _bench(lambda a: _any_nan_2d_par(a), arr, iters)
        assert bool(np.isnan(arr).any()) is False and bool(_any_nan_2d_par(arr)) is False
        arr2 = arr.copy()
        arr2[n_rows // 2, n_cols // 2] = np.nan
        assert bool(np.isnan(arr2).any()) is True and bool(_any_nan_2d_par(arr2)) is True
        print(f"{n_rows}x{n_cols:5d} f32: numpy={t_np:8.2f}us  njit={t_nb:9.2f}us  njit_par={t_pa:8.2f}us  par_speedup={t_np/t_pa:5.2f}x")


if __name__ == "__main__":
    main()
