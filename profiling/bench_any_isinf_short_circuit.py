"""Bench short-circuit any-isinf vs np.isinf(arr).any().

bench-attempt-rejected (2026-05-21, c0095 / iter141): numba @njit
short-circuit-and-skip-bool-alloc is 24-51% SLOWER on CLEAN (no inf)
arrays -- the production-dominant case in ensure_no_infinity_pd:

    CLEAN n= 100000: numpy=  28.0us  numba=  57.4us  (0.49x)
    CLEAN n=1000000: numpy= 491.2us  numba= 646.0us  (0.76x)
    CLEAN n=5000000: numpy=3996.6us  numba=3324.4us  (1.20x)
    DIRTY (inf@start) n=1M: numpy= 460us  numba=   0.6us  (831x)
    DIRTY (inf@mid)   n=1M: numpy= 488us  numba=  280us   (1.75x)
    DIRTY (inf@end)   n=1M: numpy= 488us  numba=  629us   (0.78x)

Numpy's np.isinf().any() is SIMD-vectorised inside the C kernel;
numba's @njit scalar-loop can't beat that on contiguous float64 at
typical sizes (only wins at 5M+ where the bool-alloc cost dominates).
Production callers in ensure_no_infinity_pd hit n=1M-scale columns
that are CLEAN (the function flags inf-present columns as a warning;
clean is the dominant path), so the numba version would REGRESS the
hot path 24%.

The DIRTY-at-start short-circuit (831x) is impressive but never
exercises the production hot path -- inf is rare and never
concentrated at the start.

c0095 attributed 0.185 s across 8 calls (~23 ms / call at 1M rows
across multiple float columns); the numba version would have ADDED
~0.05 s to the run.

Documented per ``feedback_document_failed_optimization_attempts`` so
the next agent doesn't re-try this same path.

Run: ``python profiling/bench_any_isinf_short_circuit.py``
"""

import time
import numpy as np
from numba import njit


def np_any_isinf(arr):
    return bool(np.isinf(arr).any())


@njit(cache=True, nogil=True)
def numba_any_isinf(arr):
    # Single-pass short-circuit; no bool-array allocation.
    n = len(arr)
    for i in range(n):
        v = arr[i]
        if v == np.inf or v == -np.inf:
            return True
    return False


def bench(label, fn, arr, n_iter=200):
    fn(arr); fn(arr)  # warmup
    times = []
    for _ in range(5):
        t = time.perf_counter()
        for _ in range(n_iter):
            fn(arr)
        times.append((time.perf_counter() - t) / n_iter)
    return min(times) * 1e6, label


if __name__ == "__main__":
    np.random.seed(0)
    # Common case: clean array, no inf -- both must scan full
    for n in [100_000, 1_000_000, 5_000_000]:
        arr = np.random.randn(n).astype(np.float64)
        t_np, _ = bench("numpy", np_any_isinf, arr)
        t_nb, _ = bench("numba", numba_any_isinf, arr)
        print(f"CLEAN n={n:>8}: numpy={t_np:7.1f}us  numba={t_nb:7.1f}us  ({t_np/t_nb:.2f}x)")

    # Dirty case: inf at start, mid, end positions -- short-circuit wins
    for n in [1_000_000]:
        for pos_name, pos in [("start", 0), ("mid", n // 2), ("end", n - 1)]:
            arr = np.random.randn(n).astype(np.float64)
            arr[pos] = np.inf
            t_np, _ = bench("numpy", np_any_isinf, arr)
            t_nb, _ = bench("numba", numba_any_isinf, arr)
            print(f"DIRTY (inf@{pos_name}) n={n:>8}: numpy={t_np:7.1f}us  numba={t_nb:7.1f}us  ({t_np/t_nb:.2f}x)")
