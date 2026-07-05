"""Microbench: standalone njit speedup of ``ks_statistic``.

The prior ``bench_ks_shared_sort.py`` confirmed sharing KS's sort with the
AUC score-desc argsort is bit-identical but UNIMPLEMENTABLE (the AUC order
lives inside the batched/GPU ``compute_batch_aucs`` and is never returned).
That sort-sharing path is dead.

This bench takes the OTHER path: keep KS's own ``np.argsort`` but fuse the
two fancy-index gathers (``yt[order]`` + ``ys[order]``, each an N-length
temporary) INTO the single-pass njit kernel by indexing through ``order``
inside the loop. No temporaries, one pass.

Variants:
  * ``numpy_current``  -- np.argsort + yt[order] + ys[order] + _ks_statistic_kernel
                          (the shipped path; two N-length gather temporaries).
  * ``fused_gather``   -- np.argsort + _ks_statistic_kernel_ordered(order, yt, ys)
                          (kernel indexes through order; zero gather temporaries).

Run: python -m mlframe.metrics._benchmarks.bench_ks_statistic_njit
"""
from __future__ import annotations

import time

import numpy as np
import numba

from mlframe.metrics.classification._classification_extras import (
    _ks_statistic_kernel,
    _ks_statistic_kernel_ordered,
    ks_statistic,
    _ks_statistic_numpy,
)


def ks_numpy_current(yt: np.ndarray, ys: np.ndarray) -> float:
    order = np.argsort(ys, kind="quicksort")
    return float(_ks_statistic_kernel(yt[order], ys[order]))


def ks_fused_gather(yt: np.ndarray, ys: np.ndarray) -> float:
    order = np.argsort(ys, kind="quicksort")
    return float(_ks_statistic_kernel_ordered(order, yt, ys))


@numba.njit(fastmath=False, cache=True, nogil=True)
def _ks_inkernel(yt: np.ndarray, ys: np.ndarray) -> float:
    n = ys.shape[0]
    n_pos = 0
    for i in range(n):
        if yt[i] != 0:
            n_pos += 1
    n_neg = n - n_pos
    if n_pos == 0 or n_neg == 0:
        return np.nan
    order = np.argsort(ys, kind="quicksort")
    s = np.empty(n, dtype=ys.dtype)
    t = np.empty(n, dtype=yt.dtype)
    for i in range(n):
        o = order[i]
        s[i] = ys[o]
        t[i] = yt[o]
    inv_pos = 1.0 / n_pos
    inv_neg = 1.0 / n_neg
    cum_pos = 0.0
    cum_neg = 0.0
    ks = 0.0
    i = 0
    while i < n:
        j = i
        cur = s[i]
        while j < n and s[j] == cur:
            if t[j] != 0:
                cum_pos += inv_pos
            else:
                cum_neg += inv_neg
            j += 1
        d = cum_pos - cum_neg
        if d < 0.0:
            d = -d
        if d > ks:
            ks = d
        i = j
    return ks


def ks_inkernel(yt: np.ndarray, ys: np.ndarray) -> float:
    return float(_ks_inkernel(yt, ys))


def _time(fn, *args, iters: int = 200) -> float:
    fn(*args)  # warm
    t0 = time.perf_counter()
    for _ in range(iters):
        fn(*args)
    return (time.perf_counter() - t0) / iters * 1e6


def main() -> None:
    rng = np.random.default_rng(7)
    print(f"{'n':>8} {'ratio':>6} {'numpy_us':>10} {'fused_us':>10} {'inker_us':>10} " f"{'fu_sp':>7} {'ik_sp':>7} {'identical':>10}")
    for n in (2_000, 20_000, 100_000, 200_000):
        for ratio in (0.5, 0.1):
            ys = rng.random(n)
            yt = (rng.random(n) < ratio).astype(np.int64)
            base = ks_numpy_current(yt, ys)
            fused = ks_fused_gather(yt, ys)
            inker = ks_inkernel(yt, ys)
            identical = (base == fused) and (base == inker)
            # median of 3 timing blocks to damp noise
            t_np = min(_time(ks_numpy_current, yt, ys) for _ in range(3))
            t_fu = min(_time(ks_fused_gather, yt, ys) for _ in range(3))
            t_ik = min(_time(ks_inkernel, yt, ys) for _ in range(3))
            print(f"{n:>8} {ratio:>6.2f} {t_np:>10.1f} {t_fu:>10.1f} {t_ik:>10.1f} " f"{t_np / t_fu:>6.2f}x {t_np / t_ik:>6.2f}x {str(identical):>10}")


if __name__ == "__main__":
    main()
