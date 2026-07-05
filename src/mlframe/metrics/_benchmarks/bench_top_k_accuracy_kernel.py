"""Bench: top_k_accuracy hit-counting loop -- pure-Python double loop vs numba kernel.

The OLD ``top_k_accuracy`` did ``np.argpartition(-p, k-1, axis=1)[:, :k]`` and
then counted hits with a pure-Python ``for i in range(n)`` outer loop + inner
``for j in range(k)`` membership scan. The argpartition is C; the hit-count loop
is interpreted Python and grows linearly with n -- the hot path at report scale.

NEW: hoist the hit-count into a numba njit kernel scanning the same ``topk_idx``
(N, k) int array against ``y_true``. Bit-identical by construction (identical
membership test, identical 0<=ti<K guard, identical break-on-first-hit).

Run:
    CUDA_VISIBLE_DEVICES="" python -m mlframe.metrics._benchmarks.bench_top_k_accuracy_kernel
"""
from __future__ import annotations

import time

import numpy as np
import numba

from .._numba_params import NUMBA_NJIT_PARAMS


def _old_top_k_accuracy(y_true, probs_NK, k=1):
    """Verbatim prior implementation (pure-Python hit-count loop)."""
    yt = np.asarray(y_true).astype(np.int64, copy=False)
    p = np.asarray(probs_NK, dtype=np.float64)
    n, K = p.shape
    if k >= K:
        return np.nan
    topk_idx = np.argpartition(-p, k - 1, axis=1)[:, :k]
    hits = 0
    for i in range(n):
        ti = yt[i]
        if 0 <= ti < K:
            for j in range(k):
                if topk_idx[i, j] == ti:
                    hits += 1
                    break
    return hits / n if n > 0 else np.nan


@numba.njit(**NUMBA_NJIT_PARAMS)
def _topk_hits_kernel(topk_idx, y_true, K):
    n, k = topk_idx.shape
    hits = 0
    for i in range(n):
        ti = y_true[i]
        if 0 <= ti < K:
            for j in range(k):
                if topk_idx[i, j] == ti:
                    hits += 1
                    break
    return hits


def _new_top_k_accuracy(y_true, probs_NK, k=1):
    yt = np.asarray(y_true).astype(np.int64, copy=False)
    p = np.asarray(probs_NK, dtype=np.float64)
    n, K = p.shape
    if k >= K:
        return np.nan
    topk_idx = np.ascontiguousarray(np.argpartition(-p, k - 1, axis=1)[:, :k])
    hits = _topk_hits_kernel(topk_idx, yt, K)
    return hits / n if n > 0 else np.nan


def _best_of(fn, *args, reps=7):
    best = float("inf")
    for _ in range(reps):
        t = time.perf_counter()
        fn(*args)
        best = min(best, time.perf_counter() - t)
    return best


def main():
    rng = np.random.default_rng(0)
    print(f"{'n':>9} {'K':>4} {'k':>3} {'old ms':>10} {'new ms':>10} {'speedup':>8}  identical")
    for n, K, k in [(10_000, 10, 3), (100_000, 20, 5), (500_000, 10, 3), (1_000_000, 50, 5)]:
        p = rng.random((n, K))
        yt = rng.integers(0, K, size=n)
        # warm both
        old = _old_top_k_accuracy(yt, p, k)
        new = _new_top_k_accuracy(yt, p, k)
        identical = old == new
        t_old = _best_of(_old_top_k_accuracy, yt, p, k)
        t_new = _best_of(_new_top_k_accuracy, yt, p, k)
        print(f"{n:>9} {K:>4} {k:>3} {t_old*1e3:>10.3f} {t_new*1e3:>10.3f} " f"{t_old/t_new:>8.2f}  {identical}")


if __name__ == "__main__":
    main()
