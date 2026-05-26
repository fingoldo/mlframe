"""Bench fast_roc_auc with argsort inside vs outside numba.

c0091/c0009 profile attributes ~5-9 s wall to ``np.argsort(y_score,
kind='stable')[::-1]`` inside ``fast_roc_auc``. The argsort runs OUTSIDE
the numba kernel (per the docstring 'np.argsort needs to stay out of
njitted func'), so each call pays Python ``_wrapfunc`` dispatch + the
view-reverse + the kwarg parse for ``kind='stable'``.

Numba's @njit DOES support ``np.argsort`` (default quicksort or
``kind='mergesort'``); ``mergesort`` is the stable algorithm. Fold the
argsort into the numba kernel so the whole function is a single
Python -> numba transition.
"""
from __future__ import annotations

import time

import numpy as np
from numba import njit


# Replica of the current shipped fast_numba_auc_nonw kernel for fair comparison.
@njit(cache=True, nogil=True)
def fast_numba_auc_nonw_local(y_true, y_score, desc_score_indices):
    y_score_sorted = y_score[desc_score_indices]
    y_true_sorted = y_true[desc_score_indices]
    last_counted_fps = 0
    last_counted_tps = 0
    tps, fps = 0, 0
    auc = 0
    l = len(y_true_sorted) - 1
    for i in range(l + 1):
        tps += y_true_sorted[i]
        fps += 1 - y_true_sorted[i]
        if i == l or y_score_sorted[i + 1] != y_score_sorted[i]:
            auc += (fps - last_counted_fps) * (last_counted_tps + tps)
            last_counted_fps = fps
            last_counted_tps = tps
    tmp = tps * fps * 2
    if tmp > 0:
        return auc / tmp
    return np.nan


def current(y_true, y_score):
    """Pre-shipped path -- argsort outside, numba kernel inside."""
    if y_score.ndim == 2:
        y_score = y_score[:, -1]
    desc_score_indices = np.argsort(y_score, kind="stable")[::-1]
    return fast_numba_auc_nonw_local(y_true, y_score, desc_score_indices)


@njit(cache=True, nogil=True)
def proposed(y_true, y_score):
    """Argsort folded into the numba kernel via mergesort (stable)."""
    desc_score_indices = np.argsort(y_score, kind="mergesort")[::-1]
    y_score_sorted = y_score[desc_score_indices]
    y_true_sorted = y_true[desc_score_indices]
    last_counted_fps = 0
    last_counted_tps = 0
    tps, fps = 0, 0
    auc = 0
    l = len(y_true_sorted) - 1
    for i in range(l + 1):
        tps += y_true_sorted[i]
        fps += 1 - y_true_sorted[i]
        if i == l or y_score_sorted[i + 1] != y_score_sorted[i]:
            auc += (fps - last_counted_fps) * (last_counted_tps + tps)
            last_counted_fps = fps
            last_counted_tps = tps
    tmp = tps * fps * 2
    if tmp > 0:
        return auc / tmp
    return np.nan


def main():
    np.random.seed(0)
    for n in (2_000, 20_000, 200_000, 1_000_000):
        y = np.random.randint(0, 2, n).astype(np.float64)
        p = np.random.rand(n).astype(np.float64)
        r1 = current(y, p)
        r2 = proposed(y, p)
        eq = abs(r1 - r2) < 1e-10
        N = max(5, 4000 // (n // 1000 + 1))
        t0 = time.perf_counter()
        for _ in range(N):
            current(y, p)
        t1 = time.perf_counter()
        for _ in range(N):
            proposed(y, p)
        t2 = time.perf_counter()
        cur_ms = (t1 - t0) * 1000 / N
        new_ms = (t2 - t1) * 1000 / N
        print(f"n={n:>9}: current={cur_ms:7.4f}ms  proposed={new_ms:7.4f}ms  speedup={cur_ms/new_ms:5.2f}x  equal={eq}")


if __name__ == "__main__":
    main()
