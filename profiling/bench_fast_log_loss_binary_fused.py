"""Bench fused vs 2-pass _fast_log_loss_binary_par.

Current production form does two prange walks over y_pred:
    Pass A: detect out-of-range values (any < 0 or > 1)
    Pass B: clip + log + sum loss

Fused form does both in one walk; if bad>0 detected, return nan.
In the common case (well-formed probs from sklearn / LightGBM / CB)
the fused form saves one full N-pass.

Per-call cumtime in c0146 / 1M-row profile: 203 ms/call across 41
calls = 8.328s total. A ~10-20% fused speedup would save ~1s.

Run: ``python profiling/bench_fast_log_loss_binary_fused.py``
"""

import time
import numpy as np
import numba

NUMBA_NJIT_PARAMS = dict(fastmath=False, cache=True, nogil=True)


@numba.njit(**NUMBA_NJIT_PARAMS, parallel=True)
def _ll_two_pass(y_true, y_pred, eps):
    n = len(y_true)
    if n == 0:
        return 0.0
    # Pass 1: bounds check
    bad = 0
    for i in numba.prange(n):
        if y_pred[i] < 0.0 or y_pred[i] > 1.0:
            bad += 1
    if bad > 0:
        return np.nan
    # Pass 2: clip + log + sum
    loss_sum = 0.0
    n_pos = 0
    for i in numba.prange(n):
        p = y_pred[i]
        if p < eps:
            p = eps
        elif p > 1 - eps:
            p = 1 - eps
        if y_true[i] == 1:
            loss_sum -= np.log(p)
            n_pos += 1
        else:
            loss_sum -= np.log(1 - p)
    if n_pos == 0 or n_pos == n:
        return np.nan
    return loss_sum / n


@numba.njit(**NUMBA_NJIT_PARAMS, parallel=True)
def _ll_one_pass(y_true, y_pred, eps):
    n = len(y_true)
    if n == 0:
        return 0.0
    # Fused: bounds check + clip + log + sum in one prange walk.
    bad = 0
    loss_sum = 0.0
    n_pos = 0
    for i in numba.prange(n):
        p = y_pred[i]
        if p < 0.0 or p > 1.0:
            bad += 1
            continue  # don't accumulate loss for bad rows; trapped below
        if p < eps:
            p = eps
        elif p > 1 - eps:
            p = 1 - eps
        if y_true[i] == 1:
            loss_sum -= np.log(p)
            n_pos += 1
        else:
            loss_sum -= np.log(1 - p)
    if bad > 0:
        return np.nan
    if n_pos == 0 or n_pos == n:
        return np.nan
    return loss_sum / n


def bench(label, fn, y_true, y_pred, eps, n_iter=50):
    fn(y_true, y_pred, eps)
    fn(y_true, y_pred, eps)
    times = []
    for _ in range(5):
        t = time.perf_counter()
        for _ in range(n_iter):
            fn(y_true, y_pred, eps)
        times.append((time.perf_counter() - t) / n_iter)
    return min(times) * 1e3, label


if __name__ == "__main__":
    np.random.seed(0)
    for N in [100_000, 500_000, 1_000_000, 5_000_000]:
        y_pred = np.random.rand(N).astype(np.float64)
        y_true = (np.random.rand(N) > 0.5).astype(np.float64)
        t2, _ = bench("2-pass", _ll_two_pass, y_true, y_pred, 1e-15)
        t1, _ = bench("1-pass", _ll_one_pass, y_true, y_pred, 1e-15)
        v2 = _ll_two_pass(y_true, y_pred, 1e-15)
        v1 = _ll_one_pass(y_true, y_pred, 1e-15)
        speedup = t2 / t1
        print(f"N={N:>8}: 2pass={t2:7.3f}ms  1pass={t1:7.3f}ms  ({speedup:.2f}x)  diff={abs(v2-v1):.3e}")
