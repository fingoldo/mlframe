"""Bench sklearn regression metrics vs numba seq vs numba par.

Decide which to ship as mlframe-native. Targets:
  mean_absolute_error / mean_squared_error / root_mean_squared_error /
  max_error / r2_score

Decision rule: ship when numba par >= 2x faster than sklearn at N=1M
AND result matches sklearn within fp64 noise (1e-12 atol).
"""

from __future__ import annotations

import sys
import time
import math
from typing import Callable

import numpy as np
import numba
from numba import prange
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    max_error,
    r2_score,
)
try:
    from sklearn.metrics import root_mean_squared_error as sk_rmse
except ImportError:
    def sk_rmse(y_true, y_pred):
        return math.sqrt(mean_squared_error(y_true, y_pred))

NUMBA_PARAMS = dict(fastmath=False, cache=True, nogil=True)


# ----------------------------------------------------------------------------
# numba kernels
# ----------------------------------------------------------------------------


@numba.njit(**NUMBA_PARAMS)
def mae_seq(y_true, y_pred):
    n = len(y_true)
    s = 0.0
    for i in range(n):
        s += abs(y_true[i] - y_pred[i])
    return s / n


@numba.njit(**NUMBA_PARAMS, parallel=True)
def mae_par(y_true, y_pred):
    n = len(y_true)
    s = 0.0
    for i in prange(n):
        s += abs(y_true[i] - y_pred[i])
    return s / n


@numba.njit(**NUMBA_PARAMS)
def mse_seq(y_true, y_pred):
    n = len(y_true)
    s = 0.0
    for i in range(n):
        d = y_true[i] - y_pred[i]
        s += d * d
    return s / n


@numba.njit(**NUMBA_PARAMS, parallel=True)
def mse_par(y_true, y_pred):
    n = len(y_true)
    s = 0.0
    for i in prange(n):
        d = y_true[i] - y_pred[i]
        s += d * d
    return s / n


@numba.njit(**NUMBA_PARAMS)
def max_err_seq(y_true, y_pred):
    n = len(y_true)
    m = 0.0
    for i in range(n):
        d = abs(y_true[i] - y_pred[i])
        if d > m:
            m = d
    return m


@numba.njit(**NUMBA_PARAMS, parallel=True)
def max_err_par(y_true, y_pred):
    """Per-thread max -- if-based max-update under prange is a race
    that drops updates between threads (numba auto-detects only += / *=
    as reductions, not min/max compares)."""
    n = len(y_true)
    nthr = numba.get_num_threads()
    per_t = np.zeros(nthr, dtype=np.float64)
    for i in prange(n):
        d = abs(y_true[i] - y_pred[i])
        tid = numba.get_thread_id()
        if d > per_t[tid]:
            per_t[tid] = d
    m = 0.0
    for t in range(nthr):
        if per_t[t] > m:
            m = per_t[t]
    return m


@numba.njit(**NUMBA_PARAMS)
def r2_seq(y_true, y_pred):
    """Two-pass: 1) mean of y_true, 2) SS_res and SS_tot."""
    n = len(y_true)
    ymean = 0.0
    for i in range(n):
        ymean += y_true[i]
    ymean /= n
    ss_res = 0.0
    ss_tot = 0.0
    for i in range(n):
        d_res = y_true[i] - y_pred[i]
        d_tot = y_true[i] - ymean
        ss_res += d_res * d_res
        ss_tot += d_tot * d_tot
    if ss_tot == 0.0:
        return 0.0  # sklearn convention
    return 1.0 - ss_res / ss_tot


@numba.njit(**NUMBA_PARAMS, parallel=True)
def r2_par(y_true, y_pred):
    n = len(y_true)
    ymean = 0.0
    for i in prange(n):
        ymean += y_true[i]
    ymean /= n
    ss_res = 0.0
    ss_tot = 0.0
    for i in prange(n):
        d_res = y_true[i] - y_pred[i]
        d_tot = y_true[i] - ymean
        ss_res += d_res * d_res
        ss_tot += d_tot * d_tot
    if ss_tot == 0.0:
        return 0.0
    return 1.0 - ss_res / ss_tot


# ----------------------------------------------------------------------------
# Bench harness
# ----------------------------------------------------------------------------


def time_op(fn: Callable, *args, repeats=5, warmup=1):
    for _ in range(warmup):
        fn(*args)
    t = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        out = fn(*args)
        t.append(time.perf_counter() - t0)
    return out, min(t)


def fmt(t):
    if t < 1e-3:
        return f"{t*1e6:7.1f}us"
    if t < 1.0:
        return f"{t*1e3:7.2f}ms"
    return f"{t:7.3f}s"


def main():
    rng = np.random.default_rng(0)
    print(f"numba: {numba.__version__}, num_threads={numba.get_num_threads()}")
    print()

    sizes = [10_000, 100_000, 1_000_000, 10_000_000]
    metrics = [
        ("MAE", mean_absolute_error, mae_seq, mae_par),
        ("MSE", mean_squared_error, mse_seq, mse_par),
        ("max_error", max_error, max_err_seq, max_err_par),
        ("R2", r2_score, r2_seq, r2_par),
    ]

    for name, sk_fn, seq_fn, par_fn in metrics:
        print(f"--- {name} ---")
        print(f"{'N':>10} | {'sklearn':>10} | {'numba seq':>10} | {'numba par':>10} | "
              f"{'seq/sk':>7} | {'par/sk':>7} | {'par/seq':>7} | {'err':>10}")
        print("-" * 105)
        for N in sizes:
            y_true = rng.standard_normal(N)
            y_pred = y_true + 0.1 * rng.standard_normal(N)
            seq_fn(y_true, y_pred)  # warm
            par_fn(y_true, y_pred)
            v_sk, t_sk = time_op(sk_fn, y_true, y_pred,
                                  repeats=3 if N >= 1_000_000 else 5)
            v_seq, t_seq = time_op(seq_fn, y_true, y_pred,
                                    repeats=3 if N >= 1_000_000 else 5)
            v_par, t_par = time_op(par_fn, y_true, y_pred,
                                    repeats=3 if N >= 1_000_000 else 5)
            err_seq = float(abs(v_sk - v_seq))
            err_par = float(abs(v_sk - v_par))
            err = max(err_seq, err_par)
            print(f"{N:>10} | {fmt(t_sk):>10} | {fmt(t_seq):>10} | {fmt(t_par):>10} | "
                  f"{t_seq/t_sk:6.2f}x | {t_par/t_sk:6.2f}x | {t_par/t_seq:6.2f}x | "
                  f"{err:.2e}")
        print()


if __name__ == "__main__":
    main()
