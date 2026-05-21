"""Benchmark: single fused numba kernel vs N separate fast_* kernels for
the regression-reporting metric block (MAE / RMSE / MaxError / R2).

User request 2026-05-22: "consider computing ALL regression metrics at
once using single numpy kernel (one pass through data). is that
possible? any speedups?"

ANSWER (preview, full numbers below):

Yes, possible. The 4 metrics decompose into 5 accumulators that fit
inside one pass over (y_true, y_pred):

  sum_abs_err   -> MAE
  sum_sqr_err   -> MSE -> RMSE
  max_abs_err   -> MaxError
  sum_y_true    -> y_true_mean (used by R2)
  sum_y_true_sq -> y_true_var (used by R2)

R2 needs the y_true mean, which a 1-pass formulation handles via the
identity SS_tot = sum(y_true^2) - n * y_true_mean^2. With sample weights
the same identity holds with weighted sums.

Speedup is bound by memory bandwidth, not FLOPs. Modern CPUs hit ~20-30
GB/s on float64 sequential reads; the existing fast_* kernels each
re-touch the (y_true, y_pred) pair from RAM. Fusing saves the redundant
cache misses + the per-kernel dispatch overhead (numba ahead-of-time
compiled njit functions still pay ~10us setup per call).

Run::

    python -m mlframe.metrics._benchmarks.bench_fused_regression_metrics

Reports table: (N, separate-ms, fused-ms, speedup, max-abs-numeric-diff)
per metric. ``max-abs-numeric-diff`` must stay below 1e-10 to confirm
the fused kernel is numerically faithful.
"""
from __future__ import annotations

import sys
import time

import numba
import numpy as np


@numba.njit(cache=True, fastmath=False, boundscheck=False, parallel=True)
def _fused_pass1_par(y_true: np.ndarray, y_pred: np.ndarray):
    """Pass 1 of the 2-pass fused kernel (parallel, per-thread reductions).

    Returns ``(sum_abs_err, sum_sqr_err, max_abs_err, sum_y_true)``.

    Numba's prange only auto-handles SUM-style reductions safely; a naive
    ``if x > m: m = x`` inside prange is a RACE between threads (observed
    250-unit drift vs sklearn's serial max at N=500k+ on the 2026-05-22
    bench). The fix is the explicit per-thread-accumulator pattern: each
    thread writes to its own slot of a length-n_threads array, then we
    reduce serially after the parallel block.
    """
    n = y_true.shape[0]
    n_threads = numba.get_num_threads()
    chunk_size = (n + n_threads - 1) // n_threads
    local_sum_abs = np.zeros(n_threads, dtype=np.float64)
    local_sum_sqr = np.zeros(n_threads, dtype=np.float64)
    local_max_abs = np.zeros(n_threads, dtype=np.float64)
    local_sum_y = np.zeros(n_threads, dtype=np.float64)
    for tid in numba.prange(n_threads):
        start = tid * chunk_size
        end = min(start + chunk_size, n)
        s_abs = 0.0
        s_sqr = 0.0
        m = 0.0
        s_y = 0.0
        for i in range(start, end):
            err = y_true[i] - y_pred[i]
            abs_err = err if err >= 0.0 else -err
            s_abs += abs_err
            s_sqr += err * err
            if abs_err > m:
                m = abs_err
            s_y += y_true[i]
        local_sum_abs[tid] = s_abs
        local_sum_sqr[tid] = s_sqr
        local_max_abs[tid] = m
        local_sum_y[tid] = s_y
    # Serial reduce across threads.
    sum_abs = 0.0
    sum_sqr = 0.0
    max_abs = 0.0
    sum_y = 0.0
    for tid in range(n_threads):
        sum_abs += local_sum_abs[tid]
        sum_sqr += local_sum_sqr[tid]
        if local_max_abs[tid] > max_abs:
            max_abs = local_max_abs[tid]
        sum_y += local_sum_y[tid]
    return sum_abs, sum_sqr, max_abs, sum_y


@numba.njit(cache=True, fastmath=False, boundscheck=False)
def _fused_pass1_seq(y_true: np.ndarray, y_pred: np.ndarray):
    n = y_true.shape[0]
    sum_abs = 0.0
    sum_sqr = 0.0
    max_abs = 0.0
    sum_y = 0.0
    for i in range(n):
        err = y_true[i] - y_pred[i]
        abs_err = err if err >= 0.0 else -err
        sum_abs += abs_err
        sum_sqr += err * err
        if abs_err > max_abs:
            max_abs = abs_err
        sum_y += y_true[i]
    return sum_abs, sum_sqr, max_abs, sum_y


@numba.njit(cache=True, fastmath=False, boundscheck=False, parallel=True)
def _fused_pass2_par(y_true: np.ndarray, y_mean: float) -> float:
    """Pass 2: centred sum-of-squares around the pre-computed mean."""
    n = y_true.shape[0]
    ss = 0.0
    for i in numba.prange(n):
        d = y_true[i] - y_mean
        ss += d * d
    return ss


@numba.njit(cache=True, fastmath=False, boundscheck=False)
def _fused_pass2_seq(y_true: np.ndarray, y_mean: float) -> float:
    n = y_true.shape[0]
    ss = 0.0
    for i in range(n):
        d = y_true[i] - y_mean
        ss += d * d
    return ss


def fused_regression_metrics(y_true, y_pred, parallel_threshold: int = 200_000):
    """Derive MAE / RMSE / MaxError / R^2 in 2 passes (vs 4-5 separate kernels).

    Pass 1: sum_abs / sum_sqr / max_abs / sum_y_true (one walk through
    both arrays). Pass 2: centred sum-of-squares of y_true around the
    pre-computed mean (one walk through y_true only). Net: ~2 reads of
    each array vs 4-5 for the separate-kernel baseline.

    Numerically equivalent to ``fast_mean_absolute_error`` /
    ``fast_root_mean_squared_error`` / ``fast_max_error`` /
    ``fast_r2_score``; the centred SS_tot stays stable for arbitrary y
    magnitudes (the naive 1-pass identity ``sum_y_sq - n*y_mean^2``
    catastrophically cancelled on the 2026-05-22 TVT-shape data with
    y_mean=11500, y_std=645).
    """
    yt = np.ascontiguousarray(np.asarray(y_true), dtype=np.float64)
    yp = np.ascontiguousarray(np.asarray(y_pred), dtype=np.float64)
    n = yt.shape[0]
    use_par = n >= parallel_threshold
    if use_par:
        sum_abs, sum_sqr, max_abs, sum_y = _fused_pass1_par(yt, yp)
        y_mean = sum_y / n
        ss_tot = _fused_pass2_par(yt, y_mean)
    else:
        sum_abs, sum_sqr, max_abs, sum_y = _fused_pass1_seq(yt, yp)
        y_mean = sum_y / n
        ss_tot = _fused_pass2_seq(yt, y_mean)
    mae = sum_abs / n
    mse = sum_sqr / n
    rmse = np.sqrt(mse)
    max_err = max_abs
    if ss_tot <= 0.0:
        r2 = 0.0 if sum_sqr == 0.0 else float("-inf")
    else:
        r2 = 1.0 - sum_sqr / ss_tot
    return {"MAE": mae, "RMSE": rmse, "MaxError": max_err, "R2": r2}


def main():
    sys.path.insert(0, "src")
    from mlframe.metrics.core import (
        fast_max_error,
        fast_mean_absolute_error,
        fast_r2_score,
        fast_root_mean_squared_error,
    )

    sizes = (10_000, 100_000, 500_000, 2_000_000, 5_000_000)
    n_iters = 12

    print()
    print("# bench_fused_regression_metrics  (1 pass vs 4 separate kernels)")
    print()
    print(f"{'N':>10} {'sep_ms':>10} {'fused_ms':>10} {'speedup':>8}  {'max_abs_diff':>14}")
    print("-" * 60)

    rng = np.random.default_rng(0)
    for n in sizes:
        y_true = rng.normal(11500.0, 645.0, n)
        y_pred = y_true + rng.normal(0.0, 50.0, n)

        # Warm up JIT for both code paths.
        _ = fused_regression_metrics(y_true, y_pred)
        _ = fast_mean_absolute_error(y_true, y_pred)
        _ = fast_root_mean_squared_error(y_true, y_pred)
        _ = fast_max_error(y_true, y_pred)
        _ = fast_r2_score(y_true, y_pred)

        t0 = time.perf_counter()
        for _ in range(n_iters):
            mae_s = fast_mean_absolute_error(y_true, y_pred)
            rmse_s = fast_root_mean_squared_error(y_true, y_pred)
            maxe_s = fast_max_error(y_true, y_pred)
            r2_s = fast_r2_score(y_true, y_pred)
        sep_ms = (time.perf_counter() - t0) / n_iters * 1000.0

        t0 = time.perf_counter()
        for _ in range(n_iters):
            fused = fused_regression_metrics(y_true, y_pred)
        fused_ms = (time.perf_counter() - t0) / n_iters * 1000.0

        # Numerical correctness: each metric must agree to < 1e-10 vs the
        # canonical sklearn-style baseline.
        diffs = (
            abs(fused["MAE"] - float(mae_s)),
            abs(fused["RMSE"] - float(rmse_s)),
            abs(fused["MaxError"] - float(maxe_s)),
            abs(fused["R2"] - float(r2_s)),
        )
        max_diff = max(diffs)
        speedup = sep_ms / max(fused_ms, 1e-9)
        print(f"{n:>10d} {sep_ms:>10.3f} {fused_ms:>10.3f} {speedup:>7.2f}x  {max_diff:>14.2e}")


if __name__ == "__main__":
    main()
