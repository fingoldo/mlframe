"""Benchmark sequential vs parallel numba kernels for mlframe metrics.

For each candidate kernel from mlframe.metrics: write a ``_par`` variant
that adds ``parallel=True`` + ``prange``, verify correctness, then time
both variants across N in {10k, 100k, 1M, 10M}.

Goal: identify which kernels actually win from parallelisation. numba
``prange`` adds a thread-pool spawn cost (~40-80 us cold, ~5-10 us warm)
that loses on sub-ms workloads but wins big on N >= 1M reductions.

Run::

    python bench_numba_parallel.py            # full sweep
    python bench_numba_parallel.py --quick    # smaller grid
"""

from __future__ import annotations

import argparse
import math
import sys
import time
from typing import Callable

import numpy as np
import numba
from numba import prange

# Reuse the canonical config from mlframe.metrics.
from mlframe.metrics import (
    fast_brier_score_loss,
    fast_log_loss_binary,
    fast_calibration_binning,
    compute_ece_and_brier_decomposition,
    compute_pr_recall_f1_metrics,
    _fast_subset_accuracy_seq,
    _fast_jaccard_score_seq,
    NUMBA_NJIT_PARAMS,
    probability_separation_score,
)

# ============================================================================
# Parallel variants under test
# ============================================================================


@numba.njit(**NUMBA_NJIT_PARAMS, parallel=True)
def fast_brier_score_loss_par(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    n = len(y_true)
    s = 0.0
    for i in prange(n):
        d = y_true[i] - y_prob[i]
        s += d * d
    return s / n


@numba.njit(**NUMBA_NJIT_PARAMS, parallel=True)
def fast_log_loss_binary_par(y_true: np.ndarray, y_pred: np.ndarray, eps: float = 1e-15) -> float:
    n = len(y_true)
    if n == 0:
        return 0.0
    # Out-of-range probability check (parallel scan via reduction-or)
    bad = 0
    for i in prange(n):
        if y_pred[i] < 0.0 or y_pred[i] > 1.0:
            bad += 1
    if bad > 0:
        return np.nan

    loss_sum = 0.0
    n_pos = 0
    for i in prange(n):
        p = y_pred[i]
        if p < eps:
            p = eps
        elif p > 1 - eps:
            p = 1 - eps
        if y_true[i] == 1:
            loss_sum -= math.log(p)
            n_pos += 1
        else:
            loss_sum -= math.log(1 - p)

    if n_pos == 0 or n_pos == n:
        return np.nan
    return loss_sum / n


@numba.njit(**NUMBA_NJIT_PARAMS, parallel=True)
def compute_pr_recall_f1_metrics_par(y_true: np.ndarray, y_pred: np.ndarray):
    n = len(y_true)
    TP = 0
    FP = 0
    FN = 0
    for i in prange(n):
        if y_true[i] == 1 and y_pred[i] == 1:
            TP += 1
        elif y_true[i] == 0 and y_pred[i] == 1:
            FP += 1
        elif y_true[i] == 1 and y_pred[i] == 0:
            FN += 1

    precision = TP / (TP + FP) if (TP + FP) > 0 else 0.0
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0.0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    return precision, recall, f1


@numba.njit(**NUMBA_NJIT_PARAMS, parallel=True)
def _fast_subset_accuracy_par(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Row-wise all-equal check, parallelised across rows."""
    N, K = y_true.shape
    correct = 0
    for i in prange(N):
        all_eq = True
        for j in range(K):
            if y_true[i, j] != y_pred[i, j]:
                all_eq = False
                break
        if all_eq:
            correct += 1
    return correct / N


@numba.njit(**NUMBA_NJIT_PARAMS, parallel=True)
def _fast_jaccard_score_par(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Per-row Jaccard, parallelised across rows."""
    N, K = y_true.shape
    total = 0.0
    for i in prange(N):
        intersect = 0.0
        union = 0.0
        for j in range(K):
            t, p = y_true[i, j], y_pred[i, j]
            if t == 1 and p == 1:
                intersect += 1.0
            if t == 1 or p == 1:
                union += 1.0
        total += (intersect / union) if union > 0 else 1.0
    return total / N


@numba.njit(**NUMBA_NJIT_PARAMS, parallel=True)
def fast_calibration_binning_par(y_true: np.ndarray, y_pred: np.ndarray, nbins: int = 100):
    """Parallel histogram. Per-thread bin accumulators avoid contention.

    NOTE: ``pmin = min(pmin, v)`` inside ``prange`` is a RACE — numba
    auto-detects ``+=`` as a reduction but NOT min/max. Use
    ``np.min`` / ``np.max`` (which numba parallelises internally) and
    keep the histogram step as the only ``prange`` loop.
    """
    n = len(y_true)
    if n == 0:
        return (
            np.zeros(0, dtype=np.float64),
            np.zeros(0, dtype=np.float64),
            np.zeros(0, dtype=np.int64),
        )

    # numba parallelises np.min/np.max under parallel=True automatically.
    pmin = np.min(y_pred)
    pmax = np.max(y_pred)
    span = pmax - pmin

    # per-thread bin buffers (use numba.get_num_threads())
    nthr = numba.get_num_threads()
    pred_sums = np.zeros((nthr, nbins), dtype=np.float64)
    true_sums = np.zeros((nthr, nbins), dtype=np.float64)
    counts = np.zeros((nthr, nbins), dtype=np.int64)

    if span > 0:
        multiplier = (nbins - 1) / span
        for i in prange(n):
            tid = numba.get_thread_id()
            v = y_pred[i]
            b = int((v - pmin) * multiplier)
            counts[tid, b] += 1
            pred_sums[tid, b] += v
            true_sums[tid, b] += y_true[i]
    else:
        for i in prange(n):
            tid = numba.get_thread_id()
            counts[tid, 0] += 1
            pred_sums[tid, 0] += y_pred[i]
            true_sums[tid, 0] += y_true[i]

    # merge per-thread buffers
    final_pred = np.zeros(nbins, dtype=np.float64)
    final_true = np.zeros(nbins, dtype=np.float64)
    final_counts = np.zeros(nbins, dtype=np.int64)
    for b in range(nbins):
        for t in range(nthr):
            final_pred[b] += pred_sums[t, b]
            final_true[b] += true_sums[t, b]
            final_counts[b] += counts[t, b]

    # Convert to mean-pred, mean-true per bin and pack like the seq variant.
    freqs_predicted = np.zeros(nbins, dtype=np.float64)
    freqs_true = np.zeros(nbins, dtype=np.float64)
    for b in range(nbins):
        if final_counts[b] > 0:
            freqs_predicted[b] = final_pred[b] / final_counts[b]
            freqs_true[b] = final_true[b] / final_counts[b]
    return freqs_predicted, freqs_true, final_counts


@numba.njit(**NUMBA_NJIT_PARAMS, parallel=True)
def compute_ece_and_brier_decomposition_par(y_true, y_pred, nbins):
    n = len(y_true)
    if n == 0:
        return 1.0, 1.0, 0.0, 0.0, 1.0

    # Same race fix: np.min/max are parallel under parallel=True. The
    # base_rate accumulation IS auto-detected as a sum-reduction by
    # numba so it stays in a prange.
    pmin = np.min(y_pred)
    pmax = np.max(y_pred)
    span = pmax - pmin

    base_rate = 0.0
    for i in prange(n):
        base_rate += y_true[i]
    base_rate /= n

    nthr = numba.get_num_threads()
    pred_sums = np.zeros((nthr, nbins), dtype=np.float64)
    true_sums = np.zeros((nthr, nbins), dtype=np.float64)
    counts = np.zeros((nthr, nbins), dtype=np.int64)

    if span > 0:
        multiplier = (nbins - 1) / span
        for i in prange(n):
            tid = numba.get_thread_id()
            v = y_pred[i]
            b = int((v - pmin) * multiplier)
            counts[tid, b] += 1
            pred_sums[tid, b] += v
            true_sums[tid, b] += y_true[i]
    else:
        for i in prange(n):
            tid = numba.get_thread_id()
            counts[tid, 0] += 1
            pred_sums[tid, 0] += y_pred[i]
            true_sums[tid, 0] += y_true[i]

    pred_sum = np.zeros(nbins, dtype=np.float64)
    true_sum = np.zeros(nbins, dtype=np.float64)
    cnt = np.zeros(nbins, dtype=np.int64)
    for b in range(nbins):
        for t in range(nthr):
            pred_sum[b] += pred_sums[t, b]
            true_sum[b] += true_sums[t, b]
            cnt[b] += counts[t, b]

    ece = 0.0
    reliability = 0.0
    resolution = 0.0
    inv_n = 1.0 / n
    for b in range(nbins):
        if cnt[b] == 0:
            continue
        w = cnt[b] * inv_n
        p_mean = pred_sum[b] / cnt[b]
        acc = true_sum[b] / cnt[b]
        diff = p_mean - acc
        ece += w * abs(diff)
        reliability += w * diff * diff
        resolution += w * (acc - base_rate) ** 2
    uncertainty = base_rate * (1.0 - base_rate)
    brier_binned = reliability - resolution + uncertainty
    return ece, reliability, resolution, uncertainty, brier_binned


# ============================================================================
# Bench harness
# ============================================================================


def time_op(fn: Callable, *args, repeats: int = 5, warmup: int = 1):
    for _ in range(warmup):
        fn(*args)
    timings = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        out = fn(*args)
        timings.append(time.perf_counter() - t0)
    return out, min(timings)


def fmt(t: float) -> str:
    if t < 1e-3:
        return f"{t*1e6:7.1f}us"
    if t < 1.0:
        return f"{t*1e3:7.2f}ms"
    return f"{t:7.3f}s"


def sweep(name: str, seq_fn, par_fn, args_for_size, sizes, atol=1e-12):
    print(f"\n--- {name} ---")
    print(f"{'N':>10} | {'seq':>10} | {'par':>10} | {'par/seq':>8} | {'max_err':>10}")
    print("-" * 65)
    for N in sizes:
        args = args_for_size(N)
        # warm both
        seq_fn(*args)
        par_fn(*args)
        out_seq, t_seq = time_op(seq_fn, *args)
        out_par, t_par = time_op(par_fn, *args)
        # err
        if isinstance(out_seq, tuple):
            err = max(
                float(abs(np.asarray(s).ravel() - np.asarray(p).ravel()).max())
                if hasattr(s, "__len__") or hasattr(s, "shape") else float(abs(s - p))
                for s, p in zip(out_seq, out_par)
            )
        elif hasattr(out_seq, "shape"):
            err = float(np.max(np.abs(np.asarray(out_seq) - np.asarray(out_par))))
        else:
            err = float(abs(out_seq - out_par))
        ratio = t_par / t_seq if t_seq > 0 else float("inf")
        print(f"{N:>10} | {fmt(t_seq):>10} | {fmt(t_par):>10} | {ratio:7.2f}x | {err:.2e}")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--quick", action="store_true", help="Smaller grid (~30s).")
    args = ap.parse_args()

    sizes = [10_000, 100_000, 1_000_000, 10_000_000] if not args.quick else [10_000, 100_000, 1_000_000]
    multilabel_sizes = [10_000, 100_000, 1_000_000] if not args.quick else [10_000, 100_000]

    print(f"numba: {numba.__version__}, num_threads = {numba.get_num_threads()}")
    print(f"numpy: {np.__version__}")

    rng = np.random.default_rng(0)

    # ---------------------------------------------------------------- Brier
    def brier_args(N):
        y = (rng.standard_normal(N) > 0).astype(np.float64)
        p = np.clip(0.5 + 0.3 * rng.standard_normal(N), 0, 1)
        return y, p
    sweep("fast_brier_score_loss", fast_brier_score_loss, fast_brier_score_loss_par,
          brier_args, sizes)

    # ---------------------------------------------------------------- Log loss
    def ll_args(N):
        y = (rng.standard_normal(N) > 0).astype(np.float64)
        p = np.clip(0.5 + 0.3 * rng.standard_normal(N), 1e-10, 1 - 1e-10)
        return y, p, 1e-15
    sweep("fast_log_loss_binary", fast_log_loss_binary, fast_log_loss_binary_par,
          ll_args, sizes)

    # ---------------------------------------------------------------- PR/F1 counts
    def prf1_args(N):
        y = (rng.standard_normal(N) > 0).astype(np.int8)
        p = (rng.standard_normal(N) > 0).astype(np.int8)
        return y, p
    sweep("compute_pr_recall_f1_metrics", compute_pr_recall_f1_metrics,
          compute_pr_recall_f1_metrics_par, prf1_args, sizes)

    # ---------------------------------------------------------------- Subset accuracy (multilabel)
    def subset_args(N, K=5):
        yt = (rng.standard_normal((N, K)) > 0).astype(np.uint8)
        yp = (rng.standard_normal((N, K)) > 0).astype(np.uint8)
        return yt, yp
    sweep("_fast_subset_accuracy (K=5)", _fast_subset_accuracy_seq,
          _fast_subset_accuracy_par, subset_args, multilabel_sizes)

    # ---------------------------------------------------------------- Jaccard (multilabel)
    sweep("_fast_jaccard_score (K=5)", _fast_jaccard_score_seq,
          _fast_jaccard_score_par, subset_args, multilabel_sizes)

    # ---------------------------------------------------------------- Calibration binning (atomic histogram)
    def calbin_args(N, nbins=10):
        y = (rng.standard_normal(N) > 0).astype(np.float64)
        p = np.clip(0.5 + 0.3 * rng.standard_normal(N), 0, 1)
        return y, p, nbins
    sweep("fast_calibration_binning (10 bins)", fast_calibration_binning,
          fast_calibration_binning_par, calbin_args, sizes)

    # ---------------------------------------------------------------- ECE + Brier decomp
    sweep("compute_ece_and_brier_decomposition (10 bins)",
          compute_ece_and_brier_decomposition,
          compute_ece_and_brier_decomposition_par,
          calbin_args, sizes)


if __name__ == "__main__":
    sys.exit(main())
