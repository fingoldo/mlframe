"""Round-2 bench: candidates not covered by bench_numba_parallel.py.

Targets the remaining numba kernels in mlframe.metrics that show up in
suite reports / charts but weren't parallelised in commit 1f84017:

- cb_logits_to_probs_binary: element-wise sigmoid (CB postproc)
- cb_logits_to_probs_multiclass: row-wise softmax (CB postproc)
- _max_abs_pct_error_kernel: regression metric, reduction over N
- probability_separation_score: masked sum + std
- format_classification_report: count loop (similar to PRF1)
- compute_grouped_group_aucs: per-group AUC scan; outer loop over groups

Decision rule: ship a `_par` variant only if speedup >= 2x at the size
where the kernel is actually called in production (typically N=200k for
single-call kernels, fewer for multi-call kernels).
"""

from __future__ import annotations

import argparse
import sys
import time
from typing import Callable

import numpy as np
import numba
from numba import prange

sys.path.insert(0, ".")
from mlframe.metrics import (  # noqa: E402
    cb_logits_to_probs_binary,
    cb_logits_to_probs_multiclass,
    _max_abs_pct_error_kernel,
    probability_separation_score,
    format_classification_report,
    NUMBA_NJIT_PARAMS,
    compute_grouped_group_aucs,
)


# ============================================================================
# Round 2 par variants under test
# ============================================================================


@numba.njit(**NUMBA_NJIT_PARAMS, parallel=True)
def cb_logits_to_probs_binary_par(logits):
    """Match seq signature: returns (N, 2) [class_0, class_1]."""
    n = len(logits)
    probs = np.empty((n, 2), dtype=np.float64)
    for i in prange(n):
        p1 = 1.0 / (1.0 + np.exp(-logits[i]))
        probs[i, 0] = 1.0 - p1
        probs[i, 1] = p1
    return probs


@numba.njit(**NUMBA_NJIT_PARAMS, parallel=True)
def cb_logits_to_probs_multiclass_par(logits_list):
    """Match seq: input shape is (n_classes, n_samples), output (n_samples, n_classes)."""
    n_classes, n_samples = logits_list.shape
    probs = np.empty((n_samples, n_classes), dtype=np.float64)
    for i in prange(n_samples):
        # numerically stable softmax
        row_max = logits_list[0, i]
        for k in range(1, n_classes):
            v = logits_list[k, i]
            if v > row_max:
                row_max = v
        denom = 0.0
        for k in range(n_classes):
            denom += np.exp(logits_list[k, i] - row_max)
        for k in range(n_classes):
            probs[i, k] = np.exp(logits_list[k, i] - row_max) / denom
    return probs


@numba.njit(**NUMBA_NJIT_PARAMS, parallel=True)
def _max_abs_pct_error_par(y_true, y_pred):
    """Per-thread max accumulator avoids the race that the simple
    ``if err > max_err: max_err = err`` form has under prange (numba
    only auto-recognises ``+=`` / ``*=`` as reductions, not ``if``-based
    max-update; that pattern can drop max updates between threads).

    Solution: per-thread max array, final reduction outside the prange.
    """
    n = len(y_true)
    nthr = numba.get_num_threads()
    per_thread_max = np.zeros(nthr, dtype=np.float64)
    n_zero = 0
    for i in prange(n):
        if y_true[i] == 0:
            n_zero += 1
            continue
        err = abs((y_pred[i] - y_true[i]) / y_true[i])
        tid = numba.get_thread_id()
        if err > per_thread_max[tid]:
            per_thread_max[tid] = err
    max_err = 0.0
    for t in range(nthr):
        if per_thread_max[t] > max_err:
            max_err = per_thread_max[t]
    return max_err, n_zero


@numba.njit(**NUMBA_NJIT_PARAMS, parallel=True)
def probability_separation_score_par(y_true, y_prob, class_label=1, std_weight=0.5):
    """Masked mean + std. Uses two prange passes (one for mean, one for
    std-deviation accumulation since std depends on mean)."""
    n = len(y_true)
    if n == 0:
        return np.nan
    # Pass 1: count + sum probs of in-class samples.
    n_in = 0
    s = 0.0
    for i in prange(n):
        if y_true[i] == class_label:
            n_in += 1
            s += y_prob[i]
    if n_in == 0:
        return np.nan
    mean = s / n_in
    if std_weight == 0.0:
        return mean if class_label == 1 else 1.0 - mean

    # Pass 2: variance.
    sse = 0.0
    for i in prange(n):
        if y_true[i] == class_label:
            d = y_prob[i] - mean
            sse += d * d
    std = np.sqrt(sse / n_in)
    addend = std * std_weight
    if class_label == 1:
        return mean - addend
    else:
        return (1.0 - mean) - addend


@numba.njit(**NUMBA_NJIT_PARAMS, parallel=True)
def _grouped_group_aucs_par_outer(sorted_group_ids, sorted_y_true, sorted_y_score):
    """Parallelise the OUTER per-group loop. Each group's AUC is a
    self-contained sequential scan, so groups are independent. Use
    a typed Dict for the result. NOTE: this kernel is benchmark-only --
    the production ``compute_grouped_group_aucs`` returns a numba
    typed Dict that we'd need to handle carefully if shipping. Keeping
    bench-only here to gauge speedup before deciding."""
    n = len(sorted_group_ids)
    if n == 0:
        return np.zeros(0, dtype=np.int64), np.zeros(0, dtype=np.float64), np.zeros(0, dtype=np.float64)

    # Find group boundaries (sequential, single pass, cheap).
    boundaries = [0]
    for i in range(1, n):
        if sorted_group_ids[i] != sorted_group_ids[i - 1]:
            boundaries.append(i)
    boundaries.append(n)
    n_groups = len(boundaries) - 1
    group_ids_out = np.empty(n_groups, dtype=np.int64)
    rocs = np.empty(n_groups, dtype=np.float64)
    prs = np.empty(n_groups, dtype=np.float64)

    for g in prange(n_groups):
        s = boundaries[g]
        e = boundaries[g + 1]
        group_ids_out[g] = sorted_group_ids[s]
        # Per-group AUC: same alg as fast_numba_aucs_simple inline.
        # Sort by score desc within group.
        gy = sorted_y_true[s:e]
        gp = sorted_y_score[s:e]
        # numba handles np.argsort inside prange; we need contiguous slices.
        order = np.argsort(gp)[::-1]
        ys = gy[order]
        ps = gp[order]
        total_pos = 0.0
        for k in range(e - s):
            total_pos += ys[k]
        total_neg = (e - s) - total_pos
        if total_pos == 0 or total_neg == 0:
            rocs[g] = np.nan
            prs[g] = np.nan
            continue
        # ROC AUC + AP via tied-aware Riemann sum.
        last_fps = 0
        last_tps = 0
        tps = 0
        fps = 0
        roc = 0.0
        prev_recall = 0.0
        ap = 0.0
        for k in range(e - s):
            tps += ys[k]
            fps += 1 - ys[k]
            if k == (e - s - 1) or ps[k + 1] != ps[k]:
                roc += (fps - last_fps) * (last_tps + tps)
                last_fps = fps
                last_tps = tps
                cur_recall = tps / total_pos
                cur_precision = tps / (tps + fps) if (tps + fps) > 0 else 0.0
                ap += (cur_recall - prev_recall) * cur_precision
                prev_recall = cur_recall
        denom = tps * fps * 2
        rocs[g] = roc / denom if denom > 0 else np.nan
        prs[g] = ap

    return group_ids_out, rocs, prs


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


def sweep(name, seq_fn, par_fn, args_for_size, sizes):
    print(f"\n--- {name} ---")
    print(f"{'N':>10} | {'seq':>10} | {'par':>10} | {'par/seq':>8} | {'max_err':>10}")
    print("-" * 65)
    for N in sizes:
        args = args_for_size(N)
        seq_fn(*args)  # warm
        par_fn(*args)
        out_seq, t_seq = time_op(seq_fn, *args)
        out_par, t_par = time_op(par_fn, *args)
        if isinstance(out_seq, tuple):
            errs = []
            for s, p in zip(out_seq, out_par):
                if hasattr(s, "shape") or hasattr(s, "__len__"):
                    errs.append(float(np.max(np.abs(np.asarray(s) - np.asarray(p)))))
                else:
                    errs.append(float(abs(s - p)))
            err = max(errs) if errs else 0.0
        elif hasattr(out_seq, "shape"):
            err = float(np.max(np.abs(np.asarray(out_seq) - np.asarray(out_par))))
        else:
            err = float(abs(out_seq - out_par))
        ratio = t_par / t_seq if t_seq > 0 else float("inf")
        print(f"{N:>10} | {fmt(t_seq):>10} | {fmt(t_par):>10} | {ratio:7.2f}x | {err:.2e}")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()

    # 10M sizes drive OOM on cb_logits_to_probs_*_par when verifying max_err
    # because the seq baseline allocates (10M, 2) probs and the par
    # variant another (10M, 2) -- 320 MiB total before subtraction. Cap
    # sweep at 1M for the round-2 set; that's already 5x the production
    # multiclass-prediction size we'd hit.
    sizes = [10_000, 100_000, 1_000_000] if not args.quick else [10_000, 100_000]
    print(f"numba: {numba.__version__}, num_threads = {numba.get_num_threads()}")

    rng = np.random.default_rng(0)

    # cb_logits_to_probs_binary
    sweep("cb_logits_to_probs_binary",
          cb_logits_to_probs_binary, cb_logits_to_probs_binary_par,
          lambda N: (rng.standard_normal(N).astype(np.float64),),
          sizes)

    # cb_logits_to_probs_multiclass (input is (n_classes, n_samples))
    sweep("cb_logits_to_probs_multiclass (K=5)",
          cb_logits_to_probs_multiclass, cb_logits_to_probs_multiclass_par,
          lambda N: (rng.standard_normal((5, N)).astype(np.float64),),
          sizes)

    # _max_abs_pct_error_kernel
    def mape_args(N):
        y = rng.standard_normal(N).astype(np.float64)
        # ensure no NaN, mostly nonzero
        y[y == 0] = 1e-3
        p = y + 0.1 * rng.standard_normal(N)
        return y, p
    sweep("_max_abs_pct_error_kernel",
          _max_abs_pct_error_kernel, _max_abs_pct_error_par,
          mape_args, sizes)

    # probability_separation_score
    def psep_args(N):
        y = (rng.standard_normal(N) > 0).astype(np.int64)
        p = np.clip(0.5 + 0.3 * rng.standard_normal(N), 0, 1)
        return y, p, 1, 0.5
    sweep("probability_separation_score",
          probability_separation_score, probability_separation_score_par,
          psep_args, sizes)


if __name__ == "__main__":
    sys.exit(main())
