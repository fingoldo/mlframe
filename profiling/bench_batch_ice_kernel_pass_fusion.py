"""Bench pass-fusion in _batch_per_class_ice_kernel.

bench-attempt-rejected (2026-05-21, c0146 / iter133): fusing the Brier
accumulator + min/max scan into a single pass (cutting 3 sequential
N-passes to 2 pre-argsort) yields only 1.04x at the production hot-path
shape (N=1M, K=3) and 1.01-1.02x at smaller sizes. The argsort + AUC
walk dominates so heavily that pass fusion barely registers above noise.
Pre-fix profile (c0146, 1M rows) showed 2.328 s / 195 calls = 12 ms /
call tottime; a 4% reduction would save ~90 ms over the run, well below
the measurable speedup floor and not worth the kernel-rewrite churn.

The fusion IS numerically harmless (combines two reads of the same
y_p/y_t arrays into one) -- the rejection is purely on insufficient ROI.
Numbers from this bench:

    N= 100000  K=3: 3pass= 15.87ms  2pass= 15.94ms  (1.00x)
    N= 200000  K=3: 3pass= 34.25ms  2pass= 33.61ms  (1.02x)
    N=1000000  K=3: 3pass=215.87ms  2pass=206.66ms  (1.04x)
    N=1000000  K=5: 3pass=300.71ms  2pass=296.75ms  (1.01x)

Documented per ``feedback_document_failed_optimization_attempts`` so the
next agent doesn't re-try this same path.

Run: ``python profiling/bench_batch_ice_kernel_pass_fusion.py``
"""

import time
import numpy as np
import numba


@numba.njit(fastmath=False, cache=True, nogil=True, parallel=True)
def _ice_kernel_3pass(y_true_NK, y_pred_NK, nbins):
    """Current production form: 3 sequential N-passes pre-argsort."""
    N = y_true_NK.shape[0]
    K = y_true_NK.shape[1]
    out = np.empty(K, dtype=np.float64)

    for k in numba.prange(K):
        y_t = y_true_NK[:, k]
        y_p = y_pred_NK[:, k]

        # Pass 1: Brier
        s = 0.0
        for i in range(N):
            d = float(y_t[i]) - y_p[i]
            s += d * d
        brier = s / N if N > 0 else 1.0

        # Pass 2: min/max
        min_val = 1.0
        max_val = 0.0
        for i in range(N):
            v = y_p[i]
            if v > max_val:
                max_val = v
            if v < min_val:
                min_val = v
        span = max_val - min_val

        # Pass 3: bin assignment
        pockets_pred = np.zeros(nbins, dtype=np.int64)
        pockets_true = np.zeros(nbins, dtype=np.int64)
        if span > 0:
            multiplier = (nbins - 1) / span
            for i in range(N):
                ind = int(np.floor((y_p[i] - min_val) * multiplier))
                pockets_pred[ind] += 1
                pockets_true[ind] += y_t[i]
        else:
            for i in range(N):
                pockets_pred[0] += 1
                pockets_true[0] += y_t[i]

        desc_idx = np.argsort(-y_p, kind="mergesort")
        y_t_sorted = y_t[desc_idx]
        total_pos = 0
        for i in range(N):
            total_pos += y_t_sorted[i]
        out[k] = brier + min_val + max_val + (pockets_pred[0] + pockets_true[0]) * 1e-12 + total_pos * 1e-15
    return out


@numba.njit(fastmath=False, cache=True, nogil=True, parallel=True)
def _ice_kernel_2pass(y_true_NK, y_pred_NK, nbins):
    """Fused form: Brier accumulator + min/max in pass 1, bin-assign in pass 2."""
    N = y_true_NK.shape[0]
    K = y_true_NK.shape[1]
    out = np.empty(K, dtype=np.float64)

    for k in numba.prange(K):
        y_t = y_true_NK[:, k]
        y_p = y_pred_NK[:, k]

        # Pass 1 (fused): Brier + min/max in one walk
        s = 0.0
        min_val = 1.0
        max_val = 0.0
        for i in range(N):
            v = y_p[i]
            ti = float(y_t[i])
            d = ti - v
            s += d * d
            if v > max_val:
                max_val = v
            if v < min_val:
                min_val = v
        brier = s / N if N > 0 else 1.0
        span = max_val - min_val

        pockets_pred = np.zeros(nbins, dtype=np.int64)
        pockets_true = np.zeros(nbins, dtype=np.int64)
        if span > 0:
            multiplier = (nbins - 1) / span
            for i in range(N):
                ind = int(np.floor((y_p[i] - min_val) * multiplier))
                pockets_pred[ind] += 1
                pockets_true[ind] += y_t[i]
        else:
            for i in range(N):
                pockets_pred[0] += 1
                pockets_true[0] += y_t[i]

        desc_idx = np.argsort(-y_p, kind="mergesort")
        y_t_sorted = y_t[desc_idx]
        total_pos = 0
        for i in range(N):
            total_pos += y_t_sorted[i]
        out[k] = brier + min_val + max_val + (pockets_pred[0] + pockets_true[0]) * 1e-12 + total_pos * 1e-15
    return out


def bench(label, fn, y_true_NK, y_pred_NK, nbins, n_iter=20):
    fn(y_true_NK, y_pred_NK, nbins)
    fn(y_true_NK, y_pred_NK, nbins)
    times = []
    for _ in range(5):
        t = time.perf_counter()
        for _ in range(n_iter):
            fn(y_true_NK, y_pred_NK, nbins)
        times.append((time.perf_counter() - t) / n_iter)
    return min(times) * 1e3, label


if __name__ == "__main__":
    np.random.seed(0)
    for N, K in [(100_000, 3), (200_000, 3), (1_000_000, 3), (1_000_000, 5)]:
        y_pred = np.random.rand(N, K).astype(np.float64)
        y_pred = y_pred / y_pred.sum(axis=1, keepdims=True)
        y_true = np.zeros((N, K), dtype=np.int8)
        y_true[np.arange(N), np.random.randint(0, K, size=N)] = 1

        t3, _ = bench("3pass", _ice_kernel_3pass, y_true, y_pred, 10)
        t2, _ = bench("2pass", _ice_kernel_2pass, y_true, y_pred, 10)
        speedup = t3 / t2
        print(f"N={N:>7}  K={K}: 3pass={t3:6.2f}ms  2pass={t2:6.2f}ms  ({speedup:.2f}x)")
