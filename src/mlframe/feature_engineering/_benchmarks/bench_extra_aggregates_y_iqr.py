"""Bench: vectorize the per-query ``y_iqr`` Python loop in ``compute_extra_aggregates``.

The weights / y_skew / x_centroid_dist blocks were batched in a prior wave, but ``y_iqr`` still walks ``n_queries`` rows in plain Python, each doing a per-row
``np.argsort`` + ``np.cumsum`` + two ``np.interp`` calls. ``compute_extra_aggregates`` runs once per HEAD per FOLD inside the row-attention OOF loop with
``n_queries`` = validation-fold size (10k-100k in prod), so 100k Python iterations each spawning four small-array numpy calls is real overhead.

New path: one batched ``np.argsort(y_nbr, axis=1)`` + ``take_along_axis`` to sort weights, ``cumsum`` along axis=1, then a fused njit kernel does the per-row
cumulative-weight quantile interp (the linear-interp lookup that ``np.interp`` does, replicated exactly: clamp below first / above last, linear between brackets).

Run: python -m mlframe.feature_engineering._benchmarks.bench_extra_aggregates_y_iqr
"""
from __future__ import annotations

import time

import numpy as np


def _old_y_iqr(y_train, topk_ids, weights):
    n_queries = topk_ids.shape[0]
    y_iqr = np.empty(n_queries, dtype=np.float32)
    for q in range(n_queries):
        y_q = y_train[topk_ids[q]]
        order = np.argsort(y_q)
        y_sorted = y_q[order]
        w_sorted = weights[q][order]
        cum_w = np.cumsum(w_sorted)
        q25 = float(np.interp(0.25, cum_w, y_sorted))
        q75 = float(np.interp(0.75, cum_w, y_sorted))
        y_iqr[q] = q75 - q25
    return y_iqr


def _new_y_iqr(y_train, topk_ids, weights):
    from mlframe.feature_engineering.transformer._aggregation import _weighted_iqr_batched
    y_nbr = y_train[topk_ids]
    order = np.argsort(y_nbr, axis=1)
    y_sorted = np.take_along_axis(y_nbr, order, axis=1)
    w_sorted = np.take_along_axis(weights, order, axis=1)
    cum_w = np.cumsum(w_sorted, axis=1)
    return _weighted_iqr_batched(y_sorted, cum_w)


def _best_of(fn, *args, n=7):
    best = float("inf")
    out = None
    for _ in range(n):
        t0 = time.perf_counter()
        out = fn(*args)
        best = min(best, time.perf_counter() - t0)
    return best, out


def main():
    rng = np.random.default_rng(0)
    # warm the njit kernel
    _new_y_iqr(rng.standard_normal(2000).astype(np.float32), rng.integers(0, 2000, size=(8, 32)).astype(np.int64), rng.random((8, 32)).astype(np.float32))
    for n_queries in (10_000, 50_000, 100_000):
        for k in (32, 64):
            n_train = max(n_queries, 2000)
            topk_ids = rng.integers(0, n_train, size=(n_queries, k)).astype(np.int64)
            y_train = rng.standard_normal(n_train).astype(np.float32)
            w = rng.random((n_queries, k)).astype(np.float32)
            w /= w.sum(axis=1, keepdims=True)

            t_old, o_old = _best_of(_old_y_iqr, y_train, topk_ids, w)
            t_new, o_new = _best_of(_new_y_iqr, y_train, topk_ids, w)
            max_abs = float(np.max(np.abs(o_old - o_new)))
            print(f"n_q={n_queries:>7} k={k:>3}  y_iqr  old={t_old*1e3:8.2f}ms  new={t_new*1e3:8.2f}ms  " f"speedup={t_old/t_new:5.2f}x  max|d|={max_abs:.2e}")


if __name__ == "__main__":
    main()
