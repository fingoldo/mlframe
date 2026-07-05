"""Bench: vectorize the per-query Python loops in ``compute_extra_aggregates``.

The current code (transformer/_aggregation.py) walks ``n_queries`` rows in plain Python for each of: the softmax-weight recompute, ``y_skew``, and
``x_centroid_dist``. The docstring claims the overhead is negligible at k=32-128, but ``compute_extra_aggregates`` runs once per HEAD per FOLD inside the
row-attention OOF loop, and ``n_queries`` is the validation-fold size (10k-100k in prod). 100k Python iterations doing a gather + matmul each is not negligible.

Each of these reductions is a fixed-k batched op that vectorizes cleanly with a single gather + broadcast:
- weights:        ``einsum('qkd,qd->qk', k_proj[topk_ids], q_proj) / temp`` then a row-wise stable softmax
- y_skew:         weighted central moments over the gathered (n_queries, k) target block
- x_centroid_dist: ``norm(q_proj - sum(w[...,None] * k_proj[topk_ids], axis=1))``

Run: python -m mlframe.feature_engineering._benchmarks.bench_extra_aggregates_vectorized
"""
from __future__ import annotations

import time

import numpy as np

from mlframe.feature_engineering.transformer._aggregation import compute_extra_aggregates


def _old_weights(q_proj, k_proj, topk_ids, softmax_temp):
    n_queries, k_count = topk_ids.shape
    weights = np.empty((n_queries, k_count), dtype=np.float32)
    for q in range(n_queries):
        ids_q = topk_ids[q]
        logits_q = (k_proj[ids_q] @ q_proj[q]) / softmax_temp
        m = float(logits_q.max())
        if not np.isfinite(m):
            weights[q] = 1.0 / k_count
            continue
        exps = np.exp(logits_q - m)
        s = float(exps.sum())
        if s <= 0.0 or not np.isfinite(s):
            weights[q] = 1.0 / k_count
        else:
            weights[q] = exps / s
    return weights


def _new_weights(q_proj, k_proj, topk_ids, softmax_temp):
    n_queries, k_count = topk_ids.shape
    gathered = k_proj[topk_ids]  # (n_queries, k, head_dim)
    logits = np.einsum("qkd,qd->qk", gathered, q_proj) / softmax_temp
    m = logits.max(axis=1, keepdims=True)
    finite_m = np.isfinite(m[:, 0])
    exps = np.exp(logits - m)
    s = exps.sum(axis=1)
    weights = exps / s[:, None]
    bad = (~finite_m) | (s <= 0.0) | (~np.isfinite(s))
    if bad.any():
        weights[bad] = 1.0 / k_count
    return weights.astype(np.float32, copy=False)


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
    head_dim = 16
    softmax_temp = 1.0
    for n_queries in (10_000, 50_000, 100_000):
        for k in (32, 64):
            n_train = max(n_queries, 2000)
            q_proj = rng.standard_normal((n_queries, head_dim)).astype(np.float32)
            k_proj = rng.standard_normal((n_train, head_dim)).astype(np.float32)
            topk_ids = rng.integers(0, n_train, size=(n_queries, k)).astype(np.int64)
            y_train = rng.standard_normal(n_train).astype(np.float32)

            t_old, w_old = _best_of(_old_weights, q_proj, k_proj, topk_ids, softmax_temp)
            t_new, w_new = _best_of(_new_weights, q_proj, k_proj, topk_ids, softmax_temp)
            max_abs = float(np.max(np.abs(w_old - w_new)))
            print(f"n_q={n_queries:>7} k={k:>3}  weights  old={t_old*1e3:8.2f}ms  new={t_new*1e3:8.2f}ms  "
                  f"speedup={t_old/t_new:5.2f}x  max|dw|={max_abs:.2e}")

            # End-to-end on the public function (y_skew + x_centroid_dist + y_iqr) before/after is compared in the identity test;
            # here we time the full call to confirm the loop dominates.
            aggs = ("y_skew", "x_centroid_dist", "y_iqr")
            t_full, _ = _best_of(compute_extra_aggregates, q_proj, k_proj, y_train, topk_ids, softmax_temp, aggs, n=3)
            print(f"                    full call ({','.join(aggs)})  {t_full*1e3:8.2f}ms")


if __name__ == "__main__":
    main()
