"""CPX23 bench — NDCG per-query double-sort + per-k n_rel_total hoist.

Two claims under test against ``ranking.py``:

(a) ``_ndcg_one_query`` sorts twice per call: ``np.argsort(-score)`` for the
    predicted order and ``-np.sort(-true)`` for the ideal order. The concern
    is that when the SAME query is evaluated at several k values, both sorts
    are recomputed per k.
(b) ``n_rel_total`` (count of relevant items) is loop-invariant per query but
    might be recounted inside the per-k loop.

Reality in the multi-k hot path (``_summary_batched_kernel``): the sort of
each group is done ONCE per group (argsort of score + sort of true), the
results ``rels_pred`` / ``rels_ideal`` are reused across every k, and
``n_rel_total`` is counted ONCE before the ``for kj in range(K)`` loop. So
both (a) and (b) are already hoisted/cached in production.

This bench measures the WIN that hoisting buys, by A/B-ing the current
production kernel against a deliberately de-optimised "naive" kernel that
re-sorts and re-counts inside the per-k loop (the shape the prompt feared).
That quantifies what the existing code already saves, and confirms there is
no further win to extract — the optimization is already present.

Run:  python -m mlframe.metrics._benchmarks.bench_ndcg_sort_count_hoist_cpx23
"""

from __future__ import annotations

import time

import numba
import numpy as np
from numba import prange

from mlframe.metrics.ranking import _summary_batched_kernel, NUMBA_NJIT_PARAMS


# --------------------------------------------------------------------------
# Naive baseline kernel: re-sort + re-count INSIDE the per-k loop (the shape
# the prompt worried the code might be in). Used only as the slow A side.
# --------------------------------------------------------------------------
@numba.njit(fastmath=False, cache=True, nogil=True, parallel=True)
def _summary_naive_resort_per_k(sorted_y_true, sorted_y_score, group_starts, eval_ks):
    n_groups = len(group_starts) - 1
    K = len(eval_ks)
    ndcg_per_group = np.full((n_groups, K), np.nan, dtype=np.float64)
    map_per_group = np.full((n_groups, K), np.nan, dtype=np.float64)
    mrr_per_group = np.full(n_groups, np.nan, dtype=np.float64)

    for i in prange(n_groups):
        s = group_starts[i]
        e = group_starts[i + 1]
        n = e - s
        if n == 0:
            continue
        y_t = sorted_y_true[s:e]
        y_sc = sorted_y_score[s:e]

        # MRR (needs one sort + count, kept outside the k-loop as in prod).
        order_m = np.argsort(-y_sc, kind="mergesort")
        rels_pred_m = y_t[order_m]
        n_rel_m = 0
        for j in range(n):
            if y_t[j] > 0:
                n_rel_m += 1
        if n_rel_m > 0:
            for j in range(n):
                if rels_pred_m[j] > 0:
                    mrr_per_group[i] = 1.0 / (j + 1.0)
                    break

        for kj in range(K):
            k = eval_ks[kj]
            limit = k if k < n else n
            # DE-OPTIMISED: re-sort BOTH orders and re-count per k.
            order = np.argsort(-y_sc, kind="mergesort")
            rels_pred = y_t[order]
            rels_ideal = -np.sort(-y_t)
            n_rel_total = 0
            for j in range(n):
                if y_t[j] > 0:
                    n_rel_total += 1

            idcg = 0.0
            for j in range(limit):
                rel = rels_ideal[j]
                if rel > 0:
                    idcg += ((2.0 ** rel) - 1.0) / np.log2(j + 2.0)
            if idcg > 0.0:
                dcg = 0.0
                for j in range(limit):
                    rel = rels_pred[j]
                    if rel > 0:
                        dcg += ((2.0 ** rel) - 1.0) / np.log2(j + 2.0)
                ndcg_per_group[i, kj] = dcg / idcg

            if n_rel_total > 0:
                n_hits = 0
                sum_prec = 0.0
                for j in range(limit):
                    if rels_pred[j] > 0:
                        n_hits += 1
                        sum_prec += n_hits / (j + 1.0)
                denom = k if k < n_rel_total else n_rel_total
                map_per_group[i, kj] = sum_prec / denom

    ndcg_sums = np.zeros(K, dtype=np.float64)
    ndcg_counts = np.zeros(K, dtype=np.int64)
    map_sums = np.zeros(K, dtype=np.float64)
    map_counts = np.zeros(K, dtype=np.int64)
    for kj in range(K):
        for i in range(n_groups):
            v = ndcg_per_group[i, kj]
            if not np.isnan(v):
                ndcg_sums[kj] += v
                ndcg_counts[kj] += 1
            v = map_per_group[i, kj]
            if not np.isnan(v):
                map_sums[kj] += v
                map_counts[kj] += 1
    mrr_sum = 0.0
    mrr_count = 0
    for i in range(n_groups):
        v = mrr_per_group[i]
        if not np.isnan(v):
            mrr_sum += v
            mrr_count += 1
    return ndcg_sums, ndcg_counts, map_sums, map_counts, mrr_sum, mrr_count


def _make_data(n_groups, qlen, seed=0):
    rng = np.random.default_rng(seed)
    n = n_groups * qlen
    y_true = rng.integers(0, 5, size=n).astype(np.float64)
    y_score = rng.standard_normal(n)
    group_starts = np.arange(0, n + 1, qlen, dtype=np.intp)
    return y_true, y_score, group_starts


def _best_of(fn, args, repeat=7):
    best = float("inf")
    out = None
    for _ in range(repeat):
        t0 = time.perf_counter()
        out = fn(*args)
        dt = time.perf_counter() - t0
        best = min(best, dt)
    return best, out


def main():
    eval_ks = np.asarray([5, 10, 20], dtype=np.int64)
    shapes = [(2000, 100), (500, 500), (200, 1000)]

    # Warm both kernels (JIT compile) on a small shape.
    yt, ys, gs = _make_data(50, 64, seed=99)
    _summary_batched_kernel(yt, ys, gs, eval_ks)
    _summary_naive_resort_per_k(yt, ys, gs, eval_ks)

    print(f"{'shape (groups x qlen)':>24} | {'naive resort/k':>16} | {'prod cached':>14} | speedup | identical")
    print("-" * 92)
    for n_groups, qlen in shapes:
        yt, ys, gs = _make_data(n_groups, qlen, seed=n_groups + qlen)
        t_naive, r_naive = _best_of(_summary_naive_resort_per_k, (yt, ys, gs, eval_ks))
        t_prod, r_prod = _best_of(_summary_batched_kernel, (yt, ys, gs, eval_ks))
        # Identity: prod must equal naive exactly (both deterministic, mergesort).
        same = all(np.array_equal(a, b) if isinstance(a, np.ndarray) else a == b
                   for a, b in zip(r_prod, r_naive))
        speedup = t_naive / t_prod
        print(f"{f'{n_groups} x {qlen}':>24} | {t_naive*1e3:13.3f} ms | {t_prod*1e3:11.3f} ms | {speedup:6.2f}x | {same}")

    print()
    print("Verdict: production _summary_batched_kernel already hoists the per-k")
    print("sort (rels_pred/rels_ideal computed once per group) AND n_rel_total")
    print("(counted once before the k-loop). The speedup column shows the win the")
    print("hoist ALREADY captures vs a naive re-sort/re-count-per-k kernel.")
    print("No further optimization to extract -> CPX23 (a) and (b) REJECTED (no-op).")


if __name__ == "__main__":
    main()
