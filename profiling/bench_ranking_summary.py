"""A/B bench: per-query Python loop vs batched numba kernel for
compute_ranking_summary on 1M-row LTR inputs.

Wave 4 1M-row fuzz profile (c0001: LTR xgb, 1M rows, 333k groups)
attributed 6 s of wall to compute_ranking_summary -- 2.33M
Python->numba transitions per call (NDCG@1/5/10 + MAP@1/5/10 + MRR
= 7 outer loops x 333k groups). The new batched kernel does all
metrics in a single numba dispatch with prange.

Usage:
    python -m mlframe.profiling.bench_ranking_summary
"""

from __future__ import annotations

import statistics
import time

import numpy as np

from mlframe.ranking_metrics import (
    compute_ranking_summary,
    _summary_batched_kernel,
    _iter_group_slices,
    _ndcg_one_query,
    _map_one_query,
    _mrr_one_query,
)


def _old_compute_ranking_summary(y_true, y_score, group_ids, eval_at=(1, 5, 10)):
    """Pre-Wave-4 form: 7 outer loops, per-group numba dispatch each iter."""
    sorted_y_true, sorted_y_score, group_starts = _iter_group_slices(y_true, y_score, group_ids)
    n_groups = len(group_starts) - 1
    out = {}
    for k in eval_at:
        out[f"ndcg@{k}"] = float("nan")
        out[f"map@{k}"] = float("nan")
    out["mrr"] = float("nan")
    if n_groups == 0:
        return out

    for label_k, kernel in (("ndcg", _ndcg_one_query), ("map", _map_one_query)):
        for k in eval_at:
            accum = 0.0
            n_valid = 0
            for i in range(n_groups):
                s, e = group_starts[i], group_starts[i + 1]
                v = kernel(sorted_y_true[s:e], sorted_y_score[s:e], k)
                if not np.isnan(v):
                    accum += v
                    n_valid += 1
            out[f"{label_k}@{k}"] = accum / n_valid if n_valid > 0 else float("nan")

    accum = 0.0
    n_valid = 0
    for i in range(n_groups):
        s, e = group_starts[i], group_starts[i + 1]
        v = _mrr_one_query(sorted_y_true[s:e], sorted_y_score[s:e])
        if not np.isnan(v):
            accum += v
            n_valid += 1
    out["mrr"] = accum / n_valid if n_valid > 0 else float("nan")
    return out


def main() -> None:
    rng = np.random.default_rng(42)
    N = 1_000_000

    # Mimic c0001's group distribution: ~3 docs / query, 333k queries.
    group_sizes = rng.integers(2, 6, size=N // 3)
    group_sizes = group_sizes[group_sizes.cumsum() < N]
    while group_sizes.sum() < N:
        group_sizes = np.append(group_sizes, N - group_sizes.sum())
    group_ids = np.repeat(np.arange(len(group_sizes)), group_sizes)
    group_ids = group_ids[:N]

    y_true = rng.integers(0, 3, N).astype(np.int64)
    y_score = rng.uniform(0, 1, N).astype(np.float64)

    print(f"# N={N:_}, n_groups={len(np.unique(group_ids)):_}, eval_at=(1, 5, 10)")
    print()

    # Validate equivalence on a small sample first
    small_idx = np.arange(10_000)
    a = _old_compute_ranking_summary(y_true[small_idx], y_score[small_idx], group_ids[small_idx])
    b = compute_ranking_summary(y_true[small_idx], y_score[small_idx], group_ids[small_idx])
    print(f"  equivalence check on n=10_000:")
    for k in a:
        va, vb = a[k], b[k]
        ok = (np.isnan(va) and np.isnan(vb)) or abs(va - vb) < 1e-10
        print(f"    {k:<10} old={va:.6f}  new={vb:.6f}  {'OK' if ok else 'MISMATCH'}")
    print()

    def bench(fn, label, n_repeat=5):
        # Warm both numba paths
        fn(y_true, y_score, group_ids)
        times = []
        for _ in range(n_repeat):
            t0 = time.perf_counter()
            fn(y_true, y_score, group_ids)
            times.append(time.perf_counter() - t0)
        m = statistics.mean(times)
        s = statistics.stdev(times) if len(times) > 1 else 0.0
        print(f"  {label:<55} {m*1000:>9.1f} ms +/- {s*1000:>6.1f} ms")

    bench(_old_compute_ranking_summary, "OLD: 7 outer loops, per-group dispatch")
    bench(compute_ranking_summary, "NEW: single batched numba kernel (prange)")


if __name__ == "__main__":
    main()
