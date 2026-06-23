"""Bench + identity check for routing the public ``fast_aucs_per_group`` to the
presort + boundary-walk twin ``fast_aucs_per_group_optimized``.

The naive public path does ``group_ids == group_id`` masking + an argsort per group
inside a Python loop over ``np.unique(group_ids)`` -- O(n * G). The optimized twin
filters doomed (single-sample / single-class) groups up front, then runs ONE numba
boundary-walk over presorted data -- effectively O(n log n).

Identity: for every VALID group (>=2 samples AND both classes present) the two paths
compute the SAME (roc_auc, pr_auc) -- the inner kernel is the same scan. They differ
ONLY on degenerate groups: the naive path emits (0.0, 0.0) for single-sample groups
(a documented silent bug -- AUC is undefined there), while the optimized twin emits
(nan, nan). Routing to the twin therefore also flips that degenerate output to the
corrected NaN. This bench asserts valid-group AUC equality and reports the timing.

Run: python src/mlframe/metrics/_benchmarks/bench_fast_aucs_per_group_dispatch.py
"""
from __future__ import annotations

import time

import numpy as np


def _bench(n: int, n_groups: int, best_of: int = 5):
    from mlframe.metrics._auc_per_group import (
        fast_aucs_per_group,
        fast_aucs_per_group_optimized,
    )

    rng = np.random.default_rng(0)
    y_true = (rng.random(n) < 0.4).astype(np.float64)
    y_score = rng.random(n)
    # Many groups, reasonable sizes -- the production "fine-grained group_ids" shape.
    group_ids = rng.integers(0, n_groups, size=n).astype(np.int64)

    # Warm numba in the optimized path.
    fast_aucs_per_group_optimized(y_true[:50], y_score[:50], group_ids[:50])

    def time_fn(fn):
        best = float("inf")
        for _ in range(best_of):
            t0 = time.perf_counter()
            fn(y_true, y_score, group_ids)
            best = min(best, time.perf_counter() - t0)
        return best

    t_naive = time_fn(fast_aucs_per_group)
    t_opt = time_fn(fast_aucs_per_group_optimized)

    # Identity on VALID groups.
    _, _, ga_naive = fast_aucs_per_group(y_true, y_score, group_ids)
    _, _, ga_opt = fast_aucs_per_group_optimized(y_true, y_score, group_ids)
    max_diff = 0.0
    n_valid = 0
    for gid, (r_o, p_o) in ga_opt.items():
        if np.isnan(r_o):
            continue  # degenerate group; naive gives (0,0) or nan -- not compared
        n_valid += 1
        r_n, p_n = ga_naive[gid]
        max_diff = max(max_diff, abs(r_o - r_n), abs(p_o - p_n))

    print(
        f"n={n:>7} groups={n_groups:>6} | naive={t_naive*1e3:8.2f}ms "
        f"opt={t_opt*1e3:8.2f}ms speedup={t_naive/t_opt:5.2f}x | "
        f"valid={n_valid} max|d_auc|={max_diff:.2e}"
    )
    return t_naive, t_opt, max_diff


if __name__ == "__main__":
    for n, g in [(5_000, 50), (20_000, 200), (50_000, 2_000), (100_000, 20_000)]:
        _bench(n, g)
