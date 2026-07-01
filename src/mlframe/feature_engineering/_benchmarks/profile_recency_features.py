"""cProfile + backend bench for the recency-weighted per-entity features (PZAD caseclients).

Run:  python -m mlframe.feature_engineering._benchmarks.profile_recency_features

Covers the three primitives added for the dunnhumby weighted-scheme lecture:
- ``per_group_recency_weighted_mean``  (cheap single pass; njit)
- ``per_group_recency_weighted_mode``  (KDE: O(n_groups * m * n_grid); the hot path)
- ``per_group_behavioral_stability``   (same KDE kernel, span-relative bandwidth)

Also A/Bs the serial vs prange KDE kernel across n_groups to justify the dispatch
threshold ``_KDE_PARALLEL_MIN_GROUPS``. Warms numba before timing (first call JITs).

Note on cProfile: numba njit bodies are mis-attributed to the Python caller frame and
deep-stack pandas/sklearn timings inflate ~10x; use the wall-time A/B numbers below,
not cProfile ``tottime``, for the optimization verdict.
"""

from __future__ import annotations

import cProfile
import pstats
import time

import numpy as np

from mlframe.feature_engineering.recency_aggregation import per_group_recency_weighted_mean
from mlframe.feature_engineering.recency_density import (
    _weighted_kde_mode_and_peak,
    _weighted_kde_mode_and_peak_parallel,
    per_group_behavioral_stability,
    per_group_recency_weighted_mode,
    _sort_into_groups,
)


def _make_panel(n_entities: int, hist: int, seed: int = 0):
    rng = np.random.default_rng(seed)
    vals, groups, order = [], [], []
    for e in range(n_entities):
        series = rng.normal(rng.uniform(0, 100), 5.0, size=hist)
        vals.append(series)
        groups.append(np.full(hist, e))
        order.append(np.arange(hist, dtype=float))
    return np.concatenate(vals), np.concatenate(groups), np.concatenate(order)


def _best_of(fn, *args, n=5, **kw):
    fn(*args, **kw)  # warm
    best = float("inf")
    for _ in range(n):
        t = time.perf_counter()
        fn(*args, **kw)
        best = min(best, time.perf_counter() - t)
    return best


def bench_backends():
    print("\n=== KDE backend A/B (serial njit vs prange), n_grid=64, hist=20 ===")
    print(f"{'n_groups':>10} {'serial_ms':>12} {'parallel_ms':>14} {'speedup':>9}")
    for n_ent in (128, 512, 2000, 10000):
        vals, groups, order = _make_panel(n_ent, 20)
        v_sorted, _, starts, ends, _ = _sort_into_groups(vals, groups, order)
        args = (v_sorted, starts, ends, 0, 1.0, -1.0, 64, 0.0)
        s = _best_of(_weighted_kde_mode_and_peak, *args)
        p = _best_of(_weighted_kde_mode_and_peak_parallel, *args)
        print(f"{n_ent:>10} {s * 1e3:>12.3f} {p * 1e3:>14.3f} {s / p:>8.2f}x")


def profile_full():
    vals, groups, order = _make_panel(10000, 20)
    # warm
    per_group_recency_weighted_mean(vals, groups, order=order, scheme="exp", param=0.6)
    per_group_recency_weighted_mode(vals, groups, order=order)
    per_group_behavioral_stability(vals, groups, order=order)

    pr = cProfile.Profile()
    pr.enable()
    for _ in range(3):
        per_group_recency_weighted_mean(vals, groups, order=order, scheme="exp", param=0.6)
        per_group_recency_weighted_mode(vals, groups, order=order)
        per_group_behavioral_stability(vals, groups, order=order)
    pr.disable()
    print("\n=== cProfile (10k entities x hist 20, x3) ===")
    pstats.Stats(pr).sort_stats("cumulative").print_stats(12)


if __name__ == "__main__":
    bench_backends()
    profile_full()
