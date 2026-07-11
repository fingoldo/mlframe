"""cProfile harness for ``inference.apply_group_zero_sum_constraint``.

Run: ``python -m mlframe.inference._benchmarks.bench_group_zero_sum_constraint``
"""
from __future__ import annotations

import cProfile
import pstats
import time
from io import StringIO

import numpy as np

from mlframe.inference.group_zero_sum_constraint import apply_group_zero_sum_constraint


def _run(n_groups: int, members_per_group: int, n_calls: int) -> None:
    rng = np.random.default_rng(0)
    n = n_groups * members_per_group
    group_ids = np.repeat(np.arange(n_groups), members_per_group)
    preds = rng.normal(0, 1, n)
    weights = rng.uniform(0.5, 2.0, n)
    for _ in range(n_calls):
        apply_group_zero_sum_constraint(preds, group_ids, weights=weights)


def _run_multi(n_groups: int, members_per_group: int, n_calls: int, preserve_rank_order: bool) -> None:
    rng = np.random.default_rng(0)
    n = n_groups * members_per_group
    group_ids = np.repeat(np.arange(n_groups), members_per_group)
    preds = rng.normal(0, 1, n)
    weights = rng.uniform(0.5, 2.0, n)
    core_pattern = np.zeros(members_per_group)
    core_pattern[: max(1, members_per_group // 3)] = 1.0
    core_coefs = np.tile(core_pattern, n_groups)
    for _ in range(n_calls):
        apply_group_zero_sum_constraint(
            preds,
            group_ids,
            weights=weights,
            extra_constraint_coefs=[core_coefs],
            extra_constraint_targets=[0.4],
            preserve_rank_order=preserve_rank_order,
        )


if __name__ == "__main__":
    for n_groups, members_per_group, n_calls in [(2_000, 20, 50), (50_000, 20, 5)]:
        t0 = time.perf_counter()
        _run(n_groups, members_per_group, n_calls)
        wall = time.perf_counter() - t0
        print(f"single-constraint n_groups={n_groups:>7,} members/group={members_per_group:>3} n_calls={n_calls:>3} -> {wall * 1000:9.2f} ms total, {wall / n_calls * 1000:9.3f} ms/call")

    for n_groups, members_per_group, n_calls, preserve in [(2_000, 20, 50, False), (2_000, 20, 50, True), (50_000, 20, 5, False)]:
        t0 = time.perf_counter()
        _run_multi(n_groups, members_per_group, n_calls, preserve)
        wall = time.perf_counter() - t0
        print(
            f"multi-constraint(preserve_rank_order={preserve}) n_groups={n_groups:>7,} members/group={members_per_group:>3} "
            f"n_calls={n_calls:>3} -> {wall * 1000:9.2f} ms total, {wall / n_calls * 1000:9.3f} ms/call"
        )

    profiler = cProfile.Profile()
    profiler.enable()
    _run(2_000, 20, 50)
    profiler.disable()
    buf = StringIO()
    stats = pstats.Stats(profiler, stream=buf).sort_stats("cumulative")
    stats.print_stats(15)
    print("--- single-constraint path ---")
    print(buf.getvalue())

    profiler_multi = cProfile.Profile()
    profiler_multi.enable()
    _run_multi(2_000, 20, 50, preserve_rank_order=False)
    profiler_multi.disable()
    buf_multi = StringIO()
    stats_multi = pstats.Stats(profiler_multi, stream=buf_multi).sort_stats("cumulative")
    stats_multi.print_stats(15)
    print("--- multi-constraint path (no rank preservation) ---")
    print(buf_multi.getvalue())

    profiler_pres = cProfile.Profile()
    profiler_pres.enable()
    _run_multi(2_000, 20, 50, preserve_rank_order=True)
    profiler_pres.disable()
    buf_pres = StringIO()
    stats_pres = pstats.Stats(profiler_pres, stream=buf_pres).sort_stats("cumulative")
    stats_pres.print_stats(15)
    print("--- multi-constraint path (preserve_rank_order=True) ---")
    print(buf_pres.getvalue())
