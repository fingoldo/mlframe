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


if __name__ == "__main__":
    for n_groups, members_per_group, n_calls in [(2_000, 20, 50), (50_000, 20, 5)]:
        t0 = time.perf_counter()
        _run(n_groups, members_per_group, n_calls)
        wall = time.perf_counter() - t0
        print(f"n_groups={n_groups:>7,} members/group={members_per_group:>3} n_calls={n_calls:>3} -> {wall * 1000:9.2f} ms total, {wall / n_calls * 1000:9.3f} ms/call")

    profiler = cProfile.Profile()
    profiler.enable()
    _run(2_000, 20, 50)
    profiler.disable()
    buf = StringIO()
    stats = pstats.Stats(profiler, stream=buf).sort_stats("cumulative")
    stats.print_stats(15)
    print(buf.getvalue())
