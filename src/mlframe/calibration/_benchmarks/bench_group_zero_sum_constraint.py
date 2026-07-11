"""cProfile harness for ``calibration.group_zero_sum_constraint.apply_group_zero_sum_constraint``.

Run: ``python -m mlframe.calibration._benchmarks.bench_group_zero_sum_constraint``
"""
from __future__ import annotations

import cProfile
import pstats
import time
from io import StringIO

import numpy as np

from mlframe.calibration.group_zero_sum_constraint import apply_group_zero_sum_constraint


def _run(n_groups: int, group_size: int, n_calls: int) -> None:
    rng = np.random.default_rng(0)
    n = n_groups * group_size
    group = np.repeat(np.arange(n_groups), group_size)
    pred = rng.normal(size=n)
    weights = rng.uniform(0.5, 1.5, size=n)
    for _ in range(n_calls):
        apply_group_zero_sum_constraint(pred, group, weights=weights)


if __name__ == "__main__":
    for n_groups, group_size, n_calls in [(2000, 20, 50), (50000, 20, 50), (50000, 50, 50)]:
        t0 = time.perf_counter()
        _run(n_groups, group_size, n_calls)
        wall = time.perf_counter() - t0
        print(f"n_groups={n_groups:>7} group_size={group_size:>3} n_calls={n_calls:>4} -> {wall * 1000:9.2f} ms")

    profiler = cProfile.Profile()
    profiler.enable()
    _run(50000, 50, 50)
    profiler.disable()
    buf = StringIO()
    stats = pstats.Stats(profiler, stream=buf).sort_stats("cumulative")
    stats.print_stats(15)
    print(buf.getvalue())
