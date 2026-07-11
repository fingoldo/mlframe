"""cProfile harness for ``calibration.group_zero_sum_constraint.apply_group_zero_sum_constraint_multi``.

Run: ``python -m mlframe.calibration._benchmarks.bench_group_zero_sum_constraint_multi``
"""
from __future__ import annotations

import cProfile
import pstats
import time
from io import StringIO

import numpy as np

from mlframe.calibration.group_zero_sum_constraint import apply_group_zero_sum_constraint_multi


def _run(n_a: int, n_b: int, n_calls: int) -> None:
    rng = np.random.default_rng(0)
    n = n_a * n_b
    group_a = np.repeat(np.arange(n_a), n_b)
    group_b = np.tile(np.arange(n_b), n_a)
    pred = rng.normal(size=n)
    weights = rng.uniform(0.5, 1.5, size=n)
    for _ in range(n_calls):
        apply_group_zero_sum_constraint_multi(pred, groups=[group_a, group_b], weights=weights)


if __name__ == "__main__":
    for n_a, n_b, n_calls in [(2000, 20, 50), (50000, 20, 10), (50000, 50, 10)]:
        t0 = time.perf_counter()
        _run(n_a, n_b, n_calls)
        wall = time.perf_counter() - t0
        print(f"n_a={n_a:>7} n_b={n_b:>3} n_calls={n_calls:>4} -> {wall * 1000:9.2f} ms")

    profiler = cProfile.Profile()
    profiler.enable()
    _run(50000, 50, 10)
    profiler.disable()
    buf = StringIO()
    stats = pstats.Stats(profiler, stream=buf).sort_stats("cumulative")
    stats.print_stats(20)
    print(buf.getvalue())
