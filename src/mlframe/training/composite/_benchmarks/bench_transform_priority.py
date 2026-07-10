"""cProfile harness for ``training.composite.transform_priority.recommend_transform_candidates``.

Run: ``python -m mlframe.training.composite._benchmarks.bench_transform_priority``
"""
from __future__ import annotations

import cProfile
import pstats
import time
from io import StringIO

import numpy as np

from mlframe.training.composite.transform_priority import recommend_transform_candidates


def _run(n: int, n_calls: int) -> None:
    rng = np.random.default_rng(0)
    y = rng.uniform(1.0, 5.0, size=n)
    base = rng.uniform(1.0, 5.0, size=n)
    candidates = ["diff", "linear_residual", "ratio", "logratio"]
    for _ in range(n_calls):
        recommend_transform_candidates(y, base, candidates)


if __name__ == "__main__":
    for n, n_calls in [(1000, 1000), (100000, 1000), (100000, 10000)]:
        t0 = time.perf_counter()
        _run(n, n_calls)
        wall = time.perf_counter() - t0
        print(f"n={n:>7} n_calls={n_calls:>6} -> {wall * 1000:9.2f} ms")

    profiler = cProfile.Profile()
    profiler.enable()
    _run(100000, 10000)
    profiler.disable()
    buf = StringIO()
    stats = pstats.Stats(profiler, stream=buf).sort_stats("cumulative")
    stats.print_stats(15)
    print(buf.getvalue())
