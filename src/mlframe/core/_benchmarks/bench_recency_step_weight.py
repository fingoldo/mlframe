"""cProfile harness for ``core.recency_step_weight.recency_step_weight``.

Run: ``python -m mlframe.core._benchmarks.bench_recency_step_weight``
"""
from __future__ import annotations

import cProfile
import pstats
import time
from io import StringIO

import numpy as np

from mlframe.core.recency_step_weight import recency_step_weight


def _run(n: int, n_calls: int) -> None:
    dates = np.arange(n)
    for _ in range(n_calls):
        recency_step_weight(dates, cutoff_date=n - 45)


if __name__ == "__main__":
    for n, n_calls in [(10000, 2000), (1000000, 2000), (10000000, 500)]:
        t0 = time.perf_counter()
        _run(n, n_calls)
        wall = time.perf_counter() - t0
        print(f"n={n:>9} n_calls={n_calls:>5} -> {wall * 1000:9.2f} ms")

    profiler = cProfile.Profile()
    profiler.enable()
    _run(10000000, 500)
    profiler.disable()
    buf = StringIO()
    stats = pstats.Stats(profiler, stream=buf).sort_stats("cumulative")
    stats.print_stats(10)
    print(buf.getvalue())
