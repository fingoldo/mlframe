"""cProfile harness for ``calibration.isotonic_risk.isotonic_overfit_risk``.

Run: ``python -m mlframe.calibration._benchmarks.bench_isotonic_risk``
"""
from __future__ import annotations

import cProfile
import pstats
import time
from io import StringIO

import numpy as np

from mlframe.calibration.isotonic_risk import isotonic_overfit_risk


def _run(n: int, n_calls: int) -> None:
    rng = np.random.default_rng(0)
    p = rng.uniform(0, 1, size=n)
    y = (rng.random(n) < p).astype(np.float64)
    for _ in range(n_calls):
        isotonic_overfit_risk(p, y)


if __name__ == "__main__":
    for n, n_calls in [(1_000, 50), (100_000, 10), (1_000_000, 3)]:
        t0 = time.perf_counter()
        _run(n, n_calls)
        wall = time.perf_counter() - t0
        print(f"n={n:>9,} -> {wall * 1000:9.2f} ms total, {wall / n_calls * 1000:8.3f} ms/call")

    profiler = cProfile.Profile()
    profiler.enable()
    _run(1_000_000, 5)
    profiler.disable()
    buf = StringIO()
    stats = pstats.Stats(profiler, stream=buf).sort_stats("cumulative")
    stats.print_stats(15)
    print(buf.getvalue())
