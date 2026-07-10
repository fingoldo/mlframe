"""cProfile harness for ``training.composite.detect_calendar_anomalies``.

Run: ``python -m mlframe.training.composite._benchmarks.bench_calendar_anomaly``
"""
from __future__ import annotations

import cProfile
import pstats
import time
from io import StringIO

import numpy as np

from mlframe.training.composite.calendar_anomaly import detect_calendar_anomalies


def _run(n_days: int, n_calls: int) -> None:
    rng = np.random.default_rng(0)
    y = 100.0 + rng.normal(0, 5, n_days)
    spikes = rng.choice(n_days, size=max(1, n_days // 200), replace=False)
    y[spikes] *= 50
    for _ in range(n_calls):
        detect_calendar_anomalies(y, window=14)


if __name__ == "__main__":
    for n_days, n_calls in [(3_650, 50), (1_000_000, 5)]:
        t0 = time.perf_counter()
        _run(n_days, n_calls)
        wall = time.perf_counter() - t0
        print(f"n_days={n_days:>9,} n_calls={n_calls:>3} -> {wall * 1000:9.2f} ms total, {wall / n_calls * 1000:9.3f} ms/call")

    profiler = cProfile.Profile()
    profiler.enable()
    _run(1_000_000, 5)
    profiler.disable()
    buf = StringIO()
    stats = pstats.Stats(profiler, stream=buf).sort_stats("cumulative")
    stats.print_stats(15)
    print(buf.getvalue())
