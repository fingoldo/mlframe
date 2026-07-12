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


def _run_tiers(n: int, n_calls: int) -> None:
    dates = np.arange(n)
    tiers = [(n - 3000, 1.2), (n - 500, 1.8), (n - 45, 3.0)]
    for _ in range(n_calls):
        recency_step_weight(dates, cutoff_date=n - 45, tiers=tiers)


def _run_smooth(n: int, n_calls: int) -> None:
    dates = np.arange(n).astype(np.float64)
    for _ in range(n_calls):
        recency_step_weight(dates, cutoff_date=float(n - 45), smooth_window=500.0)


if __name__ == "__main__":
    for label, fn in [("step", _run), ("tiers", _run_tiers), ("smooth", _run_smooth)]:
        for n, n_calls in [(10000, 2000), (1000000, 2000), (10000000, 500)]:
            t0 = time.perf_counter()
            fn(n, n_calls)
            wall = time.perf_counter() - t0
            print(f"{label:>6} n={n:>9} n_calls={n_calls:>5} -> {wall * 1000:9.2f} ms")

    for label, fn in [("step", _run), ("tiers", _run_tiers), ("smooth", _run_smooth)]:
        profiler = cProfile.Profile()
        profiler.enable()
        fn(10000000, 500)
        profiler.disable()
        buf = StringIO()
        stats = pstats.Stats(profiler, stream=buf).sort_stats("cumulative")
        stats.print_stats(10)
        print(f"--- {label} ---")
        print(buf.getvalue())
