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


def _run_auto_detect(n: int, n_calls: int) -> None:
    # Same shapes as ``_run`` but exercises the opt-in auto-detect probe (candidate list withholds the
    # multiplicative alternative on purpose, and target/base are a genuinely multiplicative pair so the
    # ratio-stationarity check has real work to do rather than short-circuiting on a sign mismatch).
    rng = np.random.default_rng(1)
    multiplier = rng.uniform(0.7, 1.3, size=n)
    base = rng.uniform(1.0, 5.0, size=n)
    y = base * multiplier * np.exp(rng.normal(scale=0.05, size=n))
    candidates = ["diff", "linear_residual"]
    for _ in range(n_calls):
        recommend_transform_candidates(y, base, candidates, auto_detect=True)


if __name__ == "__main__":
    for n, n_calls in [(1000, 1000), (100000, 1000), (100000, 10000)]:
        t0 = time.perf_counter()
        _run(n, n_calls)
        wall = time.perf_counter() - t0
        print(f"n={n:>7} n_calls={n_calls:>6} -> {wall * 1000:9.2f} ms")

        t0 = time.perf_counter()
        _run_auto_detect(n, n_calls)
        wall_auto = time.perf_counter() - t0
        print(f"n={n:>7} n_calls={n_calls:>6} (auto_detect) -> {wall_auto * 1000:9.2f} ms")

    profiler = cProfile.Profile()
    profiler.enable()
    _run(100000, 10000)
    _run_auto_detect(100000, 10000)
    profiler.disable()
    buf = StringIO()
    stats = pstats.Stats(profiler, stream=buf).sort_stats("cumulative")
    stats.print_stats(15)
    print(buf.getvalue())
