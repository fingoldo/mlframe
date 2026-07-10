"""cProfile harness for ``training.OverlappingWalkForwardCV`` / ``training.cv_stability_check``.

Run: ``python -m mlframe.training._benchmarks.bench_overlapping_walk_forward_cv``
"""
from __future__ import annotations

import cProfile
import pstats
import time
from io import StringIO

import numpy as np

from mlframe.training._overlapping_walk_forward_cv import OverlappingWalkForwardCV, cv_stability_check


def _run_splitter(n_samples: int, n_repeats: int) -> None:
    y = np.arange(n_samples)
    splitter = OverlappingWalkForwardCV(window_length=200, step=20, gap=5, test_length=10)
    for _ in range(n_repeats):
        for train_idx, test_idx in splitter.split(y):
            float(np.mean(y[train_idx]))


def _run_stability(n_calls: int) -> None:
    rng = np.random.default_rng(0)
    hp_grid = np.linspace(0, 1, 30)
    for _ in range(n_calls):
        curves = [-((hp_grid - 0.6) ** 2) + rng.normal(0, 0.02, size=len(hp_grid)) for _ in range(8)]
        cv_stability_check(curves)


if __name__ == "__main__":
    t0 = time.perf_counter()
    _run_splitter(n_samples=5_000, n_repeats=50)
    wall = time.perf_counter() - t0
    print(f"splitter: 50 repeats over 5000 samples -> {wall * 1000:.2f} ms")

    t0 = time.perf_counter()
    _run_stability(n_calls=2_000)
    wall = time.perf_counter() - t0
    print(f"cv_stability_check: 2000 calls -> {wall * 1000:.2f} ms ({wall / 2000 * 1e6:.2f} us/call)")

    profiler = cProfile.Profile()
    profiler.enable()
    _run_splitter(n_samples=5_000, n_repeats=50)
    _run_stability(n_calls=2_000)
    profiler.disable()
    buf = StringIO()
    stats = pstats.Stats(profiler, stream=buf).sort_stats("cumulative")
    stats.print_stats(20)
    print(buf.getvalue())
