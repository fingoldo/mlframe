"""cProfile harness for ``ordered_target_encode``'s ``noise_std`` parameter.

Run: ``python -m mlframe.training.feature_handling._benchmarks.bench_ordered_target_encoder_noise``
"""
from __future__ import annotations

import cProfile
import pstats
import time
from io import StringIO

import numpy as np

from mlframe.training.feature_handling.ordered_target_encoder import ordered_target_encode


def _run(n: int, n_categories: int, n_calls: int) -> None:
    rng = np.random.default_rng(0)
    cats = rng.integers(0, n_categories, n)
    order = np.arange(n)
    y = rng.normal(size=n)
    for _ in range(n_calls):
        ordered_target_encode(cats, y, order=order, smoothing=1.0, noise_std=0.5, random_state=0)


if __name__ == "__main__":
    for n, n_categories, n_calls in [(2000, 500, 20), (200000, 5000, 20), (200000, 50000, 20)]:
        t0 = time.perf_counter()
        _run(n, n_categories, n_calls)
        wall = time.perf_counter() - t0
        print(f"n={n:>7} n_categories={n_categories:>6} n_calls={n_calls:>4} -> {wall * 1000:9.2f} ms")

    profiler = cProfile.Profile()
    profiler.enable()
    _run(200000, 50000, 20)
    profiler.disable()
    buf = StringIO()
    stats = pstats.Stats(profiler, stream=buf).sort_stats("cumulative")
    stats.print_stats(15)
    print(buf.getvalue())
