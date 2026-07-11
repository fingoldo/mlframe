"""cProfile harness for ``calibration.smoothed_override.apply_smoothed_override``.

Run: ``python -m mlframe.calibration._benchmarks.bench_smoothed_override``
"""
from __future__ import annotations

import cProfile
import pstats
import time
from io import StringIO

import numpy as np

from mlframe.calibration.smoothed_override import apply_smoothed_override


def _run(n: int, n_calls: int) -> None:
    rng = np.random.default_rng(0)
    prediction = rng.normal(size=n)
    known_label = rng.normal(size=n)
    mask = rng.random(n) < 0.3
    for _ in range(n_calls):
        apply_smoothed_override(prediction, known_label, mask, a=0.9)


if __name__ == "__main__":
    for n, n_calls in [(10000, 500), (1000000, 100), (1000000, 500)]:
        t0 = time.perf_counter()
        _run(n, n_calls)
        wall = time.perf_counter() - t0
        print(f"n={n:>8} n_calls={n_calls:>5} -> {wall * 1000:9.2f} ms")

    profiler = cProfile.Profile()
    profiler.enable()
    _run(1000000, 500)
    profiler.disable()
    buf = StringIO()
    stats = pstats.Stats(profiler, stream=buf).sort_stats("cumulative")
    stats.print_stats(15)
    print(buf.getvalue())
