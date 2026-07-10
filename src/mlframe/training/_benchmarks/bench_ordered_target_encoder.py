"""cProfile harness for ``training.feature_handling.ordered_target_encode``.

Run: ``python -m mlframe.training._benchmarks.bench_ordered_target_encoder``
"""
from __future__ import annotations

import cProfile
import pstats
import time
from io import StringIO

import numpy as np

from mlframe.training.feature_handling.ordered_target_encoder import ordered_target_encode


def _run(n_rows: int, n_cats: int) -> None:
    rng = np.random.default_rng(0)
    cats = rng.integers(0, n_cats, n_rows)
    y = rng.integers(0, 2, n_rows).astype(np.float64)
    ordered_target_encode(cats, y, smoothing=1.0)


if __name__ == "__main__":
    for n_rows, n_cats in [(50_000, 500), (1_000_000, 5_000)]:
        t0 = time.perf_counter()
        _run(n_rows, n_cats)
        wall = time.perf_counter() - t0
        print(f"n_rows={n_rows:>9,} n_cats={n_cats:>6,} -> {wall * 1000:9.2f} ms")

    profiler = cProfile.Profile()
    profiler.enable()
    _run(1_000_000, 5_000)
    profiler.disable()
    buf = StringIO()
    stats = pstats.Stats(profiler, stream=buf).sort_stats("cumulative")
    stats.print_stats(15)
    print(buf.getvalue())
