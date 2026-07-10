"""cProfile harness for ``evaluation.noise_band``.

Run: ``python -m mlframe.evaluation._benchmarks.bench_noise_band``

The function is O(n_folds) with n_folds typically 3-10 in real CV loops, called potentially thousands of
times inside a greedy/RFECV selection loop. Profiled at a representative call count to confirm the per-call
cost is negligible relative to the model refit it gates.
"""
from __future__ import annotations

import cProfile
import pstats
import time
from io import StringIO

import numpy as np

from mlframe.evaluation.noise_band import cv_score_equivalence_band, is_within_noise_band


def _run(n_calls: int = 20_000, n_folds: int = 5) -> None:
    rng = np.random.default_rng(0)
    fold_sets = [0.85 + 0.01 * rng.standard_normal(n_folds) for _ in range(n_calls)]
    for folds in fold_sets:
        cv_score_equivalence_band(folds)
        is_within_noise_band(0.851, 0.852, folds)


if __name__ == "__main__":
    n_calls = 20_000
    t0 = time.perf_counter()
    _run(n_calls=n_calls)
    wall = time.perf_counter() - t0
    print(f"wall for {n_calls} paired calls: {wall * 1000:.2f} ms ({wall / n_calls * 1e6:.2f} us/call pair)")

    profiler = cProfile.Profile()
    profiler.enable()
    _run(n_calls=n_calls)
    profiler.disable()
    buf = StringIO()
    stats = pstats.Stats(profiler, stream=buf).sort_stats("cumulative")
    stats.print_stats(20)
    print(buf.getvalue())
