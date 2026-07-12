"""cProfile harness for ``calibration.prediction_band_correction``.

Run: ``python -m mlframe.calibration._benchmarks.bench_prediction_band_correction``
"""
from __future__ import annotations

import cProfile
import pstats
import time
from io import StringIO

import numpy as np

from mlframe.calibration.prediction_band_correction import apply_prediction_band_correction, assess_prediction_band_stability, find_prediction_band_shift


def _run(n: int, n_calls: int) -> None:
    rng = np.random.default_rng(0)
    y_true = rng.integers(0, 2, n).astype(float)
    y_pred = np.clip(rng.normal(loc=0.5, scale=0.2, size=n), 0, 1)
    for _ in range(n_calls):
        factor = find_prediction_band_shift(y_true, y_pred, lo=0.4, hi=1.0)
        apply_prediction_band_correction(y_pred, lo=0.4, hi=1.0, factor=factor)


def _run_stability(n: int, n_calls: int, n_bootstrap: int) -> None:
    rng = np.random.default_rng(0)
    y_true = rng.integers(0, 2, n).astype(float)
    y_pred = np.clip(rng.normal(loc=0.5, scale=0.2, size=n), 0, 1)
    for _ in range(n_calls):
        assess_prediction_band_stability(y_true, y_pred, lo=0.4, hi=1.0, n_bootstrap=n_bootstrap, random_state=0)


if __name__ == "__main__":
    for n, n_calls in [(10000, 200), (1000000, 200), (1000000, 1000)]:
        t0 = time.perf_counter()
        _run(n, n_calls)
        wall = time.perf_counter() - t0
        print(f"n={n:>8} n_calls={n_calls:>5} -> {wall * 1000:9.2f} ms")

    for n, n_calls, n_bootstrap in [(10000, 20, 500), (1000000, 20, 500), (1000000, 5, 2000)]:
        t0 = time.perf_counter()
        _run_stability(n, n_calls, n_bootstrap)
        wall = time.perf_counter() - t0
        print(f"[stability] n={n:>8} n_calls={n_calls:>4} n_bootstrap={n_bootstrap:>5} -> {wall * 1000:9.2f} ms")

    profiler = cProfile.Profile()
    profiler.enable()
    _run(1000000, 1000)
    profiler.disable()
    buf = StringIO()
    stats = pstats.Stats(profiler, stream=buf).sort_stats("cumulative")
    stats.print_stats(10)
    print(buf.getvalue())

    profiler = cProfile.Profile()
    profiler.enable()
    _run_stability(1000000, 20, 500)
    profiler.disable()
    buf = StringIO()
    stats = pstats.Stats(profiler, stream=buf).sort_stats("cumulative")
    stats.print_stats(10)
    print("[stability]")
    print(buf.getvalue())
