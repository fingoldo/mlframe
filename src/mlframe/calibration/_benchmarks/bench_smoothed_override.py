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
from mlframe.calibration.smoothed_override_backtest import backtest_override


def _run(n: int, n_calls: int) -> None:
    rng = np.random.default_rng(0)
    prediction = rng.normal(size=n)
    known_label = rng.normal(size=n)
    mask = rng.random(n) < 0.3
    for _ in range(n_calls):
        apply_smoothed_override(prediction, known_label, mask, a=0.9)


def _run_backtest(n: int, n_calls: int) -> None:
    rng = np.random.default_rng(0)
    y_true = rng.normal(size=n)
    model_pred = y_true + rng.normal(scale=0.2, size=n)
    confidence = rng.random(n)
    override_pred = y_true + np.where(confidence >= 0.5, rng.normal(scale=0.02, size=n), rng.normal(scale=1.0, size=n))
    for _ in range(n_calls):
        backtest_override(y_true, model_pred, override_pred, confidence, a=0.9, n_buckets=5)


if __name__ == "__main__":
    for n, n_calls in [(10000, 500), (1000000, 100), (1000000, 500)]:
        t0 = time.perf_counter()
        _run(n, n_calls)
        wall = time.perf_counter() - t0
        print(f"apply_smoothed_override n={n:>8} n_calls={n_calls:>5} -> {wall * 1000:9.2f} ms")

    for n, n_calls in [(10000, 200), (1000000, 20), (1000000, 100)]:
        t0 = time.perf_counter()
        _run_backtest(n, n_calls)
        wall = time.perf_counter() - t0
        print(f"backtest_override      n={n:>8} n_calls={n_calls:>5} -> {wall * 1000:9.2f} ms")

    profiler = cProfile.Profile()
    profiler.enable()
    _run(1000000, 500)
    _run_backtest(1000000, 100)
    profiler.disable()
    buf = StringIO()
    stats = pstats.Stats(profiler, stream=buf).sort_stats("cumulative")
    stats.print_stats(15)
    print(buf.getvalue())
