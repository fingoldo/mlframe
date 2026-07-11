"""cProfile harness for ``calibration.asymmetric_rescale.fit_asymmetric_rescale``.

Run: ``python -m mlframe.calibration._benchmarks.bench_asymmetric_rescale``
"""
from __future__ import annotations

import cProfile
import pstats
import time
from io import StringIO

import numpy as np

from mlframe.calibration.asymmetric_rescale import cross_validate_asymmetric_rescale, fit_asymmetric_rescale


def _neg_mse(y_true, y_pred):
    return -float(np.mean((y_true - y_pred) ** 2))


def _run(n: int, n_factors: int) -> None:
    rng = np.random.default_rng(0)
    y_true = rng.normal(size=n)
    y_pred = y_true * 0.8 + rng.normal(scale=0.1, size=n)
    fit_asymmetric_rescale(y_true, y_pred, _neg_mse, factor_range=(1.0, 2.0), n_factors=n_factors)


def _run_cv(n: int, n_factors: int, n_folds: int) -> None:
    rng = np.random.default_rng(0)
    y_true = rng.normal(size=n)
    y_pred = y_true * 0.8 + rng.normal(scale=0.1, size=n)
    cross_validate_asymmetric_rescale(y_true, y_pred, _neg_mse, n_folds=n_folds, factor_range=(1.0, 2.0), n_factors=n_factors)


if __name__ == "__main__":
    for n, n_factors in [(10000, 50), (1000000, 50), (1000000, 200)]:
        t0 = time.perf_counter()
        _run(n, n_factors)
        wall = time.perf_counter() - t0
        print(f"n={n:>8} n_factors={n_factors:>4} -> {wall * 1000:9.2f} ms")

    for n, n_factors, n_folds in [(10000, 50, 5), (1000000, 50, 5), (1000000, 50, 10)]:
        t0 = time.perf_counter()
        _run_cv(n, n_factors, n_folds)
        wall = time.perf_counter() - t0
        print(f"[cv] n={n:>8} n_factors={n_factors:>4} n_folds={n_folds:>3} -> {wall * 1000:9.2f} ms")

    profiler = cProfile.Profile()
    profiler.enable()
    _run(1000000, 200)
    profiler.disable()
    buf = StringIO()
    stats = pstats.Stats(profiler, stream=buf).sort_stats("cumulative")
    stats.print_stats(15)
    print(buf.getvalue())

    profiler_cv = cProfile.Profile()
    profiler_cv.enable()
    _run_cv(1000000, 50, 5)
    profiler_cv.disable()
    buf_cv = StringIO()
    stats_cv = pstats.Stats(profiler_cv, stream=buf_cv).sort_stats("cumulative")
    stats_cv.print_stats(15)
    print(buf_cv.getvalue())
