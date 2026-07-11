"""cProfile harness for ``preprocessing.outlier_detector_zoo.make_outlier_detector``.

Run: ``python -m mlframe.preprocessing._benchmarks.bench_outlier_detector_zoo``
"""
from __future__ import annotations

import cProfile
import pstats
import time
from io import StringIO

import numpy as np

from mlframe.preprocessing.outlier_detector_zoo import make_outlier_detector


def _make_data(n_rows: int, n_cols: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.normal(size=(n_rows, n_cols))


def _run(n_rows: int, n_cols: int, method: str) -> None:
    X = _make_data(n_rows, n_cols, seed=0)
    detector = make_outlier_detector(method)
    detector.fit_predict(X)


if __name__ == "__main__":
    for method in ("isolation_forest", "lof"):
        for n_rows, n_cols in [(5000, 10), (50000, 10)]:
            t0 = time.perf_counter()
            _run(n_rows, n_cols, method)
            wall = time.perf_counter() - t0
            print(f"method={method:>16} n_rows={n_rows:>6} n_cols={n_cols:>3} -> {wall * 1000:9.2f} ms")

    profiler = cProfile.Profile()
    profiler.enable()
    _run(50000, 10, "isolation_forest")
    profiler.disable()
    buf = StringIO()
    stats = pstats.Stats(profiler, stream=buf).sort_stats("cumulative")
    stats.print_stats(15)
    print(buf.getvalue())
