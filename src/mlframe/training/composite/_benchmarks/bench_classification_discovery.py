"""cProfile harness for ``training.composite.CompositeClassificationDiscovery``.

Run: ``python -m mlframe.training.composite._benchmarks.bench_classification_discovery``
"""
from __future__ import annotations

import cProfile
import pstats
import time
from io import StringIO

import numpy as np

from mlframe.training.composite import CompositeClassificationDiscovery


def _make(n_rows: int, n_cols: int, seed: int = 0):
    rng = np.random.default_rng(seed)
    X = rng.normal(0, 1, (n_rows, n_cols))
    logit = 2.5 * X[:, 0] + 0.5 * X[:, 1] * X[:, 2]
    y = (rng.uniform(0, 1, n_rows) < 1.0 / (1.0 + np.exp(-logit))).astype(int)
    return X, y


def _run(n_rows: int, n_cols: int) -> None:
    X, y = _make(n_rows, n_cols)
    CompositeClassificationDiscovery(random_state=0).fit(X, y)


if __name__ == "__main__":
    for n_rows, n_cols in [(3_000, 10), (10_000, 30), (30_000, 60)]:
        t0 = time.perf_counter()
        _run(n_rows, n_cols)
        wall = time.perf_counter() - t0
        print(f"n_rows={n_rows:>7,} n_cols={n_cols:>3} -> {wall:8.2f} s")

    profiler = cProfile.Profile()
    profiler.enable()
    _run(10_000, 30)
    profiler.disable()
    buf = StringIO()
    pstats.Stats(profiler, stream=buf).sort_stats("cumulative").print_stats(25)
    print(buf.getvalue())
