"""cProfile harness for ``preprocessing.apply_outlier_policy``.

Run: ``python -m mlframe.preprocessing._benchmarks.bench_outlier_policy``
"""
from __future__ import annotations

import cProfile
import pstats
import time
from io import StringIO

import numpy as np
import pandas as pd

from mlframe.preprocessing.outlier_policy import apply_outlier_policy


class _FakeLGBM:
    pass


def _run(n_rows: int, n_features: int) -> None:
    rng = np.random.default_rng(0)
    df = pd.DataFrame({f"f{i}": rng.normal(0, 1, n_rows) for i in range(n_features)})
    apply_outlier_policy(df, _FakeLGBM())
    apply_outlier_policy(df, object())


if __name__ == "__main__":
    for n_rows, n_features in [(20_000, 20), (1_000_000, 50)]:
        t0 = time.perf_counter()
        _run(n_rows, n_features)
        wall = time.perf_counter() - t0
        print(f"n_rows={n_rows:>9,} n_features={n_features:>3} -> {wall * 1000:9.2f} ms")

    profiler = cProfile.Profile()
    profiler.enable()
    _run(1_000_000, 50)
    profiler.disable()
    buf = StringIO()
    stats = pstats.Stats(profiler, stream=buf).sort_stats("cumulative")
    stats.print_stats(15)
    print(buf.getvalue())
