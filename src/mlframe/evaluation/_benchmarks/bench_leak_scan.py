"""cProfile harness for ``evaluation.leak_scan.scan_temporal_leak``.

Run: ``python -m mlframe.evaluation._benchmarks.bench_leak_scan``
"""
from __future__ import annotations

import cProfile
import pstats
import time
from io import StringIO

import numpy as np
import pandas as pd

from mlframe.evaluation.leak_scan import scan_temporal_leak


def _run(n: int, n_cols: int, n_calls: int) -> None:
    rng = np.random.default_rng(0)
    split_labels = rng.integers(0, 10, size=n)
    data = {f"col_{i}": rng.standard_normal(n) for i in range(n_cols)}
    data["leak_col"] = split_labels.astype(np.float64) * 100.0 + rng.standard_normal(n)
    X = pd.DataFrame(data)
    for _ in range(n_calls):
        scan_temporal_leak(X, split_labels)


if __name__ == "__main__":
    for n, n_cols, n_calls in [(1_000, 50, 50), (100_000, 200, 10), (1_000_000, 50, 3)]:
        t0 = time.perf_counter()
        _run(n, n_cols, n_calls)
        wall = time.perf_counter() - t0
        print(f"n={n:>9,} cols={n_cols:>4} -> {wall * 1000:8.2f} ms total, {wall / n_calls * 1e6:9.2f} us/call")

    n, n_cols, n_calls = 100_000, 200, 20
    profiler = cProfile.Profile()
    profiler.enable()
    _run(n, n_cols, n_calls)
    profiler.disable()
    buf = StringIO()
    stats = pstats.Stats(profiler, stream=buf).sort_stats("cumulative")
    stats.print_stats(15)
    print(buf.getvalue())
