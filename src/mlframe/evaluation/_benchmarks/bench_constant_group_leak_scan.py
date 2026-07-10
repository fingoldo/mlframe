"""cProfile harness for ``evaluation.constant_group_target_scan``.

Run: ``python -m mlframe.evaluation._benchmarks.bench_constant_group_leak_scan``
"""
from __future__ import annotations

import cProfile
import pstats
import time
from io import StringIO

import numpy as np
import pandas as pd

from mlframe.evaluation.constant_group_leak_scan import constant_group_target_scan


def _run(n_rows: int, n_cols: int, n_groups: int) -> None:
    rng = np.random.default_rng(0)
    cols = {f"col_{i}": rng.integers(0, n_groups, n_rows) for i in range(n_cols)}
    df = pd.DataFrame(cols)
    y = rng.integers(0, 2, n_rows).astype(float)
    constant_group_target_scan(df, y, candidate_cols=list(cols.keys()), min_group_size=20)


if __name__ == "__main__":
    for n_rows, n_cols, n_groups in [(20_000, 10, 100), (200_000, 20, 500)]:
        t0 = time.perf_counter()
        _run(n_rows, n_cols, n_groups)
        wall = time.perf_counter() - t0
        print(f"n_rows={n_rows:>9,} n_cols={n_cols:>3} n_groups={n_groups:>4} -> {wall * 1000:9.2f} ms")

    profiler = cProfile.Profile()
    profiler.enable()
    _run(20_000, 10, 100)
    profiler.disable()
    buf = StringIO()
    stats = pstats.Stats(profiler, stream=buf).sort_stats("cumulative")
    stats.print_stats(15)
    print(buf.getvalue())
