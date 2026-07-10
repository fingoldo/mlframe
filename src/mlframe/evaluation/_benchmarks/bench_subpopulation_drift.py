"""cProfile harness for ``evaluation.subpopulation_drift.subpopulation_ratio_drift_check``.

Run: ``python -m mlframe.evaluation._benchmarks.bench_subpopulation_drift``
"""
from __future__ import annotations

import cProfile
import pstats
import time
from io import StringIO

import numpy as np
import pandas as pd

from mlframe.evaluation.subpopulation_drift import subpopulation_ratio_drift_check


def _run(n: int, n_cats: int, n_calls: int) -> None:
    rng = np.random.default_rng(0)
    train_df = pd.DataFrame({"col": rng.integers(0, n_cats, size=n)})
    test_df = pd.DataFrame({"col": rng.integers(0, n_cats, size=n)})
    for _ in range(n_calls):
        subpopulation_ratio_drift_check(train_df, test_df, subgroup_col="col")


if __name__ == "__main__":
    for n, n_cats, n_calls in [(10_000, 5, 200), (1_000_000, 20, 20)]:
        t0 = time.perf_counter()
        _run(n, n_cats, n_calls)
        wall = time.perf_counter() - t0
        print(f"n={n:>9,} cats={n_cats:>3} -> {wall * 1000:9.2f} ms total, {wall / n_calls * 1000:8.3f} ms/call")

    profiler = cProfile.Profile()
    profiler.enable()
    _run(1_000_000, 20, 30)
    profiler.disable()
    buf = StringIO()
    stats = pstats.Stats(profiler, stream=buf).sort_stats("cumulative")
    stats.print_stats(15)
    print(buf.getvalue())
