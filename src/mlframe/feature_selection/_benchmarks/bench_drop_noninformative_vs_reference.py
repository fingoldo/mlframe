"""cProfile harness for ``feature_selection.drop_noninformative_vs_reference.drop_noninformative_vs_reference``.

Run: ``python -m mlframe.feature_selection._benchmarks.bench_drop_noninformative_vs_reference``
"""
from __future__ import annotations

import cProfile
import pstats
import time
from io import StringIO

import numpy as np
import pandas as pd

from mlframe.feature_selection.drop_noninformative_vs_reference import drop_noninformative_vs_reference


def _make_dataset(n: int, n_cols: int, seed: int):
    rng = np.random.default_rng(seed)
    is_treated = rng.integers(0, 2, n).astype(bool)
    df = pd.DataFrame({f"c{i}": rng.normal(size=n) for i in range(n_cols)})
    return df, ~is_treated


def _run(n: int, n_cols: int) -> None:
    df, reference_mask = _make_dataset(n, n_cols, seed=0)
    drop_noninformative_vs_reference(df, reference_mask, alpha=0.1)


if __name__ == "__main__":
    for n, n_cols in [(5000, 50), (50000, 50), (50000, 500)]:
        t0 = time.perf_counter()
        _run(n, n_cols)
        wall = time.perf_counter() - t0
        print(f"n={n:>6} n_cols={n_cols:>4} -> {wall * 1000:9.2f} ms")

    profiler = cProfile.Profile()
    profiler.enable()
    _run(50000, 500)
    profiler.disable()
    buf = StringIO()
    stats = pstats.Stats(profiler, stream=buf).sort_stats("cumulative")
    stats.print_stats(15)
    print(buf.getvalue())
