"""cProfile harness for ``filters._lof_fe.lof_scores`` (mrmr_audit_2026-07-20 fe_expansion.md
"Local Outlier Factor / k-NN local density-ratio feature").

Run: ``python -m mlframe.feature_selection._benchmarks.bench_lof_cprofile``

Cost is O(n^2) brute-force pairwise distances (the matmul trick) plus an ``np.argpartition`` per
row for k-NN selection -- expect the (n, n) distance-matrix construction to dominate at large n;
per the module's own docstring, very large n (>~200k) would need chunking or an approximate GPU-kNN
library rather than the naive dense matmul form used here.
"""

from __future__ import annotations

import cProfile
import pstats
import time
from io import StringIO

import numpy as np

from mlframe.feature_selection.filters._lof_fe import lof_scores


def _make_data(n_rows: int, p: int, seed: int):
    rng = np.random.default_rng(seed)
    return rng.standard_normal((n_rows, p))


def _run(n_rows: int, p: int, k: int) -> None:
    X = _make_data(n_rows, p, seed=0)
    lof_scores(X, k=k)


if __name__ == "__main__":
    for n_rows, p, k in [(1_000, 5, 20), (5_000, 10, 20), (10_000, 10, 20)]:
        t0 = time.perf_counter()
        _run(n_rows, p, k)
        wall = time.perf_counter() - t0
        print(f"n_rows={n_rows:>7,} p={p:>3} k={k:>3} -> {wall * 1000:9.2f} ms")

    profiler = cProfile.Profile()
    profiler.enable()
    _run(10_000, 10, 20)
    profiler.disable()
    stream = StringIO()
    stats = pstats.Stats(profiler, stream=stream)
    stats.sort_stats("cumulative")
    stats.print_stats(20)
    print(stream.getvalue())
