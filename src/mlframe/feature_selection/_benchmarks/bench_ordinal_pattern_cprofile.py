"""cProfile harness for ``filters._ordinal_pattern_fe.ordinal_pattern_ids`` (mrmr_audit_2026-07-20
fe_expansion.md "Row-wise ordinal-pattern (Bandt-Pompe permutation) encoding").

Run: ``python -m mlframe.feature_selection._benchmarks.bench_ordinal_pattern_cprofile``

Cost is one ``np.argsort(axis=1)`` (cheap, K is small: 3-5) plus a Python-level per-row dict lookup
mapping each row's sort-order tuple to its precomputed lexicographic rank -- expect the per-row loop
to dominate at large n, since it is not yet vectorized (a future optimization: a single combinatorial
number-system encode of the K columns' argsort output would replace the per-row dict lookup with one
vectorized arithmetic expression).
"""

from __future__ import annotations

import cProfile
import pstats
import time
from io import StringIO

import numpy as np

from mlframe.feature_selection.filters._ordinal_pattern_fe import ordinal_pattern_ids


def _make_data(n_rows: int, k: int, seed: int):
    rng = np.random.default_rng(seed)
    return rng.standard_normal((n_rows, k))


def _run(n_rows: int, k: int) -> None:
    X = _make_data(n_rows, k, seed=0)
    ordinal_pattern_ids(X)


if __name__ == "__main__":
    for n_rows, k in [(2_000, 3), (20_000, 3), (100_000, 4)]:
        t0 = time.perf_counter()
        _run(n_rows, k)
        wall = time.perf_counter() - t0
        print(f"n_rows={n_rows:>7,} k={k} -> {wall * 1000:9.2f} ms")

    profiler = cProfile.Profile()
    profiler.enable()
    _run(100_000, 3)
    profiler.disable()
    stream = StringIO()
    stats = pstats.Stats(profiler, stream=stream)
    stats.sort_stats("cumulative")
    stats.print_stats(20)
    print(stream.getvalue())
