"""cProfile harness for ``filters._orthogonal_xi_fe.xi_correlation_batch`` (Layer 72, Chatterjee's
Xi rank correlation, mrmr_audit_2026-07-20 fe_expansion.md).

Run: ``python -m mlframe.feature_selection._benchmarks.bench_xi_correlation_cprofile``

Cost is O(n log n) per column (one argsort dominates); expect ``np.lexsort`` / ``np.argsort`` to be
the top cumtime entries, not a numba/cupy kernel -- this scorer is pure-numpy by design (see the
module docstring's GPU-residency note: HIGH potential via ``cp.argsort``, not yet ported).
"""

from __future__ import annotations

import cProfile
import pstats
import time
from io import StringIO

import numpy as np

from mlframe.feature_selection.filters._orthogonal_xi_fe import xi_correlation_batch


def _make_data(n_rows: int, n_cols: int, seed: int):
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n_rows, n_cols))
    y = np.sin(3.0 * X[:, 0]) + 0.1 * rng.standard_normal(n_rows)
    return X, y


def _run(n_rows: int, n_cols: int) -> None:
    X, y = _make_data(n_rows, n_cols, seed=0)
    xi_correlation_batch(X, y, random_state=0)


if __name__ == "__main__":
    for n_rows, n_cols in [(2_000, 20), (20_000, 50), (100_000, 100)]:
        t0 = time.perf_counter()
        _run(n_rows, n_cols)
        wall = time.perf_counter() - t0
        print(f"n_rows={n_rows:>7,} n_cols={n_cols:>4} -> {wall * 1000:9.2f} ms")

    profiler = cProfile.Profile()
    profiler.enable()
    _run(50_000, 60)
    profiler.disable()
    stream = StringIO()
    stats = pstats.Stats(profiler, stream=stream)
    stats.sort_stats("cumulative")
    stats.print_stats(25)
    print(stream.getvalue())
