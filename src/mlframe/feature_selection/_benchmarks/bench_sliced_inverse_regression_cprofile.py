"""cProfile harness for ``filters._sliced_inverse_regression_fe.sir_direction_features``
(mrmr_audit_2026-07-20 fe_expansion.md "Sliced Inverse Regression (SIR)").

Run: ``python -m mlframe.feature_selection._benchmarks.bench_sliced_inverse_regression_cprofile``

Cost is dominated by the per-slice mean bookkeeping (a Python loop over n_slices, small) plus a
p x p generalized eigenproblem (``scipy.linalg.eigh``) -- p is typically small (a handful of
candidate columns), so the eigensolve itself should be cheap; expect the slice loop / covariance
matmuls to dominate at large n.
"""

from __future__ import annotations

import cProfile
import pstats
import time
from io import StringIO

import numpy as np

from mlframe.feature_selection.filters._sliced_inverse_regression_fe import sir_direction_features


def _make_data(n_rows: int, p: int, seed: int):
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n_rows, p))
    w = rng.uniform(0.3, 0.6, p)
    y = (X @ w > 0).astype(float)
    return X, y


def _run(n_rows: int, p: int, n_slices: int) -> None:
    X, y = _make_data(n_rows, p, seed=0)
    sir_direction_features(X, y, n_slices=n_slices, n_directions=2)


if __name__ == "__main__":
    for n_rows, p, n_slices in [(2_000, 10, 10), (20_000, 20, 15), (100_000, 30, 20)]:
        t0 = time.perf_counter()
        _run(n_rows, p, n_slices)
        wall = time.perf_counter() - t0
        print(f"n_rows={n_rows:>7,} p={p:>3} n_slices={n_slices:>3} -> {wall * 1000:9.2f} ms")

    profiler = cProfile.Profile()
    profiler.enable()
    _run(100_000, 30, 20)
    profiler.disable()
    stream = StringIO()
    stats = pstats.Stats(profiler, stream=stream)
    stats.sort_stats("cumulative")
    stats.print_stats(20)
    print(stream.getvalue())
