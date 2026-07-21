"""cProfile harness for ``filters._mahalanobis_density_fe.mahalanobis_density_feature``
(mrmr_audit_2026-07-20 fe_expansion.md "Multivariate Mahalanobis / Gaussian-copula joint density
anomaly score").

Run: ``python -m mlframe.feature_selection._benchmarks.bench_mahalanobis_density_cprofile``

Cost is dominated by ``sklearn.covariance.LedoitWolf`` (an O(n*p^2 + p^3) covariance estimate +
shrinkage) plus a batched ``np.einsum`` quadratic form -- p is small (a correlated cluster, tens
not thousands of columns) so the einsum should be cheap relative to the shrinkage fit.
"""

from __future__ import annotations

import cProfile
import pstats
import time
from io import StringIO

import numpy as np

from mlframe.feature_selection.filters._mahalanobis_density_fe import mahalanobis_density_feature


def _make_data(n_rows: int, p: int, seed: int):
    rng = np.random.default_rng(seed)
    cov = np.eye(p) + 0.3 * (np.ones((p, p)) - np.eye(p))
    return rng.multivariate_normal(np.zeros(p), cov, size=n_rows)


def _run(n_rows: int, p: int) -> None:
    X = _make_data(n_rows, p, seed=0)
    mahalanobis_density_feature(X)


if __name__ == "__main__":
    for n_rows, p in [(2_000, 10), (20_000, 20), (100_000, 25)]:
        t0 = time.perf_counter()
        _run(n_rows, p)
        wall = time.perf_counter() - t0
        print(f"n_rows={n_rows:>7,} p={p:>3} -> {wall * 1000:9.2f} ms")

    profiler = cProfile.Profile()
    profiler.enable()
    _run(100_000, 25)
    profiler.disable()
    stream = StringIO()
    stats = pstats.Stats(profiler, stream=stream)
    stats.sort_stats("cumulative")
    stats.print_stats(20)
    print(stream.getvalue())
