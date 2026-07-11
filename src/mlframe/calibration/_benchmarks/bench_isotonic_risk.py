"""cProfile harness for ``calibration.isotonic_risk.isotonic_overfit_risk``.

Run: ``python -m mlframe.calibration._benchmarks.bench_isotonic_risk``
"""
from __future__ import annotations

import cProfile
import pstats
import time
from io import StringIO

import numpy as np

from mlframe.calibration.isotonic_risk import isotonic_overfit_risk


def _run(n: int, n_calls: int) -> None:
    rng = np.random.default_rng(0)
    p = rng.uniform(0, 1, size=n)
    y = (rng.random(n) < p).astype(np.float64)
    for _ in range(n_calls):
        isotonic_overfit_risk(p, y)


def _run_remediate(n_dense: int, n_sparse: int, n_calls: int) -> None:
    # Same flagged/sparse-segment shape as the biz_value test so the profile reflects the actual
    # remediation path (Platt fit + local-density blended predict), not just the diagnostic count.
    rng = np.random.default_rng(1)
    p_lo = rng.uniform(0.02, 0.18, size=n_dense)
    y_lo = (rng.random(n_dense) < p_lo).astype(np.float64)
    p_hi = rng.uniform(0.82, 0.98, size=n_dense)
    y_hi = (rng.random(n_dense) < p_hi).astype(np.float64)
    p_mid = rng.uniform(0.4, 0.6, size=n_sparse)
    true_mid = np.clip(p_mid + 0.35 * rng.standard_normal(n_sparse), 0.0, 1.0)
    y_mid = (rng.random(n_sparse) < true_mid).astype(np.float64)

    p = np.concatenate([p_lo, p_mid, p_hi])
    y = np.concatenate([y_lo, y_mid, y_hi])
    query = rng.uniform(0.0, 1.0, size=n_dense)

    for _ in range(n_calls):
        result = isotonic_overfit_risk(p, y, remediate=True)
        if result["predict"] is not None:
            result["predict"](query)


if __name__ == "__main__":
    for n, n_calls in [(1_000, 50), (100_000, 10), (1_000_000, 3)]:
        t0 = time.perf_counter()
        _run(n, n_calls)
        wall = time.perf_counter() - t0
        print(f"n={n:>9,} -> {wall * 1000:9.2f} ms total, {wall / n_calls * 1000:8.3f} ms/call")

    for n_dense, n_sparse, n_calls in [(900, 25, 50), (45_000, 500, 10), (450_000, 5_000, 3)]:
        t0 = time.perf_counter()
        _run_remediate(n_dense, n_sparse, n_calls)
        wall = time.perf_counter() - t0
        n_total = 2 * n_dense + n_sparse
        print(
            f"remediate n={n_total:>9,} -> {wall * 1000:9.2f} ms total, {wall / n_calls * 1000:8.3f} ms/call"
        )

    profiler = cProfile.Profile()
    profiler.enable()
    _run(1_000_000, 5)
    profiler.disable()
    buf = StringIO()
    stats = pstats.Stats(profiler, stream=buf).sort_stats("cumulative")
    stats.print_stats(15)
    print(buf.getvalue())

    profiler = cProfile.Profile()
    profiler.enable()
    _run_remediate(450_000, 5_000, 5)
    profiler.disable()
    buf = StringIO()
    stats = pstats.Stats(profiler, stream=buf).sort_stats("cumulative")
    stats.print_stats(15)
    print(buf.getvalue())
