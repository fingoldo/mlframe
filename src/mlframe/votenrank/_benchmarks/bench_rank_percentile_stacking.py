"""cProfile harness for ``votenrank.rank_percentile_transform``.

Run: ``python -m mlframe.votenrank._benchmarks.bench_rank_percentile_stacking``
"""
from __future__ import annotations

import cProfile
import pstats
import time
from io import StringIO

import numpy as np

from mlframe.votenrank.rank_percentile_stacking import rank_percentile_transform


def _run(n_oof: int, n_test: int, n_calls: int, smoothing: float | None = None) -> None:
    rng = np.random.default_rng(0)
    oof = rng.normal(0, 1, n_oof)
    test_pred = rng.normal(0, 1, n_test)
    for _ in range(n_calls):
        rank_percentile_transform(oof, test_pred, smoothing=smoothing)


if __name__ == "__main__":
    for n_oof, n_test, n_calls in [(10_000, 5_000, 50), (1_000_000, 500_000, 5)]:
        t0 = time.perf_counter()
        _run(n_oof, n_test, n_calls)
        wall = time.perf_counter() - t0
        print(f"n_oof={n_oof:>9,} n_test={n_test:>9,} n_calls={n_calls:>3} -> {wall * 1000:9.2f} ms total, {wall / n_calls * 1000:9.3f} ms/call")

    # smoothed mode targets the small-reference-set regime (O(n_query * n_oof)) -- bench it there instead of
    # at the million-row scale the hard path is tuned for.
    for n_oof, n_test, n_calls in [(20, 20, 200), (1_000, 500, 20)]:
        t0 = time.perf_counter()
        _run(n_oof, n_test, n_calls, smoothing=0.35)
        wall = time.perf_counter() - t0
        print(
            f"[smoothed] n_oof={n_oof:>9,} n_test={n_test:>9,} n_calls={n_calls:>3} -> "
            f"{wall * 1000:9.2f} ms total, {wall / n_calls * 1000:9.3f} ms/call"
        )

    profiler = cProfile.Profile()
    profiler.enable()
    _run(1_000_000, 500_000, 5)
    profiler.disable()
    buf = StringIO()
    stats = pstats.Stats(profiler, stream=buf).sort_stats("cumulative")
    stats.print_stats(15)
    print(buf.getvalue())

    profiler = cProfile.Profile()
    profiler.enable()
    _run(1_000, 500, 20, smoothing=0.35)
    profiler.disable()
    buf = StringIO()
    stats = pstats.Stats(profiler, stream=buf).sort_stats("cumulative")
    stats.print_stats(15)
    print("[smoothed]")
    print(buf.getvalue())
