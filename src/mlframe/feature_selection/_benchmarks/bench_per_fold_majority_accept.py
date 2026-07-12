"""cProfile harness for ``feature_selection.filters.per_fold_majority_accept``.

Run: ``python -m mlframe.feature_selection._benchmarks.bench_per_fold_majority_accept``
"""
from __future__ import annotations

import cProfile
import pstats
import time
from io import StringIO

import numpy as np

from mlframe.feature_selection.filters._per_fold_majority_accept import per_fold_majority_accept


def _run(n_calls: int, n_folds: int, compute_agreement_score: bool = False) -> None:
    rng = np.random.default_rng(0)
    for _ in range(n_calls):
        baseline = rng.normal(0.8, 0.02, n_folds)
        candidate = baseline + rng.normal(0, 0.02, n_folds)
        per_fold_majority_accept(baseline, candidate, compute_agreement_score=compute_agreement_score)


if __name__ == "__main__":
    n_calls = 100_000
    t0 = time.perf_counter()
    _run(n_calls, n_folds=5)
    wall = time.perf_counter() - t0
    print(f"n_calls={n_calls:>7,} n_folds=5 -> {wall * 1000:9.2f} ms total, {wall / n_calls * 1e6:9.2f} us/call")

    t0 = time.perf_counter()
    _run(n_calls, n_folds=5, compute_agreement_score=True)
    wall_agreement = time.perf_counter() - t0
    print(
        f"n_calls={n_calls:>7,} n_folds=5 compute_agreement_score=True -> "
        f"{wall_agreement * 1000:9.2f} ms total, {wall_agreement / n_calls * 1e6:9.2f} us/call"
    )

    profiler = cProfile.Profile()
    profiler.enable()
    _run(20_000, n_folds=5)
    profiler.disable()
    buf = StringIO()
    stats = pstats.Stats(profiler, stream=buf).sort_stats("cumulative")
    stats.print_stats(15)
    print(buf.getvalue())

    profiler = cProfile.Profile()
    profiler.enable()
    _run(20_000, n_folds=5, compute_agreement_score=True)
    profiler.disable()
    buf = StringIO()
    stats = pstats.Stats(profiler, stream=buf).sort_stats("cumulative")
    stats.print_stats(15)
    print("compute_agreement_score=True profile:")
    print(buf.getvalue())
