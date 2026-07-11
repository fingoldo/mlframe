"""cProfile harness for ``evaluation.check_pairwise_score_correlation``.

Run: ``python -m mlframe.evaluation._benchmarks.bench_blend_source_selection``
"""
from __future__ import annotations

import cProfile
import pstats
import time
from io import StringIO

import numpy as np

from mlframe.evaluation.blend_source_selection import check_pairwise_score_correlation


def _run(n_members: int, n_calls: int) -> None:
    rng = np.random.default_rng(0)
    for _ in range(n_calls):
        a = rng.uniform(0, 1, n_members)
        b = rng.uniform(0, 1, n_members)
        check_pairwise_score_correlation(a, b)


def _run_multi_source(n_members: int, n_sources: int, n_calls: int) -> None:
    rng = np.random.default_rng(0)
    for _ in range(n_calls):
        a = rng.uniform(0, 1, n_members)
        b = rng.uniform(0, 1, n_members)
        extra = [rng.uniform(0, 1, n_members) for _ in range(n_sources - 2)]
        check_pairwise_score_correlation(a, b, oos_scores_extra=extra)


if __name__ == "__main__":
    for n_members, n_calls in [(15, 1_000), (500, 100)]:
        t0 = time.perf_counter()
        _run(n_members, n_calls)
        wall = time.perf_counter() - t0
        print(f"n_members={n_members:>4} n_calls={n_calls:>5} -> {wall * 1000:9.2f} ms total, {wall / n_calls * 1e6:9.2f} us/call")

    for n_members, n_sources, n_calls in [(15, 5, 1_000), (500, 5, 100), (500, 20, 100)]:
        t0 = time.perf_counter()
        _run_multi_source(n_members, n_sources, n_calls)
        wall = time.perf_counter() - t0
        print(
            f"[multi-source] n_members={n_members:>4} n_sources={n_sources:>3} n_calls={n_calls:>5} "
            f"-> {wall * 1000:9.2f} ms total, {wall / n_calls * 1e6:9.2f} us/call"
        )

    profiler = cProfile.Profile()
    profiler.enable()
    _run(500, 100)
    _run_multi_source(500, 20, 100)
    profiler.disable()
    buf = StringIO()
    stats = pstats.Stats(profiler, stream=buf).sort_stats("cumulative")
    stats.print_stats(15)
    print(buf.getvalue())
