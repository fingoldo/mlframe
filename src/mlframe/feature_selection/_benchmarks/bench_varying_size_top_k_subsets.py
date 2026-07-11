"""cProfile harness for ``feature_selection.varying_size_top_k_subsets.varying_size_top_k_subsets``.

Run: ``python -m mlframe.feature_selection._benchmarks.bench_varying_size_top_k_subsets``
"""
from __future__ import annotations

import cProfile
import pstats
import time
from io import StringIO

from mlframe.feature_selection.varying_size_top_k_subsets import varying_size_top_k_subsets


def _run(n_features: int, n_sizes: int, n_calls: int) -> None:
    ranked = [f"f{i}" for i in range(n_features)]
    sizes = list(range(10, n_features, max(1, n_features // n_sizes)))
    for _ in range(n_calls):
        varying_size_top_k_subsets(ranked, sizes)


if __name__ == "__main__":
    for n_features, n_sizes, n_calls in [(1000, 12, 1000), (10000, 12, 1000), (10000, 50, 5000)]:
        t0 = time.perf_counter()
        _run(n_features, n_sizes, n_calls)
        wall = time.perf_counter() - t0
        print(f"n_features={n_features:>6} n_sizes={n_sizes:>3} n_calls={n_calls:>5} -> {wall * 1000:9.2f} ms")

    profiler = cProfile.Profile()
    profiler.enable()
    _run(10000, 50, 5000)
    profiler.disable()
    buf = StringIO()
    stats = pstats.Stats(profiler, stream=buf).sort_stats("cumulative")
    stats.print_stats(15)
    print(buf.getvalue())
