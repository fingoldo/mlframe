"""cProfile harness for ``votenrank.rank_splice.segment_rank_splice``.

Run: ``python -m mlframe.votenrank._benchmarks.bench_rank_splice``
"""
from __future__ import annotations

import cProfile
import pstats
import time
from io import StringIO

import numpy as np

from mlframe.votenrank.rank_splice import segment_rank_splice


def _run(n: int, segment_frac: float, n_calls: int, blend_weight: float = 0.0) -> None:
    rng = np.random.default_rng(0)
    main_scores = rng.normal(size=n)
    segment_mask = rng.random(n) < segment_frac
    specialist_scores = rng.normal(size=int(segment_mask.sum()))
    for _ in range(n_calls):
        segment_rank_splice(main_scores, specialist_scores, segment_mask, blend_weight=blend_weight)


if __name__ == "__main__":
    for n, segment_frac, n_calls in [(10000, 0.2, 1000), (1000000, 0.2, 100), (1000000, 0.05, 100)]:
        t0 = time.perf_counter()
        _run(n, segment_frac, n_calls)
        wall = time.perf_counter() - t0
        print(f"n={n:>9} segment_frac={segment_frac:>4} n_calls={n_calls:>5} blend_weight=0.0 (hard cutover) -> {wall * 1000:9.2f} ms")

    # Soft-blend path (blend_weight != 0.0) does one extra argsort(argsort(...)) pass over the segment for the
    # main-model rank plus the blended-key re-rank -- profile it separately to see the added cost.
    for n, segment_frac, n_calls in [(10000, 0.2, 1000), (1000000, 0.2, 100), (1000000, 0.05, 100)]:
        t0 = time.perf_counter()
        _run(n, segment_frac, n_calls, blend_weight=0.5)
        wall = time.perf_counter() - t0
        print(f"n={n:>9} segment_frac={segment_frac:>4} n_calls={n_calls:>5} blend_weight=0.5 (soft blend) -> {wall * 1000:9.2f} ms")

    profiler = cProfile.Profile()
    profiler.enable()
    _run(1000000, 0.2, 200)
    profiler.disable()
    buf = StringIO()
    stats = pstats.Stats(profiler, stream=buf).sort_stats("cumulative")
    stats.print_stats(15)
    print("hard cutover (blend_weight=0.0) profile:")
    print(buf.getvalue())

    profiler_blend = cProfile.Profile()
    profiler_blend.enable()
    _run(1000000, 0.2, 200, blend_weight=0.5)
    profiler_blend.disable()
    buf_blend = StringIO()
    stats_blend = pstats.Stats(profiler_blend, stream=buf_blend).sort_stats("cumulative")
    stats_blend.print_stats(15)
    print("soft blend (blend_weight=0.5) profile:")
    print(buf_blend.getvalue())
