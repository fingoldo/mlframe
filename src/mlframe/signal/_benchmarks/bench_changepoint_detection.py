"""cProfile harness for ``signal.detect_regime_changepoints`` (default njit backend) plus an A/B against the
``ruptures`` backend.

Run: ``python -m mlframe.signal._benchmarks.bench_changepoint_detection``
"""
from __future__ import annotations

import cProfile
import pstats
import time
from io import StringIO

import numpy as np

from mlframe.signal.changepoint_detection import detect_regime_changepoints


def _make_series(n: int, n_regimes: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    seg_len = n // n_regimes
    segments = [rng.normal(rng.uniform(-10, 10), 1, seg_len) for _ in range(n_regimes)]
    return np.concatenate(segments)


def _run(n: int, n_regimes: int, backend: str, return_segment_stats: bool = False) -> None:
    y = _make_series(n, n_regimes, seed=0)
    detect_regime_changepoints(
        y,
        min_segment_length=max(10, n // (n_regimes * 4)),
        penalty=5.0,
        backend=backend,
        return_segment_stats=return_segment_stats,
    )


if __name__ == "__main__":
    _run(1_000, 3, "njit")  # warm the njit kernel before timing

    # ruptures' own l2 cost function calls np.var() per candidate window (no cumsum optimization); the njit
    # backend below is ~100x faster and is the library default. See the module docstring's performance note.
    for n, n_regimes in [(2_000, 5), (5_000, 10), (50_000, 10)]:
        t0 = time.perf_counter()
        _run(n, n_regimes, "njit")
        wall = time.perf_counter() - t0
        print(f"njit     n={n:>7,} n_regimes={n_regimes:>3} -> {wall * 1000:9.2f} ms")

    for n, n_regimes in [(2_000, 5), (5_000, 10)]:  # ruptures backend capped smaller -- multi-second at 5,000
        t0 = time.perf_counter()
        _run(n, n_regimes, "ruptures")
        wall = time.perf_counter() - t0
        print(f"ruptures n={n:>7,} n_regimes={n_regimes:>3} -> {wall * 1000:9.2f} ms")

    # return_segment_stats is O(n_regimes) post-processing on top of the njit PELT loop -- should add
    # negligible wall time even at large n / many regimes.
    for n, n_regimes in [(5_000, 10), (50_000, 10)]:
        t0 = time.perf_counter()
        _run(n, n_regimes, "njit", return_segment_stats=True)
        wall = time.perf_counter() - t0
        print(f"njit+segment_stats n={n:>7,} n_regimes={n_regimes:>3} -> {wall * 1000:9.2f} ms")

    profiler = cProfile.Profile()
    profiler.enable()
    _run(50_000, 10, "njit")
    profiler.disable()
    buf = StringIO()
    stats = pstats.Stats(profiler, stream=buf).sort_stats("cumulative")
    stats.print_stats(15)
    print(buf.getvalue())

    profiler2 = cProfile.Profile()
    profiler2.enable()
    _run(50_000, 10, "njit", return_segment_stats=True)
    profiler2.disable()
    buf2 = StringIO()
    stats2 = pstats.Stats(profiler2, stream=buf2).sort_stats("cumulative")
    stats2.print_stats(15)
    print(buf2.getvalue())
