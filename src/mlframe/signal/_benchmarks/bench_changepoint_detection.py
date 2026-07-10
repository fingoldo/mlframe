"""cProfile harness for ``signal.detect_regime_changepoints``.

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


def _run(n: int, n_regimes: int) -> None:
    y = _make_series(n, n_regimes, seed=0)
    detect_regime_changepoints(y, min_segment_length=max(10, n // (n_regimes * 4)), penalty=5.0)


if __name__ == "__main__":
    # ruptures' l2 PELT cost function calls np.var() per candidate window (no cumsum optimization) -- 50,000
    # points measured at ~29s; capped at 5,000 here so this benchmark itself completes promptly. See the
    # module docstring's performance note for the full measured numbers and the rbf-vs-l2 fix history.
    for n, n_regimes in [(2_000, 5), (5_000, 10)]:
        t0 = time.perf_counter()
        _run(n, n_regimes)
        wall = time.perf_counter() - t0
        print(f"n={n:>7,} n_regimes={n_regimes:>3} -> {wall * 1000:9.2f} ms")

    profiler = cProfile.Profile()
    profiler.enable()
    _run(5_000, 10)
    profiler.disable()
    buf = StringIO()
    stats = pstats.Stats(profiler, stream=buf).sort_stats("cumulative")
    stats.print_stats(15)
    print(buf.getvalue())
