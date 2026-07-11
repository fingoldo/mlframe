"""cProfile + A/B harness for ``feature_engineering.nested_ma_decompose.nested_ma_decompose``.

Compares the algebraic decomposition (given two precomputed MAs) against the alternative of computing the
exclusive window's average via a third ``rolling().apply()`` pass over raw data.

Run: ``python -m mlframe.feature_engineering._benchmarks.bench_nested_ma_decompose``
"""
from __future__ import annotations

import cProfile
import pstats
import time
from io import StringIO

import numpy as np
import pandas as pd

from mlframe.feature_engineering.nested_ma_decompose import nested_ma_decompose, nested_ma_decompose_chain


def _direct_exclusive_window_rolling(x: np.ndarray, window_short: int, window_long: int) -> np.ndarray:
    """Alternative: a third rolling pass computing the exclusive-window average directly from raw data."""
    s = pd.Series(x)
    total_sum = s.rolling(window_long).sum()
    recent_sum = s.rolling(window_short).sum()
    return np.asarray(((total_sum - recent_sum) / (window_long - window_short)).to_numpy())


def _run_algebraic(x: np.ndarray, window_short: int, window_long: int, n_calls: int) -> None:
    s = pd.Series(x)
    ma_short = s.rolling(window_short).mean().to_numpy()
    ma_long = s.rolling(window_long).mean().to_numpy()
    for _ in range(n_calls):
        nested_ma_decompose(ma_short, ma_long, window_short, window_long)


def _run_direct(x: np.ndarray, window_short: int, window_long: int, n_calls: int) -> None:
    for _ in range(n_calls):
        _direct_exclusive_window_rolling(x, window_short, window_long)


def _run_pairwise_chain(x: np.ndarray, windows: list, n_calls: int) -> None:
    """k-1 separate pairwise calls -- the baseline the chained mode is meant to beat."""
    s = pd.Series(x)
    mas = [s.rolling(w).mean().to_numpy() for w in windows]
    for _ in range(n_calls):
        for i in range(len(windows) - 1):
            nested_ma_decompose(mas[i], mas[i + 1], windows[i], windows[i + 1])


def _run_chain(x: np.ndarray, windows: list, n_calls: int) -> None:
    """Single vectorized multi-window call over the whole ladder."""
    s = pd.Series(x)
    mas = [s.rolling(w).mean().to_numpy() for w in windows]
    for _ in range(n_calls):
        nested_ma_decompose_chain(mas, windows)


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    for n, n_calls in [(2000, 200), (200000, 200), (2000000, 50)]:
        x = rng.normal(size=n).cumsum() + 100
        t0 = time.perf_counter()
        _run_algebraic(x, 3, 10, n_calls)
        t1 = time.perf_counter()
        _run_direct(x, 3, 10, n_calls)
        t2 = time.perf_counter()
        print(f"n={n:>8} n_calls={n_calls:>4} -> algebraic(given MAs)={(t1 - t0) * 1e3:9.2f}ms direct(recompute)={(t2 - t1) * 1e3:9.2f}ms")

    x = rng.normal(size=2000000).cumsum() + 100
    profiler = cProfile.Profile()
    profiler.enable()
    _run_algebraic(x, 3, 10, 50)
    profiler.disable()
    buf = StringIO()
    pstats.Stats(profiler, stream=buf).sort_stats("cumulative").print_stats(10)
    print(buf.getvalue())

    windows = [3, 10, 20, 45, 90, 180]
    for n, n_calls in [(2000, 200), (200000, 200), (2000000, 50)]:
        x = rng.normal(size=n).cumsum() + 100
        t0 = time.perf_counter()
        _run_pairwise_chain(x, windows, n_calls)
        t1 = time.perf_counter()
        _run_chain(x, windows, n_calls)
        t2 = time.perf_counter()
        print(f"n={n:>8} n_calls={n_calls:>4} -> pairwise({len(windows) - 1} calls)={(t1 - t0) * 1e3:9.2f}ms chain(1 call)={(t2 - t1) * 1e3:9.2f}ms")

    x = rng.normal(size=2000000).cumsum() + 100
    profiler = cProfile.Profile()
    profiler.enable()
    _run_chain(x, windows, 50)
    profiler.disable()
    buf = StringIO()
    pstats.Stats(profiler, stream=buf).sort_stats("cumulative").print_stats(10)
    print(buf.getvalue())
