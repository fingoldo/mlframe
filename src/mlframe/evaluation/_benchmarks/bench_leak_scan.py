"""cProfile harness for ``evaluation.leak_scan.scan_temporal_leak``.

Run: ``python -m mlframe.evaluation._benchmarks.bench_leak_scan``
"""
from __future__ import annotations

import cProfile
import pstats
import time
from io import StringIO

import numpy as np
import pandas as pd

from mlframe.evaluation.leak_scan import scan_temporal_leak


def _run(n: int, n_cols: int, n_calls: int) -> None:
    rng = np.random.default_rng(0)
    split_labels = rng.integers(0, 10, size=n)
    data = {f"col_{i}": rng.standard_normal(n) for i in range(n_cols)}
    data["leak_col"] = split_labels.astype(np.float64) * 100.0 + rng.standard_normal(n)
    X = pd.DataFrame(data)
    for _ in range(n_calls):
        scan_temporal_leak(X, split_labels)


def _run_derived(n: int, n_cols: int, n_calls: int, max_derived_features: int | None) -> None:
    # n_cols here is kept small relative to the raw-scan benchmark: pairwise combos are O(n_cols^2), so a
    # naive apples-to-apples n_cols=200 would generate ~20k derived features per call and dominate the
    # profile with feature-construction noise rather than the scan itself. max_derived_features additionally
    # caps the combinatorial blowup independent of n_cols, mirroring how a real caller would bound the scan.
    rng = np.random.default_rng(0)
    split_labels = rng.integers(0, 10, size=n)
    data = {f"col_{i}": rng.standard_normal(n) for i in range(n_cols)}
    data["leak_col"] = split_labels.astype(np.float64) * 100.0 + rng.standard_normal(n)
    X = pd.DataFrame(data)
    for _ in range(n_calls):
        scan_temporal_leak(X, split_labels, scan_derived=True, max_derived_features=max_derived_features)


if __name__ == "__main__":
    for n, n_cols, n_calls in [(1_000, 50, 50), (100_000, 200, 10), (1_000_000, 50, 3)]:
        t0 = time.perf_counter()
        _run(n, n_cols, n_calls)
        wall = time.perf_counter() - t0
        print(f"n={n:>9,} cols={n_cols:>4} -> {wall * 1000:8.2f} ms total, {wall / n_calls * 1e6:9.2f} us/call")

    for n, n_cols, n_calls, max_derived in [
        (1_000, 30, 20, 500),
        (100_000, 30, 5, 500),
        (1_000_000, 30, 2, 500),
    ]:
        t0 = time.perf_counter()
        _run_derived(n, n_cols, n_calls, max_derived)
        wall = time.perf_counter() - t0
        print(
            f"[derived] n={n:>9,} cols={n_cols:>4} max_derived={max_derived:>4} -> "
            f"{wall * 1000:8.2f} ms total, {wall / n_calls * 1e6:9.2f} us/call"
        )

    n, n_cols, n_calls = 100_000, 200, 20
    profiler = cProfile.Profile()
    profiler.enable()
    _run(n, n_cols, n_calls)
    profiler.disable()
    buf = StringIO()
    stats = pstats.Stats(profiler, stream=buf).sort_stats("cumulative")
    stats.print_stats(15)
    print(buf.getvalue())

    n, n_cols, n_calls, max_derived = 100_000, 30, 10, 500
    profiler = cProfile.Profile()
    profiler.enable()
    _run_derived(n, n_cols, n_calls, max_derived)
    profiler.disable()
    buf = StringIO()
    stats = pstats.Stats(profiler, stream=buf).sort_stats("cumulative")
    stats.print_stats(15)
    print("[derived-scan profile]")
    print(buf.getvalue())
