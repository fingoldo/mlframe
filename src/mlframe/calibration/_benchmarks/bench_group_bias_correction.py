"""cProfile harness for ``calibration.group_bias_correction`` (fit + apply).

Run: ``python -m mlframe.calibration._benchmarks.bench_group_bias_correction``
"""
from __future__ import annotations

import cProfile
import pstats
import time
from io import StringIO

import numpy as np

from mlframe.calibration.group_bias_correction import apply_group_bias_correction, fit_group_bias_correction


def _make_dataset(n: int, n_groups: int, seed: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    groups = rng.integers(0, n_groups, n).astype(str)
    y_true = rng.uniform(50, 150, n)
    y_pred = y_true * rng.uniform(0.7, 1.3, n_groups)[rng.integers(0, n_groups, n)]
    return y_true, y_pred, groups


def _run(n: int, n_groups: int, shrinkage_k: float | None = None) -> None:
    y_true, y_pred, groups = _make_dataset(n, n_groups, seed=0)
    ratios = fit_group_bias_correction(y_true, y_pred, groups, shrinkage_k=shrinkage_k)
    apply_group_bias_correction(y_pred, groups, ratios)


if __name__ == "__main__":
    for n, n_groups in [(50000, 50), (500000, 50), (500000, 2000)]:
        t0 = time.perf_counter()
        _run(n, n_groups)
        wall = time.perf_counter() - t0
        print(f"n={n:>7} n_groups={n_groups:>5} -> {wall * 1000:9.2f} ms")

    for n, n_groups in [(50000, 50), (500000, 50), (500000, 2000)]:
        t0 = time.perf_counter()
        _run(n, n_groups, shrinkage_k=20.0)
        wall = time.perf_counter() - t0
        print(f"[shrinkage_k=20] n={n:>7} n_groups={n_groups:>5} -> {wall * 1000:9.2f} ms")

    profiler = cProfile.Profile()
    profiler.enable()
    _run(500000, 2000)
    profiler.disable()
    buf = StringIO()
    stats = pstats.Stats(profiler, stream=buf).sort_stats("cumulative")
    stats.print_stats(20)
    print(buf.getvalue())

    profiler = cProfile.Profile()
    profiler.enable()
    _run(500000, 2000, shrinkage_k=20.0)
    profiler.disable()
    buf = StringIO()
    stats = pstats.Stats(profiler, stream=buf).sort_stats("cumulative")
    stats.print_stats(20)
    print("[shrinkage_k=20]")
    print(buf.getvalue())
