"""cProfile harness for ``calibration.optimize_decision_threshold``.

Run: ``python -m mlframe.calibration._benchmarks.bench_threshold_optimizer``
"""
from __future__ import annotations

import cProfile
import pstats
import time
from io import StringIO

import numpy as np
from sklearn.metrics import f1_score

from mlframe.calibration.threshold_optimizer import optimize_decision_threshold


def _run(n_samples: int, n_thresholds: int) -> None:
    rng = np.random.default_rng(0)
    y_true = rng.integers(0, 2, n_samples)
    y_proba = rng.uniform(0, 1, n_samples)
    optimize_decision_threshold(y_true, y_proba, metric_fn=f1_score, n_thresholds=n_thresholds)


def _run_per_segment(n_samples: int, n_thresholds: int, n_groups: int) -> None:
    """Opt-in per-segment path: adds one sweep per group on top of the global sweep."""
    rng = np.random.default_rng(0)
    y_true = rng.integers(0, 2, n_samples)
    y_proba = rng.uniform(0, 1, n_samples)
    groups = rng.integers(0, n_groups, n_samples)
    optimize_decision_threshold(y_true, y_proba, metric_fn=f1_score, n_thresholds=n_thresholds, groups=groups)


def _run_cv_stability(n_samples: int, n_thresholds: int, cv: int) -> None:
    """Opt-in CV-stability path: adds one sweep per fold on top of the global sweep."""
    rng = np.random.default_rng(0)
    y_true = rng.integers(0, 2, n_samples)
    y_proba = rng.uniform(0, 1, n_samples)
    optimize_decision_threshold(y_true, y_proba, metric_fn=f1_score, n_thresholds=n_thresholds, cv=cv)


if __name__ == "__main__":
    for n_samples, n_thresholds in [(10_000, 200), (100_000, 200)]:
        t0 = time.perf_counter()
        _run(n_samples, n_thresholds)
        wall = time.perf_counter() - t0
        print(f"global            n_samples={n_samples:>9,} n_thresholds={n_thresholds:>3} -> {wall * 1000:9.2f} ms")

    for n_samples, n_thresholds, n_groups in [(10_000, 200, 5), (100_000, 200, 5)]:
        t0 = time.perf_counter()
        _run_per_segment(n_samples, n_thresholds, n_groups)
        wall = time.perf_counter() - t0
        print(f"per_segment(g={n_groups}) n_samples={n_samples:>9,} n_thresholds={n_thresholds:>3} -> {wall * 1000:9.2f} ms")

    for n_samples, n_thresholds, cv in [(10_000, 200, 5), (100_000, 200, 5)]:
        t0 = time.perf_counter()
        _run_cv_stability(n_samples, n_thresholds, cv)
        wall = time.perf_counter() - t0
        print(f"cv_stability(k={cv}) n_samples={n_samples:>9,} n_thresholds={n_thresholds:>3} -> {wall * 1000:9.2f} ms")

    profiler = cProfile.Profile()
    profiler.enable()
    _run(10_000, 200)
    profiler.disable()
    buf = StringIO()
    stats = pstats.Stats(profiler, stream=buf).sort_stats("cumulative")
    stats.print_stats(15)
    print("=== global sweep ===")
    print(buf.getvalue())

    profiler = cProfile.Profile()
    profiler.enable()
    _run_per_segment(10_000, 200, 5)
    profiler.disable()
    buf = StringIO()
    stats = pstats.Stats(profiler, stream=buf).sort_stats("cumulative")
    stats.print_stats(15)
    print("=== per-segment sweep (5 groups) ===")
    print(buf.getvalue())

    profiler = cProfile.Profile()
    profiler.enable()
    _run_cv_stability(10_000, 200, 5)
    profiler.disable()
    buf = StringIO()
    stats = pstats.Stats(profiler, stream=buf).sort_stats("cumulative")
    stats.print_stats(15)
    print("=== CV-stability sweep (5 folds) ===")
    print(buf.getvalue())
