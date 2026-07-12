"""cProfile harness for ``calibration.sticky_state_persistence_floor``.

Run: ``python -m mlframe.calibration._benchmarks.bench_sticky_state_persistence_floor``
"""
from __future__ import annotations

import cProfile
import pstats
import time
from io import StringIO

import numpy as np

from mlframe.calibration.sticky_state_persistence_floor import (
    apply_sticky_state_persistence_floor,
    optimize_persistence_floor,
    optimize_persistence_floor_per_class,
)


def _run_apply(n: int, n_classes: int, n_calls: int) -> None:
    rng = np.random.default_rng(0)
    probs = rng.dirichlet(np.ones(n_classes), size=n)
    active = rng.integers(0, n_classes, size=n)
    for _ in range(n_calls):
        apply_sticky_state_persistence_floor(probs, active, floor=0.5)


def _run_optimize(n: int, n_classes: int) -> None:
    rng = np.random.default_rng(0)
    probs = rng.dirichlet(np.ones(n_classes), size=n)
    active = rng.integers(0, n_classes, size=n)
    y_true = rng.integers(0, n_classes, size=n)
    optimize_persistence_floor(probs, active, y_true, lambda yt, yp: float(np.mean(yt == yp)), n_thresholds=50)


def _run_apply_per_class(n: int, n_classes: int, n_calls: int) -> None:
    """Per-class floor vector path -- new opt-in mode vs. the scalar-floor path benchmarked above."""
    rng = np.random.default_rng(0)
    probs = rng.dirichlet(np.ones(n_classes), size=n)
    active = rng.integers(0, n_classes, size=n)
    floor_vec = rng.uniform(0.2, 0.7, size=n_classes)
    for _ in range(n_calls):
        apply_sticky_state_persistence_floor(probs, active, floor_vec)


def _run_optimize_per_class(n: int, n_classes: int) -> None:
    rng = np.random.default_rng(0)
    probs = rng.dirichlet(np.ones(n_classes), size=n)
    active = rng.integers(0, n_classes, size=n)
    y_true = rng.integers(0, n_classes, size=n)
    optimize_persistence_floor_per_class(probs, active, y_true, lambda yt, yp: float(np.mean(yt == yp)), n_thresholds=20, n_passes=2)


def _run_apply_common_case(n: int, n_classes: int, n_calls: int) -> None:
    """Realistic case: a peaked distribution where the active class is USUALLY already dominant, so the
    floor rarely triggers -- the regime the copy-skip optimization targets."""
    rng = np.random.default_rng(0)
    probs = rng.dirichlet(np.ones(n_classes) * 0.3, size=n)
    active = np.argmax(probs, axis=1)
    for _ in range(n_calls):
        apply_sticky_state_persistence_floor(probs, active, floor=0.3)


if __name__ == "__main__":
    for n, n_classes, n_calls in [(2000, 5, 200), (200000, 5, 200), (200000, 20, 200)]:
        t0 = time.perf_counter()
        _run_apply(n, n_classes, n_calls)
        wall = time.perf_counter() - t0
        print(f"apply (worst-case, uniform prior, high floor) n={n:>7} n_classes={n_classes:>3} n_calls={n_calls:>4} -> {wall * 1000:9.2f} ms")

    t0 = time.perf_counter()
    _run_apply_common_case(200000, 5, 200)
    print(f"apply (common case, peaked prior, floor rarely triggers) n=200000 n_classes=5 n_calls=200 -> {(time.perf_counter() - t0) * 1000:9.2f} ms")

    t0 = time.perf_counter()
    _run_optimize(50000, 10)
    print(f"optimize n=50000 n_classes=10 -> {(time.perf_counter() - t0) * 1000:9.2f} ms")

    t0 = time.perf_counter()
    _run_apply_per_class(200000, 20, 200)
    print(f"apply (per-class floor vector) n=200000 n_classes=20 n_calls=200 -> {(time.perf_counter() - t0) * 1000:9.2f} ms")

    t0 = time.perf_counter()
    _run_optimize_per_class(50000, 10)
    print(f"optimize_per_class n=50000 n_classes=10 -> {(time.perf_counter() - t0) * 1000:9.2f} ms")

    profiler = cProfile.Profile()
    profiler.enable()
    _run_apply(200000, 20, 200)
    profiler.disable()
    buf = StringIO()
    stats = pstats.Stats(profiler, stream=buf).sort_stats("cumulative")
    stats.print_stats(15)
    print(buf.getvalue())

    profiler_pc = cProfile.Profile()
    profiler_pc.enable()
    _run_apply_per_class(200000, 20, 200)
    profiler_pc.disable()
    buf_pc = StringIO()
    stats_pc = pstats.Stats(profiler_pc, stream=buf_pc).sort_stats("cumulative")
    stats_pc.print_stats(15)
    print(buf_pc.getvalue())
