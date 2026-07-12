"""cProfile harness for ``inference.logical_constraints.apply_logical_constraints``.

Run: ``python -m mlframe.inference._benchmarks.bench_logical_constraints``
"""
from __future__ import annotations

import cProfile
import pstats
import time
from io import StringIO

import numpy as np

from mlframe.inference.logical_constraints import (
    apply_logical_constraints,
    discover_logical_constraints,
    discover_logical_constraints_soft,
)


def _run(n: int, n_labels: int, n_rules: int, n_calls: int) -> None:
    rng = np.random.default_rng(0)
    preds = rng.uniform(0, 1, size=(n, n_labels))
    rules = [(i, i + 1) for i in range(0, min(n_rules * 2, n_labels - 1), 2)]
    for _ in range(n_calls):
        apply_logical_constraints(preds, rules)


def _run_soft(n: int, n_labels: int, n_rules: int, n_calls: int) -> None:
    rng = np.random.default_rng(0)
    preds = rng.uniform(0, 1, size=(n, n_labels))
    rules_soft = [(i, i + 1, 0.85) for i in range(0, min(n_rules * 2, n_labels - 1), 2)]
    for _ in range(n_calls):
        apply_logical_constraints(preds, rules_soft, mode="soft")


def _run_discover(n: int, n_labels: int, n_calls: int) -> None:
    rng = np.random.default_rng(0)
    y = (rng.random((n, n_labels)) < 0.3).astype(np.float64)
    for _ in range(n_calls):
        discover_logical_constraints(y, min_child_support=5)


def _run_discover_soft(n: int, n_labels: int, n_calls: int) -> None:
    rng = np.random.default_rng(0)
    y = (rng.random((n, n_labels)) < 0.3).astype(np.float64)
    for _ in range(n_calls):
        discover_logical_constraints_soft(y, min_child_support=5, min_confidence=0.85)


if __name__ == "__main__":
    _run(50, 4, 1, 1)  # warm both njit variants (single-thread + parallel) before timing
    _run(30_000, 4, 1, 1)

    for n, n_labels, n_rules, n_calls in [(1_000, 10, 3, 200), (100_000, 20, 5, 20), (1_000_000, 10, 3, 5)]:
        t0 = time.perf_counter()
        _run(n, n_labels, n_rules, n_calls)
        wall = time.perf_counter() - t0
        print(f"n={n:>9,} labels={n_labels:>2} rules={n_rules:>2} -> {wall * 1000:8.2f} ms total, {wall / n_calls * 1e6:8.2f} us/call")

    for n, n_labels, n_calls in [(1_000, 30, 50), (10_000, 100, 20), (100_000, 30, 5)]:
        t0 = time.perf_counter()
        _run_discover(n, n_labels, n_calls)
        wall = time.perf_counter() - t0
        print(f"discover n={n:>9,} labels={n_labels:>3} -> {wall * 1000:8.2f} ms total, {wall / n_calls * 1e6:8.2f} us/call")

    for n, n_labels, n_rules, n_calls in [(1_000, 10, 3, 200), (100_000, 20, 5, 20), (1_000_000, 10, 3, 5)]:
        t0 = time.perf_counter()
        _run_soft(n, n_labels, n_rules, n_calls)
        wall = time.perf_counter() - t0
        print(f"soft   n={n:>9,} labels={n_labels:>2} rules={n_rules:>2} -> {wall * 1000:8.2f} ms total, {wall / n_calls * 1e6:8.2f} us/call")

    for n, n_labels, n_calls in [(1_000, 30, 50), (10_000, 100, 20), (100_000, 30, 5)]:
        t0 = time.perf_counter()
        _run_discover_soft(n, n_labels, n_calls)
        wall = time.perf_counter() - t0
        print(f"discover_soft n={n:>9,} labels={n_labels:>3} -> {wall * 1000:8.2f} ms total, {wall / n_calls * 1e6:8.2f} us/call")

    n, n_labels, n_rules, n_calls = 100_000, 20, 5, 50
    profiler = cProfile.Profile()
    profiler.enable()
    _run(n, n_labels, n_rules, n_calls)
    _run_soft(n, n_labels, n_rules, n_calls)
    _run_discover(10_000, 100, 20)
    _run_discover_soft(10_000, 100, 20)
    profiler.disable()
    buf = StringIO()
    stats = pstats.Stats(profiler, stream=buf).sort_stats("cumulative")
    stats.print_stats(15)
    print(buf.getvalue())
