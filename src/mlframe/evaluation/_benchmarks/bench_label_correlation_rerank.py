"""cProfile harness for ``evaluation.label_correlation_rerank``.

Run: ``python -m mlframe.evaluation._benchmarks.bench_label_correlation_rerank``
"""
from __future__ import annotations

import cProfile
import pstats
import time
from io import StringIO

import numpy as np

from mlframe.evaluation.label_correlation_rerank import (
    detect_correlated_label_groups,
    detect_correlated_label_pairs,
    label_correlation_rerank,
    optimize_group_blend_weight,
)


def _make_data(n: int, n_labels: int, seed: int = 0):
    rng = np.random.default_rng(seed)
    latent = rng.random(n) < 0.3
    y = np.zeros((n, n_labels), dtype=int)
    y[:, 0] = latent
    y[:, 1] = latent
    for label in range(2, n_labels):
        y[:, label] = rng.random(n) < 0.15
    scores = rng.uniform(size=(n, n_labels))
    return y, scores


def _make_group_data(n: int, n_labels: int, group_size: int = 4, seed: int = 0):
    rng = np.random.default_rng(seed)
    latent = rng.random(n) < 0.3
    y = np.zeros((n, n_labels), dtype=int)
    for label in range(group_size):
        y[:, label] = latent
    for label in range(group_size, n_labels):
        y[:, label] = rng.random(n) < 0.15
    scores = rng.uniform(size=(n, n_labels))
    return y, scores


def _run(n: int, n_labels: int, n_calls: int) -> None:
    y, scores = _make_data(n, n_labels)
    for _ in range(n_calls):
        pairs = detect_correlated_label_pairs(y, min_cooccurrence_rate=0.9, min_support=5)
        label_correlation_rerank(scores, pairs)


def _run_groups(n: int, n_labels: int, n_calls: int, group_size: int = 4) -> None:
    y, scores = _make_group_data(n, n_labels, group_size=group_size)
    for _ in range(n_calls):
        groups = detect_correlated_label_groups(y, min_cooccurrence_rate=0.9, min_support=5)
        label_correlation_rerank(scores, [], correlated_groups=groups)


def _run_group_weight_optimization(n: int, n_labels: int, n_calls: int, group_size: int = 4) -> None:
    y, scores = _make_group_data(n, n_labels, group_size=group_size)
    groups = detect_correlated_label_groups(y, min_cooccurrence_rate=0.9, min_support=5)
    for _ in range(n_calls):
        weights = optimize_group_blend_weight(y, scores, groups, n_splits=3)
        label_correlation_rerank(scores, [], correlated_groups=groups, group_weights=weights)


if __name__ == "__main__":
    for n, n_labels, n_calls in [(2000, 20, 20), (50000, 20, 20), (50000, 200, 5)]:
        t0 = time.perf_counter()
        _run(n, n_labels, n_calls)
        wall = time.perf_counter() - t0
        print(f"pairs      n={n:>7} n_labels={n_labels:>4} n_calls={n_calls:>4} -> {wall * 1000:9.2f} ms")

    for n, n_labels, n_calls in [(2000, 20, 20), (50000, 20, 20), (50000, 200, 5)]:
        t0 = time.perf_counter()
        _run_groups(n, n_labels, n_calls)
        wall = time.perf_counter() - t0
        print(f"groups     n={n:>7} n_labels={n_labels:>4} n_calls={n_calls:>4} -> {wall * 1000:9.2f} ms")

    # opt-in group-weight CV search is far more expensive per call (grid x folds x rerank) -- far fewer calls/rows.
    for n, n_labels, n_calls in [(2000, 20, 3), (10000, 20, 1)]:
        t0 = time.perf_counter()
        _run_group_weight_optimization(n, n_labels, n_calls)
        wall = time.perf_counter() - t0
        print(f"cv_weight  n={n:>7} n_labels={n_labels:>4} n_calls={n_calls:>4} -> {wall * 1000:9.2f} ms")

    profiler = cProfile.Profile()
    profiler.enable()
    _run(50000, 200, 5)
    _run_groups(50000, 200, 5)
    _run_group_weight_optimization(2000, 20, 3)
    profiler.disable()
    buf = StringIO()
    stats = pstats.Stats(profiler, stream=buf).sort_stats("cumulative")
    stats.print_stats(20)
    print(buf.getvalue())
