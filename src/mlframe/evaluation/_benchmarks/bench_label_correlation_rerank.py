"""cProfile harness for ``evaluation.label_correlation_rerank``.

Run: ``python -m mlframe.evaluation._benchmarks.bench_label_correlation_rerank``
"""
from __future__ import annotations

import cProfile
import pstats
import time
from io import StringIO

import numpy as np

from mlframe.evaluation.label_correlation_rerank import detect_correlated_label_pairs, label_correlation_rerank


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


def _run(n: int, n_labels: int, n_calls: int) -> None:
    y, scores = _make_data(n, n_labels)
    for _ in range(n_calls):
        pairs = detect_correlated_label_pairs(y, min_cooccurrence_rate=0.9, min_support=5)
        label_correlation_rerank(scores, pairs)


if __name__ == "__main__":
    for n, n_labels, n_calls in [(2000, 20, 20), (50000, 20, 20), (50000, 200, 5)]:
        t0 = time.perf_counter()
        _run(n, n_labels, n_calls)
        wall = time.perf_counter() - t0
        print(f"n={n:>7} n_labels={n_labels:>4} n_calls={n_calls:>4} -> {wall * 1000:9.2f} ms")

    profiler = cProfile.Profile()
    profiler.enable()
    _run(50000, 200, 5)
    profiler.disable()
    buf = StringIO()
    stats = pstats.Stats(profiler, stream=buf).sort_stats("cumulative")
    stats.print_stats(15)
    print(buf.getvalue())
