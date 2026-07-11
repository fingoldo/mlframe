"""cProfile harness for ``models.masked_multilabel_objective``.

Run: ``python -m mlframe.models._benchmarks.bench_masked_multilabel_objective``
"""
from __future__ import annotations

import cProfile
import pstats
import time
from io import StringIO

import numpy as np

from mlframe.models.masked_multilabel_objective import flatten_masked_multilabel, masked_multilabel_logloss_objective


class _FakeDMatrix:
    def __init__(self, labels: np.ndarray) -> None:
        self._labels = labels

    def get_label(self) -> np.ndarray:
        return self._labels


def _run_flatten(n_rows: int, n_labels: int, n_calls: int) -> None:
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, size=(n_rows, n_labels)).astype(np.float64)
    mask = rng.random((n_rows, n_labels)) < 0.3
    for _ in range(n_calls):
        flatten_masked_multilabel(y, mask)


def _run_objective(n: int, n_calls: int) -> None:
    rng = np.random.default_rng(0)
    y_true = rng.choice([0.0, 1.0, 2.0], size=n)
    dtrain = _FakeDMatrix(y_true)
    pred = rng.normal(size=n)
    objective = masked_multilabel_logloss_objective()
    for _ in range(n_calls):
        objective(pred, dtrain)


if __name__ == "__main__":
    for n_rows, n_labels, n_calls in [(2000, 20, 50), (200000, 20, 50), (200000, 200, 10)]:
        t0 = time.perf_counter()
        _run_flatten(n_rows, n_labels, n_calls)
        wall = time.perf_counter() - t0
        print(f"flatten n_rows={n_rows:>7} n_labels={n_labels:>4} n_calls={n_calls:>4} -> {wall * 1000:9.2f} ms")

    for n, n_calls in [(200000, 200), (4000000, 200)]:
        t0 = time.perf_counter()
        _run_objective(n, n_calls)
        wall = time.perf_counter() - t0
        print(f"objective n={n:>8} n_calls={n_calls:>4} -> {wall * 1000:9.2f} ms")

    profiler = cProfile.Profile()
    profiler.enable()
    _run_objective(4000000, 200)
    profiler.disable()
    buf = StringIO()
    stats = pstats.Stats(profiler, stream=buf).sort_stats("cumulative")
    stats.print_stats(10)
    print(buf.getvalue())
