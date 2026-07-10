"""cProfile harness for ``training.composite.MultiTaskAuxiliaryLossRegressor``.

Run: ``python -m mlframe.training.composite._benchmarks.bench_multitask_auxiliary_loss``
"""
from __future__ import annotations

import cProfile
import pstats
import time
from io import StringIO

import numpy as np

from mlframe.training.composite import MultiTaskAuxiliaryLossRegressor


def _make_dataset(n: int, seed: int):
    rng = np.random.default_rng(seed)
    x = rng.uniform(-3, 3, n)
    y = np.sin(x) + 3.0 * np.exp(-(x**2) / 0.02) + rng.normal(scale=0.05, size=n)
    aux_binary = (np.abs(x) < 0.2).astype(np.float32)
    aux_regression = np.abs(x)
    X = x.reshape(-1, 1).astype(np.float32)
    return X, y, aux_binary, aux_regression


def _run(n: int, n_epochs: int) -> None:
    X, y, aux_binary, aux_regression = _make_dataset(n, seed=0)
    model = MultiTaskAuxiliaryLossRegressor(hidden_sizes=(32, 16), aux_task_weight=0.3, n_epochs=n_epochs, random_state=0)
    model.fit(X, y, y_aux_binary=aux_binary, y_aux_regression=aux_regression)
    model.predict(X)


if __name__ == "__main__":
    for n, n_epochs in [(500, 100), (2_000, 100), (5_000, 100)]:
        t0 = time.perf_counter()
        _run(n, n_epochs)
        wall = time.perf_counter() - t0
        print(f"n={n:>6,} n_epochs={n_epochs:>4} -> {wall * 1000:9.2f} ms")

    profiler = cProfile.Profile()
    profiler.enable()
    _run(2_000, 100)
    profiler.disable()
    buf = StringIO()
    stats = pstats.Stats(profiler, stream=buf).sort_stats("cumulative")
    stats.print_stats(15)
    print(buf.getvalue())
