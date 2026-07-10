"""cProfile harness for ``preprocessing.degradation_augment.augment_to_match_test_distribution``.

Run: ``python -m mlframe.preprocessing._benchmarks.bench_degradation_augment``
"""
from __future__ import annotations

import cProfile
import pstats
import time
from io import StringIO

import numpy as np
import pandas as pd

from mlframe.preprocessing.degradation_augment import augment_to_match_test_distribution


def _make_dataset(n_train: int, n_test: int, n_features: int, seed: int):
    rng = np.random.default_rng(seed)
    X_train = pd.DataFrame(rng.normal(scale=0.1, size=(n_train, n_features)), columns=[f"f{i}" for i in range(n_features)])
    X_test = pd.DataFrame(rng.normal(scale=1.5, size=(n_test, n_features)), columns=[f"f{i}" for i in range(n_features)])
    mask = rng.random(X_test.shape) < 0.4
    X_test = X_test.mask(mask)
    y_train = rng.integers(0, 2, size=n_train)
    return X_train, y_train, X_test


def _run(n_train: int, n_test: int, n_augments: int) -> None:
    X_train, y_train, X_test = _make_dataset(n_train, n_test, n_features=20, seed=0)
    augment_to_match_test_distribution(X_train, y_train, X_test, n_augments=n_augments)


if __name__ == "__main__":
    for n_train, n_augments in [(1000, 5), (5000, 5), (5000, 20)]:
        t0 = time.perf_counter()
        _run(n_train, 2000, n_augments)
        wall = time.perf_counter() - t0
        print(f"n_train={n_train:>6} n_augments={n_augments:>3} -> {wall * 1000:9.2f} ms")

    profiler = cProfile.Profile()
    profiler.enable()
    _run(5000, 2000, 20)
    profiler.disable()
    buf = StringIO()
    stats = pstats.Stats(profiler, stream=buf).sort_stats("cumulative")
    stats.print_stats(20)
    print(buf.getvalue())
