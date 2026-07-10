"""cProfile harness for ``evaluation.AdversarialValidator``.

Run: ``python -m mlframe.evaluation._benchmarks.bench_adversarial_validator``
"""
from __future__ import annotations

import cProfile
import pstats
import time
from io import StringIO

import numpy as np
import pandas as pd

from mlframe.evaluation import AdversarialValidator


def _make_dataset(n_train: int, n_test: int, seed: int):
    rng = np.random.default_rng(seed)
    regime = rng.random(n_train) < 0.2
    x_train = rng.normal(0, 1, n_train)
    segment_train = np.where(regime, rng.normal(3, 0.5, n_train), rng.normal(0, 0.5, n_train))
    X_train = pd.DataFrame({"x": x_train, "segment": segment_train})
    X_test = pd.DataFrame({"x": rng.normal(0, 1, n_test), "segment": rng.normal(3, 0.5, n_test)})
    return X_train, X_test


def _run(n_train: int, n_test: int) -> None:
    X_train, X_test = _make_dataset(n_train, n_test, seed=0)
    validator = AdversarialValidator(seed=0).fit(X_train, X_test)
    validator.report()
    validator.select_validation_fold(val_fraction=0.2)


if __name__ == "__main__":
    for n_train, n_test in [(1_000, 200), (5_000, 1_000), (20_000, 4_000)]:
        t0 = time.perf_counter()
        _run(n_train, n_test)
        wall = time.perf_counter() - t0
        print(f"n_train={n_train:>7,} n_test={n_test:>6,} -> {wall * 1000:9.2f} ms")

    profiler = cProfile.Profile()
    profiler.enable()
    _run(20_000, 4_000)
    profiler.disable()
    buf = StringIO()
    stats = pstats.Stats(profiler, stream=buf).sort_stats("cumulative")
    stats.print_stats(15)
    print(buf.getvalue())
