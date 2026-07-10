"""cProfile harness for ``evaluation.build_test_like_validation_fold``.

Run: ``python -m mlframe.evaluation._benchmarks.bench_adversarial_fold_selection``

Cost is dominated by the ``n_splits``-fold LightGBM cross_val_predict fit, inherent to the OOF-probability
requirement (a self-fulfilling selection without holding out folds during the classifier fit).
"""
from __future__ import annotations

import cProfile
import pstats
import time
from io import StringIO

import numpy as np
import pandas as pd

from mlframe.evaluation.adversarial_fold_selection import build_test_like_validation_fold


def _make_data(n_train: int, n_test: int, n_features: int, seed: int):
    rng = np.random.default_rng(seed)
    X_train = pd.DataFrame({f"f{i}": rng.normal(0, 1, n_train) for i in range(n_features)})
    X_test = pd.DataFrame({f"f{i}": rng.normal(0.5, 1, n_test) for i in range(n_features)})
    return X_train, X_test


def _run(n_train: int, n_test: int, n_features: int) -> None:
    X_train, X_test = _make_data(n_train, n_test, n_features, seed=0)
    build_test_like_validation_fold(X_train, X_test, val_fraction=0.2, seed=0)


if __name__ == "__main__":
    for n_train, n_test, n_features in [(5_000, 2_000, 10), (50_000, 20_000, 20)]:
        t0 = time.perf_counter()
        _run(n_train, n_test, n_features)
        wall = time.perf_counter() - t0
        print(f"n_train={n_train:>7,} n_test={n_test:>7,} n_features={n_features:>3} -> {wall * 1000:9.2f} ms")

    profiler = cProfile.Profile()
    profiler.enable()
    _run(5_000, 2_000, 10)
    profiler.disable()
    buf = StringIO()
    stats = pstats.Stats(profiler, stream=buf).sort_stats("cumulative")
    stats.print_stats(15)
    print(buf.getvalue())
