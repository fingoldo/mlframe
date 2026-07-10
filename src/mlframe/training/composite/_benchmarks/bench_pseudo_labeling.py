"""cProfile harness for ``training.composite.PseudoLabelingLoop``.

Run: ``python -m mlframe.training.composite._benchmarks.bench_pseudo_labeling``
"""
from __future__ import annotations

import cProfile
import pstats
import time
from io import StringIO

import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier

from mlframe.training.composite import PseudoLabelingLoop


def _make_dataset(n: int, seed: int, d: int = 6):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, d))
    w = np.zeros(d)
    w[:3] = [1.5, -1.0, 0.5]
    p = 1.0 / (1.0 + np.exp(-(X @ w)))
    y = (rng.random(n) < p).astype(float)
    return pd.DataFrame(X, columns=[f"f{i}" for i in range(d)]), y


def _run(n_unlabeled: int) -> None:
    X_labeled, y_labeled = _make_dataset(35, 0)
    X_unlabeled, _ = _make_dataset(n_unlabeled, 1)
    loop = PseudoLabelingLoop(estimator_factory=lambda: DecisionTreeClassifier(max_depth=4, random_state=0), task="classification", n_rounds=2, n_splits=5, confidence_threshold=0.8)
    loop.fit(X_labeled, y_labeled, X_unlabeled)


if __name__ == "__main__":
    for n_unlabeled in [1_000, 5_000, 20_000]:
        t0 = time.perf_counter()
        _run(n_unlabeled)
        wall = time.perf_counter() - t0
        print(f"n_unlabeled={n_unlabeled:>7,} -> {wall * 1000:9.2f} ms")

    profiler = cProfile.Profile()
    profiler.enable()
    _run(20_000)
    profiler.disable()
    buf = StringIO()
    stats = pstats.Stats(profiler, stream=buf).sort_stats("cumulative")
    stats.print_stats(15)
    print(buf.getvalue())
