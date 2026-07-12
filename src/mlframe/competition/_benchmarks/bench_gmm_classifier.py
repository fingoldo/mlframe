"""cProfile benchmark for ``mlframe.competition.gmm_classifier``.

COMPETITION/EXPLORATORY USE ONLY -- see ``mlframe.competition`` package docstring.

Run directly: ``python -m mlframe.competition._benchmarks.bench_gmm_classifier``
"""
from __future__ import annotations

import cProfile
import pstats
import time

import numpy as np

from mlframe.competition.gmm_classifier import GaussianMixtureClassifier


def _make_dataset(n: int, n_features: int, seed: int):
    rng = np.random.default_rng(seed)
    n_per_bucket = n // 4
    class0_centers = [np.full(n_features, -0.5), np.full(n_features, 0.5)]
    class1_centers = [np.full(n_features, -0.5), np.full(n_features, 0.5)]
    X_parts, y_parts = [], []
    for center in class0_centers:
        X_parts.append(rng.multivariate_normal(center, np.eye(n_features) * 0.5, size=n_per_bucket))
        y_parts.append(np.zeros(n_per_bucket, dtype=int))
    for center in class1_centers:
        X_parts.append(rng.multivariate_normal(center, np.eye(n_features) * 3.5, size=n_per_bucket))
        y_parts.append(np.ones(n_per_bucket, dtype=int))
    X = np.concatenate(X_parts, axis=0)
    y = np.concatenate(y_parts, axis=0)
    perm = rng.permutation(len(y))
    return X[perm], y[perm]


def _run_once() -> None:
    for n, n_features in [(2_000, 10), (5_000, 20), (10_000, 30)]:
        X_train, y_train = _make_dataset(n, n_features, seed=0)
        X_test, _ = _make_dataset(n, n_features, seed=1)
        clf = GaussianMixtureClassifier(n_components_per_class=2, random_state=0)
        clf.fit(X_train, y_train)
        clf.predict_proba(X_test)
        clf.predict(X_test)


def main() -> None:
    profiler = cProfile.Profile()
    t0 = time.perf_counter()
    profiler.enable()
    _run_once()
    profiler.disable()
    wall = time.perf_counter() - t0

    stats = pstats.Stats(profiler).sort_stats("cumulative")
    print(f"wall time: {wall:.4f}s")
    stats.print_stats(30)


if __name__ == "__main__":
    main()
