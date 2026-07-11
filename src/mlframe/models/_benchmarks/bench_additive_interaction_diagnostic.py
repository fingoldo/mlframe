"""cProfile harness for ``models.additive_interaction_diagnostic``.

Run: ``python -m mlframe.models._benchmarks.bench_additive_interaction_diagnostic``

Cost is dominated by ``2 * n_folds`` LightGBM fits (one full-interaction, one additive-only per fold),
inherent to the diagnostic's comparison methodology. The opt-in ``per_feature_report`` path adds
``2 * n_features * n_folds`` further fits (one full + one additive retrain per dropped feature per fold), so
it is benchmarked separately at a deliberately small ``n_features`` -- LOFO cost scales linearly with feature
count and is not meant for wide feature sets in one call.
"""
from __future__ import annotations

import cProfile
import pstats
import time
from io import StringIO

import numpy as np
from sklearn.metrics import r2_score
from sklearn.model_selection import KFold

from mlframe.models.additive_interaction_diagnostic import additive_interaction_diagnostic


def _run(n_rows: int, n_features: int, n_splits: int) -> None:
    rng = np.random.default_rng(0)
    X = rng.normal(0, 1, (n_rows, n_features))
    y = X[:, 0] * 2 + rng.normal(0, 0.3, n_rows)
    splits = list(KFold(n_splits, shuffle=True, random_state=0).split(X))
    additive_interaction_diagnostic(X, y, splits, metric_fn=r2_score, objective="regression")


def _run_per_feature(n_rows: int, n_features: int, n_splits: int) -> None:
    rng = np.random.default_rng(0)
    X = rng.normal(0, 1, (n_rows, n_features))
    y = X[:, 0] * X[:, 1] * 3 + X[:, 2] * 1.5 + rng.normal(0, 0.3, n_rows)
    splits = list(KFold(n_splits, shuffle=True, random_state=0).split(X))
    additive_interaction_diagnostic(X, y, splits, metric_fn=r2_score, objective="regression", per_feature_report=True)


if __name__ == "__main__":
    for n_rows, n_features, n_splits in [(2_000, 10, 5), (20_000, 20, 5)]:
        t0 = time.perf_counter()
        _run(n_rows, n_features, n_splits)
        wall = time.perf_counter() - t0
        print(f"n_rows={n_rows:>7,} n_features={n_features:>3} n_splits={n_splits} -> {wall * 1000:9.2f} ms")

    # kept deliberately small: cost is 2 * n_features * n_splits extra fits on top of the base 2 * n_splits.
    for n_rows, n_features, n_splits in [(2_000, 5, 3), (2_000, 8, 5)]:
        t0 = time.perf_counter()
        _run_per_feature(n_rows, n_features, n_splits)
        wall = time.perf_counter() - t0
        print(f"[per_feature_report] n_rows={n_rows:>7,} n_features={n_features:>3} n_splits={n_splits} -> {wall * 1000:9.2f} ms")

    profiler = cProfile.Profile()
    profiler.enable()
    _run(2_000, 10, 5)
    profiler.disable()
    buf = StringIO()
    stats = pstats.Stats(profiler, stream=buf).sort_stats("cumulative")
    stats.print_stats(15)
    print(buf.getvalue())

    profiler = cProfile.Profile()
    profiler.enable()
    _run_per_feature(2_000, 5, 3)
    profiler.disable()
    buf = StringIO()
    stats = pstats.Stats(profiler, stream=buf).sort_stats("cumulative")
    stats.print_stats(15)
    print("[per_feature_report]")
    print(buf.getvalue())
