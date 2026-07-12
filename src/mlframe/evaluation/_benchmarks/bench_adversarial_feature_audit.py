"""cProfile harness for ``evaluation.adversarial_validation_feature_audit``.

Run: ``python -m mlframe.evaluation._benchmarks.bench_adversarial_feature_audit``

The function's cost is dominated by LightGBM fits (1 for the adversarial classifier's CV, then 1 baseline +
1-per-ablated-feature inside the pseudo-split -- an audit-cadence tool, not a hot inner-loop kernel).
"""
from __future__ import annotations

import cProfile
import pstats
import time
from io import StringIO

import numpy as np
import pandas as pd

from mlframe.evaluation.adversarial_feature_audit import adversarial_validation_feature_audit


def _make_data(n_train: int, n_test: int, n_features: int, seed: int):
    rng = np.random.default_rng(seed)
    y_train = rng.integers(0, 2, size=n_train)
    cols_train = {f"f{i}": rng.normal(0, 1, size=n_train) + (0.5 if i == 0 else 0.0) * y_train for i in range(n_features)}
    cols_test = {f"f{i}": rng.normal(0, 1, size=n_test) + 1.0 for i in range(n_features)}
    return pd.DataFrame(cols_train), y_train, pd.DataFrame(cols_test)


def _run(n_train: int, n_test: int, n_features: int, top_k: int, stability_folds: int | None = None) -> None:
    X_train, y_train, X_test = _make_data(n_train, n_test, n_features, seed=0)
    adversarial_validation_feature_audit(
        X_train,
        y_train,
        X_test,
        top_k_features=top_k,
        seed=0,
        lgbm_params={"n_estimators": 50, "verbosity": -1},
        stability_folds=stability_folds,
    )


if __name__ == "__main__":
    for n_train, n_test, n_features, top_k in [(3_000, 1_000, 10, 5), (20_000, 5_000, 20, 10)]:
        t0 = time.perf_counter()
        _run(n_train, n_test, n_features, top_k)
        wall = time.perf_counter() - t0
        print(f"n_train={n_train:>7,} n_test={n_test:>6,} n_features={n_features:>3} top_k={top_k:>2} -> {wall * 1000:9.2f} ms")

    # stability_folds re-runs the whole pseudo-split ablation N times -- roughly N x the single-split cost
    # (the adversarial-AUC ranking itself is computed once and shared across folds).
    for n_train, n_test, n_features, top_k, stability_folds in [(3_000, 1_000, 10, 5, 5)]:
        t0 = time.perf_counter()
        _run(n_train, n_test, n_features, top_k, stability_folds=stability_folds)
        wall = time.perf_counter() - t0
        print(
            f"n_train={n_train:>7,} n_test={n_test:>6,} n_features={n_features:>3} top_k={top_k:>2} "
            f"stability_folds={stability_folds:>2} -> {wall * 1000:9.2f} ms"
        )

    profiler = cProfile.Profile()
    profiler.enable()
    _run(3_000, 1_000, 10, 5)
    profiler.disable()
    buf = StringIO()
    stats = pstats.Stats(profiler, stream=buf).sort_stats("cumulative")
    stats.print_stats(15)
    print(buf.getvalue())

    profiler_stability = cProfile.Profile()
    profiler_stability.enable()
    _run(3_000, 1_000, 10, 5, stability_folds=5)
    profiler_stability.disable()
    buf_stability = StringIO()
    stats_stability = pstats.Stats(profiler_stability, stream=buf_stability).sort_stats("cumulative")
    stats_stability.print_stats(15)
    print(buf_stability.getvalue())
