"""cProfile harness for ``evaluation.cv_informativeness_check``.

Run: ``python -m mlframe.evaluation._benchmarks.bench_cv_informativeness``

Cost is dominated by ``n_folds`` model fits, inherent to a leave-one-group-out sanity check.
"""
from __future__ import annotations

import cProfile
import pstats
import time
from io import StringIO

import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

from mlframe.evaluation.cv_informativeness import cv_informativeness_check


def _neg_rmse(y_true, y_pred):
    return -float(np.sqrt(mean_squared_error(y_true, y_pred)))


def _make_group_splits(group_ids: np.ndarray):
    for g in np.unique(group_ids):
        test_idx = np.flatnonzero(group_ids == g)
        train_idx = np.flatnonzero(group_ids != g)
        yield train_idx, test_idx


def _run(n_groups: int, rows_per_group: int, check_trend: bool = False) -> None:
    rng = np.random.default_rng(0)
    group_ids = np.repeat(np.arange(n_groups), rows_per_group)
    X = rng.normal(0, 1, (n_groups * rows_per_group, 5))
    y = 2.0 * X[:, 0] + rng.normal(0, 0.5, n_groups * rows_per_group)
    cv_informativeness_check(
        X, y, _make_group_splits(group_ids), model_factory=lambda: LinearRegression(), metric_fn=_neg_rmse,
        check_trend=check_trend,
    )


def _run_variable_sizes(group_sizes: list, check_trend: bool = True) -> None:
    # variable-size groups are the realistic shape for the trend diagnostic (constant-size groups give no
    # size axis to correlate against).
    rng = np.random.default_rng(0)
    group_ids = np.repeat(np.arange(len(group_sizes)), group_sizes)
    n = int(sum(group_sizes))
    X = rng.normal(0, 1, (n, 5))
    y = 2.0 * X[:, 0] + rng.normal(0, 0.5, n)
    cv_informativeness_check(
        X, y, _make_group_splits(group_ids), model_factory=lambda: LinearRegression(), metric_fn=_neg_rmse,
        check_trend=check_trend,
    )


if __name__ == "__main__":
    for n_groups, rows_per_group in [(10, 2_000), (20, 5_000)]:
        t0 = time.perf_counter()
        _run(n_groups, rows_per_group)
        wall = time.perf_counter() - t0
        print(f"n_groups={n_groups:>3} rows/group={rows_per_group:>6,} -> {wall * 1000:9.2f} ms")

    t0 = time.perf_counter()
    _run_variable_sizes([500, 1_000, 2_000, 5_000, 10_000], check_trend=True)
    wall = time.perf_counter() - t0
    print(f"check_trend=True variable-size groups -> {wall * 1000:9.2f} ms")

    profiler = cProfile.Profile()
    profiler.enable()
    _run(10, 2_000)
    profiler.disable()
    buf = StringIO()
    stats = pstats.Stats(profiler, stream=buf).sort_stats("cumulative")
    stats.print_stats(15)
    print(buf.getvalue())

    profiler = cProfile.Profile()
    profiler.enable()
    _run_variable_sizes([500, 1_000, 2_000, 5_000, 10_000], check_trend=True)
    profiler.disable()
    buf = StringIO()
    stats = pstats.Stats(profiler, stream=buf).sort_stats("cumulative")
    stats.print_stats(15)
    print("--- check_trend=True profile ---")
    print(buf.getvalue())
