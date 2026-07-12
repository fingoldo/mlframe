"""cProfile harness for ``evaluation.compare_cv_schemes``.

Run: ``python -m mlframe.evaluation._benchmarks.bench_compare_cv_schemes``

Cost is dominated by ``(n_folds_per_scheme * n_schemes) + 1`` model fits, inherent to the CV-comparison
methodology.
"""
from __future__ import annotations

import cProfile
import pstats
import time
from io import StringIO
from typing import Dict, Iterable, Optional, Tuple

import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold

from mlframe.evaluation.compare_cv_schemes import compare_cv_schemes


def _rmse(y_true, y_pred):
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def _run(n_rows: int, n_splits: int, significance_alpha: Optional[float] = None) -> None:
    rng = np.random.default_rng(0)
    X = rng.normal(0, 1, (n_rows, 5))
    y = X[:, 0] * 2 + rng.normal(0, 0.5, n_rows)

    hist_idx = np.arange(int(n_rows * 0.8))
    future_idx = np.arange(int(n_rows * 0.8), n_rows)
    kfold_splits = [(hist_idx[tr], hist_idx[te]) for tr, te in KFold(n_splits, shuffle=True, random_state=0).split(hist_idx)]
    schemes: Dict[str, Iterable[Tuple[np.ndarray, np.ndarray]]] = {"kfold": kfold_splits}
    if significance_alpha is not None:
        # A second scheme (different random_state) is required for the significance path to have anything to
        # pair-test the point-estimate winner against.
        kfold_splits_b = [(hist_idx[tr], hist_idx[te]) for tr, te in KFold(n_splits, shuffle=True, random_state=1).split(hist_idx)]
        schemes["kfold_b"] = kfold_splits_b

    compare_cv_schemes(
        X, y, schemes=schemes, ooo_time_idx=(hist_idx, future_idx),
        model_factory=lambda: RandomForestRegressor(n_estimators=50, max_depth=6, random_state=0), metric_fn=_rmse,
        significance_alpha=significance_alpha,
    )


if __name__ == "__main__":
    for n_rows, n_splits in [(2_000, 5), (20_000, 5)]:
        t0 = time.perf_counter()
        _run(n_rows, n_splits)
        wall = time.perf_counter() - t0
        print(f"n_rows={n_rows:>7,} n_splits={n_splits} -> {wall * 1000:9.2f} ms")

    for n_rows, n_splits in [(2_000, 5), (20_000, 5)]:
        t0 = time.perf_counter()
        _run(n_rows, n_splits, significance_alpha=0.05)
        wall = time.perf_counter() - t0
        print(f"n_rows={n_rows:>7,} n_splits={n_splits} significance_alpha=0.05 -> {wall * 1000:9.2f} ms")

    profiler = cProfile.Profile()
    profiler.enable()
    _run(2_000, 5)
    profiler.disable()
    buf = StringIO()
    stats = pstats.Stats(profiler, stream=buf).sort_stats("cumulative")
    stats.print_stats(15)
    print(buf.getvalue())

    profiler = cProfile.Profile()
    profiler.enable()
    _run(2_000, 5, significance_alpha=0.05)
    profiler.disable()
    buf = StringIO()
    stats = pstats.Stats(profiler, stream=buf).sort_stats("cumulative")
    stats.print_stats(15)
    print(buf.getvalue())
