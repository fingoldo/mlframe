"""cProfile harness for ``preprocessing.outlier_detector_zoo.make_outlier_detector``.

Run: ``python -m mlframe.preprocessing._benchmarks.bench_outlier_detector_zoo``
"""
from __future__ import annotations

import cProfile
import pstats
import time
from io import StringIO

import numpy as np

from mlframe.preprocessing.outlier_detector_zoo import make_ensemble_outlier_scores, make_outlier_detector, select_outlier_threshold


def _make_data(n_rows: int, n_cols: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.normal(size=(n_rows, n_cols))


def _run(n_rows: int, n_cols: int, method: str) -> None:
    X = _make_data(n_rows, n_cols, seed=0)
    detector = make_outlier_detector(method)
    detector.fit_predict(X)


def _run_ensemble(n_rows: int, n_cols: int) -> None:
    X = _make_data(n_rows, n_cols, seed=0)
    make_ensemble_outlier_scores(X, methods=("isolation_forest", "lof"), random_state=0)


def _run_threshold(n_rows: int, n_cols: int, method: str) -> None:
    X = _make_data(n_rows, n_cols, seed=0)
    detector = make_outlier_detector("isolation_forest", random_state=0)
    detector.fit(X)
    scores = -detector.decision_function(X)
    select_outlier_threshold(scores, method=method, contamination=0.1, percentile=90.0)


if __name__ == "__main__":
    for method in ("isolation_forest", "lof"):
        for n_rows, n_cols in [(5000, 10), (50000, 10)]:
            t0 = time.perf_counter()
            _run(n_rows, n_cols, method)
            wall = time.perf_counter() - t0
            print(f"method={method:>16} n_rows={n_rows:>6} n_cols={n_cols:>3} -> {wall * 1000:9.2f} ms")

    # Ensemble mode pays for its slowest member (LOF's O(n*k) k-NN search) -- wall time is roughly
    # IsolationForest + LOF combined, not free. Measured to confirm the cost story, not to optimize it away:
    # the value is detection quality (see the biz_value test), and callers who need to stay fast at large n
    # should use "lof"/"isolation_forest" alone instead of "ensemble".
    for n_rows, n_cols in [(5000, 10), (50000, 10)]:
        t0 = time.perf_counter()
        _run_ensemble(n_rows, n_cols)
        wall = time.perf_counter() - t0
        print(f"method={'ensemble':>16} n_rows={n_rows:>6} n_cols={n_cols:>3} -> {wall * 1000:9.2f} ms")

    # select_outlier_threshold is a pure numpy score->flag conversion (argpartition/percentile), several
    # orders of magnitude cheaper than fitting the detector that produced the scores -- measured to confirm
    # it never becomes the bottleneck, not to optimize it.
    for method in ("contamination", "percentile", "iqr"):
        for n_rows, n_cols in [(5000, 10), (50000, 10)]:
            t0 = time.perf_counter()
            _run_threshold(n_rows, n_cols, method)
            wall = time.perf_counter() - t0
            print(f"method={'threshold_' + method:>16} n_rows={n_rows:>6} n_cols={n_cols:>3} -> {wall * 1000:9.2f} ms")

    profiler = cProfile.Profile()
    profiler.enable()
    _run(50000, 10, "isolation_forest")
    profiler.disable()
    buf = StringIO()
    stats = pstats.Stats(profiler, stream=buf).sort_stats("cumulative")
    stats.print_stats(15)
    print(buf.getvalue())
