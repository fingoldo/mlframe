"""cProfile harness for ``feature_selection.drop_near_noise_univariate_auc.drop_near_noise_univariate_auc``.

Run: ``python -m mlframe.feature_selection._benchmarks.bench_drop_near_noise_univariate_auc``
"""
from __future__ import annotations

import cProfile
import pstats
import time
from io import StringIO

from mlframe.feature_selection.drop_near_noise_univariate_auc import drop_near_noise_univariate_auc
from mlframe._bench_data_shared import make_binary_noise_dataset as _make_dataset


def _run(n_rows: int, n_cols: int) -> None:
    df, y = _make_dataset(n_rows, n_cols, seed=0)
    drop_near_noise_univariate_auc(df, y, tolerance=0.03)


def _run_bootstrap(n_rows: int, n_cols: int, n_bootstrap: int) -> None:
    df, y = _make_dataset(n_rows, n_cols, seed=0)
    drop_near_noise_univariate_auc(df, y, tolerance=0.03, n_bootstrap=n_bootstrap, random_state=0)


if __name__ == "__main__":
    for n_rows, n_cols in [(5000, 50), (50000, 50), (50000, 500)]:
        t0 = time.perf_counter()
        _run(n_rows, n_cols)
        wall = time.perf_counter() - t0
        print(f"single-pass n_rows={n_rows:>6} n_cols={n_cols:>4} -> {wall * 1000:9.2f} ms")

    for n_rows, n_cols, n_bootstrap in [(5000, 50, 20), (50000, 50, 20), (50000, 500, 20)]:
        t0 = time.perf_counter()
        _run_bootstrap(n_rows, n_cols, n_bootstrap)
        wall = time.perf_counter() - t0
        print(f"bootstrap({n_bootstrap:>2}) n_rows={n_rows:>6} n_cols={n_cols:>4} -> {wall * 1000:9.2f} ms")

    profiler = cProfile.Profile()
    profiler.enable()
    _run(50000, 500)
    profiler.disable()
    buf = StringIO()
    stats = pstats.Stats(profiler, stream=buf).sort_stats("cumulative")
    stats.print_stats(15)
    print("single-pass profile:")
    print(buf.getvalue())

    profiler = cProfile.Profile()
    profiler.enable()
    _run_bootstrap(50000, 500, 20)
    profiler.disable()
    buf = StringIO()
    stats = pstats.Stats(profiler, stream=buf).sort_stats("cumulative")
    stats.print_stats(15)
    print("bootstrap-stability profile:")
    print(buf.getvalue())
