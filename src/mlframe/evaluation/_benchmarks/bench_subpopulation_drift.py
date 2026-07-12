"""cProfile harness for ``evaluation.subpopulation_drift.subpopulation_ratio_drift_check``.

Run: ``python -m mlframe.evaluation._benchmarks.bench_subpopulation_drift``
"""
from __future__ import annotations

import cProfile
import pstats
import time
from io import StringIO

import numpy as np
import pandas as pd

from mlframe.evaluation.subpopulation_drift import rank_subpopulation_drift_severity, subpopulation_ratio_drift_check


def _run(n: int, n_cats: int, n_calls: int) -> None:
    rng = np.random.default_rng(0)
    train_df = pd.DataFrame({"col": rng.integers(0, n_cats, size=n)})
    test_df = pd.DataFrame({"col": rng.integers(0, n_cats, size=n)})
    for _ in range(n_calls):
        subpopulation_ratio_drift_check(train_df, test_df, subgroup_col="col")


def _run_severity_score(n: int, n_cats: int, n_calls: int) -> None:
    rng = np.random.default_rng(0)
    train_df = pd.DataFrame({"col": rng.integers(0, n_cats, size=n)})
    test_df = pd.DataFrame({"col": rng.integers(0, n_cats, size=n)})
    for _ in range(n_calls):
        subpopulation_ratio_drift_check(train_df, test_df, subgroup_col="col", include_severity_score=True)


def _run_rank_multi_col(n: int, n_cats: int, n_cols: int, n_calls: int) -> None:
    rng = np.random.default_rng(0)
    cols = [f"col_{i}" for i in range(n_cols)]
    train_df = pd.DataFrame({c: rng.integers(0, n_cats, size=n) for c in cols})
    test_df = pd.DataFrame({c: rng.integers(0, n_cats, size=n) for c in cols})
    for _ in range(n_calls):
        rank_subpopulation_drift_severity(train_df, test_df, subgroup_cols=cols)


if __name__ == "__main__":
    for n, n_cats, n_calls in [(10_000, 5, 200), (1_000_000, 20, 20)]:
        t0 = time.perf_counter()
        _run(n, n_cats, n_calls)
        wall = time.perf_counter() - t0
        print(f"n={n:>9,} cats={n_cats:>3} -> {wall * 1000:9.2f} ms total, {wall / n_calls * 1000:8.3f} ms/call")

    for n, n_cats, n_calls in [(10_000, 5, 200), (1_000_000, 20, 20)]:
        t0 = time.perf_counter()
        _run_severity_score(n, n_cats, n_calls)
        wall = time.perf_counter() - t0
        print(f"[severity_score] n={n:>9,} cats={n_cats:>3} -> {wall * 1000:9.2f} ms total, {wall / n_calls * 1000:8.3f} ms/call")

    for n, n_cats, n_cols, n_calls in [(10_000, 5, 10, 50), (1_000_000, 20, 10, 5)]:
        t0 = time.perf_counter()
        _run_rank_multi_col(n, n_cats, n_cols, n_calls)
        wall = time.perf_counter() - t0
        print(
            f"[rank_multi_col] n={n:>9,} cats={n_cats:>3} cols={n_cols:>3} -> "
            f"{wall * 1000:9.2f} ms total, {wall / n_calls * 1000:8.3f} ms/call"
        )

    profiler = cProfile.Profile()
    profiler.enable()
    _run(1_000_000, 20, 30)
    _run_severity_score(1_000_000, 20, 30)
    _run_rank_multi_col(1_000_000, 20, 10, 3)
    profiler.disable()
    buf = StringIO()
    stats = pstats.Stats(profiler, stream=buf).sort_stats("cumulative")
    stats.print_stats(15)
    print(buf.getvalue())
