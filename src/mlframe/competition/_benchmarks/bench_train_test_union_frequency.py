"""cProfile benchmark for mlframe.competition.train_test_union_frequency.

COMPETITION/EXPLORATORY ONLY - see module docstring under src/mlframe/competition/.
Run: python -m mlframe.competition._benchmarks.bench_train_test_union_frequency
"""

from __future__ import annotations

import cProfile
import pstats
import time
from io import StringIO

import numpy as np
import pandas as pd

from mlframe.competition.train_test_union_frequency import train_test_union_frequency_encode


def _make_bench_version_series(n_train: int, n_test: int, n_versions: int, seed: int = 0) -> tuple:
    rng = np.random.default_rng(seed)
    majors = rng.integers(1, 6, size=n_versions)
    minors = rng.integers(0, 10, size=n_versions)
    patches = rng.integers(0, 10, size=n_versions)
    versions = np.array([f"{ma}.{mi}.{pa}" for ma, mi, pa in zip(majors, minors, patches)])

    train_series = pd.Series(rng.choice(versions, size=n_train))
    test_series = pd.Series(rng.choice(versions, size=n_test))
    return train_series, test_series


def run_once(n_train: int, n_test: int, n_versions: int, hierarchical: bool) -> float:
    train_series, test_series = _make_bench_version_series(n_train, n_test, n_versions)

    start = time.perf_counter()
    train_test_union_frequency_encode(
        train_series, test_series, hierarchical_split_sep="." if hierarchical else None
    )
    return time.perf_counter() - start


def main() -> None:
    for n_train, n_test, n_versions in [(5_000, 2_000, 50), (50_000, 20_000, 200), (500_000, 200_000, 1000)]:
        elapsed_flat = run_once(n_train, n_test, n_versions, hierarchical=False)
        elapsed_hier = run_once(n_train, n_test, n_versions, hierarchical=True)
        print(
            f"n_train={n_train} n_test={n_test} n_versions={n_versions}: "
            f"flat={elapsed_flat:.3f}s hierarchical={elapsed_hier:.3f}s"
        )

    train_series, test_series = _make_bench_version_series(50_000, 20_000, 200)

    profiler = cProfile.Profile()
    profiler.enable()
    train_test_union_frequency_encode(train_series, test_series, hierarchical_split_sep=".")
    profiler.disable()

    stream = StringIO()
    stats = pstats.Stats(profiler, stream=stream).sort_stats("cumulative")
    stats.print_stats(30)
    print(stream.getvalue())


if __name__ == "__main__":
    main()
