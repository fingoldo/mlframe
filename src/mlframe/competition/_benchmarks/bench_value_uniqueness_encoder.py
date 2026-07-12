"""cProfile benchmark for mlframe.competition.value_uniqueness_encoder.

COMPETITION/EXPLORATORY ONLY - see module docstring under src/mlframe/competition/.
Run: python -m mlframe.competition._benchmarks.bench_value_uniqueness_encoder
"""

from __future__ import annotations

import cProfile
import pstats
import time
from io import StringIO

import numpy as np
import pandas as pd

from mlframe.competition.value_uniqueness_encoder import value_uniqueness_encoder


def _make_bench_dataset(n_train: int, n_test: int, n_cols: int, seed: int = 0) -> tuple:
    rng = np.random.default_rng(seed)
    train = pd.DataFrame(
        {f"col_{j}": rng.integers(0, n_train // 3, size=n_train) for j in range(n_cols)}
    )
    test = pd.DataFrame(
        {f"col_{j}": rng.integers(0, n_train // 3, size=n_test) for j in range(n_cols)}
    )
    real_mask = rng.random(n_test) < 0.8
    y_train = rng.integers(0, 2, size=n_train)
    return train, test, real_mask, y_train


def run_once(n_train: int, n_test: int, n_cols: int) -> float:
    train, test, real_mask, y_train = _make_bench_dataset(n_train, n_test, n_cols)
    columns = list(train.columns)

    start = time.perf_counter()
    value_uniqueness_encoder(train, test, real_test_mask=real_mask, y_train=y_train, columns=columns)
    return time.perf_counter() - start


def main() -> None:
    for n_train, n_test, n_cols in [(5_000, 2_000, 5), (50_000, 20_000, 5), (200_000, 50_000, 3)]:
        elapsed = run_once(n_train, n_test, n_cols)
        print(f"n_train={n_train} n_test={n_test} n_cols={n_cols}: {elapsed:.3f}s wall")

    train, test, real_mask, y_train = _make_bench_dataset(50_000, 20_000, 5)
    columns = list(train.columns)

    profiler = cProfile.Profile()
    profiler.enable()
    value_uniqueness_encoder(train, test, real_test_mask=real_mask, y_train=y_train, columns=columns)
    profiler.disable()

    stream = StringIO()
    stats = pstats.Stats(profiler, stream=stream).sort_stats("cumulative")
    stats.print_stats(30)
    print(stream.getvalue())


if __name__ == "__main__":
    main()
