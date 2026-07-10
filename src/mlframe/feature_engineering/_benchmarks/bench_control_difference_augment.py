"""cProfile harness for ``feature_engineering.control_difference_augment``.

Run: ``python -m mlframe.feature_engineering._benchmarks.bench_control_difference_augment``
"""
from __future__ import annotations

import cProfile
import pstats
import time
from io import StringIO

import numpy as np
import pandas as pd

from mlframe.feature_engineering.control_difference_augment import control_difference_augment


def _make_data(n_treated: int, n_control: int, n_features: int, seed: int):
    rng = np.random.default_rng(seed)
    cols = [f"f{i}" for i in range(n_features)]
    treated_df = pd.DataFrame(rng.normal(0, 1, (n_treated, n_features)), columns=cols)
    treated_df["y"] = rng.integers(0, 2, n_treated)
    control_df = pd.DataFrame(rng.normal(0, 1, (n_control, n_features)), columns=cols)
    return treated_df, control_df


def _run(n_treated: int, n_control: int, n_features: int, n_augmented_per_treated: int) -> None:
    treated_df, control_df = _make_data(n_treated, n_control, n_features, seed=0)
    control_difference_augment(
        treated_df, control_df, feature_cols=[f"f{i}" for i in range(n_features)], n_augmented_per_treated=n_augmented_per_treated, random_state=0
    )


if __name__ == "__main__":
    for n_treated, n_control, n_features, n_aug in [(1_000, 5_000, 50, 10), (10_000, 20_000, 100, 10)]:
        t0 = time.perf_counter()
        _run(n_treated, n_control, n_features, n_aug)
        wall = time.perf_counter() - t0
        print(f"n_treated={n_treated:>7,} n_control={n_control:>7,} n_features={n_features:>3} n_aug={n_aug:>3} -> {wall * 1000:9.2f} ms")

    profiler = cProfile.Profile()
    profiler.enable()
    _run(10_000, 20_000, 100, 10)
    profiler.disable()
    buf = StringIO()
    stats = pstats.Stats(profiler, stream=buf).sort_stats("cumulative")
    stats.print_stats(15)
    print(buf.getvalue())
