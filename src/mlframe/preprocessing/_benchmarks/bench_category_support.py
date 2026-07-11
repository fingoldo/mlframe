"""cProfile harness for ``preprocessing.category_support.train_test_support_screen``.

Run: ``python -m mlframe.preprocessing._benchmarks.bench_category_support``
"""
from __future__ import annotations

import cProfile
import pstats
import time
from io import StringIO

import numpy as np
import pandas as pd

from mlframe.preprocessing.category_support import train_test_support_screen


def _make_data(n: int, n_cols: int, n_cats: int, seed: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    train = {f"col_{i}": rng.integers(0, n_cats, size=n) for i in range(n_cols)}
    test = {f"col_{i}": rng.integers(0, n_cats, size=n) for i in range(n_cols)}
    return pd.DataFrame(train), pd.DataFrame(test)


def _make_data_with_target(n: int, n_cols: int, n_cats: int, seed: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Near-uniform per-category counts (frequency encoding would collapse) plus a shared target column, so the
    ``enable_smoothed_target_encoding_fallback`` path is actually exercised (not just the base screen)."""
    rng = np.random.default_rng(seed)
    train = {f"col_{i}": rng.integers(0, n_cats, size=n) for i in range(n_cols)}
    test = {f"col_{i}": rng.integers(0, n_cats, size=n) for i in range(n_cols)}
    train["y"] = rng.integers(0, 2, size=n).astype(float)
    test["y"] = rng.integers(0, 2, size=n).astype(float)
    return pd.DataFrame(train), pd.DataFrame(test)


def _run(n: int, n_cols: int, n_cats: int, n_calls: int) -> None:
    train_df, test_df = _make_data(n, n_cols, n_cats, seed=0)
    for _ in range(n_calls):
        train_test_support_screen(train_df, test_df)


def _run_with_target_encoding_fallback(n: int, n_cols: int, n_cats: int, n_calls: int) -> None:
    train_df, test_df = _make_data_with_target(n, n_cols, n_cats, seed=0)
    cols = [c for c in train_df.columns if c != "y"]
    for _ in range(n_calls):
        train_test_support_screen(
            train_df,
            test_df,
            categorical_cols=cols,
            target_col="y",
            enable_smoothed_target_encoding_fallback=True,
        )


if __name__ == "__main__":
    for n, n_cols, n_cats, n_calls in [(10_000, 20, 100, 20), (500_000, 50, 1000, 3)]:
        t0 = time.perf_counter()
        _run(n, n_cols, n_cats, n_calls)
        wall = time.perf_counter() - t0
        print(f"n={n:>9,} cols={n_cols:>3} cats={n_cats:>5} -> {wall * 1000:9.2f} ms total, {wall / n_calls * 1000:8.2f} ms/call")

    for n, n_cols, n_cats, n_calls in [(10_000, 20, 100, 20), (500_000, 50, 1000, 3)]:
        t0 = time.perf_counter()
        _run_with_target_encoding_fallback(n, n_cols, n_cats, n_calls)
        wall = time.perf_counter() - t0
        print(
            f"[target_encoding_fallback] n={n:>9,} cols={n_cols:>3} cats={n_cats:>5} -> "
            f"{wall * 1000:9.2f} ms total, {wall / n_calls * 1000:8.2f} ms/call"
        )

    profiler = cProfile.Profile()
    profiler.enable()
    _run(500_000, 50, 1000, 5)
    _run_with_target_encoding_fallback(500_000, 50, 1000, 5)
    profiler.disable()
    buf = StringIO()
    stats = pstats.Stats(profiler, stream=buf).sort_stats("cumulative")
    stats.print_stats(20)
    print(buf.getvalue())
