"""cProfile harness for ``evaluation.detect_expanding_window_feature_leakage``.

Run: ``python -m mlframe.evaluation._benchmarks.bench_expanding_window_leakage``
"""
from __future__ import annotations

import cProfile
import pstats
import time
from io import StringIO

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

from mlframe.evaluation import detect_expanding_window_feature_leakage


def _make_dataset(n: int, n_cats: int, seed: int):
    rng = np.random.default_rng(seed)
    cat_rate = rng.uniform(0.5, 5.0, n_cats)
    cat = rng.choice(n_cats, size=n, p=cat_rate / cat_rate.sum())
    y = cat_rate[cat] * 3.0 + rng.normal(scale=1.0, size=n)
    return pd.DataFrame({"t": np.arange(n, dtype=float), "cat": cat}), y


def _fit_transform(fit_df: pd.DataFrame, transform_df: pd.DataFrame) -> np.ndarray:
    counts = fit_df["cat"].value_counts()
    return np.asarray(transform_df["cat"].map(counts).fillna(0).to_numpy(dtype=np.float64))


def _run(n: int, n_cats: int) -> None:
    df, y = _make_dataset(n, n_cats, seed=0)
    detect_expanding_window_feature_leakage(df, "t", y, _fit_transform, lambda: LinearRegression(), n_splits=5, scoring="r2")


if __name__ == "__main__":
    for n, n_cats in [(2_000, 20), (20_000, 30), (100_000, 50)]:
        t0 = time.perf_counter()
        _run(n, n_cats)
        wall = time.perf_counter() - t0
        print(f"n={n:>8,} n_cats={n_cats:>3} -> {wall * 1000:9.2f} ms")

    profiler = cProfile.Profile()
    profiler.enable()
    _run(100_000, 50)
    profiler.disable()
    buf = StringIO()
    stats = pstats.Stats(profiler, stream=buf).sort_stats("cumulative")
    stats.print_stats(15)
    print(buf.getvalue())
