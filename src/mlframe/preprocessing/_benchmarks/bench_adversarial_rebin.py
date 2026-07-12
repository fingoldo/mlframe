"""cProfile harness for ``preprocessing.adversarial_rebin_categorical``.

Run: ``python -m mlframe.preprocessing._benchmarks.bench_adversarial_rebin``
"""
from __future__ import annotations

import cProfile
import pstats
import time
from io import StringIO

import numpy as np
import pandas as pd

from mlframe.preprocessing.adversarial_rebin import adversarial_rebin_categorical


def _make_data(n_rows: int, n_cats: int, seed: int):
    rng = np.random.default_rng(seed)
    cats = [f"cat_{i}" for i in range(n_cats)]
    train = pd.Series(rng.choice(cats, size=n_rows))
    test = pd.Series(rng.choice(cats, size=n_rows))
    return train, test


def _run(n_rows: int, n_cats: int) -> None:
    train, test = _make_data(n_rows, n_cats, seed=0)
    adversarial_rebin_categorical(train, test)


def _make_numeric_data(n_rows: int, seed: int):
    rng = np.random.default_rng(seed)
    train = pd.Series(rng.normal(size=n_rows))
    test = pd.Series(rng.normal(size=n_rows))
    return train, test


def _run_continuous(n_rows: int, n_quantile_bins: int) -> None:
    train, test = _make_numeric_data(n_rows, seed=0)
    adversarial_rebin_categorical(train, test, mode="continuous", n_quantile_bins=n_quantile_bins)


if __name__ == "__main__":
    for n_rows, n_cats in [(20_000, 500), (500_000, 2_000)]:
        t0 = time.perf_counter()
        _run(n_rows, n_cats)
        wall = time.perf_counter() - t0
        print(f"n_rows={n_rows:>9,} n_cats={n_cats:>6,} -> {wall * 1000:9.2f} ms")

    for n_rows, n_quantile_bins in [(20_000, 20), (500_000, 50)]:
        t0 = time.perf_counter()
        _run_continuous(n_rows, n_quantile_bins)
        wall = time.perf_counter() - t0
        print(f"[continuous] n_rows={n_rows:>9,} n_quantile_bins={n_quantile_bins:>4} -> {wall * 1000:9.2f} ms")

    profiler = cProfile.Profile()
    profiler.enable()
    _run(500_000, 2_000)
    profiler.disable()
    buf = StringIO()
    stats = pstats.Stats(profiler, stream=buf).sort_stats("cumulative")
    stats.print_stats(15)
    print(buf.getvalue())

    profiler = cProfile.Profile()
    profiler.enable()
    _run_continuous(500_000, 50)
    profiler.disable()
    buf = StringIO()
    stats = pstats.Stats(profiler, stream=buf).sort_stats("cumulative")
    stats.print_stats(15)
    print("[continuous]")
    print(buf.getvalue())
