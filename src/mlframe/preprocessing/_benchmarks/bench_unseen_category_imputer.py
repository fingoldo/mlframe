"""cProfile harness for ``preprocessing.unseen_category_imputer.UnseenCategoryImputer``.

Run: ``python -m mlframe.preprocessing._benchmarks.bench_unseen_category_imputer``
"""
from __future__ import annotations

import cProfile
import pstats
import time
from io import StringIO

import numpy as np
import pandas as pd

from mlframe.preprocessing.unseen_category_imputer import UnseenCategoryImputer


def _make_frame(n: int, n_categories: int, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    cats = [f"cat_{i}" for i in range(n_categories)]
    return pd.DataFrame({f"col_{c}": rng.choice(cats + ["__unseen__"], n) for c in range(5)})


def _run(n: int, n_categories: int, n_calls: int) -> None:
    train_df = _make_frame(n, n_categories, seed=0)
    test_df = _make_frame(n, n_categories, seed=1)
    columns = list(train_df.columns)
    for _ in range(n_calls):
        UnseenCategoryImputer(columns=columns).fit(train_df).transform(test_df)


if __name__ == "__main__":
    for n, n_categories, n_calls in [(2000, 50, 20), (200000, 50, 20), (200000, 500, 20)]:
        t0 = time.perf_counter()
        _run(n, n_categories, n_calls)
        wall = time.perf_counter() - t0
        print(f"n={n:>7} n_categories={n_categories:>4} n_calls={n_calls:>4} -> {wall * 1000:9.2f} ms")

    profiler = cProfile.Profile()
    profiler.enable()
    _run(200000, 500, 20)
    profiler.disable()
    buf = StringIO()
    stats = pstats.Stats(profiler, stream=buf).sort_stats("cumulative")
    stats.print_stats(15)
    print(buf.getvalue())
