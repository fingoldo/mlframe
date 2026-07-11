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
    df = pd.DataFrame({f"col_{c}": rng.choice(cats + ["__unseen__"], n) for c in range(5)})
    df["val"] = rng.normal(size=n)
    return df


def _run(n: int, n_categories: int, n_calls: int, similarity_mode: str = "mode") -> None:
    train_df = _make_frame(n, n_categories, seed=0)
    test_df = _make_frame(n, n_categories, seed=1)
    columns = [c for c in train_df.columns if c != "val"]
    for _ in range(n_calls):
        if similarity_mode == "nearest":
            UnseenCategoryImputer(columns=columns, similarity_mode="nearest", value_column="val").fit(train_df).transform(test_df)
        else:
            UnseenCategoryImputer(columns=columns).fit(train_df).transform(test_df)


if __name__ == "__main__":
    for similarity_mode in ("mode", "nearest"):
        for n, n_categories, n_calls in [(2000, 50, 20), (200000, 50, 20), (200000, 500, 20)]:
            t0 = time.perf_counter()
            _run(n, n_categories, n_calls, similarity_mode)
            wall = time.perf_counter() - t0
            print(f"mode={similarity_mode:<8} n={n:>7} n_categories={n_categories:>4} n_calls={n_calls:>4} -> {wall * 1000:9.2f} ms")

    for similarity_mode in ("mode", "nearest"):
        profiler = cProfile.Profile()
        profiler.enable()
        _run(200000, 500, 20, similarity_mode)
        profiler.disable()
        buf = StringIO()
        stats = pstats.Stats(profiler, stream=buf).sort_stats("cumulative")
        stats.print_stats(15)
        print(f"--- cProfile similarity_mode={similarity_mode} ---")
        print(buf.getvalue())
