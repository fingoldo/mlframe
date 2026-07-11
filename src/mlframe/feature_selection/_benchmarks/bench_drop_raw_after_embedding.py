"""cProfile harness for ``feature_selection.drop_raw_after_embedding.drop_raw_after_embedding``.

Run: ``python -m mlframe.feature_selection._benchmarks.bench_drop_raw_after_embedding``
"""
from __future__ import annotations

import cProfile
import pstats
import time
from io import StringIO

import numpy as np
import pandas as pd

from mlframe.feature_selection.drop_raw_after_embedding import drop_raw_after_embedding


def _make_dataset(n_rows: int, n_raw_cols: int, n_derived_per_raw: int, seed: int):
    rng = np.random.default_rng(seed)
    data = {}
    raw_to_derived = {}
    for r in range(n_raw_cols):
        raw_name = f"raw_{r}"
        data[raw_name] = rng.integers(0, 500, n_rows).astype(str)
        derived_names = []
        for d in range(n_derived_per_raw):
            derived_name = f"raw_{r}_enc_{d}"
            data[derived_name] = rng.normal(size=n_rows)
            derived_names.append(derived_name)
        raw_to_derived[raw_name] = derived_names
    return pd.DataFrame(data), raw_to_derived


def _run(n_rows: int, n_raw_cols: int, n_derived_per_raw: int, verify: bool = False) -> None:
    df, raw_to_derived = _make_dataset(n_rows, n_raw_cols, n_derived_per_raw, seed=0)
    if verify:
        rng = np.random.default_rng(0)
        y = rng.integers(0, 2, n_rows)
        drop_raw_after_embedding(df, raw_to_derived, verify_against=(y, 0.5))
    else:
        drop_raw_after_embedding(df, raw_to_derived)


if __name__ == "__main__":
    for n_rows, n_raw_cols, n_derived_per_raw in [(50000, 10, 2), (500000, 10, 2), (500000, 50, 3)]:
        t0 = time.perf_counter()
        _run(n_rows, n_raw_cols, n_derived_per_raw)
        wall = time.perf_counter() - t0
        print(f"n_rows={n_rows:>7} n_raw_cols={n_raw_cols:>3} n_derived_per_raw={n_derived_per_raw:>2} -> {wall * 1000:9.2f} ms")

    for n_rows, n_raw_cols, n_derived_per_raw in [(50000, 10, 2), (500000, 10, 2), (500000, 50, 3)]:
        t0 = time.perf_counter()
        _run(n_rows, n_raw_cols, n_derived_per_raw, verify=True)
        wall = time.perf_counter() - t0
        print(f"[verify_against] n_rows={n_rows:>7} n_raw_cols={n_raw_cols:>3} n_derived_per_raw={n_derived_per_raw:>2} -> {wall * 1000:9.2f} ms")

    profiler = cProfile.Profile()
    profiler.enable()
    _run(500000, 50, 3)
    profiler.disable()
    buf = StringIO()
    stats = pstats.Stats(profiler, stream=buf).sort_stats("cumulative")
    stats.print_stats(15)
    print(buf.getvalue())

    profiler_verify = cProfile.Profile()
    profiler_verify.enable()
    _run(500000, 50, 3, verify=True)
    profiler_verify.disable()
    buf_verify = StringIO()
    stats_verify = pstats.Stats(profiler_verify, stream=buf_verify).sort_stats("cumulative")
    stats_verify.print_stats(15)
    print(buf_verify.getvalue())
