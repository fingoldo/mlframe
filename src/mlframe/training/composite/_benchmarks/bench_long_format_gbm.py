"""cProfile harness for ``training.composite.melt_to_long_gbm_features``.

Run: ``python -m mlframe.training.composite._benchmarks.bench_long_format_gbm``
"""
from __future__ import annotations

import cProfile
import pstats
import time
from io import StringIO

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

from mlframe.training.composite import melt_to_long_gbm_features


def _make_dataset(n: int, d: int, seed: int):
    rng = np.random.default_rng(seed)
    X = pd.DataFrame(rng.normal(size=(n, d)), columns=[f"f{j}" for j in range(d)])
    y = rng.normal(size=n)
    return X, y


def _run(n: int, d: int, context_columns: list[str] | None = None) -> None:
    X, y = _make_dataset(n, d, seed=0)
    melt_to_long_gbm_features(X, y, model_factory=lambda: LinearRegression(), n_splits=5, context_columns=context_columns)


if __name__ == "__main__":
    for n, d in [(500, 20), (2_000, 50), (5_000, 100)]:
        t0 = time.perf_counter()
        _run(n, d)
        wall = time.perf_counter() - t0
        print(f"n={n:>7,} d={d:>4} (long rows={n*d:>9,}) -> {wall * 1000:9.2f} ms  [pure long]")

    # context_columns path -- broadcasts a handful of companion feature values onto every long-format row
    # (see the module docstring's additive-target SNR-loss fix); profile it separately since it adds a
    # per-row join/concat the pure path doesn't pay.
    for n, d in [(500, 20), (2_000, 50), (5_000, 100)]:
        ctx_cols = [f"f{j}" for j in range(min(5, d))]
        t0 = time.perf_counter()
        _run(n, d, context_columns=ctx_cols)
        wall = time.perf_counter() - t0
        print(f"n={n:>7,} d={d:>4} (long rows={n*d:>9,}) -> {wall * 1000:9.2f} ms  [context_columns={len(ctx_cols)}]")

    profiler = cProfile.Profile()
    profiler.enable()
    _run(5_000, 100)
    profiler.disable()
    buf = StringIO()
    stats = pstats.Stats(profiler, stream=buf).sort_stats("cumulative")
    stats.print_stats(15)
    print("pure long format:")
    print(buf.getvalue())

    profiler_ctx = cProfile.Profile()
    profiler_ctx.enable()
    _run(5_000, 100, context_columns=[f"f{j}" for j in range(5)])
    profiler_ctx.disable()
    buf_ctx = StringIO()
    stats_ctx = pstats.Stats(profiler_ctx, stream=buf_ctx).sort_stats("cumulative")
    stats_ctx.print_stats(15)
    print("context_columns-augmented:")
    print(buf_ctx.getvalue())
