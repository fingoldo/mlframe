"""cProfile harness for ``training.composite.dual_direction.DualDirectionCompositeEstimator``.

Run: ``python -m mlframe.training.composite._benchmarks.bench_dual_direction``
"""
from __future__ import annotations

import cProfile
import pstats
import time
from io import StringIO

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

from mlframe.training.composite.dual_direction import DualDirectionCompositeEstimator


def _make_data(n: int, seed: int):
    rng = np.random.default_rng(seed)
    x1 = rng.uniform(0, 1, n)
    x2 = rng.uniform(0, 1, n)
    shape = np.clip(0.1 + 0.8 * x1, 0, 1)
    scale = 10.0 + 20.0 * x2
    y = shape * scale + rng.normal(scale=0.3, size=n)
    df = pd.DataFrame({"x1": x1, "x2": x2})
    return df, y, scale


def _run(n: int, n_splits: int) -> None:
    df, y, scale = _make_data(n, seed=0)
    est = DualDirectionCompositeEstimator(scale_estimator=Ridge(), shape_estimator=Ridge(), n_splits=n_splits)
    est.fit(df, y, scale)
    est.predict(df)


if __name__ == "__main__":
    for n, n_splits in [(2000, 5), (50000, 5), (50000, 10)]:
        t0 = time.perf_counter()
        _run(n, n_splits)
        wall = time.perf_counter() - t0
        print(f"n={n:>6} n_splits={n_splits:>2} -> {wall * 1000:9.2f} ms")

    profiler = cProfile.Profile()
    profiler.enable()
    _run(50000, 10)
    profiler.disable()
    buf = StringIO()
    stats = pstats.Stats(profiler, stream=buf).sort_stats("cumulative")
    stats.print_stats(15)
    print(buf.getvalue())
