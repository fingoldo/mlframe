"""cProfile harness for ``training.composite.MultiStageMetaFeatureStacker``.

Run: ``python -m mlframe.training.composite._benchmarks.bench_stacking_multi_stage``
"""
from __future__ import annotations

import cProfile
import pstats
import time
from io import StringIO

import numpy as np
from sklearn.linear_model import LinearRegression

from mlframe.training.composite import MultiStageMetaFeatureStacker


def _make_dataset(n: int, seed: int):
    rng = np.random.default_rng(seed)
    z = rng.normal(size=n)
    X = np.column_stack([np.sin(z * 2) + rng.normal(scale=0.3, size=n), z**2 + rng.normal(scale=0.3, size=n), rng.normal(size=n)])
    y_aux = z + rng.normal(scale=0.2, size=n)
    y_primary = 5 * z + rng.normal(scale=0.5, size=n)
    return X, y_primary, y_aux


def _run(n: int) -> None:
    X, y_primary, y_aux = _make_dataset(n, seed=0)
    stacker = MultiStageMetaFeatureStacker(
        stage1_estimator_factories={"aux": lambda: LinearRegression()}, stage2_estimator=LinearRegression(), n_splits=5,
    )
    stacker.fit(X, y_primary, {"aux": y_aux})
    stacker.predict(X)


if __name__ == "__main__":
    for n in [1_000, 10_000, 100_000]:
        t0 = time.perf_counter()
        _run(n)
        wall = time.perf_counter() - t0
        print(f"n={n:>7,} -> {wall * 1000:9.2f} ms")

    profiler = cProfile.Profile()
    profiler.enable()
    _run(100_000)
    profiler.disable()
    buf = StringIO()
    stats = pstats.Stats(profiler, stream=buf).sort_stats("cumulative")
    stats.print_stats(15)
    print(buf.getvalue())
