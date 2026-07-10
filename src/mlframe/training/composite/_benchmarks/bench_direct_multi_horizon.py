"""cProfile harness for ``training.composite.DirectMultiHorizonEnsemble``.

Run: ``python -m mlframe.training.composite._benchmarks.bench_direct_multi_horizon``
"""
from __future__ import annotations

import cProfile
import pstats
import time
from io import StringIO

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

from mlframe.training.composite import DirectMultiHorizonEnsemble


def _make_dataset(n: int, horizon: int, seed: int):
    rng = np.random.default_rng(seed)
    z0 = rng.normal(size=n)
    X0 = pd.DataFrame({"x": z0 + rng.normal(scale=0.3, size=n)})
    Z = np.zeros((n, horizon + 1))
    Z[:, 0] = z0
    for h in range(1, horizon + 1):
        Z[:, h] = 0.9 * Z[:, h - 1] + rng.normal(scale=0.25, size=n)
    Y = Z[:, 1:]
    return X0, Y


def _run(n: int, horizon: int) -> None:
    X0, Y = _make_dataset(n, horizon, seed=0)
    est = DirectMultiHorizonEnsemble(estimator_factory=lambda: LinearRegression(), horizon_blocks=[[h] for h in range(horizon)])
    est.fit(X0, Y)
    est.predict(X0)


if __name__ == "__main__":
    for n, horizon in [(1_000, 12), (10_000, 12), (100_000, 28)]:
        t0 = time.perf_counter()
        _run(n, horizon)
        wall = time.perf_counter() - t0
        print(f"n={n:>7,} horizon={horizon:>3} -> {wall * 1000:9.2f} ms")

    profiler = cProfile.Profile()
    profiler.enable()
    _run(100_000, 28)
    profiler.disable()
    buf = StringIO()
    stats = pstats.Stats(profiler, stream=buf).sort_stats("cumulative")
    stats.print_stats(15)
    print(buf.getvalue())
