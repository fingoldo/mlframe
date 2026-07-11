"""cProfile harness for ``training.neural.trunk_residual_mlp.TrunkResidualMLPRegressor``.

Run: ``python -m mlframe.training.neural._benchmarks.bench_trunk_residual_mlp``
"""
from __future__ import annotations

import cProfile
import pstats
import time
from io import StringIO

import numpy as np

from mlframe.training.neural.trunk_residual_mlp import TrunkResidualMLPRegressor


def _make_data(n: int, n_features: int, seed: int = 0):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, n_features)).astype(np.float32)
    y = (X[:, :3].sum(axis=1) + rng.normal(scale=0.5, size=n)).astype(np.float32)
    return X, y


def _run_fit(n: int, n_features: int, n_blocks: int, n_epochs: int) -> None:
    X, y = _make_data(n, n_features)
    TrunkResidualMLPRegressor(trunk_dim=12, n_blocks=n_blocks, n_epochs=n_epochs, random_state=0).fit(X, y)


if __name__ == "__main__":
    for n, n_features, n_blocks, n_epochs in [(200, 10, 5, 100), (200, 10, 15, 100), (500, 10, 15, 200)]:
        t0 = time.perf_counter()
        _run_fit(n, n_features, n_blocks, n_epochs)
        wall = time.perf_counter() - t0
        print(f"n={n:>4} n_features={n_features:>3} n_blocks={n_blocks:>3} n_epochs={n_epochs:>4} -> {wall * 1000:9.2f} ms")

    profiler = cProfile.Profile()
    profiler.enable()
    _run_fit(500, 10, 15, 200)
    profiler.disable()
    buf = StringIO()
    stats = pstats.Stats(profiler, stream=buf).sort_stats("cumulative")
    stats.print_stats(15)
    print(buf.getvalue())
