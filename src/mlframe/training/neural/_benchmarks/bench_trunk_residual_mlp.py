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


def _run_seed_ensemble(n: int, n_features: int, n_blocks: int, n_epochs: int, n_seeds: int) -> None:
    """Profile the opt-in seed-ensemble path (fit_seed_ensemble + predict_std + variance curve) -- ``n_seeds``
    sequential fits dominate wall time here (each an independent full-batch training loop); the point is to
    confirm the diagnostic-curve bookkeeping around them stays negligible in comparison.
    """
    X, y = _make_data(n, n_features)
    model = TrunkResidualMLPRegressor(trunk_dim=12, n_blocks=n_blocks, n_epochs=n_epochs, random_state=0)
    model.fit_seed_ensemble(X, y, n_seeds=n_seeds, base_random_state=0)
    model.predict_std(X)
    TrunkResidualMLPRegressor.seed_ensemble_variance_curve(
        X, y, X, k_values=(1, 2, 4, 8), base_random_state=0, trunk_dim=12, n_blocks=n_blocks, n_epochs=n_epochs,
    )


if __name__ == "__main__":
    for n, n_features, n_blocks, n_epochs in [(200, 10, 5, 100), (200, 10, 15, 100), (500, 10, 15, 200)]:
        t0 = time.perf_counter()
        _run_fit(n, n_features, n_blocks, n_epochs)
        wall = time.perf_counter() - t0
        print(f"n={n:>4} n_features={n_features:>3} n_blocks={n_blocks:>3} n_epochs={n_epochs:>4} -> {wall * 1000:9.2f} ms")

    t0 = time.perf_counter()
    _run_seed_ensemble(n=200, n_features=10, n_blocks=5, n_epochs=100, n_seeds=8)
    wall = time.perf_counter() - t0
    print(f"seed_ensemble n=200 n_blocks=5 n_epochs=100 n_seeds=8 -> {wall * 1000:9.2f} ms")

    profiler = cProfile.Profile()
    profiler.enable()
    _run_fit(500, 10, 15, 200)
    profiler.disable()
    buf = StringIO()
    stats = pstats.Stats(profiler, stream=buf).sort_stats("cumulative")
    stats.print_stats(15)
    print(buf.getvalue())

    profiler = cProfile.Profile()
    profiler.enable()
    _run_seed_ensemble(n=200, n_features=10, n_blocks=5, n_epochs=100, n_seeds=8)
    profiler.disable()
    buf = StringIO()
    stats = pstats.Stats(profiler, stream=buf).sort_stats("cumulative")
    stats.print_stats(15)
    print(buf.getvalue())
