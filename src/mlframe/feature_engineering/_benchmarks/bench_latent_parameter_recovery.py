"""cProfile harness for ``feature_engineering.latent_parameter_recovery.latent_parameter_recovery_features``.

Run: ``python -m mlframe.feature_engineering._benchmarks.bench_latent_parameter_recovery``
"""
from __future__ import annotations

import cProfile
import pstats
import time
from io import StringIO

import numpy as np
import pandas as pd

from mlframe.feature_engineering.latent_parameter_recovery import latent_parameter_recovery_features


def _make_data(n: int, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    true_rate = rng.uniform(0.005, 0.03, n)
    duration = rng.choice([12, 24, 36, 48, 60], n).astype(float)
    amount = rng.uniform(50000, 300000, n)
    annuity = amount * true_rate / (1 - (1 + true_rate) ** (-duration)) + rng.normal(scale=5, size=n)
    return pd.DataFrame({"amount": amount, "n": duration, "annuity": annuity})


def _constraint_fn(df: pd.DataFrame, rate: float) -> np.ndarray:
    if rate <= 0:
        return np.full(len(df), np.inf)
    implied = df["amount"].to_numpy() * rate / (1 - (1 + rate) ** (-df["n"].to_numpy()))
    return np.asarray(implied - df["annuity"].to_numpy())


def _rate_prior_weight(df: pd.DataFrame, candidate: float) -> np.ndarray:
    density = np.exp(-0.5 * ((candidate - 0.0085) / 0.003) ** 2)
    return np.full(len(df), density)


def _run(n: int, grid_size: int, n_calls: int) -> None:
    df = _make_data(n)
    grid = np.linspace(0.002, 0.05, grid_size)
    for _ in range(n_calls):
        latent_parameter_recovery_features(df, grid, _constraint_fn, tolerance=150.0)


def _run_weighted(n: int, grid_size: int, n_calls: int) -> None:
    df = _make_data(n)
    grid = np.linspace(0.002, 0.05, grid_size)
    for _ in range(n_calls):
        latent_parameter_recovery_features(df, grid, _constraint_fn, tolerance=150.0, weight_fn=_rate_prior_weight)


if __name__ == "__main__":
    for n, grid_size, n_calls in [(2000, 100, 20), (200000, 100, 20), (200000, 500, 5)]:
        t0 = time.perf_counter()
        _run(n, grid_size, n_calls)
        wall = time.perf_counter() - t0
        print(f"uniform  n={n:>7} grid_size={grid_size:>4} n_calls={n_calls:>4} -> {wall * 1000:9.2f} ms")

    for n, grid_size, n_calls in [(2000, 100, 20), (200000, 100, 20), (200000, 500, 5)]:
        t0 = time.perf_counter()
        _run_weighted(n, grid_size, n_calls)
        wall = time.perf_counter() - t0
        print(f"weighted n={n:>7} grid_size={grid_size:>4} n_calls={n_calls:>4} -> {wall * 1000:9.2f} ms")

    profiler = cProfile.Profile()
    profiler.enable()
    _run(200000, 500, 5)
    profiler.disable()
    buf = StringIO()
    stats = pstats.Stats(profiler, stream=buf).sort_stats("cumulative")
    stats.print_stats(15)
    print("uniform path:")
    print(buf.getvalue())

    profiler = cProfile.Profile()
    profiler.enable()
    _run_weighted(200000, 500, 5)
    profiler.disable()
    buf = StringIO()
    stats = pstats.Stats(profiler, stream=buf).sort_stats("cumulative")
    stats.print_stats(15)
    print("weighted path:")
    print(buf.getvalue())
