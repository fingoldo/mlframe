"""cProfile harness for ``feature_engineering.transformer.compute_denoising_autoencoder_features`` /
``swap_noise_augment``.

Run: ``python -m mlframe.feature_engineering._benchmarks.bench_denoising_autoencoder``
"""
from __future__ import annotations

import cProfile
import pstats
import time
from io import StringIO
from typing import Literal

import numpy as np
from sklearn.model_selection import KFold

from mlframe.feature_engineering.transformer import compute_denoising_autoencoder_features, swap_noise_augment


def _make_dataset(n: int, d: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    latent = rng.normal(size=(n, 3))
    W = rng.normal(size=(3, d))
    return latent @ W + rng.normal(scale=0.2, size=(n, d))


def _run_dae(n: int, extract_layers: Literal["bottleneck", "multi"] = "bottleneck") -> None:
    X = _make_dataset(n, d=12, seed=0)
    y = np.zeros(n)  # unused, API symmetry only
    splitter = KFold(n_splits=5, shuffle=True, random_state=0)
    compute_denoising_autoencoder_features(
        X, y, X_query=None, splitter=splitter, seed=0, hidden_size=16, bottleneck_dim=4, max_iter=150, extract_layers=extract_layers
    )


if __name__ == "__main__":
    X = _make_dataset(50_000, d=20, seed=0)
    for _ in range(3):
        t0 = time.perf_counter()
        swap_noise_augment(X, swap_prob=0.15, rng=np.random.default_rng(0))
        wall = time.perf_counter() - t0
        print(f"swap_noise_augment n=50,000 d=20 -> {wall * 1000:9.3f} ms")

    for n in [500, 2_000, 10_000]:
        t0 = time.perf_counter()
        _run_dae(n)
        wall = time.perf_counter() - t0
        print(f"compute_denoising_autoencoder_features n={n:>7,} (bottleneck) -> {wall * 1000:9.2f} ms")

    for n in [500, 2_000, 10_000]:
        t0 = time.perf_counter()
        _run_dae(n, extract_layers="multi")
        wall = time.perf_counter() - t0
        print(f"compute_denoising_autoencoder_features n={n:>7,} (multi)       -> {wall * 1000:9.2f} ms")

    profiler = cProfile.Profile()
    profiler.enable()
    _run_dae(10_000)
    profiler.disable()
    buf = StringIO()
    stats = pstats.Stats(profiler, stream=buf).sort_stats("cumulative")
    stats.print_stats(15)
    print(buf.getvalue())

    profiler_multi = cProfile.Profile()
    profiler_multi.enable()
    _run_dae(10_000, extract_layers="multi")
    profiler_multi.disable()
    buf_multi = StringIO()
    stats_multi = pstats.Stats(profiler_multi, stream=buf_multi).sort_stats("cumulative")
    stats_multi.print_stats(15)
    print("--- extract_layers='multi' profile ---")
    print(buf_multi.getvalue())
