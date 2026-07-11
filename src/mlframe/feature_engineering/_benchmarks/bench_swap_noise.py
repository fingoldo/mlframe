"""cProfile harness for ``feature_engineering.transformer.swap_noise.swap_noise_augment``, including the
per-column ``column_swap_probs`` opt-in path.

Run: ``python -m mlframe.feature_engineering._benchmarks.bench_swap_noise``
"""
from __future__ import annotations

import cProfile
import pstats
import time
from io import StringIO

import numpy as np

from mlframe.feature_engineering.transformer.swap_noise import swap_noise_augment


def _make_dataset(n: int, d: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    latent = rng.normal(size=(n, 3))
    W = rng.normal(size=(3, d))
    return latent @ W + rng.normal(scale=0.2, size=(n, d))


if __name__ == "__main__":
    X = _make_dataset(50_000, d=20, seed=0)
    per_col = np.linspace(0.05, 0.55, 20)

    for _ in range(3):
        t0 = time.perf_counter()
        swap_noise_augment(X, swap_prob=0.15, rng=np.random.default_rng(0))
        wall = time.perf_counter() - t0
        print(f"swap_noise_augment uniform    n=50,000 d=20 -> {wall * 1000:9.3f} ms")

    for _ in range(3):
        t0 = time.perf_counter()
        swap_noise_augment(X, rng=np.random.default_rng(0), column_swap_probs=per_col)
        wall = time.perf_counter() - t0
        print(f"swap_noise_augment per-column n=50,000 d=20 -> {wall * 1000:9.3f} ms")

    profiler = cProfile.Profile()
    profiler.enable()
    for _ in range(20):
        swap_noise_augment(X, rng=np.random.default_rng(0), column_swap_probs=per_col)
    profiler.disable()
    buf = StringIO()
    stats = pstats.Stats(profiler, stream=buf).sort_stats("cumulative")
    stats.print_stats(20)
    print(buf.getvalue())
