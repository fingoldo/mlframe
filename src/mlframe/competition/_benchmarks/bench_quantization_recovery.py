"""cProfile benchmark for ``mlframe.competition.quantization_recovery``.

COMPETITION/EXPLORATORY USE ONLY -- see ``mlframe.competition`` package docstring.

Run directly: ``python -m mlframe.competition._benchmarks.bench_quantization_recovery``
"""
from __future__ import annotations

import cProfile
import pstats
import time

import numpy as np

from mlframe.competition.quantization_recovery import (
    derounded_feature,
    detect_quantization_step,
    rank_features_by_quantization_confidence,
)


def _make_scaled_noised_integer_feature(rng: np.random.Generator, n: int, true_step: float, noise_scale: float, int_high: int):
    true_int = rng.integers(0, int_high, size=n)
    noise = rng.normal(0.0, noise_scale, size=n)
    return true_int * true_step + noise, true_int


def _run_once() -> None:
    rng = np.random.default_rng(42)

    sizes = [1_000, 10_000, 100_000]
    for n in sizes:
        x, true_int = _make_scaled_noised_integer_feature(rng, n=n, true_step=0.037, noise_scale=0.037 * 0.03, int_high=max(50, n // 20))
        step = detect_quantization_step(x)
        derounded_feature(x, step)

    features = {}
    for i in range(20):
        x, _ = _make_scaled_noised_integer_feature(rng, n=5_000, true_step=0.01 * (i + 1), noise_scale=0.01 * (i + 1) * 0.03, int_high=300)
        features[f"feat_{i}"] = x
    rank_features_by_quantization_confidence(features)


def main() -> None:
    profiler = cProfile.Profile()
    t0 = time.perf_counter()
    profiler.enable()
    _run_once()
    profiler.disable()
    wall = time.perf_counter() - t0

    stats = pstats.Stats(profiler).sort_stats("cumulative")
    print(f"wall time: {wall:.4f}s")
    stats.print_stats(30)


if __name__ == "__main__":
    main()
