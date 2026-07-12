"""cProfile benchmark for the COMPETITION-ONLY FloatPrecisionDenoiser.fit/transform.

Not for production use — profiles a Kaggle-only denoising utility (mlframe.competition).
Run directly: `python bench_float_precision_denoise.py`.
"""

from __future__ import annotations

import cProfile
import pstats
import io

import numpy as np

from mlframe.competition.float_precision_denoise import FloatPrecisionDenoiser


def _make_amex_like(n: int, scale: float, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    true_int = rng.integers(0, 100, size=n)
    true_value = true_int / scale
    noise = rng.uniform(0.0, 0.2 / scale, size=n)
    return true_value + noise


def run_once(n: int) -> None:
    noisy = _make_amex_like(n=n, scale=100.0, seed=0)
    denoiser = FloatPrecisionDenoiser(max_decimal_pow=6, max_denominator=1000, use_floor=True)
    denoiser.fit_transform(noisy)


def main() -> None:
    profiler = cProfile.Profile()
    profiler.enable()
    for n in (1_000, 10_000, 100_000):
        for _ in range(3):
            run_once(n)
    profiler.disable()

    stream = io.StringIO()
    stats = pstats.Stats(profiler, stream=stream).sort_stats("cumulative")
    stats.print_stats(30)
    print(stream.getvalue())


if __name__ == "__main__":
    main()
