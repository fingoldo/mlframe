"""cProfile benchmark for the COMPETITION-ONLY frequency_power_interaction.

Not for production use — profiles a Kaggle-only power-interaction feature builder
(mlframe.competition). Run directly: `python bench_frequency_power_interaction.py`.
"""

from __future__ import annotations

import cProfile
import io
import pstats

import numpy as np

from mlframe.competition.frequency_power_interaction import frequency_power_interaction


def _make_feature(n: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return np.round(rng.uniform(-3.0, 3.0, size=n), 1)


def run_once(n: int) -> None:
    x = _make_feature(n=n, seed=0)
    frequency_power_interaction(x, feature_range=(-4.0, 4.0), count_clip_range=(1.0, 3.0))


def main() -> None:
    profiler = cProfile.Profile()
    profiler.enable()
    for n in (1_000, 10_000, 100_000, 500_000):
        for _ in range(3):
            run_once(n)
    profiler.disable()

    stream = io.StringIO()
    stats = pstats.Stats(profiler, stream=stream).sort_stats("cumtime")
    stats.print_stats(30)
    print(stream.getvalue())


if __name__ == "__main__":
    main()
