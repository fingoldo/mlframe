"""cProfile benchmark for the COMPETITION-ONLY inject_noise_and_recenter.

Not for production use — profiles a Kaggle-only adversarial-trend-defeating
utility (mlframe.competition). Run directly: `python bench_trend_noise_decorrelation.py`.
"""

from __future__ import annotations

import cProfile
import io
import pstats

import numpy as np

from mlframe.competition.trend_noise_decorrelation import inject_noise_and_recenter


def _make_segment(n: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    base = rng.normal(0.0, 1.0, size=n)
    peak_positions = rng.choice(n, size=max(1, n // 500), replace=False)
    base[peak_positions] += rng.uniform(3.0, 8.0, size=peak_positions.size)
    return base + np.linspace(0.0, 6.0, n)


def run_once(n: int, seed: int) -> None:
    segment = _make_segment(n, seed)
    inject_noise_and_recenter(segment, noise_std=0.5, random_state=seed)


    # Profiled at n up to 1M: cumtime is dominated by numpy's median (partition) and the RNG draw,
    # both already C-speed vectorized ops; the function is called far below the ~100x/fit threshold
    # that would justify njit/parallel — no actionable speedup here.


def main() -> None:
    profiler = cProfile.Profile()
    profiler.enable()
    for n in (150_000, 500_000, 1_000_000):
        for i in range(5):
            run_once(n, seed=i)
    profiler.disable()

    stream = io.StringIO()
    stats = pstats.Stats(profiler, stream=stream).sort_stats("cumtime")
    stats.print_stats(30)
    print(stream.getvalue())


if __name__ == "__main__":
    main()
