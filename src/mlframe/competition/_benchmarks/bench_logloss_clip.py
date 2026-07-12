"""cProfile benchmark for the COMPETITION-ONLY clip_probabilities_for_logloss.

Not for production use — profiles a Kaggle-only log-loss metric-gaming utility
(mlframe.competition). Run directly: `python bench_logloss_clip.py`.
"""

from __future__ import annotations

import cProfile
import io
import pstats

import numpy as np

from mlframe.competition.logloss_clip import clip_probabilities_for_logloss


def _make_probs(n: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.beta(a=0.5, b=0.5, size=n)


def run_clip_once(n: int) -> None:
    probs = _make_probs(n=n, seed=0)
    clip_probabilities_for_logloss(probs, lower=0.05, upper=0.95)


def main() -> None:
    profiler = cProfile.Profile()
    profiler.enable()
    for n in (1_000, 10_000, 100_000, 1_000_000):
        for _ in range(5):
            run_clip_once(n)
    profiler.disable()

    stream = io.StringIO()
    stats = pstats.Stats(profiler, stream=stream).sort_stats("cumtime")
    stats.print_stats(30)
    print(stream.getvalue())


if __name__ == "__main__":
    main()
