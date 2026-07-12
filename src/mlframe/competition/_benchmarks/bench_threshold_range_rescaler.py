"""cProfile benchmark for the COMPETITION-ONLY ThresholdRangeRescaler.

Not for production use — profiles a Kaggle-only post-hoc "magic correction" grid search
(mlframe.competition). Run directly: `python bench_threshold_range_rescaler.py`.
"""

from __future__ import annotations

import cProfile
import io
import pstats

import numpy as np

from mlframe.competition.threshold_range_rescaler import ThresholdRangeRescaler


def _make_dataset(n: int, seed: int):
    rng = np.random.default_rng(seed)
    is_revolving = rng.random(n) < 0.35
    latent = rng.normal(size=n)
    true_prob = 1.0 / (1.0 + np.exp(-latent))
    y = (rng.random(n) < true_prob).astype(np.int64)
    pred = np.clip(true_prob + rng.normal(scale=0.05, size=n), 1e-6, 1.0 - 1e-6)
    inflate_hit = is_revolving & (pred > 0.4)
    pred[inflate_hit] = np.clip(pred[inflate_hit] * 1.6, 1e-6, 1.0 - 1e-6)
    subgroups = {"revolving_loan": is_revolving, "other_loan": ~is_revolving}
    return pred, y, subgroups


def run_once(n: int, n_grid: int) -> None:
    pred, y, subgroups = _make_dataset(n=n, seed=0)
    rescaler = ThresholdRangeRescaler(
        thresholds=np.linspace(0.1, 0.9, n_grid),
        multipliers=np.linspace(0.5, 1.5, n_grid),
        n_splits=5,
        max_corrections=3,
        random_state=0,
    )
    rescaler.fit(pred, y, subgroups)


def main() -> None:
    profiler = cProfile.Profile()
    profiler.enable()
    for n, n_grid in ((2_000, 9), (10_000, 17), (50_000, 17)):
        run_once(n=n, n_grid=n_grid)
    profiler.disable()

    stream = io.StringIO()
    stats = pstats.Stats(profiler, stream=stream).sort_stats("cumtime")
    stats.print_stats(30)
    print(stream.getvalue())


if __name__ == "__main__":
    main()
