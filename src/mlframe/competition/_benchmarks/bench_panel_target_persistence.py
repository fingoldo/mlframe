"""cProfile benchmark for ``mlframe.competition.panel_target_persistence``.

COMPETITION/EXPLORATORY USE ONLY -- see ``mlframe.competition`` package docstring.

Run directly: ``python -m mlframe.competition._benchmarks.bench_panel_target_persistence``
"""
from __future__ import annotations

import cProfile
import pstats
import time

import numpy as np

from mlframe.competition.panel_target_persistence import (
    check_target_persistence,
    lag_target_within_group,
    lead_target_within_group,
)


def _make_persistent_panel(rng: np.random.Generator, n_groups: int, group_size: int, flip_prob: float = 0.02):
    group_ids = np.repeat(np.arange(n_groups), group_size)
    order = np.tile(np.arange(group_size), n_groups)
    y = np.empty(group_ids.size, dtype=float)
    for g in range(n_groups):
        state = float(rng.integers(0, 2))
        seq = np.empty(group_size, dtype=float)
        for i in range(group_size):
            if i > 0 and rng.random() < flip_prob:
                state = 1.0 - state
            seq[i] = state
        y[g * group_size : (g + 1) * group_size] = seq
    return group_ids, order, y


def _run_once() -> None:
    rng = np.random.default_rng(42)

    for n_groups, group_size in [(500, 10), (2_000, 10), (5_000, 20)]:
        group_ids, order, y = _make_persistent_panel(rng, n_groups, group_size)
        check_target_persistence(group_ids, order, y)
        lag_target_within_group(group_ids, order, y)
        lead_target_within_group(group_ids, order, y)


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
