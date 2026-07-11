"""cProfile harness for ``feature_engineering.recency_aggregation.per_group_recency_weighted_agg``.

Run: ``python -m mlframe.feature_engineering._benchmarks.bench_recency_weighted_agg``
"""
from __future__ import annotations

import cProfile
import pstats
import time
from io import StringIO

import numpy as np

from mlframe.feature_engineering.recency_aggregation import per_group_recency_weighted_agg


def _make_panel(n_entities: int, hist: int, seed: int = 0):
    rng = np.random.default_rng(seed)
    values = rng.normal(0.0, 1.0, size=n_entities * hist)
    group_ids = np.repeat(np.arange(n_entities), hist)
    order = np.tile(np.arange(hist, dtype=np.float64), n_entities)
    return values, group_ids, order


DECAY_FAMILY = [0.5, 1.0, 2.0, 5.0, 10.0]  # a typical small family of decay rates for one base column.


def _run(n_entities: int, hist: int, n_calls: int) -> None:
    values, group_ids, order = _make_panel(n_entities, hist)
    for _ in range(n_calls):
        for agg in ("mean", "sum", "min", "max", "std", "var"):
            per_group_recency_weighted_agg(values, group_ids, agg=agg, order=order, scheme="poly", param=1.0)


def _run_multi_decay(n_entities: int, hist: int, n_calls: int) -> None:
    """Same family of decay rates via the opt-in ``params=`` multi-decay path (one sort/boundary pass per call)."""
    values, group_ids, order = _make_panel(n_entities, hist)
    for _ in range(n_calls):
        for agg in ("mean", "sum", "min", "max", "std", "var"):
            per_group_recency_weighted_agg(values, group_ids, agg=agg, order=order, scheme="poly", params=DECAY_FAMILY)


def _run_multi_decay_via_single_calls(n_entities: int, hist: int, n_calls: int) -> None:
    """Same family of decay rates via len(DECAY_FAMILY) separate single-``param`` calls (re-sorts every time) -- the baseline the multi-decay path is meant to beat."""
    values, group_ids, order = _make_panel(n_entities, hist)
    for _ in range(n_calls):
        for agg in ("mean", "sum", "min", "max", "std", "var"):
            for param in DECAY_FAMILY:
                per_group_recency_weighted_agg(values, group_ids, agg=agg, order=order, scheme="poly", param=param)


if __name__ == "__main__":
    # warm numba dispatch cache before timing.
    _run(100, 8, 1)
    _run_multi_decay(100, 8, 1)
    _run_multi_decay_via_single_calls(100, 8, 1)

    for n_entities, hist, n_calls in [(2000, 12, 20), (50000, 12, 20), (50000, 30, 20)]:
        t0 = time.perf_counter()
        _run(n_entities, hist, n_calls)
        wall = time.perf_counter() - t0
        print(f"n_entities={n_entities:>7} hist={hist:>3} n_calls={n_calls:>4} -> {wall * 1000:9.2f} ms")

    print(f"\nmulti-decay path ({len(DECAY_FAMILY)} decay rates per call) vs {len(DECAY_FAMILY)} single-param calls:")
    for n_entities, hist, n_calls in [(2000, 12, 20), (50000, 12, 20), (50000, 30, 20)]:
        t0 = time.perf_counter()
        _run_multi_decay(n_entities, hist, n_calls)
        wall_multi = time.perf_counter() - t0

        t0 = time.perf_counter()
        _run_multi_decay_via_single_calls(n_entities, hist, n_calls)
        wall_single = time.perf_counter() - t0

        speedup = wall_single / wall_multi if wall_multi > 0 else float("nan")
        print(
            f"n_entities={n_entities:>7} hist={hist:>3} n_calls={n_calls:>4} -> "
            f"multi={wall_multi * 1000:9.2f} ms  single_calls={wall_single * 1000:9.2f} ms  speedup={speedup:5.2f}x"
        )

    profiler = cProfile.Profile()
    profiler.enable()
    _run(50000, 30, 20)
    profiler.disable()
    buf = StringIO()
    stats = pstats.Stats(profiler, stream=buf).sort_stats("cumulative")
    stats.print_stats(15)
    print(buf.getvalue())

    profiler = cProfile.Profile()
    profiler.enable()
    _run_multi_decay(50000, 30, 20)
    profiler.disable()
    buf = StringIO()
    stats = pstats.Stats(profiler, stream=buf).sort_stats("cumulative")
    print("multi-decay path cProfile:")
    stats.print_stats(15)
    print(buf.getvalue())
