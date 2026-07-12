"""cProfile harness for ``feature_engineering.fuzzy_entity.fuzzy_entity_group_features``.

Run: ``python -m mlframe.feature_engineering._benchmarks.bench_fuzzy_entity``
"""
from __future__ import annotations

import cProfile
import pstats
import time
from io import StringIO

import numpy as np

from mlframe.feature_engineering.fuzzy_entity import fuzzy_entity_group_features


def _run(n_entities: int, events_per_entity: int, n_values: int, n_calls: int) -> None:
    rng = np.random.default_rng(0)
    entity_ids = np.repeat(np.arange(n_entities), events_per_entity)
    n = entity_ids.shape[0]
    values = rng.integers(0, n_values, size=n)
    order = np.arange(n, dtype=np.float64)
    for _ in range(n_calls):
        fuzzy_entity_group_features(entity_ids, values, time_order=order)


def _make_noisy_keys(n_entities: int, events_per_entity: int, seed: int, typo_rate: float = 0.3) -> np.ndarray:
    """Typo'd string keys, each ``"cust_NNNNNN"`` (11 chars) with only the LAST char perturbable -- matching
    ``fuzzy_block_prefix_len=10`` below (all but the last char). A fixed-length prefix must actually cover the
    id's distinguishing digits to keep blocks small; blocking on a short prefix of a zero-padded id (e.g. the
    first 6 of 11 chars) would put ~all entities in one giant block since the leading digits are mostly zero.
    """
    rng = np.random.default_rng(seed)
    keys = np.empty(n_entities * events_per_entity, dtype=object)
    pos = 0
    for e in range(n_entities):
        # spaced out (x97) so a last-digit typo lands on unused id space rather than another real entity's
        # canonical key -- see the matching comment in tests/feature_engineering/test_biz_val_fuzzy_entity.py.
        canonical_key = f"cust_{(e * 97) % 1_000_000:06d}"
        for _ in range(events_per_entity):
            if rng.random() < typo_rate:
                chars = list(canonical_key)
                chars[-1] = str(rng.integers(0, 10))
                keys[pos] = "".join(chars)
            else:
                keys[pos] = canonical_key
            pos += 1
    return keys


def _run_fuzzy(n_entities: int, events_per_entity: int, n_values: int, n_calls: int) -> None:
    rng = np.random.default_rng(1)
    keys = _make_noisy_keys(n_entities, events_per_entity, seed=1)
    n = keys.shape[0]
    values = np.asarray(rng.integers(0, n_values, size=n))
    order = np.arange(n, dtype=np.float64)
    for _ in range(n_calls):
        fuzzy_entity_group_features(
            keys, values, time_order=order, fuzzy_key_matching=True, fuzzy_max_distance=1, fuzzy_block_prefix_len=10
        )


if __name__ == "__main__":
    for n_entities, events_per_entity, n_values, n_calls in [(1_000, 10, 50, 20), (50_000, 10, 500, 3)]:
        t0 = time.perf_counter()
        _run(n_entities, events_per_entity, n_values, n_calls)
        wall = time.perf_counter() - t0
        n_rows = n_entities * events_per_entity
        print(f"exact       rows={n_rows:>9,} entities={n_entities:>7,} -> {wall * 1000:9.2f} ms total, {wall / n_calls * 1000:8.3f} ms/call")

    # Fuzzy path pays an extra blocked-pairwise-edit-distance clustering cost on TOP of the exact-match
    # pipeline above, so it's benched separately -- capped well below the 50k-row exact-match benchmark to
    # keep this script fast, and sized so effective blocking (see ``_make_noisy_keys``) keeps wall time low.
    for n_entities, events_per_entity, n_values, n_calls in [(1_000, 10, 50, 10), (10_000, 10, 200, 3)]:
        t0 = time.perf_counter()
        _run_fuzzy(n_entities, events_per_entity, n_values, n_calls)
        wall = time.perf_counter() - t0
        n_rows = n_entities * events_per_entity
        print(f"fuzzy       rows={n_rows:>9,} entities={n_entities:>7,} -> {wall * 1000:9.2f} ms total, {wall / n_calls * 1000:8.3f} ms/call")

    profiler = cProfile.Profile()
    profiler.enable()
    _run(50_000, 10, 500, 5)
    profiler.disable()
    buf = StringIO()
    stats = pstats.Stats(profiler, stream=buf).sort_stats("cumulative")
    stats.print_stats(15)
    print(buf.getvalue())

    profiler_fuzzy = cProfile.Profile()
    profiler_fuzzy.enable()
    _run_fuzzy(10_000, 10, 200, 3)
    profiler_fuzzy.disable()
    buf_fuzzy = StringIO()
    stats_fuzzy = pstats.Stats(profiler_fuzzy, stream=buf_fuzzy).sort_stats("cumulative")
    stats_fuzzy.print_stats(15)
    print(buf_fuzzy.getvalue())
