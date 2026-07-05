"""Microbenchmark for the recurrent numpy-coercion cache (POLARS-PANDAS-CHURN).

Compares:
    (a) baseline: ``np.asarray(features.to_numpy(), dtype=np.float32)`` repeated 3x per split
    (b) cached: one coercion + 2 cache hits per split via ``_coerce_features_to_float32(cache=...)``

The recurrent path predicts on train / val / test for every recurrent member; without the cache
each member re-coerces the same frame 3x. With ``ctx._recurrent_numpy_cache`` the second and
third hits return the same ndarray.

Run: ``python -m mlframe.training._benchmarks.bench_recurrent_numpy_cache``
"""
from __future__ import annotations

import statistics
import time
from typing import Dict

import numpy as np
import pandas as pd


def _baseline(df: pd.DataFrame, trials: int) -> float:
    samples = []
    for _ in range(trials):
        t0 = time.perf_counter()
        # Three identical re-coercions, one per split (train/val/test).
        for _split in ("train", "val", "test"):
            _ = np.asarray(df.to_numpy(), dtype=np.float32)
        samples.append((time.perf_counter() - t0) * 1000.0)
    return statistics.median(samples)


def _cached(df: pd.DataFrame, trials: int) -> float:
    from mlframe.training.core._phase_recurrent import _coerce_features_to_float32
    samples = []
    for _ in range(trials):
        cache: Dict = {}
        t0 = time.perf_counter()
        for _split in ("train", "val", "test"):
            _ = _coerce_features_to_float32(df, cache=cache, cache_key=(_split, id(df)))
        samples.append((time.perf_counter() - t0) * 1000.0)
    return statistics.median(samples)


def _cached_warm(df: pd.DataFrame, trials: int) -> float:
    """Single-split key reused 3x -- the real-world case where one member hits the same (split, id) thrice."""
    from mlframe.training.core._phase_recurrent import _coerce_features_to_float32
    samples = []
    for _ in range(trials):
        cache: Dict = {}
        t0 = time.perf_counter()
        for _ in range(3):
            _ = _coerce_features_to_float32(df, cache=cache, cache_key=("train", id(df)))
        samples.append((time.perf_counter() - t0) * 1000.0)
    return statistics.median(samples)


def run(combinations=None, trials: int = 7) -> None:
    combinations = combinations or [
        (10_000, 16),
        (100_000, 16),
        (100_000, 64),
        (250_000, 64),
    ]
    rng = np.random.default_rng(20260516)
    print(f"# Recurrent feature-frame coercion (median of {trials} trials, ms)")
    print("| n_rows | n_cols | (a) baseline 3x | (b) cached 3-split | (c) cached 3-hit same key | speedup b/a | speedup c/a |")
    print("|--------|--------|------------------|--------------------|---------------------------|-------------|-------------|")
    for n, k in combinations:
        df = pd.DataFrame({f"f{i}": rng.standard_normal(n).astype(np.float64) for i in range(k)})
        a = _baseline(df, trials)
        b = _cached(df, trials)
        c = _cached_warm(df, trials)
        print(f"| {n} | {k} | {a:.3f} | {b:.3f} | {c:.3f} | {a/b:.2f}x | {a/c:.2f}x |")


if __name__ == "__main__":
    run()
