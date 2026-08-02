"""Shared micro-benchmark timing helper used by ``_benchmarks/`` scripts across sibling
feature_selection subpackages (feature_selection/_benchmarks, feature_selection/filters/_benchmarks):
independently duplicated across those scripts, consolidated here so a fix can't silently drift out
of sync across copies.
"""
from __future__ import annotations

import time
from typing import Callable

import numpy as np


def best_of(fn: Callable, *a, reps: int) -> float:
    """Run ``fn(*a)`` ``reps`` times and return the best (minimum) wall-clock time."""
    t = []
    for _ in range(reps):
        s = time.perf_counter()
        fn(*a)
        t.append(time.perf_counter() - s)
    return min(t)


def median_time(fn: Callable, n_runs: int = 7) -> float:
    """Warm ``fn()`` once, then return the median wall-clock time of ``n_runs`` zero-arg calls."""
    fn()
    ts = []
    for _ in range(n_runs):
        t0 = time.perf_counter()
        fn()
        ts.append(time.perf_counter() - t0)
    return float(np.median(ts))


def best_of_args(fn: Callable, args: tuple, reps: int) -> float:
    """Run ``fn(*args)`` ``reps`` times and return the best (minimum) wall-clock time (args passed as an explicit
    tuple rather than varargs, matching this call convention's existing call sites)."""
    best = 1e18
    for _ in range(reps):
        t = time.perf_counter()
        fn(*args)
        best = min(best, time.perf_counter() - t)
    return best
