"""Shared micro-benchmark timing helper used by ``_benchmarks/`` scripts across sibling
feature_selection subpackages (feature_selection/_benchmarks, feature_selection/filters/_benchmarks):
independently duplicated across those scripts, consolidated here so a fix can't silently drift out
of sync across copies.
"""
from __future__ import annotations

import time
from typing import Callable


def best_of(fn: Callable, *a, reps: int) -> float:
    """Run ``fn(*a)`` ``reps`` times and return the best (minimum) wall-clock time."""
    t = []
    for _ in range(reps):
        s = time.perf_counter()
        fn(*a)
        t.append(time.perf_counter() - s)
    return min(t)
