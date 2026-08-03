"""Shared micro-benchmark timing helper for the hermite_fe polyeval bench cluster."""
from __future__ import annotations

import time


def best_of(fn, x, c, reps):
    """Warm-run ``fn(x, c)`` ``reps`` times, returning the fastest single-call wall time."""
    best = 1e9
    for _ in range(reps):
        t = time.perf_counter()
        fn(x, c)
        best = min(best, time.perf_counter() - t)
    return best
