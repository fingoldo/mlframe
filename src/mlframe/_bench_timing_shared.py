"""Shared micro-benchmark timing helper used by several unrelated ``_benchmarks/`` packages
(training, metrics): independently duplicated across those scripts, consolidated here so a fix
can't silently drift out of sync across copies.
"""
from __future__ import annotations

import time
from typing import Callable


def best_of_median_seconds(fn: Callable, n: int = 5) -> tuple[float, float]:
    """Run zero-arg ``fn()`` ``n`` times and return ``(best, median)`` wall-clock seconds, no warm-up
    call before the loop. Shared by the ranking-metrics dispatch bench pair."""
    import numpy as np

    ts = []
    for _ in range(n):
        t = time.perf_counter()
        fn()
        ts.append(time.perf_counter() - t)
    return min(ts), float(np.median(ts))


def time_call(fn: Callable, *args, iters: int) -> float:
    """Warm ``fn(*args)`` once, then return its mean wall-clock time in microseconds over ``iters`` calls."""
    fn(*args)
    t0 = time.perf_counter()
    for _ in range(iters):
        fn(*args)
    return (time.perf_counter() - t0) / iters * 1e6


def best_of_seconds_with_output(fn: Callable, *args, reps: int = 7):
    """Best (minimum) wall-clock time in seconds over ``reps`` calls, plus the LAST call's return value."""
    best = float("inf")
    out = None
    for _ in range(reps):
        t0 = time.perf_counter()
        out = fn(*args)
        best = min(best, time.perf_counter() - t0)
    return best, out


def best_of_seconds_zero_arg(fn: Callable, repeats: int = 3) -> float:
    """Warm ``fn()`` once (no args), then return its best (minimum) wall-clock time in seconds over ``repeats`` calls."""
    fn()
    best = float("inf")
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn()
        best = min(best, time.perf_counter() - t0)
    return best


def best_of_seconds(fn: Callable, *args, reps: int = 5) -> float:
    """Warm ``fn(*args)`` once, then return its best (minimum) wall-clock time in seconds over ``reps`` calls (no output captured)."""
    fn(*args)
    best = float("inf")
    for _ in range(reps):
        t0 = time.perf_counter()
        fn(*args)
        best = min(best, time.perf_counter() - t0)
    return best


def best_of_seconds_no_warmup(fn: Callable, *args, repeat: int = 5) -> float:
    """Best (minimum) wall-clock time in seconds over ``repeat`` calls, no warm-up call before the loop."""
    best = float("inf")
    for _ in range(repeat):
        t0 = time.perf_counter()
        fn(*args)
        best = min(best, time.perf_counter() - t0)
    return best


def best_of_seconds_args_no_warmup(fn: Callable, args: tuple, reps: int = 5) -> float:
    """Best (minimum) wall-clock time in seconds over ``reps`` calls of ``fn(*args)``, ``args`` passed as an
    explicit tuple (not varargs); no warm-up call before the loop."""
    best = float("inf")
    for _ in range(reps):
        t0 = time.perf_counter()
        fn(*args)
        best = min(best, time.perf_counter() - t0)
    return best


def time_once(fn: Callable) -> float:
    """Wall-clock time in seconds of a single (unwarmed) ``fn()`` call."""
    t0 = time.perf_counter()
    fn()
    return time.perf_counter() - t0


def best_of_ms(fn: Callable, *args, repeat: int = 5) -> float:
    """Warm ``fn(*args)`` once, then return its best (minimum) wall-clock time in milliseconds over ``repeat`` calls."""
    fn(*args)
    best = float("inf")
    for _ in range(repeat):
        t0 = time.perf_counter()
        fn(*args)
        dt = (time.perf_counter() - t0) * 1000.0
        if dt < best:
            best = dt
    return best
