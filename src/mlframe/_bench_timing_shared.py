"""Shared micro-benchmark timing helper used by several unrelated ``_benchmarks/`` packages
(training, metrics): independently duplicated across those scripts, consolidated here so a fix
can't silently drift out of sync across copies.
"""
from __future__ import annotations

import time
from typing import Callable


def time_call(fn: Callable, *args, iters: int) -> float:
    """Warm ``fn(*args)`` once, then return its mean wall-clock time in microseconds over ``iters`` calls."""
    fn(*args)
    t0 = time.perf_counter()
    for _ in range(iters):
        fn(*args)
    return (time.perf_counter() - t0) / iters * 1e6
