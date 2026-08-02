"""Shared cProfile-harness ``main()`` used by several ``bench_*.py`` scripts in this directory:
independently duplicated across those scripts, consolidated here so a fix can't silently drift out
of sync across copies. Each benchmark script stays independently runnable from this same
``_benchmarks/`` directory -- only the literal duplicated body moves here.
"""
from __future__ import annotations

import cProfile
import pstats
import time
from typing import Callable


def cprofile_main(run_once: Callable[[], None]) -> None:
    """Profile a single call to ``run_once`` with cProfile, print wall time + top-30 cumulative stats."""
    profiler = cProfile.Profile()
    t0 = time.perf_counter()
    profiler.enable()
    run_once()
    profiler.disable()
    wall = time.perf_counter() - t0

    stats = pstats.Stats(profiler).sort_stats("cumulative")
    print(f"wall time: {wall:.4f}s")
    stats.print_stats(30)
