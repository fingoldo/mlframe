"""Shared cProfile-report helpers for the training/_benchmarks/ profiling harness family: small
utilities independently duplicated across those scripts, consolidated here so a fix can't
silently drift out of sync across copies.
"""
from __future__ import annotations

import io
import pstats


def profile_table(profiler: pstats.Stats, sort_key: str, top_n: int) -> str:
    """Render the top ``top_n`` rows of ``profiler`` sorted by ``sort_key`` (e.g. "cumulative"/"tottime") as text."""
    s = io.StringIO()
    pstats.Stats(profiler, stream=s).sort_stats(sort_key).print_stats(top_n)
    return s.getvalue()
