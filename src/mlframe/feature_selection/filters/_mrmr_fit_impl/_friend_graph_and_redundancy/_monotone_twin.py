"""Survivor choice between two selected raw columns that are monotone twins."""
from __future__ import annotations

from typing import Optional


def monotone_twin_to_drop(candidate: int, kept: int, cached_mis: dict[tuple, float]) -> Optional[int]:
    """Index of the twin to drop (the lower screening MI; a tie drops ``candidate``), or ``None`` to keep both.

    ``cached_mis`` holds only what the greedy screen scored, so a raw column a rescue pass re-added can be missing. Reading a miss as 0.0
    made that rescued column lose every comparison; with either relevance unknown the drop is skipped, the conservative direction.
    """
    if (candidate,) not in cached_mis or (kept,) not in cached_mis:
        return None
    rel_c = float(cached_mis[(candidate,)])
    rel_k = float(cached_mis[(kept,)])
    return kept if rel_c > rel_k + 1e-12 else candidate
