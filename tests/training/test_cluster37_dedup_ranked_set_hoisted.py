"""Cluster 37 regression: the ranked set in the composite dedup priority is built once (audit 2026-11-20, finding 37-F32).

``set(ranked)`` used to be rebuilt for every element of the comprehension. The set is now built once and reused.
"""

from pathlib import Path

from mlframe.training.core import _phase_composite_discovery_dedup as dedup


def test_ranked_set_is_hoisted_out_of_the_comprehension():
    """The comprehension tests membership against a prebuilt set, not a fresh set() each time (37-F32)."""
    src = Path(dedup.__file__).resolve().read_text(encoding="utf-8")
    assert "if n not in set(ranked)" not in src
    assert "ranked_set = set(ranked)" in src
    assert "if n not in ranked_set" in src
