"""Cluster 13 regression: the bit-identity check reports each seed on its own (audit 2026-11-20, finding 13-F36).

Source-level pin, since the check runs the full cluster-aggregate step: each seed has its own flag, mismatches set that flag, and the
per-seed line prints the flag rather than the running overall result.
"""

from pathlib import Path

from mlframe.feature_selection import _benchmarks as bench

_SCRIPT = Path(bench.__file__).resolve().parent / "profile_cluster_aggregate.py"


def test_per_seed_flag_is_printed_and_combined():
    """The seed line prints seed_ok and the overall result folds it in (13-F36)."""
    text = _SCRIPT.read_text(encoding="utf-8")
    assert "seed_ok = True" in text
    assert "seed_ok = False" in text
    assert "ok = ok and seed_ok" in text
    assert "if seed_ok else 'NO'" in text
