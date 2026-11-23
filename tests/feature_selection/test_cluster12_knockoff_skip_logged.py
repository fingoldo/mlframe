"""Cluster 12 regression: the knockoff cut announces when it is skipped for having fewer than 5 features (audit 2026-11-20, finding 12-F97).

Source-level pin, since the bench needs real data: the early return in knockoff_cut prints a SKIPPED line first.
"""

from pathlib import Path

_BENCH = Path(__file__).resolve().parents[2] / "src" / "mlframe" / "feature_selection" / "_benchmarks" / "fs_hybrid" / "round4_knockoff_fdr_bench.py"


def test_knockoff_skip_is_printed_before_return():
    """A block with fewer than 5 features prints SKIPPED instead of returning silently (12-F97)."""
    text = _BENCH.read_text(encoding="utf-8")
    skip = text.index("SKIPPED:")
    guard = text.index("if len(feats) < 5:")
    assert guard < skip
    assert "the knockoff cut needs at least 5" in text
