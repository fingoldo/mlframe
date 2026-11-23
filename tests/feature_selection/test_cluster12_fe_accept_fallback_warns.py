"""Cluster 12 regression: the FE acceptance benches announce the positional base fallback (audit 2026-11-20, finding 12-F94).

Source-level pin, since the benches need real data: when no feature has a nonzero importance, the base falls back to the first 25
columns by position, and a labelled WARNING is printed in both the standard and frugal benches.
"""

from pathlib import Path

_FS = Path(__file__).resolve().parents[2] / "src" / "mlframe" / "feature_selection" / "_benchmarks" / "fs_hybrid"


def test_both_benches_warn_on_positional_fallback():
    """Each bench prints a WARNING naming the positional fallback when ranked is empty (12-F94)."""
    for name in ("round4_fe_accept_bench.py", "round4_fe_accept_frugal_bench.py"):
        text = (_FS / name).read_text(encoding="utf-8")
        assert "base = list(Xtr.columns)[:25]" in text
        assert "falls back to the first 25 columns by position" in text
