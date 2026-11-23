"""Cluster 12 regression: the bootstrap RFECV seed is unique per (outer seed, bootstrap index) (audit 2026-11-20, finding 12-F91).

A bare bootstrap index gave every outer seed the same RFECV random streams. Source-level pin, since the bench needs real data: the
call passes ``1000 * sd + b`` and no longer passes bare ``b``.
"""

from pathlib import Path

_BENCH = Path(__file__).resolve().parents[2] / "src" / "mlframe" / "feature_selection" / "_benchmarks" / "fs_hybrid" / "round2_rfecv_crossseed_bench.py"


def test_bootstrap_rfecv_seed_depends_on_outer_seed():
    """The bootstrap fit is seeded by 1000 * sd + b, not by b alone (12-F91)."""
    text = _BENCH.read_text(encoding="utf-8")
    assert "fit_support(Xb, yb, 1000 * sd + b)" in text
    assert "fit_support(Xb, yb, b))" not in text
