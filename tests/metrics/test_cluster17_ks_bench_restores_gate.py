"""Cluster 17 regression: the KS bench restores the production gate even when a timed run raises (audit 2026-11-20, finding 17-F30).

Source-level pin: the gate mutation sits inside try and the restore is in the finally clause.
"""

from pathlib import Path

from mlframe.metrics._benchmarks import bench_ks_desc_order_inline_gather_iter72 as bench

_SRC = Path(bench.__file__).resolve()


def test_gate_restore_is_in_a_finally_clause():
    """The original gate value is restored from a finally block (17-F30)."""
    text = _SRC.read_text(encoding="utf-8")
    assert "    try:\n        ex._KS_INLINE_ORDERED_MIN_N = 10**18" in text
    assert "    finally:\n        ex._KS_INLINE_ORDERED_MIN_N = orig" in text
