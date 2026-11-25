"""Cluster 13 regression: par_threshold uses the same never-wins sentinel as cuda_threshold (audit 2026-11-20, finding 13-F29).

Source-level pin, since the sweep needs real timings: when njit_par never beats njit, the fallback is swept_max * 100, not the fixed
50_000 source default that routed large inputs to the slower parallel path.
"""

from pathlib import Path

from mlframe.feature_selection import _benchmarks as bench

_SWEEPS = Path(bench.__file__).resolve().parent / "kernel_tuning_cache" / "_auto_tune_sweeps_a.py"


def test_par_threshold_fallback_is_a_never_wins_sentinel():
    """The par_threshold fallback is swept_max * 100 and the 50_000 default is gone (13-F29)."""
    text = _SWEEPS.read_text(encoding="utf-8")
    assert "par_threshold = max(timing_table) * 100" in text
    assert "par_threshold = 50_000" not in text
