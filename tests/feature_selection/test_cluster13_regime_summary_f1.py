"""Cluster 13 regression: the regime summary F1 is the mean of per-cell F1, not the F1 of mean precision and recall (audit 2026-11-20, finding 13-F31).

Source-level pin, since the bench needs real runs: the F1 is averaged over per-row F1 values, and the old formula over the means is gone.
"""

from pathlib import Path

from mlframe.feature_selection import _benchmarks as bench

_SCRIPT = Path(bench.__file__).resolve().parent / "fs_quality" / "mrmr_synergy_regime_bench.py"


def test_summary_f1_is_mean_of_cell_f1():
    """The summary averages per-cell F1 (13-F31)."""
    text = _SCRIPT.read_text(encoding="utf-8")
    assert "_mean([2 * r['precision'] * r['recall'] / (r['precision'] + r['recall'])" in text
    assert "f1 = 2 * prec * rec / (prec + rec)" not in text
