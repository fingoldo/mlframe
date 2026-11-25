"""Cluster 20 regression: the interval-band panel has one empty-row guard (audit 2026-11-20, finding 20-F31).

Source-level pin: the unreachable second guard, which repeated the first with a different label, is gone; the first guard still annotates
an empty interval panel.
"""

from pathlib import Path

from mlframe.reporting.charts import quantile as qt

_SRC = Path(qt.__file__).resolve()


def test_single_empty_row_guard_remains():
    """The duplicate guard with the 'Pinball loss by alpha' label is removed (20-F31)."""
    text = _SRC.read_text(encoding="utf-8")
    assert text.count('_no_rows_annotation("Pinball loss by alpha"') == 1  # only the pinball panel keeps its guard
    assert text.count('_no_rows_annotation("Prediction intervals"') == 1
