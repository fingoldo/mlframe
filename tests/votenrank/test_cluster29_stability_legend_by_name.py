"""Cluster 29 regression: the stability plot legend is relabelled by method name, not by position (audit 2026-11-20, finding 29-F36).

Positional indexing into the legend texts breaks when the number of plotted methods changes. The source must match each entry by its label.
"""

from pathlib import Path

from mlframe.votenrank import stability_exp

_SRC = Path(stability_exp.__file__).resolve().read_text(encoding="utf-8")


def test_legend_is_not_relabelled_by_position():
    """No positional get_texts()[k] indexing remains (29-F36)."""
    assert "get_texts()[0]" not in _SRC
    assert "get_texts()[3]" not in _SRC


def test_legend_entries_are_matched_by_label():
    """The loop matches the optimality-gap and mean labels by name (29-F36)."""
    assert '"optimality_gap" in label' in _SRC
    assert '"mean" in label' in _SRC
