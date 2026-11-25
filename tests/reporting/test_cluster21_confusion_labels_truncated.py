"""Cluster 21 regression: the confusion-margins heatmap truncates its tick labels as the plain heatmap does (audit 2026-11-20, finding 21-F25).

Source-level pin, since the margins draw needs a full figure: the margin tick labels pass through truncate_bar_label. The duplicated
cell-text loop is left for a separate change (it differs in its colour-bound handling).
"""

from pathlib import Path

from mlframe.reporting.renderers import _matplotlib_heatmap as mh

_SRC = Path(mh.__file__).resolve()


def test_margin_tick_labels_are_truncated():
    """The margins path wraps its tick labels in truncate_bar_label, as _heatmap does (21-F25)."""
    text = _SRC.read_text(encoding="utf-8")
    assert 'set_xticklabels([truncate_bar_label(p.col_labels[i]) for i in _xt], rotation=45, ha="right", fontsize=8)' in text
    assert "set_yticklabels([truncate_bar_label(p.row_labels[i]) for i in _yt], fontsize=8)" in text
