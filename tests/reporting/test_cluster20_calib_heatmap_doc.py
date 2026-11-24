"""Cluster 20 regression: the calibration heatmap docstring describes the binned ECE the code computes (audit 2026-11-20, finding 20-F38).

Pinned: the per-cell description names the binned ECE and the [0, 1] score assumption, and no longer says each cell is |mean(score) -
mean(target)|.
"""

from pathlib import Path

from mlframe.reporting.charts import calibration_heatmap_2d as ch

_SRC = Path(ch.__file__).resolve()


def test_docstring_names_the_binned_ece():
    """The per-cell description says binned ECE and states the score range (20-F38)."""
    text = _SRC.read_text(encoding="utf-8")
    assert "shows its binned ECE" in text
    assert "scores\n    are expected in [0, 1]" in text


def test_docstring_no_longer_claims_mean_gap():
    """The stale |mean(y_score) - mean(y_true)| per-cell claim is gone (20-F38)."""
    text = _SRC.read_text(encoding="utf-8")
    assert "cell shows ``|mean(y_score) - mean(y_true)|``" not in text
