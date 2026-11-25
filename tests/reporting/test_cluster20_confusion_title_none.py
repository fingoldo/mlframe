"""Cluster 20 regression: the hidden-values note is not appended to a missing title (audit 2026-11-20, finding 20-F29).

An absent title used to become the literal text "None" in front of the note. With no title the note is the whole title; with a title the
note follows it on a new line.
"""

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from mlframe.reporting.charts import confusion_matrix_plot as cm

_CLASSES = 25  # 25 x 25 = 625 cells, above the readable limit, so the note is drawn


def _render_title(title):
    """Draw an oversized confusion matrix with ``title`` and return the axes title text."""
    labels = np.arange(_CLASSES)
    y_true = np.repeat(labels, 2)
    y_pred = np.roll(y_true, 1)
    fig, ax = plt.subplots()
    try:
        cm.plot_confusion_matrix(y_true, y_pred, ax=ax, title=title, include_values=True)
        return ax.get_title()
    finally:
        plt.close(fig)


def test_missing_title_is_not_rendered_as_the_word_none():
    """With no title the axes title is the hidden-values note alone, with no literal ``None`` in it (20-F29)."""
    text = _render_title(None)
    assert text.startswith("(per-cell values hidden")
    assert "None" not in text


def test_given_title_is_kept_above_the_note():
    """Control: a supplied title is kept and the note follows it on its own line (20-F29)."""
    text = _render_title("Held-out")
    assert text.startswith("Held-out\n(per-cell values hidden")
