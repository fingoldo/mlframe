"""Cluster 20 regression: a large confusion matrix with no title does not get the literal "None" in its title (audit 2026-11-20, finding 20-F29).

Source-level pin: the hidden-values note is built separately and joined to the title only when a title exists.
"""

from pathlib import Path

from mlframe.reporting.charts import confusion_matrix_plot as cm

_SRC = Path(cm.__file__).resolve()


def test_hidden_values_note_is_not_appended_to_a_none_title():
    """The note is joined to the title only when the title is set (20-F29)."""
    text = _SRC.read_text(encoding="utf-8")
    assert 'title = f"{title}\\n{note}" if title else note' in text
    assert 'title = f"{title}\\n(per-cell values hidden' not in text
