"""Cluster 20 regression: multilabel legend entries shorten long label names as multiclass legends do (audit 2026-11-20, finding 20-F34).

A long class name used to overflow a multilabel legend. The shortening helper is shared with the multiclass charts, and a legend entry for
a long label is now bounded in length.
"""

from mlframe.reporting.charts import multilabel as ml
from mlframe.reporting.charts.multiclass import _LEGEND_CLASS_NAME_MAXLEN


def test_legend_name_helper_shortens_long_labels():
    """A label longer than the legend limit is truncated (20-F34)."""
    long_label = "x" * (_LEGEND_CLASS_NAME_MAXLEN * 3)
    out = ml._legend_name(long_label)
    assert len(str(out)) <= _LEGEND_CLASS_NAME_MAXLEN + 3


def test_short_labels_are_unchanged():
    """Control: a short label is kept as given (20-F34)."""
    assert ml._legend_name("cat") == "cat"
