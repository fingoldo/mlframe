"""Cluster 20 regression: rows with a missing group label are reported when they are left out of every group (audit 2026-11-20, finding 20-F41).

pd.factorize codes a missing label -1, and those rows belong to no group. A WARNING now says how many were excluded; a fully labelled input is
silent and the grouping is unchanged.
"""

import logging

import numpy as np

from mlframe.reporting.charts import _grouping as gr


def test_missing_labels_are_reported(caplog):
    """Rows with a NaN label are excluded with a WARNING naming the count (20-F41)."""
    labels = np.array(["a", np.nan, "b", "a"], dtype=object)
    with caplog.at_level(logging.WARNING):
        _order, bounds, uniq = gr.group_indices_by_label(labels)
    grouped = sum(bounds[i + 1] - bounds[i] for i in range(len(uniq)))
    assert grouped == 3  # the missing row belongs to no group
    assert any("1 of 4 rows have a missing label" in r.getMessage() for r in caplog.records)


def test_fully_labelled_input_is_silent(caplog):
    """Control: all labels present means no warning and every row grouped (20-F41)."""
    labels = np.array(["a", "b", "a"], dtype=object)
    with caplog.at_level(logging.WARNING):
        order, _bounds, _uniq = gr.group_indices_by_label(labels)
    assert order.size == 3
    assert not any("missing label" in r.getMessage() for r in caplog.records)
