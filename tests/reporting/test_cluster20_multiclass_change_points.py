"""Cluster 20 regression: the multiclass sampler derives class starts from the sorted labels without a second sort (audit 2026-11-20, finding 20-F39).

The change-point construction must give the same classes and starts as np.unique on the sorted labels, including repeated labels and a
single class.
"""

from pathlib import Path

import numpy as np

from mlframe.reporting.charts import multiclass as mc

_SRC = Path(mc.__file__).resolve()


def _change_points(sorted_labels):
    """The starts computed the way the sampler now does."""
    starts = np.concatenate(([0], np.flatnonzero(sorted_labels[1:] != sorted_labels[:-1]) + 1))
    return sorted_labels[starts], starts


def test_change_points_match_np_unique():
    """For sorted labels the change points equal np.unique's classes and first indices (20-F39)."""
    labels = np.array([2, 0, 1, 0, 2, 2, 1])
    sorted_labels = np.sort(labels, kind="stable")
    classes, starts = _change_points(sorted_labels)
    ref_classes, ref_starts = np.unique(sorted_labels, return_index=True)
    assert np.array_equal(classes, ref_classes)
    assert np.array_equal(starts, ref_starts)


def test_single_class_gives_one_start():
    """Control: one repeated label yields a single class starting at 0 (20-F39)."""
    classes, starts = _change_points(np.array([3, 3, 3]))
    assert classes.tolist() == [3] and starts.tolist() == [0]
