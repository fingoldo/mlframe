"""Cluster 18 regression: the default AUC drops rows with a NaN label before scoring (audit 2026-11-20, finding 18-F8).

A NaN label used to count as a negative in the AUC the greedy walk optimises. Such rows are now removed from both the labels and the blend,
so the score equals the score on the labelled rows alone.
"""

import numpy as np

from mlframe.models.ensembling.selection import _score_blend


def test_nan_label_row_does_not_change_the_score():
    """Adding a NaN-labelled row gives the same AUC as the labelled rows alone (18-F8)."""
    blend = np.array([0.9, 0.2, 0.7, 0.4, 0.6, 0.1])
    y_clean = np.array([1.0, 0.0, 1.0, 0.0, 1.0, 0.0])
    with_nan_blend = np.append(blend, 0.8)
    with_nan_y = np.append(y_clean, np.nan)
    assert np.isclose(_score_blend(with_nan_blend, with_nan_y, None), _score_blend(blend, y_clean, None))


def test_clean_labels_are_unchanged():
    """Control: with no NaN label the score is the plain AUC of the blend (18-F8)."""
    blend = np.array([0.9, 0.2, 0.7, 0.4])
    y = np.array([1.0, 0.0, 1.0, 0.0])
    assert _score_blend(blend, y, None) == 1.0
