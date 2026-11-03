"""Multiclass risk-coverage matches the argmax column to the class label, not to the label's numeric value."""

from __future__ import annotations

import numpy as np
import pytest

from mlframe.reporting.charts.risk_coverage import _prepare_classification


def _proba(idx, n_classes=3, sharp=0.9):
    """Probability matrix that puts ``sharp`` on the column ``idx`` of every row."""
    p = np.full((len(idx), n_classes), (1 - sharp) / (n_classes - 1))
    p[np.arange(len(idx)), idx] = sharp
    return p


def test_non_contiguous_integer_labels_are_matched_by_sorted_order():
    """A perfect classifier on labels 3/5/7 is right on every row, where the old index-vs-value comparison scored 0."""
    idx = np.array([0, 1, 2, 0, 1, 2])
    y = np.array([3, 5, 7, 3, 5, 7])[: len(idx)]
    _, correct = _prepare_classification(y, _proba(idx), task="multiclass")
    assert correct.tolist() == [1.0] * 6


def test_string_labels_are_accepted():
    """String labels no longer crash the float cast."""
    idx = np.array([0, 1, 2, 2])
    y = np.array(["cat", "dog", "emu", "dog"])
    _, correct = _prepare_classification(y, _proba(idx), task="multiclass")
    assert correct.tolist() == [1.0, 1.0, 1.0, 0.0]


def test_zero_based_labels_keep_their_meaning_when_a_class_is_absent():
    """Labels 0..K-1 with one class missing from the sample still map to their own column."""
    idx = np.array([0, 2, 2, 0])
    y = np.array([0, 2, 2, 0])
    _, correct = _prepare_classification(y, _proba(idx), task="multiclass")
    assert correct.tolist() == [1.0] * 4


def test_unmatchable_label_set_raises():
    """Four distinct labels against three columns cannot be matched and the error says so."""
    with pytest.raises(ValueError, match="cannot be matched"):
        _prepare_classification(np.array([10, 20, 30, 40]), _proba(np.array([0, 1, 2, 0])), task="multiclass")
