"""The forward-selection patience test uses the Nadeau-Bengio correction for overlapping CV folds.

With three folds whose score differences are 0.03, 0.01 and 0.02, the plain paired t-test calls the candidate significant at the 0.05 level.
Those folds share training rows, so the correction widens the variance and the candidate is not significant.
"""

from __future__ import annotations

import numpy as np

import importlib

fs = importlib.import_module("mlframe.feature_selection.forward_select")


def test_overlapping_folds_are_not_called_significant_on_a_plain_t_test_margin():
    """The corrected test rejects a difference that the uncorrected paired test would accept."""
    best = np.array([0.83, 0.81, 0.82])
    current = np.array([0.80, 0.80, 0.80])
    _, significant = fs._paired_significance(best, current, 0.05, 1)
    assert significant is False


def test_a_consistent_gain_is_still_significant():
    """A candidate that beats the current subset on every fold by a clear margin is still selected."""
    best = np.array([0.90, 0.91, 0.92, 0.90, 0.91])
    current = np.array([0.70, 0.71, 0.70, 0.69, 0.70])
    p, significant = fs._paired_significance(best, current, 0.05, 1)
    assert significant is True
    assert p < 0.05
