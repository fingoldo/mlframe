"""REPORTING_B-9 (2026-08-05 audit): ``_is_binary_score`` guarded its label check on the pre-NaN-filter
``yt.size`` rather than the post-filter size. For an all-NaN ``y_true``, ``np.all(np.isin([], (0, 1)))`` is
vacuously True, so ``label_ok=True`` was wrongly reported for a target carrying zero real label
information, instead of the intended False.
"""

from __future__ import annotations

import numpy as np

from mlframe.reporting.charts.shap_per_instance import _is_binary_score


def test_all_nan_y_true_is_not_reported_as_binary():
    """An all-NaN y_true must never be flagged as a valid binary label (label_ok must be False)."""
    y_true = np.full(10, np.nan)
    y_score = np.linspace(0.0, 1.0, 10)
    assert _is_binary_score(y_true, y_score) is False


def test_real_binary_labels_still_detected():
    """Sanity: a real {0,1} label array with valid [0,1] scores must still be detected as binary."""
    y_true = np.array([0, 1, 0, 1, 1, 0])
    y_score = np.array([0.1, 0.9, 0.2, 0.8, 0.7, 0.3])
    assert _is_binary_score(y_true, y_score) is True
