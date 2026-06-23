"""Numerical-stability regression test for the precision kernel.

Pre-fix ``fast_precision`` divided hits/allpreds elementwise with no zero-denominator
guard, so a class that is never predicted (allpreds==0) yielded nan that was returned
verbatim. (The companion ``fast_classification_report`` divides the same way but its
result is sanitised by a downstream ``np.nan_to_num`` before return, so only the
precision entry point was observably corrupted.)
"""
from __future__ import annotations

import warnings

import numpy as np

from mlframe.metrics._core_precision_mape import fast_precision, fast_classification_report


def test_fast_precision_absent_predicted_class_is_finite():
    # nclasses=3 but class 2 is never predicted -> allpreds[2]==0.
    y_true = np.array([0, 0, 1, 1], dtype=np.int64)
    y_pred = np.array([0, 1, 1, 0], dtype=np.int64)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = fast_precision(y_true, y_pred, nclasses=3)
    assert np.isfinite(res), "precision of an unpredicted class must be finite (0), not nan"
    assert res == 0.0


def test_fast_classification_report_per_class_arrays_finite():
    # Class 2 absent from both arrays: the per-class precision/recall/f1 entries
    # for the absent class must be finite zeros, not propagated inf/nan.
    y_true = np.array([0, 0, 1, 1], dtype=np.int64)
    y_pred = np.array([0, 0, 1, 0], dtype=np.int64)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        out = fast_classification_report(y_true, y_pred, nclasses=3)
    _, _, _, _, _, precisions, recalls, f1s, macro, weighted = out
    for arr in (precisions, recalls, f1s, np.asarray(macro), np.asarray(weighted)):
        assert np.all(np.isfinite(arr)), f"non-finite values in classification report: {arr}"
