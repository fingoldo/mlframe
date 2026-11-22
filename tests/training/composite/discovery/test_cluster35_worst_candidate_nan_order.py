"""Audit 2026-11-20 finding 35-F34: the worst-candidate choice uses the judge's fill and does not depend on candidate order.

A NaN score used to make max() order-dependent, and the fill used every fit row instead of the rows the judge used.
"""

import numpy as np

from mlframe.training.composite.discovery._yscale_holdout_gate import _select_worst_candidate


def test_worst_candidate_is_the_larger_error_after_the_fill():
    """The candidate with the larger RMSE after the fill is returned."""
    y_eval = np.array([1.0, 2.0, 3.0])
    fit_values = np.array([1.0, 2.0, 3.0])
    good = np.array([1.0, 2.0, 3.0])
    bad = np.array([10.0, 10.0, 10.0])
    assert _select_worst_candidate([good, bad], y_eval, fit_values) is bad


def test_nan_candidate_is_worst_regardless_of_order():
    """A candidate whose score is NaN is the worst, whichever position it sits in."""
    y_eval = np.array([1.0, 2.0, 3.0])
    fit_values = np.array([1.0, 2.0, 3.0])
    ok = np.array([1.1, 2.1, 3.1])
    nan_candidate = np.array([np.nan, np.nan, np.nan])
    first = _select_worst_candidate([nan_candidate, ok], y_eval, fit_values)
    second = _select_worst_candidate([ok, nan_candidate], y_eval, fit_values)
    assert first is nan_candidate and second is nan_candidate
