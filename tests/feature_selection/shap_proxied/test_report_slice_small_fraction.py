"""A small report fraction of a small holdout does not crash the fit; it falls back to an unstratified slice."""

from __future__ import annotations

import numpy as np
import pandas as pd

from mlframe.feature_selection.shap_proxied_fs._shap_proxied_report_slice import split_report_slice


def _data(n=40):
    """Frame, binary target and the holdout index covering all rows."""
    rng = np.random.default_rng(0)
    X = pd.DataFrame(rng.normal(size=(n, 2)), columns=["a", "b"])
    y = np.tile([0, 1], n // 2)
    return np.arange(n), X, y


def test_one_row_slice_of_a_binary_holdout_does_not_raise():
    """30 rows, fraction 0.03 -> 1 report row: stratifying 2 classes over 1 row raised in train_test_split."""
    idx, X, y = _data(30)
    keep, rep = split_report_slice(idx, X, y, 0.03, True, 0)
    assert rep is not None and len(rep[1]) == 1 and len(keep) == 29


def test_fraction_of_one_or_more_disables_the_slice():
    """fraction >= 1 would leave nothing for selection: the slice is skipped."""
    idx, X, y = _data(40)
    keep, rep = split_report_slice(idx, X, y, 1.0, True, 0)
    assert rep is None and len(keep) == 40


def test_normal_fraction_still_stratifies():
    """40 rows, fraction 0.25: ten report rows with both classes present (negative control)."""
    idx, X, y = _data(40)
    _keep, rep = split_report_slice(idx, X, y, 0.25, True, 0)
    assert rep is not None and set(rep[1].tolist()) == {0, 1}


def test_grouped_report_slice_shares_no_group_with_the_rest_of_the_holdout():
    """With group labels the slice is carved by whole groups, so the reported figure is not scored on groups the selection holdout also saw."""
    rng = np.random.default_rng(0)
    n = 200
    X = pd.DataFrame(rng.normal(size=(n, 2)), columns=["a", "b"])
    y = np.tile([0, 1], n // 2)
    groups = np.repeat(np.arange(40), 5)
    idx = np.arange(n)
    keep, rep = split_report_slice(idx, X, y, 0.25, True, 0, groups)
    assert rep is not None
    rep_rows = np.setdiff1d(idx, keep)
    assert not set(groups[keep]) & set(groups[rep_rows])
