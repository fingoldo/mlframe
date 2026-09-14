"""A qcut failure inside the shared target encoder must be visible (mrmr_audit_2026-09-14 IMPL-5).

``encode_y_for_classif_mi`` quantile-bins a continuous target before densifying it. When ``pd.qcut`` raises, it falls back
to plain densify, which turns a continuous target into roughly one class per distinct value -- its own module docstring
names that as signal-destroying for classification MI. The fallback is kept so the fit still runs, but it used to be
logged at debug. Now that nineteen FE sites route through this helper, the degradation has to surface as a warning.
"""

import logging

import numpy as np
import pandas as pd

from mlframe.feature_selection.filters._y_encoding import encode_y_for_classif_mi
from mlframe.utils.log_throttle import reset_throttle_counts


def test_a_qcut_failure_is_warned_not_logged_at_debug(monkeypatch, caplog):
    """Force the continuous branch's qcut to raise; the fallback must still return codes AND warn."""
    reset_throttle_counts("encode_y_qcut_failed")  # the call site is throttled; start from a known budget

    def _boom(*args, **kwargs):
        """Stand in for a qcut that fails on a degenerate distribution."""
        raise ValueError("synthetic qcut failure")

    monkeypatch.setattr(pd, "qcut", _boom)
    y = np.random.default_rng(0).normal(size=200)  # > 32 distinct values -> the qcut branch
    with caplog.at_level(logging.WARNING):
        codes = encode_y_for_classif_mi(y)

    assert codes.dtype == np.int64 and codes.shape == (200,), "the fallback must still return usable codes"
    warned = [r for r in caplog.records if r.levelno >= logging.WARNING and "pd.qcut binning of a continuous target failed" in r.getMessage()]
    assert warned, "a qcut failure degraded the target silently"
    assert "ValueError" in warned[0].getMessage(), "the warning must name the exception type"


def test_a_successful_qcut_emits_no_warning(caplog):
    """Control: the ordinary continuous path must stay quiet, so the warning above means a real failure."""
    reset_throttle_counts("encode_y_qcut_failed")
    y = np.random.default_rng(1).normal(size=200)
    with caplog.at_level(logging.WARNING):
        encode_y_for_classif_mi(y)
    assert not [r for r in caplog.records if "pd.qcut binning of a continuous target failed" in r.getMessage()]
