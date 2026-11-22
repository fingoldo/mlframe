"""Audit 2026-11-20 finding 35-F15: the sample-weight fallback logs at WARNING, cannot hit an unbound kwargs name, and rejects non-finite weights.

Before the fix a signature failure reached the handler with ``_kwargs`` unbound (UnboundLocalError), the unweighted fallback was logged at DEBUG,
and NaN weights were passed on silently.
"""

import logging
from unittest import mock

import numpy as np
import pytest

from mlframe.training.composite.ensemble import _maybe_pass_sample_weight


class _Recorder:
    """A fit-callable whose fit records whether it received sample_weight."""

    def __init__(self, raise_weighted=False):
        self.calls = []
        self.raise_weighted = raise_weighted

    def fit(self, X, y, sample_weight=None):
        """Record the call; raise for a weighted call when asked."""
        self.calls.append(sample_weight is not None)
        if sample_weight is not None and self.raise_weighted:
            raise ValueError("weighted fit broke")
        return self


def test_non_finite_sample_weight_is_refused():
    """NaN weights raise ValueError at entry."""
    X, y = np.zeros((4, 1)), np.arange(4.0)
    with pytest.raises(ValueError, match="non-finite"):
        _maybe_pass_sample_weight(_Recorder(), X, y, np.array([1.0, np.nan, 1.0, 1.0]))


def test_signature_failure_does_not_crash_the_handler():
    """inspect.signature raising no longer reaches an unbound _kwargs: the sample-weight-only retry runs."""
    X, y = np.zeros((4, 1)), np.arange(4.0)
    est = _Recorder()
    with mock.patch("inspect.signature", side_effect=ValueError("no signature")):
        out = _maybe_pass_sample_weight(est, X, y, np.ones(4))
    assert out is est
    assert est.calls and est.calls[-1] is True


def test_unweighted_fallback_is_logged_at_warning(caplog):
    """When the weighted fit raises, the fallback to the unweighted fit is logged at WARNING."""
    X, y = np.zeros((4, 1)), np.arange(4.0)
    est = _Recorder(raise_weighted=True)
    with caplog.at_level(logging.WARNING):
        _maybe_pass_sample_weight(est, X, y, np.ones(4))
    assert any(r.levelno == logging.WARNING and "UNWEIGHTED" in r.getMessage() for r in caplog.records)
