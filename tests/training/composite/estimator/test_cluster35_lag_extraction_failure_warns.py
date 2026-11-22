"""Audit 2026-11-20 finding 35-F24: a failed lag-column extraction is logged at WARNING, not DEBUG.

Deep-OOD rows fall back to the median instead of the causal lag, and that changes predictions, so it must be visible.
"""

import logging
from types import SimpleNamespace
from unittest import mock

import numpy as np

from mlframe.training.composite.estimator import _soft_shrink


def test_failed_lag_extraction_is_logged_at_warning(caplog):
    """When the lag column cannot be extracted, the function returns None and a WARNING names the column."""
    self_ = SimpleNamespace(base_column="b")
    X = np.zeros((4, 2))
    with (
        mock.patch.object(_soft_shrink, "_resolve_lag_column", return_value="lag_col"),
        mock.patch("mlframe.training.composite.estimator._extract_base", side_effect=KeyError("lag_col")),
    ):
        with caplog.at_level(logging.WARNING):
            out = _soft_shrink._resolve_lag_values(self_, X, np.zeros(4), 4)
    assert out is None
    assert any(r.levelno == logging.WARNING and "lag_col" in r.getMessage() for r in caplog.records)
