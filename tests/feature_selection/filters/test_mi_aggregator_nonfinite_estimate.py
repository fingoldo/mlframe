"""Regression: genie_mi_panel must not let a non-finite estimator output poison or zero the ensemble."""
import logging

import numpy as np
import pytest

from mlframe.feature_selection.filters._mi_aggregator import genie_mi_panel
from mlframe.utils.log_throttle import reset_throttle_counts


def test_genie_panel_drops_nan_estimate(caplog):
    """A NaN-returning member is dropped (with a warning) and the finite member decides the value."""
    reset_throttle_counts()
    x = np.arange(10.0)
    est = {"good": lambda a, b: 0.4, "bad": lambda a, b: float("nan")}
    with caplog.at_level(logging.WARNING):
        out = genie_mi_panel(x, x, est)
    assert out == pytest.approx(0.4, abs=1e-6)
    assert any("non-finite" in r.getMessage() for r in caplog.records)
