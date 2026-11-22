"""Audit 2026-11-20 finding 35-F23: a degenerate fit does not change the requested calibration method for later fits.

Before the fix the constant-prediction fit left method="linear", and a later valid fit on the same object silently fit a linear map.
"""

import numpy as np

from mlframe.training.composite.ensemble import OutputCalibrator


def test_valid_fit_after_degenerate_fit_uses_the_requested_method():
    """isotonic requested: fit on constant predictions, then fit on valid data -> the second fit is isotonic."""
    cal = OutputCalibrator("isotonic")
    cal.fit(np.full(20, 0.5), np.linspace(0, 1, 20))
    assert cal.method == "linear"
    rng = np.random.default_rng(0)
    p = rng.uniform(size=200)
    y = (p + 0.05 * rng.normal(size=200) > 0.5).astype(float)
    cal.fit(p, y)
    assert cal.method == "isotonic"
