"""A heavy residual tail must not stretch the residual histogram into two bars near 0."""

from __future__ import annotations

import numpy as np

from mlframe.reporting.charts.regression import _resid_hist_panel


def test_heavy_residuals_use_asinh_bins():
    rng = np.random.default_rng(0)
    yt = np.where(rng.random(20000) < 0.7, 0.0, rng.lognormal(1, 1.5, 20000))
    panel = _resid_hist_panel(yt, np.abs(yt * 0.6 + rng.normal(size=yt.size)))
    assert panel.xscale == "asinh"
    assert abs(float(np.sum(panel.values)) - 1.0) < 1e-9
    assert (np.asarray(panel.values) > 1e-3).sum() >= 20


def test_gaussian_residuals_stay_linear():
    rng = np.random.default_rng(1)
    y = rng.normal(size=5000)
    panel = _resid_hist_panel(y, y + rng.normal(scale=0.3, size=y.size))
    assert panel.xscale == "linear"
