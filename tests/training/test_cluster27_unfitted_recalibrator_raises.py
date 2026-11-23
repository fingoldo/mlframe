"""Cluster 27 regression: an unfitted distributional recalibrator raises instead of returning the identity silently (audit 2026-11-20, finding 27-F21)."""

import numpy as np
import pytest

from mlframe.training._regression_calibration import DistributionalRecalibrator


def test_unfitted_recalibrate_raises():
    """Calling recalibrate before fit is an error, not a silent pass-through (27-F21)."""
    with pytest.raises(RuntimeError, match="before fit"):
        DistributionalRecalibrator().recalibrate(np.array([0.1, 0.5, 0.9]))


def test_fitted_recalibrate_still_maps():
    """Control: a fitted recalibrator returns a mapped array of the same length (27-F21)."""
    rng = np.random.default_rng(0)
    pit = rng.uniform(size=200)
    rec = DistributionalRecalibrator().fit(pit)
    out = rec.recalibrate(np.array([0.2, 0.8]))
    assert out.shape == (2,)
