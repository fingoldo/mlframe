"""BinaryPostCalibrator fits a private copy; the calibrator parameter stays unfitted (tracker 2026-11-02 wave 14, SKL14-7b)."""

import numpy as np
from sklearn.isotonic import IsotonicRegression

from mlframe.calibration.post import BinaryPostCalibrator


def test_fit_leaves_the_constructor_parameter_unfitted():
    """sklearn estimators must not learn state in constructor parameters; the fitted copy lives in calibrator_."""
    rng = np.random.default_rng(0)
    p = rng.random(200)
    y = (rng.random(200) < p).astype(int)
    iso = IsotonicRegression(out_of_bounds="clip")
    cal = BinaryPostCalibrator(iso, transform_method_name="predict").fit(p, y)
    assert not hasattr(iso, "X_thresholds_")
    assert hasattr(cal.calibrator_, "X_thresholds_")
    assert cal.predict_proba(np.array([0.1, 0.9])).shape == (2, 2)
