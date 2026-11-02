"""BinaryPostCalibrator.predict_proba before fit raises NotFittedError (tracker 2026-11-02 wave 14, SKL14-7)."""

import numpy as np
import pytest
from sklearn.exceptions import NotFittedError
from sklearn.isotonic import IsotonicRegression

from mlframe.calibration.post import BinaryPostCalibrator


def test_predict_proba_before_fit():
    """The unfitted library calibrator used to raise its own error, or AttributeError on the Venn-Abers path."""
    cal = BinaryPostCalibrator(IsotonicRegression(), transform_method_name="predict")
    with pytest.raises(NotFittedError):
        cal.predict_proba(np.array([0.2, 0.8]))
