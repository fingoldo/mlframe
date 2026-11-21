"""apply_cutpoints refuses a non-finite prediction instead of grading it as the top class (audit 2026-11-20, finding 17-F6)."""
import numpy as np
import pytest

from mlframe.metrics.classification._ordinal_cutpoints import apply_cutpoints


def test_nan_prediction_is_refused():
    """A NaN used to come back as the highest grade (digitize sorts it last)."""
    with pytest.raises(ValueError, match="non-finite"):
        apply_cutpoints(np.array([np.nan, 0.2]), np.array([0.5]), 2)


def test_finite_predictions_are_graded():
    """Finite predictions keep their grades."""
    assert apply_cutpoints(np.array([0.1, 0.9]), np.array([0.5]), 2).tolist() == [0, 1]
