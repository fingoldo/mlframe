"""Audit 2026-11-20 finding 35-F54: predict before fit raises sklearn's NotFittedError, which is also a ValueError."""

import numpy as np
import pytest
from sklearn.exceptions import NotFittedError

from mlframe.training.composite.dual_direction import DualDirectionCompositeEstimator


def test_predict_before_fit_raises_not_fitted_error():
    """The unfitted estimator raises NotFittedError, and it is still a ValueError for older callers."""
    est = DualDirectionCompositeEstimator.__new__(DualDirectionCompositeEstimator)
    est.__dict__.clear()
    with pytest.raises(NotFittedError):
        est.predict(np.zeros((2, 1)))
    assert issubclass(NotFittedError, ValueError)
