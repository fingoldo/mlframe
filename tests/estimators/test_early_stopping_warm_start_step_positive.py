"""``warm_start_step`` below 1 is rejected instead of refitting without end (HNG4-2).

With a step of 0 the warm-start loop's count stays at 0 and, with no runtime limit, the estimator is refitted for ever. The stub estimator
here fails on its fifth fit, so a regression shows up as a quick RuntimeError rather than a hang.
"""

import numpy as np
import pytest
from sklearn.base import BaseEstimator, ClassifierMixin

from mlframe.estimators.early_stopping import EarlyStoppingWrapper


class _CountingForest(ClassifierMixin, BaseEstimator):
    """A warm-start classifier stand-in that raises on its fifth fit, so an endless loop cannot hang the test run."""

    def __init__(self, n_estimators=1, warm_start=False):
        """Store the two parameters the wrapper sets."""
        self.n_estimators = n_estimators
        self.warm_start = warm_start

    def fit(self, X, y):
        """Count fits; refuse to run a fifth time."""
        self.fits_ = getattr(self, "fits_", 0) + 1
        if self.fits_ > 4:
            raise RuntimeError("the warm-start loop refitted without end")
        self.classes_ = np.unique(y)
        return self

    def predict(self, X):
        """Predict the first class everywhere."""
        return np.full(len(X), self.classes_[0])


@pytest.mark.parametrize("step", [0, -3])
def test_non_positive_warm_start_step_is_rejected(step):
    """A step under 1 raises a ValueError that names the parameter."""
    X = np.random.RandomState(0).randn(60, 3)
    y = (X[:, 0] > 0).astype(int)
    wrapper = EarlyStoppingWrapper(_CountingForest(), max_iter=10, warm_start_step=step, validation_fraction=0.2)
    with pytest.raises(ValueError, match="warm_start_step"):
        wrapper.fit(X, y)
