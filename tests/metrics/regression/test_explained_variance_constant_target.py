"""Explained variance on a constant target follows sklearn's force_finite rule (tracker 2026-11-02 wave 12, NJ12-6)."""

import numpy as np
import pytest
from sklearn.metrics import explained_variance_score

from mlframe.metrics.regression._regression_extras import fast_explained_variance


@pytest.mark.parametrize("pred", [[2.0, 2.0, 2.0], [1.0, 2.0, 3.0]])
def test_constant_target_matches_sklearn(pred):
    """A perfect fit on a constant target gives 1.0 (it was NaN); an imperfect one gives 0.0."""
    y = np.array([2.0, 2.0, 2.0])
    p = np.array(pred)
    assert fast_explained_variance(y, p) == pytest.approx(explained_variance_score(y, p))
