"""The usability scorers' multi-class log loss equals sklearn's ``log_loss`` over the full label set, with and without a class absent from the fold."""

import numpy as np
import pytest
from sklearn.metrics import log_loss

from mlframe.feature_selection.filters._usability_aware_selection import _usability_scorers


@pytest.mark.parametrize("n_classes", [2, 3, 5])
@pytest.mark.parametrize("dtype", [np.float64, np.float32])
def test_logloss_equals_sklearn(n_classes, dtype):
    """Probabilities with exact 0 and 1 entries (clipped) score the same as sklearn's value."""
    rng = np.random.default_rng(n_classes)
    y = rng.integers(0, n_classes, 5000)
    _mk, logloss, y_enc, k = _usability_scorers(True, y)
    proba = rng.random((5000, k)).astype(dtype)
    proba /= proba.sum(axis=1, keepdims=True)
    proba[:4, 0] = 0.0
    proba[:4, 1:] = 1.0 / (k - 1)
    proba[4:7, 0] = 1.0
    proba[4:7, 1:] = 0.0
    assert logloss(y_enc, proba) == pytest.approx(log_loss(y_enc, proba, labels=np.arange(k)), abs=1e-6)


def test_fewer_proba_columns_than_classes_still_uses_sklearn():
    """A model that saw fewer classes than the label set goes to sklearn's path instead of mis-indexing."""
    y = np.array([0, 1, 2, 0, 1, 2] * 50)
    _mk, logloss, y_enc, _k = _usability_scorers(True, y)
    proba = np.full((300, 3), 1 / 3)
    assert logloss(y_enc, proba) == pytest.approx(np.log(3.0))
