"""RFECV re-validates its parameters at fit, so set_params and grid search cannot bypass the constructor checks (tracker 2026-11-02 wave 14, SKL14-8)."""

import numpy as np
import pytest
from sklearn.linear_model import LogisticRegression

from mlframe.feature_selection.wrappers import RFECV


def test_invalid_value_set_after_construction_is_rejected_at_fit():
    """stability_threshold=5.0 is outside (0, 1]; the constructor refuses it, set_params used to let it through."""
    sel = RFECV(estimator=LogisticRegression(), stability_selection=True, verbose=0)
    sel.set_params(stability_threshold=5.0)
    rng = np.random.default_rng(0)
    X = rng.normal(size=(60, 4))
    y = (X[:, 0] > 0).astype(int)
    with pytest.raises(ValueError, match="stability_threshold"):
        sel.fit(X, y)
