"""The identity estimators raise ``NotFittedError`` before fit, as sklearn estimators do.

``predict`` used to return the selected columns from an unfitted instance, and ``predict_proba`` failed with an ``AttributeError`` on the missing
``classes_`` instead.
"""

from __future__ import annotations

import numpy as np
import pytest
from sklearn.exceptions import NotFittedError

from mlframe.estimators.custom import IdentityClassifier, IdentityRegressor


def test_identity_regressor_predict_before_fit_raises():
    """An unfitted identity regressor refuses to predict."""
    X = np.arange(12.0).reshape(4, 3)
    with pytest.raises(NotFittedError):
        IdentityRegressor(feature_indices=[0]).predict(X)


def test_identity_classifier_predict_proba_before_fit_raises_not_fitted():
    """An unfitted identity classifier raises NotFittedError from predict_proba, not an AttributeError on classes_."""
    X = np.arange(12.0).reshape(4, 3)
    with pytest.raises(NotFittedError):
        IdentityClassifier(feature_indices=[0]).predict_proba(X)


def test_identity_regressor_predicts_after_fit():
    """After fit the selected column comes back verbatim."""
    X = np.arange(12.0).reshape(4, 3)
    out = IdentityRegressor(feature_indices=[1]).fit(X, np.zeros(4)).predict(X)
    np.testing.assert_array_equal(out, X[:, [1]])
