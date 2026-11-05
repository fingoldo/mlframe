"""``IdentityEstimator`` records the width and names of its input at fit, as sklearn estimators do.

Without ``n_features_in_`` the sklearn contract checks fail, and a downstream caller cannot confirm the column layout it fitted on.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from mlframe.estimators.custom import IdentityClassifier, IdentityEstimator


def test_identity_regressor_records_width_and_names_from_a_frame():
    """A DataFrame fit records its column count and names."""
    X = pd.DataFrame({"a": [1.0, 2.0], "b": [3.0, 4.0], "c": [5.0, 6.0]})
    est = IdentityEstimator(feature_names=["a"]).fit(X, np.array([0.0, 1.0]))
    assert est.n_features_in_ == 3
    assert list(est.feature_names_in_) == ["a", "b", "c"]


def test_identity_regressor_records_width_from_an_array():
    """An array fit records its column count and no names."""
    est = IdentityEstimator(feature_indices=[0]).fit(np.zeros((4, 2)), np.zeros(4))
    assert est.n_features_in_ == 2
    assert not hasattr(est, "feature_names_in_")


def test_identity_classifier_records_width_and_classes():
    """The classifier variant records the input width alongside its classes."""
    est = IdentityClassifier(feature_indices=[0]).fit(np.zeros((4, 3)), np.array([0, 1, 1, 0]))
    assert est.n_features_in_ == 3
    assert list(est.classes_) == [0, 1]
