"""Two contract fixes in the custom estimators.

``MyDecorrelator`` fit on an array must leave ``feature_names_in_`` absent, not set to None. ``PureRandomClassifier.predict_proba`` must refuse
input whose width differs from the fitted width instead of answering for it.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from mlframe.estimators.custom import MyDecorrelator, PureRandomClassifier


def test_decorrelator_fit_on_an_array_has_no_feature_names_attribute():
    """An array fit records the width but no names, so the attribute is absent."""
    X = np.random.default_rng(0).normal(size=(50, 3))
    est = MyDecorrelator(threshold=0.99).fit(X)
    assert not hasattr(est, "feature_names_in_")
    assert est.n_features_in_ == 3


def test_decorrelator_fit_on_a_frame_records_the_names():
    """A frame fit keeps its column names."""
    X = pd.DataFrame(np.random.default_rng(0).normal(size=(50, 3)), columns=["a", "b", "c"])
    est = MyDecorrelator(threshold=0.99).fit(X)
    assert list(est.feature_names_in_) == ["a", "b", "c"]


def test_random_classifier_refuses_a_different_width_at_predict():
    """Predicting on 7 columns after a 3-column fit raises ValueError."""
    rng = np.random.default_rng(0)
    clf = PureRandomClassifier(random_state=0).fit(rng.normal(size=(20, 3)), rng.integers(0, 2, 20))
    with pytest.raises(ValueError):
        clf.predict_proba(rng.normal(size=(5, 7)))


def test_random_classifier_predicts_at_the_fitted_width():
    """With the fitted width the probabilities come back as one row per sample."""
    rng = np.random.default_rng(0)
    clf = PureRandomClassifier(random_state=0).fit(rng.normal(size=(20, 3)), rng.integers(0, 2, 20))
    assert clf.predict_proba(rng.normal(size=(5, 3))).shape == (5, 2)
