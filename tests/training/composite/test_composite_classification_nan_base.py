"""The composite classifier's default base-margin LogisticRegression must tolerate NaN features.

Without an imputer in front of the base LogisticRegression, a fit on any NaN-carrying feature frame raised
"Input X contains NaN", so composite classification could not run on data with missing values.
"""

from __future__ import annotations

import numpy as np
import pytest


def test_default_base_margin_fits_with_nan_features():
    """Composite classification fits and predicts on a feature matrix with NaN cells."""
    pytest.importorskip("lightgbm")
    from lightgbm import LGBMClassifier
    from mlframe.training.composite.classification import CompositeClassificationEstimator

    rng = np.random.default_rng(0)
    X = rng.normal(size=(300, 4))
    X[::7, 1] = np.nan
    y = (X[:, 0] > 0).astype(int)
    model = CompositeClassificationEstimator(base_estimator=LGBMClassifier(n_estimators=20, verbose=-1))
    model.fit(X, y)
    assert model.predict_proba(X[:5]).shape == (5, 2)
