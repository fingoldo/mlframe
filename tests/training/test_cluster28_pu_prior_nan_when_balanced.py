"""Audit 2026-11-20 finding 28-F20: the elkan_noto prior is NaN when the proxy was balanced, and a number when it was not.

A balanced proxy's mean is not P(s=1), so the old value looked like an estimate and was not. The test fits both ways on the same data and checks the
attribute.
"""

import numpy as np
from sklearn.linear_model import LogisticRegression

from mlframe.training.pu_learning import PULearningWrapper


def _data():
    """Labelled and unlabelled rows with an unbiased subset, enough for the elkan_noto estimate."""
    rng = np.random.default_rng(0)
    X = rng.standard_normal((1000, 3))
    y_true = (X[:, 0] > 0).astype(int)
    s = y_true * (rng.random(1000) < 0.5)
    unbiased = rng.random(1000) < 0.6
    return X, s, unbiased


def test_balanced_proxy_gives_nan_prior_and_unbalanced_gives_a_number():
    """balance_proxy=True reports NaN; balance_proxy=False reports a prior in [0, 1]."""
    X, s, unbiased = _data()
    balanced = PULearningWrapper(LogisticRegression(), strategy="elkan_noto", balance_proxy=True).fit(X, s, is_unbiased=unbiased)
    plain = PULearningWrapper(LogisticRegression(), strategy="elkan_noto", balance_proxy=False).fit(X, s, is_unbiased=unbiased)
    assert np.isnan(balanced.estimated_prior_)
    assert 0.0 <= plain.estimated_prior_ <= 1.0
