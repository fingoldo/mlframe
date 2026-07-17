"""Regression test for CON5: BorutaShap's default surrogate forest was created with no ``random_state``, so two runs with the same configured ``random_state`` produced
DIFFERENT forests (the docstring claimed determinism from a fixed seed). The fix passes ``self.random_state`` into RandomForestClassifier/Regressor.
"""

from __future__ import annotations

import numpy as np


def _make(random_state, classification):
    """Helper that make."""
    from mlframe.feature_selection.boruta_shap import BorutaShap

    # sklearn clone-ability: the verbatim ``model`` param stays None; ``check_model`` resolves the default
    # seeded RandomForest into the learned ``model_`` attribute -- which is where the CON5 seed lands.
    bs = BorutaShap(model=None, classification=classification, random_state=random_state)
    bs.check_model()
    return bs.model_


def test_con5_default_forest_is_seeded_classifier():
    """Con5 default forest is seeded classifier."""
    rng = np.random.default_rng(0)
    X = rng.normal(size=(120, 6))
    y = (X[:, 0] + X[:, 1] > 0).astype(int)

    m1 = _make(7, classification=True)
    m2 = _make(7, classification=True)
    m3 = _make(11, classification=True)
    m1.fit(X, y)
    m2.fit(X, y)
    m3.fit(X, y)

    assert np.array_equal(m1.feature_importances_, m2.feature_importances_), "same random_state must give identical forest importances"
    assert not np.array_equal(m1.feature_importances_, m3.feature_importances_), "different random_state should differ"


def test_con5_default_forest_is_seeded_regressor():
    """Con5 default forest is seeded regressor."""
    rng = np.random.default_rng(1)
    X = rng.normal(size=(120, 6))
    y = X[:, 0] * 2.0 + rng.normal(scale=0.1, size=120)

    m1 = _make(3, classification=False)
    m2 = _make(3, classification=False)
    m1.fit(X, y)
    m2.fit(X, y)

    assert np.array_equal(m1.feature_importances_, m2.feature_importances_)
