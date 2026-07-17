"""Regression test (audit5 sklearn-P1): PureRandomClassifier.predict_proba consumed the fitted RNG state, so
two identical predict calls returned DIFFERENT outputs -- violating sklearn's predict determinism (a fitted
estimator must give the same output for the same X). It now derives a fresh RNG from a concrete fit-time seed.
"""

import numpy as np
import pytest

from mlframe.estimators.custom import PureRandomClassifier


@pytest.mark.parametrize("n_classes", [2, 3, 5])
def test_predict_proba_and_predict_are_deterministic_across_calls(n_classes):
    """Predict proba and predict are deterministic across calls."""
    rng = np.random.default_rng(0)
    X = rng.standard_normal((80, 4))
    y = rng.integers(0, n_classes, size=80)
    clf = PureRandomClassifier(random_state=7).fit(X, y)

    p1 = clf.predict_proba(X)
    p2 = clf.predict_proba(X)
    assert np.array_equal(p1, p2), "predict_proba differs across identical calls (RNG state consumed)"
    assert np.array_equal(clf.predict(X), clf.predict(X)), "predict differs across identical calls"
    # proba rows are valid distributions
    assert np.allclose(p1.sum(axis=1), 1.0)
    assert p1.shape == (80, n_classes)


def test_refit_resets_the_predict_seed():
    """Re-fitting fully resets predict-time RNG (fit idempotence): same random_state -> same predict."""
    X = np.random.default_rng(1).standard_normal((40, 3))
    y = np.random.default_rng(2).integers(0, 2, 40)
    a = PureRandomClassifier(random_state=3).fit(X, y).predict_proba(X)
    b = PureRandomClassifier(random_state=3).fit(X, y).predict_proba(X)
    assert np.array_equal(a, b), "two estimators with the same seed must predict identically"
