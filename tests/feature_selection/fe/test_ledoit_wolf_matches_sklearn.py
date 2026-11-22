"""``ledoit_wolf_fit`` gives scikit-learn's ``LedoitWolf`` location, covariance, precision and shrinkage, and the Mahalanobis family keeps its values."""

from __future__ import annotations

import numpy as np
import pytest

from mlframe.feature_selection.filters._ledoit_wolf import ledoit_wolf_fit


@pytest.mark.parametrize("n,p", [(40, 30), (300, 8), (2000, 25), (500, 1)])
def test_matches_sklearn_ledoit_wolf(n, p):
    """Includes the wide case (n close to p) the shrinkage exists for, and a single column."""
    from sklearn.covariance import LedoitWolf

    rng = np.random.default_rng(n + p)
    X = rng.standard_normal((n, p)) @ rng.standard_normal((p, p)) + rng.normal(size=p)
    ref = LedoitWolf().fit(X)
    location, covariance, precision, shrinkage = ledoit_wolf_fit(X)
    np.testing.assert_allclose(location, ref.location_, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(covariance, ref.covariance_, rtol=1e-9, atol=1e-12)
    assert shrinkage == pytest.approx(float(ref.shrinkage_), abs=1e-12)
    np.testing.assert_allclose(precision, ref.get_precision(), rtol=1e-6, atol=1e-8)


def test_a_constant_column_does_not_break_the_fit():
    """A zero-variance column gives a finite precision through the pseudo-inverse, as in scikit-learn."""
    from sklearn.covariance import LedoitWolf

    rng = np.random.default_rng(1)
    X = rng.standard_normal((200, 5))
    X[:, 2] = 4.0
    ref = LedoitWolf().fit(X).get_precision()
    assert np.isfinite(ledoit_wolf_fit(X)[2]).all()
    np.testing.assert_allclose(ledoit_wolf_fit(X)[2], ref, rtol=1e-6, atol=1e-8)
