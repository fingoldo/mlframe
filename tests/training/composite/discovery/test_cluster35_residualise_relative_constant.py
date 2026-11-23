"""Audit 2026-11-20 finding 35-F46: _residualise treats a column as constant only relative to its own magnitude.

An absolute 1e-12 called a column of tiny values constant even when it varied; a truly constant column must still be detected.
"""

import numpy as np

from mlframe.training.composite.discovery.screening import _residualise


def test_tiny_varying_column_is_regressed_not_treated_as_constant():
    """x around 1e-13 with spread about 1e-13 is a real regressor: the residual is not just y minus its mean."""
    rng = np.random.default_rng(0)
    x = 1e-13 * (1.0 + rng.normal(size=200))
    y = 3.0 * x + 1e-14 * rng.normal(size=200)
    res = _residualise(y, x)
    assert np.std(res) < 0.5 * np.std(y - y.mean())


def test_truly_constant_column_still_centres_y():
    """A constant x returns y minus its mean."""
    y = np.arange(10.0)
    x = np.full(10, 4.0)
    np.testing.assert_allclose(_residualise(y, x), y - y.mean())
