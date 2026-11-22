"""Audit 2026-11-20 finding 35-F26: smeared_prediction uses the same residual quantiles as the fit path.

It used to compute unweighted quantiles over every row, so a fit with sample weights was gated with a different predictor.
"""

import numpy as np

from mlframe.training.composite.estimator import _smearing


class _Const:
    """A model predicting a constant."""

    def predict(self, X):
        """Return the constant for every row."""
        return np.full(len(X), 0.5)


def test_smeared_prediction_uses_residual_quantiles_of_the_fit_path():
    """The quantiles smeared_prediction uses equal residual_quantiles on the same inputs."""
    rng = np.random.default_rng(0)
    n = 300
    X = rng.normal(size=(n, 2))
    t_fit = rng.normal(size=n)
    t_hat = np.zeros(n)
    seen = {}
    real = _smearing.smeared_inverse

    def spy(inverse, t, q):
        """Record the quantiles that the gate uses, then delegate."""
        seen["q"] = q
        return real(inverse, t, q)

    orig = _smearing.smeared_inverse
    _smearing.smeared_inverse = spy
    try:
        _smearing.smeared_prediction("log_y", _Const(), X, t_fit, t_hat, lambda t: np.exp(t))
    finally:
        _smearing.smeared_inverse = orig
    expected = _smearing.residual_quantiles(_Const(), X, t_fit, seed=0)
    np.testing.assert_allclose(seen["q"], expected)
