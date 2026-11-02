"""The hurdle's Duan smearing factor averages over the weighted population the magnitude regressor was fit on."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from mlframe.training.composite.hurdle import HurdleRegressor


class _Zero:
    """Regressor predicting 0, so the log residual equals the log target."""

    def predict(self, X):
        """Return zeros."""
        return np.zeros(len(X))


def test_factor_is_the_weighted_mean_of_exp_residuals():
    """Residuals 0 and log(3) with weights 9 and 1 average to (9*1 + 1*3) / 10 = 1.2, not the unweighted 2.0."""
    fake = SimpleNamespace(smearing="insample", regressor_=_Zero())
    target = np.array([0.0, np.log(3.0)])
    got = HurdleRegressor._smearing_factor(fake, np.zeros((2, 1)), target, np.array([9.0, 1.0]))
    assert abs(got - 1.2) < 1e-12


def test_no_weights_keeps_the_plain_mean():
    """Without weights the factor is the plain mean (negative control)."""
    fake = SimpleNamespace(smearing="insample", regressor_=_Zero())
    got = HurdleRegressor._smearing_factor(fake, np.zeros((2, 1)), np.array([0.0, np.log(3.0)]), None)
    assert abs(got - 2.0) < 1e-12
