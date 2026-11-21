"""The power-zero Tweedie deviance refuses mismatched lengths instead of broadcasting (audit 2026-11-20, finding 17-F3)."""
import numpy as np
import pytest

from mlframe.metrics.regression._regression_deviance import fast_tweedie_deviance


def test_power_zero_rejects_a_column_prediction():
    """A (N,) target with an (N, 1) prediction used to broadcast to N*N pairs and average them."""
    y = np.arange(1.0, 6.0)
    p = y.reshape(-1, 1)
    with pytest.raises(ValueError):
        fast_tweedie_deviance(y, p, power=0.0)


def test_power_zero_matches_mse_on_matching_lengths():
    """Matching lengths keep the squared-error value."""
    y = np.array([1.0, 2.0, 3.0])
    p = np.array([1.0, 2.5, 2.0])
    assert abs(fast_tweedie_deviance(y, p, power=0.0) - np.mean((y - p) ** 2)) < 1e-12
