"""A NaN probability makes the sequential and parallel log-loss kernels agree on NaN (tracker 2026-11-02 wave 12, NJ12-5)."""

import numpy as np

from mlframe.metrics._log_loss_and_separation import fast_log_loss_binary


def test_nan_probability_gives_nan_for_small_n():
    """yp=[.2, nan, .3, .9] scored 0.1713 on the sequential path because max/min turned NaN into 1-eps."""
    assert np.isnan(fast_log_loss_binary(np.array([0, 1, 0, 1]), np.array([0.2, np.nan, 0.3, 0.9])))
