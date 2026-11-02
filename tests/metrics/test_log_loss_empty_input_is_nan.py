"""An empty slice must not score as a perfect log loss (tracker 2026-11-02 wave 8, NAN8-5)."""

import numpy as np

from mlframe.metrics._log_loss_and_separation import fast_log_loss_binary


def test_empty_log_loss_is_nan_not_zero():
    """Zero rows give NaN, matching the Brier score, instead of the best possible loss 0.0."""
    assert np.isnan(fast_log_loss_binary(np.array([], dtype=np.int64), np.array([], dtype=np.float64)))
