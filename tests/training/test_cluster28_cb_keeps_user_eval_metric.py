"""Audit 2026-11-20 finding 28-F44: a CatBoost metric the user set is not overwritten by the MTR loss default.

The MTR path used to set eval_metric to MultiRMSE whatever the user had chosen. The test gives a continuous 2-D target and a model that
already has eval_metric="AUC", and checks set_params receives only the loss.
"""

import numpy as np

from mlframe.training import _training_loop_cb_utils as cb


class CatBoostRegressor:
    """A CatBoost-named model that records the set_params calls it receives."""

    def __init__(self, eval_metric):
        """Hold the user's eval_metric and an empty record of set_params calls."""
        self._params = {"eval_metric": eval_metric}
        self.calls = []

    def get_params(self):
        """Return the current parameters."""
        return dict(self._params)

    def set_params(self, **kwargs):
        """Record the call and apply it."""
        self.calls.append(kwargs)
        self._params.update(kwargs)


def test_user_eval_metric_survives_the_mtr_loss_default():
    """With eval_metric set by the user, set_params is called with the loss alone."""
    model = CatBoostRegressor(eval_metric="AUC")
    cb._ensure_cb_mtr_loss(model, np.random.default_rng(0).standard_normal((20, 3)))
    assert model.calls == [{"loss_function": "MultiRMSE"}]
