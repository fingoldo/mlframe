"""Audit 2026-11-20 finding 28-F43: when CatBoost rejects set_params, the rejection is logged before the private fallback is written.

The rejection used to be discarded silently. The test uses a continuous 2-D target and a model whose set_params raises, and checks a
WARNING names the rejection and the private fallback is still written.
"""

import logging

import numpy as np

from mlframe.training import _training_loop_cb_utils as cb


class CatBoostRegressor:
    """A CatBoost-like model whose set_params refuses the loss and records the private fallback it is given."""

    def __init__(self):
        """Start with an empty private parameter map."""
        self._init_params = {}

    def set_params(self, **kwargs):
        """Refuse every parameter change, as a CatBoost build may."""
        raise ValueError("loss_function rejected")


def test_rejected_set_params_is_logged_and_the_fallback_still_written(caplog):
    """A WARNING names the rejection, and the private _init_params fallback receives MultiRMSE."""
    model = CatBoostRegressor()
    target = np.random.default_rng(0).standard_normal((20, 3))
    with caplog.at_level(logging.WARNING):
        cb._ensure_cb_mtr_loss(model, target)
    assert model._init_params.get("loss_function") == "MultiRMSE"
    assert any(r.levelno == logging.WARNING and "set_params rejected" in r.getMessage() for r in caplog.records)
