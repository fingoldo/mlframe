"""Cluster 29 regression: zero restarts or zero trials in the dual-optimizer blend raise a clear ValueError (audit 2026-11-20, finding 29-F35).

Previously n_restarts=0 hit a bare assert and n_optuna_trials=0 raised a raw error from study.best_params. Both now name the parameter that must be
at least 1; a normal call still returns a weight vector.
"""

import inspect  # noqa: F401

import numpy as np
import pytest

from mlframe.votenrank import dual_optimizer_blend as dob


def _mse(y_true, y_pred):
    """Mean squared error, the loss the blend minimises."""
    return float(np.mean((np.asarray(y_true) - np.asarray(y_pred)) ** 2))


def _preds(n_models=3, n_rows=40, seed=0):
    """Random model predictions and a target vector."""
    rng = np.random.default_rng(seed)
    y = rng.normal(size=n_rows)
    preds = np.stack([y + rng.normal(size=n_rows) * s for s in (0.1, 0.3, 0.5)[:n_models]])
    return preds, y


def _public_blend():
    """The public dual-optimizer weight blend."""
    return dob.dual_optimizer_weight_blend


def test_zero_restarts_is_a_clear_error():
    """n_restarts=0 raises a ValueError naming n_restarts, not a bare AssertionError (29-F35)."""
    fn = _public_blend()
    preds, y = _preds()
    with pytest.raises(ValueError, match="restarts"):
        fn(list(preds), y, _mse, n_restarts=0)


def test_zero_trials_is_a_clear_error():
    """n_optuna_trials=0 raises a ValueError naming it (29-F35)."""
    fn = _public_blend()
    preds, y = _preds()
    with pytest.raises(ValueError, match="n_optuna_trials"):
        fn(list(preds), y, _mse, n_optuna_trials=0)
