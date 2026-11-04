"""The honest RMSE gate forwards and inverts a grouped transform with the discovery's group labels."""

from __future__ import annotations

import types

import numpy as np
import pytest

from mlframe.training.composite.discovery._honest_rmse_gate import _spec_holdout_prediction
from mlframe.training.composite.transforms import get_transform
from mlframe.training.composite.transforms._call_gateway import call_transform


def _bed():
    """Group-level base levels, a fitted grouped linear-residual transform and a stand-in tiny model that predicts the mean transformed target."""
    rng = np.random.default_rng(0)
    n_groups, per = 6, 100
    groups = np.repeat(np.arange(n_groups), per)
    base = 100.0 + 50.0 * groups + rng.normal(0, 3, groups.size)
    y = base + 2.0 * rng.normal(size=groups.size)
    transform = get_transform("linear_residual_grouped")
    params = call_transform(transform, "fit", y, base, groups=groups)
    fit_idx = np.arange(0, groups.size, 2)
    eval_idx = np.arange(1, groups.size, 2)
    self = types.SimpleNamespace(_group_ids_for_rerank=groups)
    spec = types.SimpleNamespace(name="s", transform_name="linear_residual_grouped")
    valid = np.ones(fit_idx.size, dtype=bool)
    fit_predict = lambda t, valid_mask=None: np.full(eval_idx.size, float(np.mean(t)))
    return self, spec, transform, params, y, base, fit_idx, eval_idx, valid, fit_predict


def test_grouped_transform_is_scored_not_rejected_for_missing_groups():
    """Forward and inverse used to be called without groups and raised 'requires groups' for every grouped spec."""
    self, spec, transform, params, y, base, fit_idx, eval_idx, valid, fit_predict = _bed()
    y_hat = _spec_holdout_prediction(
        self, spec, transform, params, y[fit_idx], base[fit_idx], base[eval_idx], valid, fit_idx, eval_idx, fit_predict, {"q": None}
    )
    assert y_hat.shape == (eval_idx.size,) and np.isfinite(y_hat).all()
    assert np.corrcoef(y_hat, y[eval_idx])[0, 1] > 0.9  # the group levels carry through the inverse


def test_missing_group_ids_give_a_clear_error():
    """Without group ids the spec cannot be scored, and the reason says so (the caller turns it into a rejection)."""
    self, spec, transform, params, y, base, fit_idx, eval_idx, valid, fit_predict = _bed()
    self._group_ids_for_rerank = None
    with pytest.raises(ValueError, match="needs group ids"):
        _spec_holdout_prediction(self, spec, transform, params, y[fit_idx], base[fit_idx], base[eval_idx], valid, fit_idx, eval_idx, fit_predict, {"q": None})
