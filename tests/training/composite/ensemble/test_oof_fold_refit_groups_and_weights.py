"""The ensemble's per-fold transform refit passes the fold's groups and weights, and a grouped component survives OOF.

The refit called ``transform.fit(y, base)`` and ``transform.forward(y, base, params)`` without groups: a grouped transform
raised in the fit (the fold then reused the full-train params, which saw the fold's holdout) and again in the forward, so
a grouped component never produced an OOF column and dropped out of the ensemble. Weighted fits ran unweighted.
"""

from __future__ import annotations

import dataclasses

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LinearRegression

from mlframe.training.composite import CompositeTargetEstimator
from mlframe.training.composite.ensemble import compute_oof_holdout_predictions
from mlframe.training.composite.transforms.registry import _TRANSFORMS_REGISTRY


def _grouped_component(n: int = 600, seed: int = 0):
    """A fitted ``linear_residual_grouped`` wrapper, its spec, and the frame it was fitted on."""
    rng = np.random.default_rng(seed)
    grp = rng.integers(0, 4, n)
    base = rng.uniform(1.0, 10.0, n)
    feat = rng.normal(size=n)
    y = (1.0 + 0.3 * grp) * base + 0.5 * feat + rng.normal(0.0, 0.1, n)
    X = pd.DataFrame({"base": base, "feat": feat, "grp": grp})
    est = CompositeTargetEstimator(base_estimator=LinearRegression(), transform_name="linear_residual_grouped", base_column="base", group_column="grp")
    est.fit(X, y)
    spec = {"name": "y-lrg-base", "transform_name": "linear_residual_grouped", "base_column": "base", "fitted_params": dict(est.fitted_params_)}
    return est, spec, X, y, grp.astype(np.int64), base


@pytest.fixture
def fit_spy(monkeypatch):
    """Record the ``groups`` / ``sample_weight`` lengths every ``linear_residual_grouped.fit`` call receives."""
    calls: list[dict] = []
    t = _TRANSFORMS_REGISTRY["linear_residual_grouped"]
    orig = t.fit

    def _spy(y, base, *args, **kwargs):
        calls.append({"n": len(y), "groups": None if kwargs.get("groups") is None else len(kwargs["groups"]), "sw": kwargs.get("sample_weight") is not None})
        return orig(y, base, *args, **kwargs)

    _spy.__signature__ = __import__("inspect").signature(orig)
    monkeypatch.setitem(_TRANSFORMS_REGISTRY, "linear_residual_grouped", dataclasses.replace(t, fit=_spy))
    return calls


def test_the_kfold_refit_passes_fold_groups_and_the_grouped_component_survives(fit_spy):
    """Every per-fold refit receives groups of fold length, and the grouped component gets an OOF column."""
    est, spec, X, y, grp, base = _grouped_component()
    fit_spy.clear()  # the component's own fit above is not a per-fold refit
    _preds, _y_hold, surviving = compute_oof_holdout_predictions(
        component_models=[est], component_names=["lrg"], component_specs=[spec], train_X=X, y_train_full=y,
        base_train_full_per_spec={"y-lrg-base": base}, holdout_frac=0.2, random_state=0, kfold=3, group_ids=grp,
    )
    assert fit_spy, "the per-fold refit never reached the transform's fit"
    assert all(c["groups"] == c["n"] for c in fit_spy), f"a refit ran without fold-length groups: {fit_spy}"
    assert surviving == ["lrg"], "the grouped component must produce an OOF column instead of dropping out"


def test_the_refit_passes_the_fold_weights_where_the_fit_takes_them(fit_spy):
    """A weighted suite refits each fold weighted, like the deployed spec."""
    est, spec, X, y, grp, base = _grouped_component()
    fit_spy.clear()  # the component's own fit above is not a per-fold refit
    compute_oof_holdout_predictions(
        component_models=[est], component_names=["lrg"], component_specs=[spec], train_X=X, y_train_full=y,
        base_train_full_per_spec={"y-lrg-base": base}, holdout_frac=0.2, random_state=0, kfold=3, group_ids=grp,
        sample_weight=np.ones(len(y)),
    )
    accepts_weights = "sample_weight" in __import__("inspect").signature(_TRANSFORMS_REGISTRY["linear_residual_grouped"].fit).parameters
    assert fit_spy and all(c["sw"] == accepts_weights for c in fit_spy)
