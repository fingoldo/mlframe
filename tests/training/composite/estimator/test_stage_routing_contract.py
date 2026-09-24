"""Every wrapper and predict entry point feeds the inner its own pipeline stage and the base its raw stage.

The inner of a composite wrapper was trained on the pre_pipeline output while its base must stay on the raw frame; routes
that fed the inner raw X (RMSE 165.8 against a 0.99 oracle) or the base the scaled X (RMSE 419) shipped, and the wrap-pass
watchdog could not see it because it re-ran the same route. Here an independent oracle: the inner raises when it receives a
frame whose per-column fingerprint is not its training stage's, and the y-scale RMSE exposes a base read at the wrong stage.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from mlframe.training.composite import CompositeTargetEstimator
from mlframe.training.composite.transforms import get_transform


class StageSentinelInner(BaseEstimator, RegressorMixin):
    """A linear inner that remembers the per-column mean/std of the frame it was fit on and refuses any other stage."""

    def fit(self, X, y):
        """Fit a linear model and fingerprint the training frame."""
        X = pd.DataFrame(X)
        self.columns_ = list(X.columns)
        self.mean_, self.std_ = X.mean().to_numpy(), X.std().to_numpy()
        self.model_ = LinearRegression().fit(X.to_numpy(), np.asarray(y))
        return self

    def predict(self, X):
        """Predict after checking the incoming frame is the training stage (same columns, means within 1 sd of the fit's)."""
        X = pd.DataFrame(X)
        assert list(X.columns) == self.columns_, f"inner got columns {list(X.columns)}, trained on {self.columns_}"
        drift = np.abs(X.mean().to_numpy() - self.mean_) / np.maximum(self.std_, 1e-9)
        assert np.all(drift < 1.0), f"inner got a frame from another pipeline stage (mean drift {drift.round(1)} sd)"
        return self.model_.predict(X.to_numpy())


def _fixture(n: int = 400, seed: int = 0):
    """Raw frame with a large-scale base, a fitted StandardScaler stage, y, and the linear_residual T and params."""
    rng = np.random.default_rng(seed)
    X = pd.DataFrame({"b": rng.uniform(100.0, 200.0, n), "x0": rng.normal(5.0, 2.0, n), "x1": rng.normal(-3.0, 1.0, n)})
    y = 2.0 * X["b"].to_numpy() + 3.0 * X["x0"].to_numpy() + rng.normal(0.0, 0.5, n)
    pp = Pipeline([("scale", StandardScaler())]).set_output(transform="pandas").fit(X)
    t = get_transform("linear_residual")
    params = t.fit(y, X["b"].to_numpy())
    T = t.forward(y, X["b"].to_numpy(), params)
    inner = StageSentinelInner().fit(pp.transform(X), T)
    return X, y, pp, params, inner


def _rmse(p, y):
    """Root mean squared error."""
    return float(np.sqrt(np.mean((np.asarray(p, dtype=float) - y) ** 2)))


def _wrapper(inner, params, y, X, pp=None):
    """A composite wrapper around the sentinel inner, optionally carrying the inner's pipeline."""
    return CompositeTargetEstimator.from_fitted_inner(
        fitted_inner=inner, transform_name="linear_residual", base_column="b", transform_fitted_params=params, y_train=y,
        inner_pre_pipeline=pp, base_train=X["b"].to_numpy(),
    )


def test_a_wrapper_with_its_pipeline_routes_raw_to_the_base_and_the_stage_to_the_inner():
    """``predict(raw X)`` on a wrapper carrying ``inner_pre_pipeline``: the inner sees its stage, the base stays raw."""
    X, y, pp, params, inner = _fixture()
    assert _rmse(_wrapper(inner, params, y, X, pp).predict(X), y) < 1.0


def test_the_suite_predict_route_feeds_a_pipeline_less_wrapper_correctly():
    """``composite_predict`` (predict_from_models' route) computes the entry's stage for the inner and keeps the base raw."""
    from types import SimpleNamespace

    from mlframe.training.core._predict_composite_routing import composite_predict

    X, y, pp, params, inner = _fixture()
    entry = SimpleNamespace(pre_pipeline=pp)
    pred = composite_predict(_wrapper(inner, params, y, X), entry, X, None, lambda f: f)
    assert _rmse(pred, y) < 1.0


def test_the_prepipeline_shim_and_the_ct_ensemble_feed_their_components_the_stage():
    """A ``PrePipelinePredictShim`` transforms raw X for its model, and a CT ensemble of shims predicts through them."""
    from mlframe.training.composite import CompositeCrossTargetEnsemble
    from mlframe.training.composite.post_shim import PrePipelinePredictShim

    X, y, pp, params, inner = _fixture()
    raw_inner = StageSentinelInner().fit(pp.transform(X), y)
    shim = PrePipelinePredictShim(model=raw_inner, pre_pipeline=pp, name="raw")
    assert _rmse(shim.predict(X), y) < 1.0
    comp = PrePipelinePredictShim(model=_wrapper(inner, params, y, X, pp), pre_pipeline=None, name="comp")
    P = np.column_stack([shim.predict(X), comp.predict(X)])
    ens = CompositeCrossTargetEnsemble.from_nnls_stack(component_models=[shim, comp], component_names=["raw", "comp"], component_predictions=P, y_train=y)
    assert _rmse(ens.predict(X), y) < 1.0


def test_the_sentinel_catches_both_misroutes():
    """Canary: raw X to the inner raises, and a scaled base inverts T to the wrong level."""
    X, y, pp, params, inner = _fixture()
    with pytest.raises(AssertionError, match="another pipeline stage"):
        inner.predict(X)
    t = get_transform("linear_residual")
    wrong = t.inverse(inner.predict(pp.transform(X)), pp.transform(X)["b"].to_numpy(), params)
    assert _rmse(wrong, y) > 50.0


def _entry_and_spec(pp, params):
    """A suite model entry whose inner was trained on ``pp``'s output, and the composite spec it belongs to."""
    from types import SimpleNamespace

    spec = {"name": "y-linres-b", "transform_name": "linear_residual", "base_column": "b", "fitted_params": dict(params), "target_col": "y"}
    return SimpleNamespace(pre_pipeline=pp, model=None), spec


def test_the_wrap_pass_builder_routes_both_stages():
    """``build_composite_wrapper``, which the end-of-target wrap pass and the per-model hook both use, wires the entry's stage."""
    from mlframe.training.core._composite_wrap_helpers import build_composite_wrapper

    X, y, pp, params, inner = _fixture()
    entry, spec = _entry_and_spec(pp, params)
    wrapper = build_composite_wrapper(entry=entry, inner=inner, spec=spec, y_train=y, train_df=X)
    assert _rmse(wrapper.predict(X), y) < 1.0


def test_the_wrap_watchdog_reads_the_inner_at_its_stage(caplog):
    """The wrap-pass watchdog predicts the inner itself; a wrong stage would trip the sentinel and surface as its warning."""
    import logging

    from mlframe.training.core._composite_wrap_helpers import build_composite_wrapper
    from mlframe.training.core._composite_wrap_watchdog import run_wrap_watchdog

    X, y, pp, params, inner = _fixture()
    entry, spec = _entry_and_spec(pp, params)
    wrapper = build_composite_wrapper(entry=entry, inner=inner, spec=spec, y_train=y, train_df=X)
    with caplog.at_level(logging.WARNING):
        run_wrap_watchdog(wrapper, spec, X, y, composite_name="y-linres-b", split_name="val")
    assert not [r for r in caplog.records if r.levelno >= logging.WARNING], [r.getMessage() for r in caplog.records]


def test_the_per_model_hook_wraps_the_entry_at_its_stage(caplog):
    """The per-model y-scale hook swallows errors, so the check is the wrapper it leaves behind and the absence of warnings."""
    import logging

    from mlframe.training.core._phase_composite_wrapping import emit_per_model_composite_y_scale_test

    X, y, pp, params, inner = _fixture()
    entry, spec = _entry_and_spec(pp, params)
    entry.model = inner
    idx = np.arange(len(y))
    with caplog.at_level(logging.WARNING):
        emit_per_model_composite_y_scale_test(entry=entry, composite_spec=spec, orig_target_name="y", composite_name="y-linres-b",
                                              target_name="y-linres-b", y_full=y, test_idx=idx[300:], test_df_pd=X.iloc[300:],
                                              train_idx=idx[:300], train_df=X.iloc[:300])
    assert isinstance(entry.model, CompositeTargetEstimator), "the hook did not wrap the entry"
    assert _rmse(entry.model.predict(X), y) < 1.0
    assert not [r for r in caplog.records if r.levelno >= logging.WARNING], [r.getMessage() for r in caplog.records]


def test_the_moe_wrapper_feeds_every_expert_its_own_stage():
    """The deployed MoE wrapper predicts each expert on raw X; the composite and the shimmed raw model reach their stages."""
    from mlframe.training.composite._moe_gate import MoESelectionGate
    from mlframe.training.composite.post_shim import PrePipelinePredictShim
    from mlframe.training.core._phase_composite_post_lag_predict import _LagPredictDeployableModel
    from mlframe.training.core._phase_composite_post_moe import _MoEGatedDeployableModel

    X, y, pp, params, inner = _fixture()
    raw = PrePipelinePredictShim(model=StageSentinelInner().fit(pp.transform(X), y), pre_pipeline=pp, name="raw")
    moe = _MoEGatedDeployableModel(composite_model=_wrapper(inner, params, y, X, pp), raw_model=raw, lag_model=_LagPredictDeployableModel("b"),
                                   gate=MoESelectionGate(failsafe="lag"), group_column=None)
    moe.gate.fit(y, moe._expert_preds(X))
    assert _rmse(moe.predict(X), y) < 1.0


def test_the_oof_refits_feed_the_cloned_inner_its_stage():
    """The honest-OOF path refits a clone of the inner per fold and predicts the holdout at the same stage."""
    from mlframe.training.composite.ensemble import compute_oof_holdout_predictions

    X, y, pp, params, inner = _fixture()
    spec = {"name": "y-linres-b", "transform_name": "linear_residual", "base_column": "b", "fitted_params": dict(params)}
    oof, y_hold, surviving = compute_oof_holdout_predictions(
        component_models=[_wrapper(inner, params, y, X, pp)], component_names=["c"], component_specs=[spec], train_X=X, y_train_full=y,
        base_train_full_per_spec={"y-linres-b": X["b"].to_numpy()}, holdout_frac=0.2, random_state=0, kfold=3,
    )
    assert surviving == ["c"], "the component dropped out of OOF (a sentinel assertion inside a fold is one way)"
    assert _rmse(oof[:, 0], y_hold) < 1.0


# Classes that hold a fitted inner (``self.estimator_``) and extract a base themselves: each routes two stages, so each
# needs an entry-point test above.
_TWO_STAGE_CLASSES = {
    "composite/estimator/_estimator.py::CompositeTargetEstimator": "test_a_wrapper_with_its_pipeline_routes_raw_to_the_base_and_the_stage_to_the_inner",
}


def test_every_two_stage_class_is_covered():
    """A new class holding ``self.estimator_`` and calling a base extractor must join the stage-routing contract."""
    import ast
    from pathlib import Path

    import mlframe

    root = Path(mlframe.__file__).resolve().parent / "training"
    found = set()
    for path in sorted(root.rglob("*.py")):
        if "_benchmarks" in path.parts:
            continue
        for cls in (n for n in ast.walk(ast.parse(path.read_text(encoding="utf-8"))) if isinstance(n, ast.ClassDef)):
            nodes = list(ast.walk(cls))
            holds = any(isinstance(a, ast.Attribute) and a.attr == "estimator_" and isinstance(a.value, ast.Name) and a.value.id == "self" for a in nodes)
            names = [getattr(c.func, "attr", None) or getattr(c.func, "id", "") or "" for c in nodes if isinstance(c, ast.Call)]
            if holds and any("extract" in n and "base" in n for n in names):
                found.add(f"{path.relative_to(root).as_posix()}::{cls.name}")
    assert found == set(_TWO_STAGE_CLASSES), sorted(found ^ set(_TWO_STAGE_CLASSES))
    assert all(t in globals() for t in _TWO_STAGE_CLASSES.values())
