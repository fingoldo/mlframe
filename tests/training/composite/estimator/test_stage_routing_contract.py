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
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from mlframe.training.composite import CompositeTargetEstimator
from mlframe.training.composite.transforms import get_transform


class StageSentinelInner:
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
