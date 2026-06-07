"""Regression coverage for ``PrePipelinePredictShim`` and shim-aware OOF refit.

The shim previously lived as a local class inside ``run_composite_post_processing``; ``sklearn.clone`` could not clone it (no ``get_params``) so every cross-target NNLS component was excluded from ensemble weights -- producing a "built ensemble" with zero weighted members. These tests pin:

1. The shim is sklearn-compatible (``clone`` produces a fresh shim with the inner cloned but the fitted ``pre_pipeline`` shared).
2. The shim routes ``fit`` and ``predict`` through ``pre_pipeline.transform``.
3. ``compute_oof_holdout_predictions`` does NOT drop shim-wrapped components.
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd
import pytest
from sklearn.base import BaseEstimator, RegressorMixin, clone
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from mlframe.training.composite.ensemble import compute_oof_holdout_predictions
from mlframe.training.composite.post_shim import PrePipelinePredictShim


class _RecordingScaler(BaseEstimator):
    """StandardScaler-like estimator that records every ``transform`` call so tests can assert pre_pipeline routing.
    """

    def __init__(self) -> None:
        self.calls = 0
        self.mean_: float = 0.0
        self.scale_: float = 1.0

    def fit(self, X, y=None):  # noqa: D401
        arr = np.asarray(X, dtype=np.float64)
        self.mean_ = float(arr.mean())
        self.scale_ = float(arr.std() or 1.0)
        return self

    def transform(self, X):
        self.calls += 1
        return (np.asarray(X, dtype=np.float64) - self.mean_) / self.scale_


class _StubEstimator(BaseEstimator, RegressorMixin):
    """Records training data identity so the test can assert clone semantics."""

    def __init__(self, slope: float = 1.0) -> None:
        self.slope = slope

    def fit(self, X, y, sample_weight=None):
        arr = np.asarray(X, dtype=np.float64)
        self.fit_input_mean_ = float(arr.mean())
        self.fit_n_ = int(arr.shape[0])
        self.coef_ = float(np.asarray(y).mean()) * self.slope
        return self

    def predict(self, X):
        n = int(np.asarray(X).shape[0])
        return np.full(n, self.coef_, dtype=np.float64)


def _make_X_y(n: int = 200, seed: int = 7) -> tuple[pd.DataFrame, np.ndarray]:
    rng = np.random.default_rng(seed)
    X = pd.DataFrame({"a": rng.normal(10.0, 2.0, n), "b": rng.normal(-3.0, 0.5, n)})
    y = X["a"].values * 0.3 + X["b"].values * -1.1 + rng.normal(0, 0.1, n)
    return X, y


class TestShimSemantics:
    def test_fit_routes_through_pre_pipeline(self) -> None:
        scaler = StandardScaler().fit(np.array([[1.0], [2.0], [3.0], [4.0]]))
        inner = _StubEstimator()
        shim = PrePipelinePredictShim(model=inner, pre_pipeline=scaler, name="t")
        X = np.array([[5.0], [6.0], [7.0]])
        y = np.array([0.1, 0.2, 0.3])
        shim.fit(X, y)
        expected_mean = float(scaler.transform(X).mean())
        assert inner.fit_input_mean_ == pytest.approx(expected_mean, abs=1e-9)

    def test_predict_routes_through_pre_pipeline(self) -> None:
        rec = _RecordingScaler().fit(np.array([1.0, 2.0, 3.0]).reshape(-1, 1))
        inner = _StubEstimator()
        inner.fit(np.array([[1.0]]), np.array([0.5]))
        shim = PrePipelinePredictShim(model=inner, pre_pipeline=rec, name="t")
        pre_calls = rec.calls
        shim.predict(np.array([[10.0], [20.0]]))
        assert rec.calls == pre_calls + 1

    def test_passthrough_when_pre_pipeline_none(self) -> None:
        inner = _StubEstimator()
        shim = PrePipelinePredictShim(model=inner, pre_pipeline=None, name="np")
        X = np.array([[1.0], [2.0], [3.0]])
        shim.fit(X, np.array([0.0, 0.0, 0.0]))
        assert inner.fit_input_mean_ == pytest.approx(float(X.mean()), abs=1e-9)


class TestSklearnClone:
    def test_clone_returns_shim_with_cloned_inner(self) -> None:
        inner = _StubEstimator(slope=2.5)
        pp = StandardScaler().fit(np.array([[1.0], [2.0], [3.0]]))
        shim = PrePipelinePredictShim(model=inner, pre_pipeline=pp, name="A")
        cloned = clone(shim)
        assert isinstance(cloned, PrePipelinePredictShim)
        assert cloned.name == "A"
        assert cloned.model is not inner
        assert isinstance(cloned.model, _StubEstimator)
        assert cloned.model.slope == 2.5
        # pre_pipeline must be SHARED (refitting would shift the scaling distribution).
        assert cloned.pre_pipeline is pp

    def test_clone_inner_has_no_fitted_state(self) -> None:
        inner = _StubEstimator()
        inner.fit(np.array([[1.0]]), np.array([0.5]))
        shim = PrePipelinePredictShim(model=inner, pre_pipeline=None, name="B")
        cloned = clone(shim)
        assert not hasattr(cloned.model, "coef_")

    def test_get_params_set_params_roundtrip(self) -> None:
        inner = _StubEstimator(slope=1.5)
        shim = PrePipelinePredictShim(model=inner, pre_pipeline=None, name="C")
        params = shim.get_params(deep=False)
        assert params["model"] is inner
        assert params["name"] == "C"
        new_inner = _StubEstimator(slope=4.0)
        shim.set_params(model=new_inner)
        assert shim.model is new_inner


class TestOofRefitWithShim:
    """Repro: pre-fix this triggered 'OOF refit failed ... Cannot clone object _PrePipelinePredictShim' for every component."""

    def test_raw_shim_component_survives_oof_refit(self, caplog) -> None:
        X, y = _make_X_y(n=400, seed=11)
        pp = Pipeline([("scaler", StandardScaler())]).fit(X)
        inner_raw = LinearRegression().fit(pp.transform(X), y)
        shim = PrePipelinePredictShim(model=inner_raw, pre_pipeline=pp, name="raw#0")

        with caplog.at_level(logging.WARNING, logger="mlframe.training.composite.ensemble"):
            matrix, y_hold, surviving = compute_oof_holdout_predictions(
                component_models=[shim],
                component_names=["raw#0"],
                component_specs=[None],
                train_X=X,
                y_train_full=y,
                base_train_full_per_spec={},
                holdout_frac=0.2,
                random_state=0,
            )

        assert "OOF refit failed" not in caplog.text
        assert surviving == ["raw#0"]
        assert matrix.shape[0] > 0
        assert matrix.shape[1] == 1
        assert np.all(np.isfinite(matrix))

    def test_multiple_shims_all_survive(self, caplog) -> None:
        X, y = _make_X_y(n=400, seed=13)
        pp = Pipeline([("scaler", StandardScaler())]).fit(X)
        Xt = pp.transform(X)
        shims = [
            PrePipelinePredictShim(
                model=LinearRegression().fit(Xt, y),
                pre_pipeline=pp,
                name=f"raw#{i}",
            )
            for i in range(3)
        ]

        with caplog.at_level(logging.WARNING, logger="mlframe.training.composite.ensemble"):
            matrix, _, surviving = compute_oof_holdout_predictions(
                component_models=shims,
                component_names=[s.name for s in shims],
                component_specs=[None] * 3,
                train_X=X,
                y_train_full=y,
                base_train_full_per_spec={},
                holdout_frac=0.25,
                random_state=1,
            )

        assert "OOF refit failed" not in caplog.text
        assert surviving == ["raw#0", "raw#1", "raw#2"]
        assert matrix.shape[1] == 3
