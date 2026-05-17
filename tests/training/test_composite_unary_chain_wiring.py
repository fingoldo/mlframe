"""End-to-end wiring tests for Pack J/K transforms in CompositeTargetEstimator.

Pins:
- 4 unary y-transforms (``cbrt_y``, ``log_y``, ``yeo_johnson_y``, ``quantile_normal_y``) are registered with ``requires_base=False`` and run through ``CompositeTargetEstimator.fit / predict`` WITHOUT a base column.
- 4 chain transforms (``chain_linres_*``, ``chain_monres_*``) are registered with ``requires_base=True`` and run through the wrapper WITH a base column.
- ``CompositeTargetDiscoveryConfig.transforms`` default now lists all 14 (6 legacy + 4 unary + 4 chain).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import Ridge

from mlframe.training.composite_estimator import CompositeTargetEstimator
from mlframe.training.composite_transforms import _TRANSFORMS_REGISTRY, get_transform
from mlframe.training.configs import CompositeTargetDiscoveryConfig


class TestRegistry:
    def test_unary_registered_with_requires_base_false(self) -> None:
        for name in ("cbrt_y", "log_y", "yeo_johnson_y", "quantile_normal_y"):
            t = get_transform(name)
            assert t.requires_base is False, f"{name} must declare requires_base=False"

    def test_chain_registered_with_requires_base_true(self) -> None:
        for name in (
            "chain_linres_cbrt", "chain_linres_yj",
            "chain_monres_cbrt", "chain_monres_yj",
        ):
            t = get_transform(name)
            assert t.requires_base is True, f"{name} must declare requires_base=True (chain inherits from bivariate)"

    def test_discovery_default_includes_new_transforms(self) -> None:
        cfg = CompositeTargetDiscoveryConfig()
        for name in (
            "cbrt_y", "log_y", "yeo_johnson_y", "quantile_normal_y",
            "chain_linres_cbrt", "chain_linres_yj",
            "chain_monres_cbrt", "chain_monres_yj",
        ):
            assert name in cfg.transforms, f"{name} missing from default discovery transforms list"


class TestUnaryWrapper:
    """CompositeTargetEstimator must work with unary transforms WITHOUT a base column."""

    def _make_data(self, n: int = 500, seed: int = 0) -> tuple[pd.DataFrame, np.ndarray]:
        rng = np.random.default_rng(seed)
        X = pd.DataFrame({"a": rng.normal(0, 1, n), "b": rng.normal(0, 1, n)})
        # Heavy-tailed target
        y = rng.laplace(0.0, 5.0, n) + X["a"].values * 0.3
        return X, y

    @pytest.mark.parametrize("transform_name", ["cbrt_y", "log_y", "yeo_johnson_y", "quantile_normal_y"])
    def test_unary_fit_predict_no_base(self, transform_name: str) -> None:
        X, y = self._make_data(seed=1)
        wrapper = CompositeTargetEstimator(
            base_estimator=Ridge(alpha=1e-3),
            transform_name=transform_name,
            base_column="",  # unary: empty base is OK
        )
        wrapper.fit(X, y)
        preds = np.asarray(wrapper.predict(X), dtype=np.float64).reshape(-1)
        assert preds.shape == y.shape
        assert np.all(np.isfinite(preds))
        # Sanity: predictions should be roughly in the train y range (clipping makes this tight).
        assert preds.min() >= float(y.min()) * 10.0 - 1.0  # very loose; just guards against blow-up


class TestChainWrapper:
    """Chain transforms (bivariate + unary) must work like a normal bivariate composite."""

    def _make_data(self, n: int = 500, seed: int = 0) -> tuple[pd.DataFrame, np.ndarray]:
        rng = np.random.default_rng(seed)
        X = pd.DataFrame({"a": rng.normal(0, 1, n), "b": rng.normal(0, 1, n)})
        # Linear-in-a + heavy-tail residual (production-shape).
        y = X["a"].values * 0.7 + 5.0 + rng.laplace(0.0, 2.0, n)
        return X, y

    @pytest.mark.parametrize("transform_name", ["chain_linres_cbrt", "chain_linres_yj"])
    def test_chain_linres_fit_predict(self, transform_name: str) -> None:
        X, y = self._make_data(seed=2)
        wrapper = CompositeTargetEstimator(
            base_estimator=Ridge(alpha=1e-3),
            transform_name=transform_name,
            base_column="a",
        )
        wrapper.fit(X, y)
        preds = np.asarray(wrapper.predict(X), dtype=np.float64).reshape(-1)
        assert preds.shape == y.shape
        assert np.all(np.isfinite(preds))
        mae = float(np.mean(np.abs(preds - y)))
        # Chain composite should give bounded MAE on the production-shape synthetic
        # (Ridge inner on a linear+laplace target).
        assert mae < float(np.std(y)) * 1.5, (
            f"chain {transform_name} MAE={mae:.3f} > 1.5 * std(y)={float(np.std(y)) * 1.5:.3f}"
        )
