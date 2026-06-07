"""F-34: suite-side dispatch for MULTI_TARGET_REGRESSION target type.

Tests:
  * Strategy native-support flags (CatBoost, XGBoost, MLP, Linear = True;
    HGB / LightGBM = False -> wrapper).
  * ``get_multi_target_objective_kwargs`` per strategy returns the
    canonical native kwargs (MultiRMSE for CB, multi_output_tree+hist
    for XGB).
  * ``wrap_multi_target`` returns estimator unchanged for native, wraps
    in ``sklearn.multioutput.MultiOutputRegressor`` for non-native.
  * Auto-route helper: ``multilabel_strategy="multi_target_regression"``
    moves (N, K>=2) regression targets to
    ``target_by_type[TargetTypes.MULTI_TARGET_REGRESSION]`` instead of
    expanding to K independent 1-D targets.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
from sklearn.linear_model import LinearRegression
from sklearn.multioutput import MultiOutputRegressor

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from mlframe.training import TargetTypes
from mlframe.training.strategies import ModelPipelineStrategy
from mlframe.training.strategies import HGBStrategy
from mlframe.training.strategies import LinearModelStrategy, NeuralNetStrategy
from mlframe.training.strategies import CatBoostStrategy, TreeModelStrategy
from mlframe.training.strategies import XGBoostStrategy


# ----- Strategy gate flags -----------------------------------------------


def test_native_multi_target_flags():
    """Native MTR strategies: CatBoost, XGBoost, NeuralNet, LinearModel.
    Non-native: TreeModel base (LightGBM), HGB.
    ModelPipelineStrategy itself is abstract; the default-False contract
    is verified by inspecting the property descriptor."""
    assert CatBoostStrategy().supports_native_multi_target is True
    assert XGBoostStrategy().supports_native_multi_target is True
    assert NeuralNetStrategy().supports_native_multi_target is True
    assert LinearModelStrategy().supports_native_multi_target is True
    # LightGBM is TreeModelStrategy NOT extended -> default False
    assert TreeModelStrategy().supports_native_multi_target is False
    assert HGBStrategy().supports_native_multi_target is False
    # Base default (descriptor-level check; ABC can't be instantiated).
    assert ModelPipelineStrategy.supports_native_multi_target.fget(None) is False


# ----- get_multi_target_objective_kwargs ---------------------------------


def test_catboost_returns_multirmse():
    """CatBoost MultiRMSE for joint K-target regression."""
    cb = CatBoostStrategy()
    kwargs = cb.get_multi_target_objective_kwargs()
    assert kwargs == {"loss_function": "MultiRMSE"}


def test_xgboost_returns_multi_output_tree():
    """XGBoost native multi-output trees require tree_method='hist'."""
    xgb = XGBoostStrategy()
    kwargs = xgb.get_multi_target_objective_kwargs()
    assert kwargs == {
        "multi_strategy": "multi_output_tree",
        "tree_method": "hist",
    }


def test_non_native_returns_empty_kwargs():
    """Non-native strategies return {} -- the wrap_multi_target path
    is used instead to add MultiOutputRegressor."""
    for cls in (HGBStrategy, TreeModelStrategy):
        assert cls().get_multi_target_objective_kwargs() == {}


# ----- wrap_multi_target --------------------------------------------------


def test_wrap_native_returns_unchanged():
    """Native strategies pass the estimator through unchanged
    (the native loss_function handles K targets)."""
    est = LinearRegression()
    wrapped = CatBoostStrategy().wrap_multi_target(est)
    assert wrapped is est


def test_wrap_non_native_uses_multioutputregressor():
    """Non-native strategies wrap in sklearn MultiOutputRegressor."""
    est = LinearRegression()
    wrapped = HGBStrategy().wrap_multi_target(est)
    assert isinstance(wrapped, MultiOutputRegressor)
    assert wrapped.estimator is est


def test_wrap_non_native_actually_fits_predicts_n_k():
    """End-to-end sanity on the wrapper: fit a single-target regressor
    wrapped via MultiOutputRegressor on (N, K) y, predict (N, K)."""
    rng = np.random.default_rng(0)
    X = rng.normal(size=(100, 4)).astype(np.float32)
    coefs = rng.normal(size=(4, 3))
    y = (X @ coefs + 0.05 * rng.normal(size=(100, 3))).astype(np.float32)

    wrapped = HGBStrategy().wrap_multi_target(LinearRegression())
    wrapped.fit(X, y)
    preds = wrapped.predict(X)
    assert preds.shape == (100, 3)


# ----- Auto-route helper --------------------------------------------------


def test_auto_route_multi_target_regression():
    """multilabel_strategy='multi_target_regression' moves (N, K>=2)
    regression targets to TargetTypes.MULTI_TARGET_REGRESSION; the
    1-D regression targets in the same bucket are untouched."""
    from types import SimpleNamespace

    from mlframe.training.core._phase_helpers import (
        _defensive_copy_and_expand_multilabel_regression,
    )

    rng = np.random.default_rng(0)
    target_by_type = {
        TargetTypes.REGRESSION: {
            "single_target": rng.normal(size=80).astype(np.float32),
            "multi_target": rng.normal(size=(80, 3)).astype(np.float32),
        },
    }
    config = SimpleNamespace(multilabel_strategy="multi_target_regression")
    metadata = {}

    out = _defensive_copy_and_expand_multilabel_regression(
        target_by_type=target_by_type,
        composite_target_discovery_config=config,
        metadata=metadata,
    )

    # Single-target regression stays under REGRESSION.
    assert "single_target" in out[TargetTypes.REGRESSION]
    assert "multi_target" not in out[TargetTypes.REGRESSION]
    # (N, K) target routed to MULTI_TARGET_REGRESSION.
    assert TargetTypes.MULTI_TARGET_REGRESSION in out
    assert "multi_target" in out[TargetTypes.MULTI_TARGET_REGRESSION]
    routed = out[TargetTypes.MULTI_TARGET_REGRESSION]["multi_target"]
    assert routed.shape == (80, 3)

    # Metadata records the routing.
    assert "multi_target_regression_routing" in metadata
    assert "multi_target" in metadata["multi_target_regression_routing"][
        str(TargetTypes.MULTI_TARGET_REGRESSION)
    ]


def test_per_target_strategy_still_expands():
    """Default multilabel_strategy='per_target' keeps the existing
    behaviour: (N, K) regression targets are expanded to K independent
    1-D targets under REGRESSION."""
    from types import SimpleNamespace

    from mlframe.training.core._phase_helpers import (
        _defensive_copy_and_expand_multilabel_regression,
    )

    rng = np.random.default_rng(0)
    target_by_type = {
        TargetTypes.REGRESSION: {
            "mt": rng.normal(size=(80, 3)).astype(np.float32),
        },
    }
    config = SimpleNamespace(multilabel_strategy="per_target")
    metadata = {}

    out = _defensive_copy_and_expand_multilabel_regression(
        target_by_type=target_by_type,
        composite_target_discovery_config=config,
        metadata=metadata,
    )

    # Expanded into 3 sub-targets, no MTR bucket created.
    assert TargetTypes.MULTI_TARGET_REGRESSION not in out
    sub_names = {"mt_out0", "mt_out1", "mt_out2"}
    assert sub_names.issubset(out[TargetTypes.REGRESSION].keys())
    assert "mt" not in out[TargetTypes.REGRESSION]


def test_invalid_multilabel_strategy_raises():
    """Pydantic validator rejects unknown strategies."""
    from mlframe.training._composite_target_discovery_config import (
        CompositeTargetDiscoveryConfig,
    )
    with pytest.raises(ValueError, match=r"multilabel_strategy must be one of"):
        CompositeTargetDiscoveryConfig(multilabel_strategy="invalid_strategy")
