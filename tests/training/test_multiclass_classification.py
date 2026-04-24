"""Tests for multiclass classification support (target_type=MULTICLASS_CLASSIFICATION).

Phase A foundation tests — verify the dispatch helpers (`_canonical_predict_proba_shape`,
`_predict_from_probs`, `_classif_objective_kwargs`), TargetTypes helper properties,
and per-strategy native multiclass dispatch (CB, XGB, LGB).

End-to-end suite tests live in test_multiclass_classification_e2e.py once the
plumbing is wired through `train_mlframe_models_suite`.
"""
from __future__ import annotations

import numpy as np
import pytest

from mlframe.training.configs import (
    TargetTypes,
    MultilabelDispatchConfig,
    EnsemblingConfig,
)
from mlframe.training.helpers import (
    _canonical_predict_proba_shape,
    _predict_from_probs,
    _classif_objective_kwargs,
    _maybe_wrap_multilabel,
    _compute_chain_orders,
    get_training_configs,
)
from mlframe.training.strategies import (
    CatBoostStrategy,
    XGBoostStrategy,
    TreeModelStrategy,
    HGBStrategy,
    LinearModelStrategy,
)


# ---------------------------------------------------------------------------
# TargetTypes helpers — guards the 8 call-sites that previously hardcoded
# `target_type == BINARY_CLASSIFICATION`.
# ---------------------------------------------------------------------------


def test_target_types_is_classification_predicates():
    assert TargetTypes.REGRESSION.is_regression is True
    assert TargetTypes.REGRESSION.is_classification is False
    assert TargetTypes.BINARY_CLASSIFICATION.is_classification is True
    assert TargetTypes.BINARY_CLASSIFICATION.is_binary is True
    assert TargetTypes.BINARY_CLASSIFICATION.is_multi_output is False
    assert TargetTypes.MULTICLASS_CLASSIFICATION.is_classification is True
    assert TargetTypes.MULTICLASS_CLASSIFICATION.is_multiclass is True
    assert TargetTypes.MULTICLASS_CLASSIFICATION.is_multi_output is True
    assert TargetTypes.MULTILABEL_CLASSIFICATION.is_classification is True
    assert TargetTypes.MULTILABEL_CLASSIFICATION.is_multilabel is True
    assert TargetTypes.MULTILABEL_CLASSIFICATION.is_multi_output is True


def test_target_types_mutual_exclusion():
    """Only one of {is_binary, is_regression, is_multiclass, is_multilabel} is True."""
    for tt in TargetTypes:
        flags = [tt.is_binary, tt.is_regression, tt.is_multiclass, tt.is_multilabel]
        assert sum(flags) == 1, f"{tt!r}: expected exactly 1 flag True, got {flags}"


# ---------------------------------------------------------------------------
# _canonical_predict_proba_shape
# ---------------------------------------------------------------------------


def test_canonical_passthrough_NK():
    arr = np.array([[0.1, 0.7, 0.2], [0.5, 0.3, 0.2]])
    out = _canonical_predict_proba_shape(arr)
    np.testing.assert_array_equal(out, arr)


def test_canonical_1d_sigmoid_to_NK():
    arr = np.array([0.7, 0.3, 0.5])
    out = _canonical_predict_proba_shape(arr)
    assert out.shape == (3, 2)
    np.testing.assert_allclose(out[:, 1], arr)
    np.testing.assert_allclose(out[:, 0], 1.0 - arr)


def test_canonical_list_multioutput_to_NK():
    """MultiOutputClassifier returns List[(N, 2)]; canonicalize stacks class-1 cols."""
    per_label = [
        np.array([[0.7, 0.3], [0.4, 0.6]]),  # label 0
        np.array([[0.2, 0.8], [0.9, 0.1]]),  # label 1
        np.array([[0.5, 0.5], [0.5, 0.5]]),  # label 2
    ]
    out = _canonical_predict_proba_shape(per_label)
    assert out.shape == (2, 3)
    np.testing.assert_allclose(out[:, 0], [0.3, 0.6])
    np.testing.assert_allclose(out[:, 1], [0.8, 0.1])
    np.testing.assert_allclose(out[:, 2], [0.5, 0.5])


def test_canonical_constant_label_column_emits_zeros():
    """When a per-label estimator's predict_proba returns (N, 1) (constant
    label, only class 0 in training), canonicalizer must emit zeros not crash."""
    per_label = [
        np.array([[0.7, 0.3], [0.4, 0.6]]),  # normal binary
        np.array([[1.0], [1.0]]),             # constant — only class 0
    ]
    out = _canonical_predict_proba_shape(per_label)
    assert out.shape == (2, 2)
    np.testing.assert_allclose(out[:, 1], 0.0)


# ---------------------------------------------------------------------------
# _predict_from_probs
# ---------------------------------------------------------------------------


def test_predict_from_probs_binary():
    probs = np.array([[0.3, 0.7], [0.6, 0.4]])
    out = _predict_from_probs(probs, TargetTypes.BINARY_CLASSIFICATION)
    np.testing.assert_array_equal(out, [1, 0])


def test_predict_from_probs_binary_with_classes():
    probs = np.array([[0.3, 0.7], [0.6, 0.4]])
    out = _predict_from_probs(probs, TargetTypes.BINARY_CLASSIFICATION, classes_=np.array(["neg", "pos"]))
    np.testing.assert_array_equal(out, ["pos", "neg"])


def test_predict_from_probs_multiclass_argmax():
    probs = np.array([[0.1, 0.7, 0.2], [0.5, 0.3, 0.2]])
    out = _predict_from_probs(probs, TargetTypes.MULTICLASS_CLASSIFICATION)
    np.testing.assert_array_equal(out, [1, 0])


def test_predict_from_probs_multilabel_scalar_threshold():
    probs = np.array([[0.6, 0.3, 0.7]])
    out = _predict_from_probs(probs, TargetTypes.MULTILABEL_CLASSIFICATION, threshold=0.5)
    np.testing.assert_array_equal(out, [[1, 0, 1]])


def test_predict_from_probs_multilabel_per_label_threshold():
    probs = np.array([[0.6, 0.3, 0.7]])
    out = _predict_from_probs(
        probs, TargetTypes.MULTILABEL_CLASSIFICATION,
        threshold=np.array([0.5, 0.2, 0.8]),
    )
    np.testing.assert_array_equal(out, [[1, 1, 0]])


def test_predict_from_probs_multilabel_nan_safe():
    """NaN probs treated as 'below threshold' rather than propagating."""
    probs = np.array([[0.6, np.nan, 0.7]])
    out = _predict_from_probs(probs, TargetTypes.MULTILABEL_CLASSIFICATION, threshold=0.5)
    np.testing.assert_array_equal(out, [[1, 0, 1]])


def test_predict_from_probs_regression_raises():
    probs = np.array([[0.5, 0.5]])
    with pytest.raises(ValueError, match="not a classification"):
        _predict_from_probs(probs, TargetTypes.REGRESSION)


# ---------------------------------------------------------------------------
# _classif_objective_kwargs
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("flavor,expected", [
    ("catboost", {}),
    ("xgboost", {"objective": "binary:logistic"}),
    ("lightgbm", {"objective": "binary"}),
    ("hgb", {}),
    ("linear", {}),
])
def test_classif_kwargs_binary(flavor, expected):
    assert _classif_objective_kwargs(flavor, TargetTypes.BINARY_CLASSIFICATION, n_classes=2) == expected


@pytest.mark.parametrize("flavor,expected_keys", [
    ("catboost", {"loss_function"}),
    ("xgboost", {"objective", "num_class"}),
    ("lightgbm", {"objective", "num_class"}),
    ("hgb", set()),
    ("linear", {"multi_class", "solver"}),
])
def test_classif_kwargs_multiclass(flavor, expected_keys):
    out = _classif_objective_kwargs(flavor, TargetTypes.MULTICLASS_CLASSIFICATION, n_classes=5)
    assert set(out.keys()) == expected_keys
    if "num_class" in expected_keys:
        assert out["num_class"] == 5


def test_classif_kwargs_multilabel_cb_native_others_empty():
    assert _classif_objective_kwargs("catboost", TargetTypes.MULTILABEL_CLASSIFICATION, n_classes=3) == {"loss_function": "MultiLogloss"}
    for flavor in ("xgboost", "lightgbm", "hgb", "linear"):
        assert _classif_objective_kwargs(flavor, TargetTypes.MULTILABEL_CLASSIFICATION, n_classes=3) == {}


# ---------------------------------------------------------------------------
# Strategy feature flags
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("strategy_cls,native_mc,native_ml", [
    (CatBoostStrategy, True, True),
    (XGBoostStrategy, True, False),
    (TreeModelStrategy, True, False),  # LGB
    (HGBStrategy, True, False),
    (LinearModelStrategy, True, False),
])
def test_strategy_native_flags(strategy_cls, native_mc, native_ml):
    s = strategy_cls()
    assert s.supports_native_multiclass is native_mc
    assert s.supports_native_multilabel is native_ml


def test_strategy_get_classif_objective_kwargs_dispatches():
    s = CatBoostStrategy()
    out = s.get_classif_objective_kwargs(TargetTypes.MULTICLASS_CLASSIFICATION, n_classes=5)
    assert out == {"loss_function": "MultiClass"}


# ---------------------------------------------------------------------------
# get_training_configs target_type plumbing
# ---------------------------------------------------------------------------


def test_get_training_configs_binary_default_unchanged():
    """Default behavior preserved — XGB binary objective, no num_class."""
    c = get_training_configs(iterations=10, early_stopping_rounds=2)
    assert c.XGB_GENERAL_CLASSIF["objective"] == "binary:logistic"
    assert "num_class" not in c.XGB_GENERAL_CLASSIF
    assert "_mlframe_target_type" not in c.XGB_GENERAL_CLASSIF
    # CB binary: no loss_function injected (CB auto-detects)
    assert "loss_function" not in c.CB_CLASSIF


def test_get_training_configs_multiclass_injects_objectives():
    c = get_training_configs(
        iterations=10, early_stopping_rounds=2,
        target_type=TargetTypes.MULTICLASS_CLASSIFICATION, n_classes=5,
    )
    assert c.XGB_GENERAL_CLASSIF["objective"] == "multi:softprob"
    assert c.XGB_GENERAL_CLASSIF["num_class"] == 5
    assert c.LGB_GENERAL_PARAMS["objective"] == "multiclass"
    assert c.LGB_GENERAL_PARAMS["num_class"] == 5
    assert c.CB_CLASSIF["loss_function"] == "MultiClass"


def test_get_training_configs_multilabel_cb_only_native():
    c = get_training_configs(
        iterations=10, early_stopping_rounds=2,
        target_type=TargetTypes.MULTILABEL_CLASSIFICATION, n_classes=3,
    )
    # CB native gets MultiLogloss
    assert c.CB_CLASSIF["loss_function"] == "MultiLogloss"
    # XGB / LGB stay binary — wrapper handles multilabel for them
    assert c.XGB_GENERAL_CLASSIF["objective"] == "binary:logistic"
    assert c.LGB_GENERAL_PARAMS.get("objective") in ("binary", None)


# ---------------------------------------------------------------------------
# _compute_chain_orders + _ChainEnsemble basics
# ---------------------------------------------------------------------------


def test_compute_chain_orders_random():
    orders = _compute_chain_orders(n_labels=5, n_chains=3, seeds=[0, 1, 2])
    assert len(orders) == 3
    for o in orders:
        assert sorted(o.tolist()) == [0, 1, 2, 3, 4]


def test_compute_chain_orders_by_frequency():
    """Rare-first ordering."""
    y = np.array([
        [1, 0, 0],
        [1, 0, 1],
        [1, 1, 1],
    ])  # label freq: [3, 1, 2]
    orders = _compute_chain_orders(n_labels=3, n_chains=2, order_strategy="by_frequency", y=y)
    assert orders[0].tolist() == [1, 2, 0]


def test_compute_chain_orders_user():
    user = [[0, 1, 2], [2, 1, 0]]
    orders = _compute_chain_orders(n_labels=3, n_chains=2, order_strategy="user", user_orders=user)
    assert orders[0].tolist() == [0, 1, 2]
    assert orders[1].tolist() == [2, 1, 0]


def test_compute_chain_orders_by_frequency_requires_y():
    with pytest.raises(ValueError, match="requires y"):
        _compute_chain_orders(n_labels=3, n_chains=2, order_strategy="by_frequency")


# ---------------------------------------------------------------------------
# Configs dataclasses
# ---------------------------------------------------------------------------


def test_multilabel_dispatch_config_defaults():
    cfg = MultilabelDispatchConfig()
    assert cfg.strategy == "auto"
    assert cfg.n_chains == 3
    assert cfg.cv == 5  # cross-validated chain (Review Tier 1 #2 fix)
    assert cfg.wrapper_n_jobs == "auto"


def test_ensembling_config_defaults():
    cfg = EnsemblingConfig()
    assert cfg.force_legacy is False
    assert cfg.quantile_budget_bytes == 500 * 1024 * 1024
    assert cfg.accumulator == "welford"
