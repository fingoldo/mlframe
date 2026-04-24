"""Tests for XGB native multilabel opt-in (Session 2).

XGBoost 3.x has experimental ``multi_strategy='multi_output_tree'`` for native
multilabel. Marked WIP by upstream until v3.1 stable; opted in via
``MultilabelDispatchConfig.force_native_xgb_multilabel=True``.

Verifies:
- Default (flag=False): XGBoostStrategy uses MultiOutputClassifier wrapper
- Opt-in (flag=True): XGBoostStrategy returns estimator unchanged + native
  kwargs are injected (objective='binary:logistic',
  multi_strategy='multi_output_tree', tree_method='hist')
- Toggling flag does not break XGB multiclass dispatch (which has nothing
  to do with multilabel)
"""
from __future__ import annotations

import numpy as np
import pytest

from mlframe.training.configs import (
    TargetTypes,
    MultilabelDispatchConfig,
)
from mlframe.training.strategies import XGBoostStrategy


class _FakeXGBEstimator:
    """Stand-in for XGBClassifier — sklearn-compatible enough for dispatch tests."""
    def __init__(self):
        pass
    def get_params(self, deep=True):
        return {}
    def set_params(self, **kwargs):
        return self


def test_xgb_default_multilabel_uses_wrapper():
    """Without force_native_xgb_multilabel, MultiOutputClassifier wraps XGB."""
    s = XGBoostStrategy()
    cfg = MultilabelDispatchConfig()  # default: force_native_xgb_multilabel=False
    assert cfg.force_native_xgb_multilabel is False
    wrapped = s.wrap_multilabel(
        _FakeXGBEstimator(), TargetTypes.MULTILABEL_CLASSIFICATION,
        multilabel_config=cfg, n_labels=3,
    )
    # Wrapper path: type is MultiOutputClassifier, not the bare estimator.
    from sklearn.multioutput import MultiOutputClassifier
    assert isinstance(wrapped, MultiOutputClassifier)


def test_xgb_force_native_returns_estimator_unchanged():
    """With force_native_xgb_multilabel=True, return estimator unchanged."""
    s = XGBoostStrategy()
    cfg = MultilabelDispatchConfig(force_native_xgb_multilabel=True)
    estimator = _FakeXGBEstimator()
    wrapped = s.wrap_multilabel(
        estimator, TargetTypes.MULTILABEL_CLASSIFICATION,
        multilabel_config=cfg, n_labels=3,
    )
    # Native path: SAME object returned, no wrapper.
    assert wrapped is estimator


def test_xgb_force_native_kwargs_injected():
    """get_classif_objective_kwargs returns native multi-output-tree kwargs."""
    s = XGBoostStrategy()
    cfg = MultilabelDispatchConfig(force_native_xgb_multilabel=True)
    kwargs = s.get_classif_objective_kwargs(
        TargetTypes.MULTILABEL_CLASSIFICATION, n_classes=3,
        multilabel_config=cfg,
    )
    assert kwargs["objective"] == "binary:logistic"
    assert kwargs["multi_strategy"] == "multi_output_tree"
    assert kwargs["tree_method"] == "hist"


def test_xgb_default_multilabel_kwargs_empty():
    """Without flag, kwargs dict is empty (wrapper path handles dispatch)."""
    s = XGBoostStrategy()
    cfg = MultilabelDispatchConfig()  # force=False
    kwargs = s.get_classif_objective_kwargs(
        TargetTypes.MULTILABEL_CLASSIFICATION, n_classes=3,
        multilabel_config=cfg,
    )
    assert kwargs == {}


def test_xgb_no_config_falls_back_to_wrapper():
    """When multilabel_config is None (caller didn't pass one), falls back
    to wrapper path — does NOT silently activate native."""
    s = XGBoostStrategy()
    estimator = _FakeXGBEstimator()
    wrapped = s.wrap_multilabel(
        estimator, TargetTypes.MULTILABEL_CLASSIFICATION,
        multilabel_config=None, n_labels=3,
    )
    from sklearn.multioutput import MultiOutputClassifier
    assert isinstance(wrapped, MultiOutputClassifier)


def test_xgb_multiclass_unaffected_by_native_multilabel_flag():
    """Toggling force_native_xgb_multilabel does not affect multiclass path."""
    s = XGBoostStrategy()
    cfg_off = MultilabelDispatchConfig(force_native_xgb_multilabel=False)
    cfg_on = MultilabelDispatchConfig(force_native_xgb_multilabel=True)
    kw_off = s.get_classif_objective_kwargs(
        TargetTypes.MULTICLASS_CLASSIFICATION, n_classes=5, multilabel_config=cfg_off,
    )
    kw_on = s.get_classif_objective_kwargs(
        TargetTypes.MULTICLASS_CLASSIFICATION, n_classes=5, multilabel_config=cfg_on,
    )
    # Both should give the same multiclass kwargs (multi:softprob + num_class)
    assert kw_off == kw_on
    assert kw_on["objective"] == "multi:softprob"
    assert kw_on["num_class"] == 5


def test_xgb_force_native_with_strategy_native_explicit_works():
    """multilabel_config.strategy='native' + force_native_xgb_multilabel=True
    works — caller explicitly asked for native and the flag enables it."""
    s = XGBoostStrategy()
    cfg = MultilabelDispatchConfig(
        strategy="native",
        force_native_xgb_multilabel=True,
    )
    estimator = _FakeXGBEstimator()
    # Should NOT raise (XGB native is enabled via the flag)
    wrapped = s.wrap_multilabel(
        estimator, TargetTypes.MULTILABEL_CLASSIFICATION,
        multilabel_config=cfg, n_labels=3,
    )
    assert wrapped is estimator


def test_xgb_native_binary_unaffected():
    """Binary classification path unchanged — neither multilabel flag matters."""
    s = XGBoostStrategy()
    cfg = MultilabelDispatchConfig(force_native_xgb_multilabel=True)
    kw = s.get_classif_objective_kwargs(
        TargetTypes.BINARY_CLASSIFICATION, n_classes=2, multilabel_config=cfg,
    )
    assert kw == {"objective": "binary:logistic"}
    # No multi_strategy / tree_method injected on binary
    assert "multi_strategy" not in kw
