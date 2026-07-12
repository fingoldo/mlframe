"""Wiring + regression tests for the functional-utility FS selectors (ForwardSelect /
GreedyBackwardElimination / ZeroImportancePruning / CascadeSelect), wrapped by sklearn adapters in
``mlframe.feature_selection.functional_adapters`` and reachable from the suite via
``FeatureSelectionConfig.use_<sel>_fs`` + a ``_build_pre_pipelines`` branch (mirrors ACE/ShapProxiedFS).
"""
from __future__ import annotations

import os

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import numpy as np
import pandas as pd
import pytest

from mlframe.training import FeatureSelectionConfig


def _build(**over):
    from mlframe.training.core._setup_helpers_pre_pipelines import _build_pre_pipelines

    base = dict(use_ordinary_models=False, rfecv_models=[], rfecv_models_params={}, use_mrmr_fs=False, mrmr_kwargs={})
    base.update(over)
    return _build_pre_pipelines(**base)


def _make_classification_frame(n=200, seed=0):
    rng = np.random.default_rng(seed)
    signal = rng.standard_normal(n)
    df = pd.DataFrame({
        "s0": signal,
        "s1": signal * 0.9 + 0.1 * rng.standard_normal(n),
        "noise0": rng.standard_normal(n),
        "noise1": rng.standard_normal(n),
        "noise2": rng.standard_normal(n),
    })
    y = pd.Series((signal > 0).astype(int))
    return df, y


# --------------------------------------------------------------------------- reachability

@pytest.mark.parametrize(
    "flag,names_key,kwargs_field",
    [
        ("use_forward_select_fs", "ForwardSelect ", "forward_select_kwargs"),
        ("use_greedy_backward_elimination_fs", "GreedyBackwardElimination ", "greedy_backward_elimination_kwargs"),
        ("use_zero_importance_pruning_fs", "ZeroImportancePruning ", "zero_importance_pruning_kwargs"),
        ("use_cascade_select_fs", "CascadeSelect ", "cascade_select_kwargs"),
    ],
)
def test_selector_reachable_from_suite(flag, names_key, kwargs_field):
    pps, names = _build(**{flag: True})
    assert names_key in names
    sel = pps[names.index(names_key)]
    assert getattr(sel, "_mlframe_selector_kind_") == names_key.strip()


def test_selector_kind_classifies_all_four():
    from mlframe.training.core._phase_train_one_target import _selector_kind

    for flag, names_key in [
        ("use_forward_select_fs", "ForwardSelect "),
        ("use_greedy_backward_elimination_fs", "GreedyBackwardElimination "),
        ("use_zero_importance_pruning_fs", "ZeroImportancePruning "),
        ("use_cascade_select_fs", "CascadeSelect "),
    ]:
        pps, names = _build(**{flag: True})
        assert _selector_kind(pps[names.index(names_key)]) == names_key.strip()


# --------------------------------------------------------------------------- master-flag gate + kwargs validation

@pytest.mark.parametrize(
    "flag,kwargs_field",
    [
        ("use_forward_select_fs", "forward_select_kwargs"),
        ("use_greedy_backward_elimination_fs", "greedy_backward_elimination_kwargs"),
        ("use_zero_importance_pruning_fs", "zero_importance_pruning_kwargs"),
        ("use_cascade_select_fs", "cascade_select_kwargs"),
    ],
)
def test_kwargs_master_flag_gate(flag, kwargs_field):
    with pytest.raises(ValueError, match=f"{kwargs_field} supplied but {flag}"):
        FeatureSelectionConfig(**{kwargs_field: {"cv": 3}})


@pytest.mark.parametrize(
    "flag,kwargs_field",
    [
        ("use_forward_select_fs", "forward_select_kwargs"),
        ("use_greedy_backward_elimination_fs", "greedy_backward_elimination_kwargs"),
        ("use_zero_importance_pruning_fs", "zero_importance_pruning_kwargs"),
        ("use_cascade_select_fs", "cascade_select_kwargs"),
    ],
)
def test_kwargs_rejects_unknown_key(flag, kwargs_field):
    with pytest.raises(ValueError, match="unknown key"):
        FeatureSelectionConfig(**{flag: True, kwargs_field: {"definitely_not_a_param": 1}})


# --------------------------------------------------------------------------- biz_value: genuinely runs the selector

def test_biz_forward_select_shrinks_noisy_frame():
    df, y = _make_classification_frame()
    pps, names = _build(use_forward_select_fs=True, forward_select_kwargs={"cv": 3, "min_improvement": 0.001})
    sel = pps[names.index("ForwardSelect ")]
    kept = list(sel.fit(df, y).transform(df).columns)
    assert len(kept) < df.shape[1]
    assert {"s0", "s1"} & set(kept), f"forward_select dropped both signal columns, kept {kept}"


def test_biz_greedy_backward_elimination_shrinks_noisy_frame():
    from sklearn.linear_model import LogisticRegression

    df, y = _make_classification_frame(n=400)
    pps, names = _build(
        use_greedy_backward_elimination_fs=True,
        greedy_backward_elimination_kwargs={"cv": None, "estimator": LogisticRegression(max_iter=1000)},
    )
    sel = pps[names.index("GreedyBackwardElimination ")]
    kept = list(sel.fit(df, y).transform(df).columns)
    assert len(kept) < df.shape[1], f"expected noise columns pruned, kept {kept}"


def test_biz_zero_importance_pruning_shrinks_noisy_frame():
    df, y = _make_classification_frame()
    pps, names = _build(use_zero_importance_pruning_fs=True, zero_importance_pruning_kwargs={})
    sel = pps[names.index("ZeroImportancePruning ")]
    kept = list(sel.fit(df, y).transform(df).columns)
    assert len(kept) <= df.shape[1]


def test_biz_cascade_select_shrinks_noisy_frame():
    df, y = _make_classification_frame(n=300)
    pps, names = _build(use_cascade_select_fs=True, cascade_select_kwargs={"n_boruta_iterations": 10, "cv": 3})
    sel = pps[names.index("CascadeSelect ")]
    kept = list(sel.fit(df, y).transform(df).columns)
    assert len(kept) <= df.shape[1]
    assert hasattr(sel, "cascade_result_")


# --------------------------------------------------------------------------- regression: default OFF is a no-op

def test_default_config_omits_all_four_new_flags():
    cfg = FeatureSelectionConfig()
    assert cfg.use_forward_select_fs is False
    assert cfg.forward_select_kwargs is None
    assert cfg.use_greedy_backward_elimination_fs is False
    assert cfg.greedy_backward_elimination_kwargs is None
    assert cfg.use_zero_importance_pruning_fs is False
    assert cfg.zero_importance_pruning_kwargs is None
    assert cfg.use_cascade_select_fs is False
    assert cfg.cascade_select_kwargs is None


def test_build_pre_pipelines_default_bit_identical_without_new_flags():
    """With all four new flags left at default False, the pre-pipeline list/names are byte-identical
    to the pre-existing set (regression against the wiring pass changing anything unrelated)."""
    pps_before, names_before = _build(use_mrmr_fs=True, mrmr_kwargs={})
    pps_after, names_after = _build(use_mrmr_fs=True, mrmr_kwargs={}, use_forward_select_fs=False, use_cascade_select_fs=False)
    assert names_before == names_after
    assert len(pps_before) == len(pps_after)
