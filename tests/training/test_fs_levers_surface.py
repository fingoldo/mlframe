"""Unit tests for first-class feature-selection lever fields (D-surface).

Levers map to MRMR/RFECV constructor keys; unset -> byte-identical (no merge); set -> folded into
the kwargs dict; setting both a lever AND the same kwarg key -> raise (silent-override guard).
"""

from __future__ import annotations

import pytest

from mlframe.training.configs import FeatureSelectionConfig


def test_unset_levers_are_byte_identical():
    cfg = FeatureSelectionConfig()
    assert cfg.mrmr_kwargs is None
    assert cfg.rfecv_kwargs is None


def test_mrmr_lever_folds_into_kwargs():
    cfg = FeatureSelectionConfig(use_mrmr_fs=True, mrmr_mi_normalization="su", mrmr_redundancy_aggregator="jmim")
    assert cfg.mrmr_kwargs["mi_normalization"] == "su"
    assert cfg.mrmr_kwargs["redundancy_aggregator"] == "jmim"


def test_rfecv_enable_flags_fold_into_kwargs():
    cfg = FeatureSelectionConfig(
        rfecv_models=["lgb"],
        rfecv_enable_permutation_importance=True,
        rfecv_enable_stability_selection=True,
        rfecv_n_features_selection_rule="one_se_min",
    )
    assert cfg.rfecv_kwargs["importance_getter"] == "permutation"
    assert cfg.rfecv_kwargs["stability_selection"] is True
    assert cfg.rfecv_kwargs["n_features_selection_rule"] == "one_se_min"


def test_rfecv_must_include_exclude_groups_fold():
    cfg = FeatureSelectionConfig(
        rfecv_models=["lgb"],
        rfecv_must_include=["a", "b"],
        rfecv_must_exclude=["leak"],
        rfecv_feature_groups={"g": ["a", "b"]},
    )
    assert cfg.rfecv_kwargs["must_include"] == ["a", "b"]
    assert cfg.rfecv_kwargs["must_exclude"] == ["leak"]
    assert cfg.rfecv_kwargs["feature_groups"] == {"g": ["a", "b"]}


def test_conflict_between_lever_and_explicit_kwarg_raises():
    with pytest.raises(ValueError, match="mi_normalization"):
        FeatureSelectionConfig(
            use_mrmr_fs=True,
            mrmr_mi_normalization="su",
            mrmr_kwargs={"mi_normalization": "none"},
        )


def test_lever_preserves_other_explicit_kwargs():
    cfg = FeatureSelectionConfig(
        rfecv_models=["lgb"],
        rfecv_swap_top_k=3,
        rfecv_kwargs={"n_jobs": 1},
    )
    assert cfg.rfecv_kwargs["swap_top_k"] == 3
    assert cfg.rfecv_kwargs["n_jobs"] == 1


def test_mrmr_lever_without_master_flag_raises():
    # An MRMR lever folds into mrmr_kwargs, which is ignored unless use_mrmr_fs=True -> loud error.
    with pytest.raises(ValueError, match="use_mrmr_fs"):
        FeatureSelectionConfig(mrmr_mi_normalization="su")
