"""The former first-class FS levers are plain typed fields of the selector sub-configs now.

``FeatureSelectionConfig(rfecv=..., mrmr=...)`` holds one strict sub-config per selector; every ``RFECV.__init__`` / ``MRMR.__init__`` parameter is a
field, ``to_kwargs()`` returns only what the caller wrote (the suite keeps its own defaults for the rest), and a ``model_dump`` round trip
rebuilds an equal config with the same written fields.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from mlframe.training.configs import FeatureSelectionConfig


def test_a_default_config_runs_no_selector_and_keeps_the_pre_screen():
    """Nothing supervised is on by default; only the unsupervised pre-screen is."""
    cfg = FeatureSelectionConfig()
    assert [cfg.selector_kwargs(n) for n in ("rfecv", "mrmr", "boruta_shap", "shap_proxied_fs", "ace")] == [None] * 5
    assert cfg.pre_screen.enable is True


def test_mrmr_levers_are_constructor_parameters():
    """The MRMR levers are ordinary fields and come back through ``to_kwargs`` once written."""
    cfg = FeatureSelectionConfig(mrmr={"mi_normalization": "su", "redundancy_aggregator": "jmim"})
    assert cfg.selector_kwargs("mrmr") == {"mi_normalization": "su", "redundancy_aggregator": "jmim"}


def test_rfecv_levers_are_constructor_parameters():
    """The RFECV levers are ordinary fields, including the permutation importance source."""
    cfg = FeatureSelectionConfig(
        rfecv={"models": ["lgb"], "stability_selection": True, "n_features_selection_rule": "one_se_min", "importance_getter": "permutation"}
    )
    kwargs = cfg.selector_kwargs("rfecv")
    assert kwargs["importance_getter"] == "permutation"
    assert kwargs["stability_selection"] is True
    assert kwargs["n_features_selection_rule"] == "one_se_min"
    assert "models" not in kwargs and "cluster" not in kwargs


def test_must_include_exclude_and_groups_pass_through():
    """Feature-level levers reach the kwargs unchanged."""
    cfg = FeatureSelectionConfig(rfecv={"models": ["lgb"], "must_include": ["a", "b"], "must_exclude": ["leak"], "feature_groups": {"g": ["a", "b"]}})
    kwargs = cfg.selector_kwargs("rfecv")
    assert kwargs["must_include"] == ["a", "b"]
    assert kwargs["must_exclude"] == ["leak"]
    assert kwargs["feature_groups"] == {"g": ["a", "b"]}


def test_a_feature_cannot_be_both_included_and_excluded():
    """The overlap is a config error instead of an RFECV surprise at fit time."""
    with pytest.raises(ValidationError, match="must_include and must_exclude"):
        FeatureSelectionConfig(rfecv={"models": ["lgb"], "must_include": ["a"], "must_exclude": ["a"]})


def test_a_grouped_feature_cannot_be_excluded():
    """A feature group member that is also excluded contradicts the all-or-nothing group decision."""
    with pytest.raises(ValidationError, match="feature group"):
        FeatureSelectionConfig(rfecv={"models": ["lgb"], "must_exclude": ["a"], "feature_groups": {"g": ["a", "b"]}})


def test_group_mi_knobs_need_group_aware_mi():
    """The group-MI aggregation knobs mean nothing without ``group_aware_mi=True``."""
    with pytest.raises(ValidationError, match="group_aware_mi"):
        FeatureSelectionConfig(mrmr={"group_mi_aggregate": "equal"})
    assert FeatureSelectionConfig(mrmr={"group_aware_mi": True, "group_mi_aggregate": "equal"}).mrmr.group_mi_aggregate == "equal"


def test_unwritten_fields_are_not_forwarded():
    """A field the caller did not write keeps the suite's own default: only written fields reach the selector."""
    cfg = FeatureSelectionConfig(mrmr={"verbose": 0}, rfecv={"models": ["cb"], "swap_top_k": 3, "n_jobs": 1})
    assert cfg.selector_kwargs("mrmr") == {"verbose": 0}
    assert cfg.selector_kwargs("rfecv") == {"swap_top_k": 3, "n_jobs": 1}


def test_dump_round_trip_keeps_the_written_fields():
    """``model_dump`` is sparse for selector configs, so rebuilding from it gives an equal config with the same written fields."""
    cfg = FeatureSelectionConfig(rfecv={"models": ["lgb"], "must_include": ["a"]}, mrmr={"cpt_test": True}, boruta_shap=True)
    rebuilt = FeatureSelectionConfig(**cfg.model_dump())
    assert rebuilt == cfg
    assert rebuilt.selector_kwargs("mrmr") == {"cpt_test": True}
    assert rebuilt.selector_kwargs("boruta_shap").keys() >= {"cluster_reduce"}
