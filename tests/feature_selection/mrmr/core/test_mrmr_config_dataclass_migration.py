"""Regression tests for audits/mrmr_audit_2026-07-16/10_config_dataclass_proposal.md (finding #1).

Purely additive migration: nested pydantic config dataclasses (FastSearchConfig,
StabilitySelectionConfig, SynergyRedundancyConfig, GroupAwareConfig, DCDConfig, HybridOrthConfig)
as NEW optional MRMR ctor kwargs, expanding onto the existing flat attrs when passed. The flat-kwarg
path (every existing caller) must be entirely unaffected when no config is passed.
"""
from __future__ import annotations

import pickle  # nosec B403 -- test-only local pickle round-trip, never untrusted/network data

import numpy as np
import pandas as pd
import pydantic
import pytest
from sklearn.base import clone

from mlframe.feature_selection.filters.mrmr import MRMR
from mlframe.feature_selection.filters.mrmr._mrmr_config_dataclasses import (
    DCDConfig,
    FastSearchConfig,
    GroupAwareConfig,
    HybridOrthConfig,
    HybridOrthScorersConfig,
    StabilitySelectionConfig,
    SynergyRedundancyConfig,
)


def test_default_construction_configs_are_none_and_flat_attrs_unchanged():
    """No config passed -> every config attr is None, flat attrs keep their normal defaults."""
    m = MRMR(verbose=0)
    assert m.dcd_config is None
    assert m.stability_config is None
    assert m.dcd_enable is True  # normal flat default, untouched


def test_dcd_config_overrides_flat_attrs():
    """A passed DCDConfig overrides its cluster's flat attrs; unset fields keep their own default."""
    m = MRMR(verbose=0, dcd_config=DCDConfig(dcd_enable=False, dcd_min_cluster_size=3))
    assert m.dcd_enable is False
    assert m.dcd_min_cluster_size == 3
    # a field NOT set on the config still takes ITS OWN default (not the flat ctor kwarg's).
    assert m.dcd_max_cluster_size == 12


def test_stability_config_overrides_flat_attrs():
    """A passed StabilitySelectionConfig overrides stability_selection_method / stability_n_bootstrap."""
    m = MRMR(verbose=0, stability_config=StabilitySelectionConfig(stability_selection_method="cluster", stability_n_bootstrap=7))
    assert m.stability_selection_method == "cluster"
    assert m.stability_n_bootstrap == 7


def test_synergy_config_overrides_flat_attrs():
    """A passed SynergyRedundancyConfig overrides redundancy_aggregator / bur_lambda."""
    m = MRMR(verbose=0, synergy_config=SynergyRedundancyConfig(redundancy_aggregator="jmim", bur_lambda=0.5))
    assert m.redundancy_aggregator == "jmim"
    assert m.bur_lambda == 0.5


def test_group_aware_config_overrides_flat_attrs():
    """A passed GroupAwareConfig overrides group_aware_mi / group_mi_min_rows."""
    m = MRMR(verbose=0, group_aware_config=GroupAwareConfig(group_aware_mi=True, group_mi_min_rows=5))
    assert m.group_aware_mi is True
    assert m.group_mi_min_rows == 5


def test_fast_search_config_overrides_flat_attrs():
    """A passed FastSearchConfig overrides fe_fast_search / fe_max_steps."""
    m = MRMR(verbose=0, fast_search_config=FastSearchConfig(fe_fast_search=True, fe_max_steps=2))
    assert m.fe_fast_search is True
    assert m.fe_max_steps == 2


def test_hybrid_orth_config_overrides_flat_attrs_including_nested_scorers():
    """A passed HybridOrthConfig overrides top-level fields and the nested scorers sub-config."""
    cfg = HybridOrthConfig(top_k=9, scorers=HybridOrthScorersConfig(ksg_enable=True, ksg_n_neighbors=7))
    m = MRMR(verbose=0, hybrid_orth_config=cfg)
    assert m.fe_hybrid_orth_top_k == 9
    assert m.fe_hybrid_orth_ksg_enable is True
    assert m.fe_hybrid_orth_ksg_n_neighbors == 7
    # unset fields keep their own dataclass default, not whatever the flat ctor kwarg default was.
    assert m.fe_hybrid_orth_pair_enable is True


def test_configs_are_pydantic_validated_at_construction_not_fit_time():
    """A typo'd literal value raises pydantic.ValidationError immediately at Config(...) construction,
    not minutes into a later fit() call -- the whole point of finding #1's migration."""
    with pytest.raises(pydantic.ValidationError):
        StabilitySelectionConfig(stability_selection_method="bogus")
    with pytest.raises(pydantic.ValidationError):
        SynergyRedundancyConfig(redundancy_aggregator="typo")
    with pytest.raises(pydantic.ValidationError):
        DCDConfig(dcd_swap_alpha=1.5)  # out of (0, 1) range


def test_configs_reject_unknown_fields():
    """extra='forbid' catches a misspelled field name at construction time."""
    with pytest.raises(pydantic.ValidationError):
        DCDConfig(dcd_enalbe=False)  # typo


def test_get_params_includes_config_slots_and_clone_preserves_config():
    """get_params() exposes the raw config object; clone() copies both the config and derived flat attrs."""
    m = MRMR(verbose=0, dcd_config=DCDConfig(dcd_enable=False))
    params = m.get_params()
    assert "dcd_config" in params
    assert params["dcd_config"].dcd_enable is False
    c = clone(m)
    assert c.dcd_config is not None
    assert c.dcd_config.dcd_enable is False
    assert c.dcd_enable is False


def test_pickle_round_trip_preserves_config_and_derived_flat_attrs():
    """A pickle round-trip preserves both the raw config object and its derived flat attrs."""
    m = MRMR(verbose=0, dcd_config=DCDConfig(dcd_enable=False, dcd_min_cluster_size=3))
    m2 = pickle.loads(pickle.dumps(m))  # nosec B301 -- round-trip of a locally-created, trusted object
    assert m2.dcd_config is not None
    assert m2.dcd_config.dcd_enable is False
    assert m2.dcd_enable is False
    assert m2.dcd_min_cluster_size == 3


def test_legacy_pickle_missing_config_keys_restores_none_and_flat_value_intact():
    """A pickle predating this migration (state dict missing the new config keys entirely) restores
    every config attr to None (the ctor default) via __setstate__'s existing fresh-instance catch-all,
    while the flat attr it pickled with (dcd_enable=False, set via the plain flat kwarg, not a config)
    survives unchanged."""
    m = MRMR(verbose=0, dcd_enable=False)
    state = m.__getstate__()
    for key in ("fast_search_config", "stability_config", "synergy_config", "group_aware_config", "dcd_config", "hybrid_orth_config"):
        state.pop(key, None)
    m2 = MRMR()
    m2.__setstate__(state)
    assert m2.dcd_config is None
    assert m2.dcd_enable is False


def test_fitting_with_a_config_object_actually_changes_selection_behavior():
    """End-to-end: dcd_config=DCDConfig(dcd_enable=False) actually disables DCD for a real fit, same
    as passing dcd_enable=False directly -- the config path is not just cosmetic."""
    rng = np.random.default_rng(0)
    n = 300
    X = pd.DataFrame({"a": rng.standard_normal(n), "b": rng.standard_normal(n), "c": rng.standard_normal(n)})
    y = pd.Series((X["a"] > 0).astype(int))

    m_flat = MRMR(verbose=0, dcd_enable=False, min_features_fallback=1)
    m_flat.fit(X, y)

    m_cfg = MRMR(verbose=0, dcd_config=DCDConfig(dcd_enable=False), min_features_fallback=1)
    m_cfg.fit(X, y)

    assert list(m_flat.get_feature_names_out()) == list(m_cfg.get_feature_names_out())
