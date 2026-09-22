"""Config contracts that did not hold: a round-trip that failed, and a safety flag whose default contradicted its twin.

CFG-04: `FeatureSelectionConfig(rfecv_models=['cb'], rfecv_swap_top_k=3).model_dump()` carried the lever AND its
folded `rfecv_kwargs` entry, and re-validating that dump raised "set BOTH as a first-class lever field AND inside
rfecv_kwargs" - so any cache key, JSON reload or sweep harness crashed on a config nobody mis-specified.

CFG-03: `PreprocessingConfig.skip_infinity_checks` defaulted to True (protection off) against `DataConfig`'s False,
and nothing reads it.
"""

from __future__ import annotations

import warnings

import pytest

from mlframe.training.configs import FeatureSelectionConfig, PreprocessingConfig


def test_a_dumped_feature_selection_config_revalidates():
    cfg = FeatureSelectionConfig(rfecv_models=["cb"], rfecv_swap_top_k=3)
    again = FeatureSelectionConfig(**cfg.model_dump())
    assert again.rfecv_kwargs == cfg.rfecv_kwargs == {"swap_top_k": 3}


def test_a_genuine_conflict_still_raises():
    with pytest.raises(ValueError, match="BOTH as a first-class lever"):
        FeatureSelectionConfig(rfecv_models=["cb"], rfecv_swap_top_k=3, rfecv_kwargs={"swap_top_k": 5})


def test_the_preprocessing_skip_flag_default_matches_the_data_config():
    from mlframe.training.configs import DataConfig

    assert PreprocessingConfig().skip_infinity_checks is False
    assert DataConfig.model_fields["skip_infinity_checks"].default is False


def test_setting_the_inert_preprocessing_flag_warns():
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        PreprocessingConfig(skip_infinity_checks=True)
    assert any("has no effect" in str(w.message) for w in caught)


def test_leaving_it_alone_does_not_warn():
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        PreprocessingConfig()
    assert not any("has no effect" in str(w.message) for w in caught)
