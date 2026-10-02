"""Config contracts that did not hold: a round-trip that failed, and a safety flag whose default contradicted its twin.

CFG-04: a dumped `FeatureSelectionConfig` once failed to re-validate because a lever lived in two places; each lever has one
home now (`rfecv={...}`), so the dump rebuilds an equal config.

CFG-03: `PreprocessingConfig.skip_infinity_checks` defaulted to True (protection off) against `DataConfig`'s False,
and nothing reads it.
"""

from __future__ import annotations

import warnings

import pytest

from mlframe.training.configs import FeatureSelectionConfig, PreprocessingConfig


def test_a_dumped_feature_selection_config_revalidates():
    """A dumped feature selection config revalidates to an equal config with the same written levers."""
    cfg = FeatureSelectionConfig(rfecv={"models": ["cb"], "swap_top_k": 3})
    again = FeatureSelectionConfig(**cfg.model_dump())
    assert again == cfg
    assert again.selector_kwargs("rfecv") == cfg.selector_kwargs("rfecv") == {"swap_top_k": 3}


def test_the_previous_flat_spelling_of_a_lever_is_rejected():
    """There is one spelling of each lever: the flat ``rfecv_swap_top_k`` of the previous layout raises and names the new one."""
    with pytest.raises(ValueError, match="rfecv"):
        FeatureSelectionConfig(rfecv_swap_top_k=3)


def test_the_preprocessing_skip_flag_default_matches_the_data_config():
    """The preprocessing skip flag default matches the data config."""
    from mlframe.training.configs import DataConfig

    assert PreprocessingConfig().skip_infinity_checks is False
    assert DataConfig.model_fields["skip_infinity_checks"].default is False


def test_setting_the_inert_preprocessing_flag_warns():
    """Setting the inert preprocessing flag warns."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        PreprocessingConfig(skip_infinity_checks=True)
    assert any("has no effect" in str(w.message) for w in caught)


def test_leaving_it_alone_does_not_warn():
    """Leaving it alone does not warn."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        PreprocessingConfig()
    assert not any("has no effect" in str(w.message) for w in caught)
