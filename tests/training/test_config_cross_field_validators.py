"""Sensors for wave-13 cross-field config validators.

Each validator catches a "kwargs validated as well-formed, master toggle is False,
kwargs silently ignored" bug class -- recurring 5+ times across the codebase per
the wave-13 audit agent.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from mlframe.training.configs import (
    FeatureSelectionConfig,
    FeatureTypesConfig,
    OutputConfig,
    MultilabelDispatchConfig,
)


# ---- FeatureSelectionConfig ------------------------------------------------


def test_fsc_mrmr_kwargs_without_master_flag_raises():
    """REGRESSION: pre-fix mrmr_kwargs set but use_mrmr_fs=False -> kwargs silently ignored.

    Uses a valid MRMR.__init__ kwarg so the field validator passes and the
    cross-field model validator gets to fire (the case under test).
    """
    with pytest.raises(ValidationError, match="use_mrmr_fs=False"):
        FeatureSelectionConfig(
            use_mrmr_fs=False,
            mrmr_kwargs={"verbose": 0},
        )


def test_fsc_mrmr_kwargs_with_master_flag_passes():
    """Fsc mrmr kwargs with master flag passes."""
    cfg = FeatureSelectionConfig(
        use_mrmr_fs=True,
        mrmr_kwargs={"verbose": 0},
    )
    assert cfg.use_mrmr_fs is True


def test_fsc_rfecv_kwargs_without_models_raises():
    """Fsc rfecv kwargs without models raises."""
    with pytest.raises(ValidationError, match="rfecv_models"):
        FeatureSelectionConfig(
            rfecv_models=None,
            rfecv_kwargs={"verbose": 0},
        )


def test_fsc_rfecv_kwargs_with_models_passes():
    """Fsc rfecv kwargs with models passes."""
    cfg = FeatureSelectionConfig(
        rfecv_models=["cb"],
        rfecv_kwargs={"verbose": 0},
    )
    assert cfg.rfecv_models == ["cb"]


def test_fsc_boruta_kwargs_without_master_flag_raises():
    """Fsc boruta kwargs without master flag raises."""
    with pytest.raises(ValidationError, match="use_boruta_shap=False"):
        FeatureSelectionConfig(
            use_boruta_shap=False,
            boruta_shap_kwargs={"n_trials": 50},
        )


def test_fsc_default_kwargs_none_passes():
    """Bare defaults must keep working (no kwargs supplied = no contradiction)."""
    FeatureSelectionConfig()


# ---- FeatureTypesConfig ----------------------------------------------------


def test_ftc_text_features_with_master_off_raises():
    """REGRESSION: explicit text_features=[...] + use_text_features=False silently
    dropped the list and routed columns through cat path -- CatBoost burned minutes
    on degenerate ordinal encoding of high-card text."""
    with pytest.raises(ValidationError, match="text_features.+use_text_features=False"):
        FeatureTypesConfig(
            text_features=["job_description"],
            use_text_features=False,
        )


def test_ftc_text_features_with_master_on_passes():
    """Ftc text features with master on passes."""
    cfg = FeatureTypesConfig(
        text_features=["job_description"],
        use_text_features=True,
    )
    assert cfg.text_features == ["job_description"]


def test_ftc_no_text_features_master_off_passes():
    """The compose-pattern that triggered the bug must still be expressible: a
    later preset can turn off use_text_features as long as the text_features list
    is also cleared. Without that no operator can have a 'no text at all' preset."""
    FeatureTypesConfig(text_features=None, use_text_features=False)


# ---- OutputConfig ---------------------------------------------------------


def test_oc_bare_defaults_pass():
    """BC: legacy callers using OutputConfig() with default save_charts=True +
    empty data_dir must keep working (silent no-save by design)."""
    OutputConfig()


def test_oc_explicit_save_charts_true_no_data_dir_raises():
    """REGRESSION: explicit save_charts=True override + empty data_dir was
    silently no-op pre-fix."""
    with pytest.raises(ValidationError, match="data_dir is empty"):
        OutputConfig(save_charts=True)


def test_oc_explicit_save_charts_true_with_data_dir_passes():
    """Oc explicit save charts true with data dir passes."""
    cfg = OutputConfig(save_charts=True, data_dir="out/")
    assert cfg.save_charts is True


def test_oc_explicit_save_charts_false_passes():
    """Oc explicit save charts false passes."""
    OutputConfig(save_charts=False, data_dir="")


# ---- MultilabelDispatchConfig ---------------------------------------------


def test_mlc_invalid_strategy_raises():
    """Typo in strategy was silently accepted pre-fix; now caught."""
    with pytest.raises(ValidationError, match="strategy="):
        MultilabelDispatchConfig(strategy="wrappr")  # typo


def test_mlc_invalid_chain_order_strategy_raises():
    """Mlc invalid chain order strategy raises."""
    with pytest.raises(ValidationError, match="chain_order_strategy="):
        MultilabelDispatchConfig(chain_order_strategy="bogus")


def test_mlc_user_strategy_without_orderings_raises():
    """REGRESSION: pre-fix chain_order_strategy='user' + missing chain_order_user
    silently fell back to default ordering."""
    with pytest.raises(ValidationError, match="chain_order_user is None"):
        MultilabelDispatchConfig(chain_order_strategy="user")


def test_mlc_user_strategy_with_wrong_count_raises():
    """Mlc user strategy with wrong count raises."""
    with pytest.raises(ValidationError, match="orderings but n_chains"):
        MultilabelDispatchConfig(
            chain_order_strategy="user",
            n_chains=3,
            chain_order_user=[[0, 1]],  # only 1 ordering for 3 chains
        )


def test_mlc_user_strategy_with_matching_count_passes():
    """Mlc user strategy with matching count passes."""
    cfg = MultilabelDispatchConfig(
        chain_order_strategy="user",
        n_chains=2,
        chain_order_user=[[0, 1], [1, 0]],
    )
    assert cfg.chain_order_strategy == "user"


def test_mlc_default_random_passes():
    """Mlc default random passes."""
    MultilabelDispatchConfig()  # bare defaults: strategy='auto', chain_order_strategy='random'
