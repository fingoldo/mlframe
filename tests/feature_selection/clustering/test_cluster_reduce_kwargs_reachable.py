"""Regression: the GroupAwareMRMR cluster-medoid pre-reduction wrap (default-ON for
both RFECV and BorutaShap) must be configurable through the public FeatureSelectionConfig.

``registry._instantiate_rfecv`` / ``_instantiate_boruta_shap`` pop ``cluster_reduce`` /
``cluster_corr_threshold`` / ``cluster_min_reduction`` from the kwargs to drive the wrap.
Pre-fix the FeatureSelectionConfig kwargs validators rejected those keys (they aren't
RFECV/BorutaShap ``__init__`` params), so the default-ON wrap was unreachable AND
un-fuzzable through the public path. The validators now whitelist them (the same
mechanism rfecv_kwargs already used for ``cv_n_splits``).
"""

from __future__ import annotations

import pytest

from mlframe.training.configs import FeatureSelectionConfig

_CLUSTER_KWARGS = {"cluster_reduce": True, "cluster_corr_threshold": 0.85, "cluster_min_reduction": 0.1}


def test_boruta_shap_kwargs_accepts_cluster_reduce_keys():
    cfg = FeatureSelectionConfig(use_boruta_shap=True, boruta_shap_kwargs=dict(_CLUSTER_KWARGS))
    assert cfg.boruta_shap_kwargs["cluster_reduce"] is True
    assert cfg.boruta_shap_kwargs["cluster_corr_threshold"] == 0.85


def test_rfecv_cluster_reduce_reachable_via_first_class_fields():
    """The suite RFECV cluster-medoid wrap is reachable + fuzzable through the DEDICATED first-class config
    fields (``rfecv_cluster_reduce`` / ``rfecv_cluster_corr_threshold`` / ``rfecv_cluster_min_reduction``),
    NOT through ``rfecv_kwargs``. The suite builds RFECV directly in ``configure_training_params`` and forwards
    ``rfecv_kwargs`` VERBATIM to ``RFECV(**rfecv_kwargs)``, so a ``cluster_reduce`` key there would TypeError in
    ``RFECV.__init__``; ``_build_pre_pipelines`` reads the first-class fields and applies the GroupAwareMRMR wrap
    itself. (Contrast BorutaShap, whose registry factory POPS the cluster keys before constructing, so they ARE
    whitelisted in ``boruta_shap_kwargs``.) The pre-fix shape -- whitelisting cluster keys in ``rfecv_kwargs`` --
    was the stale proxy; the real reach is the first-class fields."""
    cfg = FeatureSelectionConfig(
        rfecv_models=["lgb"],
        rfecv_cluster_reduce=False,
        rfecv_cluster_corr_threshold=0.9,
        rfecv_cluster_min_reduction=0.1,
    )
    assert cfg.rfecv_cluster_reduce is False
    assert cfg.rfecv_cluster_corr_threshold == 0.9
    assert cfg.rfecv_cluster_min_reduction == 0.1
    # rfecv_kwargs correctly REJECTS cluster keys (they would TypeError in the verbatim-forwarded RFECV.__init__).
    with pytest.raises(ValueError, match="unknown key"):
        FeatureSelectionConfig(rfecv_models=["lgb"], rfecv_kwargs={"cluster_reduce": False})


@pytest.mark.parametrize(
    "field,enable",
    [
        ("boruta_shap_kwargs", {"use_boruta_shap": True}),
        ("rfecv_kwargs", {"rfecv_models": ["lgb"]}),
    ],
)
def test_validator_still_rejects_genuinely_unknown_keys(field, enable):
    """The whitelist must not weaken the guard against truly-unknown keys."""
    with pytest.raises(ValueError, match="unknown key"):
        FeatureSelectionConfig(**enable, **{field: {"totally_bogus_key": 1}})


def test_cluster_reduce_keys_drive_registry_wrap():
    """End-to-end: the keys the validator now allows actually toggle the registry wrap."""
    pytest.importorskip("shap")
    from mlframe.feature_selection.boruta_shap import BorutaShap
    from mlframe.feature_selection.filters.group_aware import GroupAwareMRMR
    from mlframe.feature_selection.registry import _instantiate_boruta_shap

    wrapped = _instantiate_boruta_shap(cluster_reduce=True, cluster_corr_threshold=0.85, cluster_min_reduction=0.1)
    bare = _instantiate_boruta_shap(cluster_reduce=False)
    assert isinstance(wrapped, GroupAwareMRMR), "cluster_reduce=True must yield the GroupAwareMRMR medoid wrap"
    assert isinstance(bare, BorutaShap) and not isinstance(bare, GroupAwareMRMR), "cluster_reduce=False must yield bare BorutaShap"
