"""Regression: the CorrelatedFeaturesSelector cluster-medoid pre-reduction wrap (default-ON for
both RFECV and BorutaShap) must be configurable through the public FeatureSelectionConfig.

``registry._instantiate_rfecv`` / ``_instantiate_boruta_shap`` pop ``cluster_reduce`` /
``cluster_corr_threshold`` / ``cluster_min_reduction`` from the kwargs to drive the wrap.
Pre-fix the FeatureSelectionConfig kwargs validators rejected those keys (they aren't
RFECV/BorutaShap ``__init__`` params), so the default-ON wrap was unreachable AND
un-fuzzable through the public path. The selector sub-configs carry them as a typed ``cluster`` group.
"""

from __future__ import annotations

import pytest

from mlframe.training.configs import FeatureSelectionConfig

_CLUSTER = {"enable": True, "corr_threshold": 0.85, "min_reduction": 0.1}


def test_boruta_shap_cluster_group_reaches_the_registry_kwargs():
    """The BorutaShap cluster group is typed and turns into the ``cluster_*`` keys the registry factory pops."""
    cfg = FeatureSelectionConfig(boruta_shap={"cluster": dict(_CLUSTER)})
    kwargs = cfg.selector_kwargs("boruta_shap")
    assert kwargs["cluster_reduce"] is True
    assert kwargs["cluster_corr_threshold"] == 0.85
    assert kwargs["cluster_min_reduction"] == 0.1


def test_rfecv_cluster_reduce_reachable_via_the_cluster_group():
    """The suite RFECV cluster-medoid wrap is configured through ``rfecv.cluster``, not through RFECV constructor parameters.

    The suite builds RFECV directly and ``_build_pre_pipelines`` applies the CorrelatedFeaturesSelector wrap itself, so a ``cluster_reduce``
    constructor key would be a TypeError in ``RFECV.__init__``; the strict config rejects it at construction instead.
    """
    cfg = FeatureSelectionConfig(rfecv={"models": ["lgb"], "cluster": {"enable": False, "corr_threshold": 0.9, "min_reduction": 0.1}})
    args = cfg.pre_pipeline_kwargs()
    assert args["rfecv_cluster_reduce"] is False
    assert args["rfecv_cluster_corr_threshold"] == 0.9
    assert args["rfecv_cluster_min_reduction"] == 0.1
    with pytest.raises(ValueError, match="cluster_reduce"):
        FeatureSelectionConfig(rfecv={"cluster_reduce": False, "models": ["lgb"]})


@pytest.mark.parametrize(
    "field",
    ["boruta_shap", "rfecv"],
)
def test_validator_still_rejects_genuinely_unknown_keys(field):
    """An unknown key inside any selector sub-config raises when the config is created."""
    with pytest.raises(ValueError, match="totally_bogus_key"):
        FeatureSelectionConfig(**{field: {"totally_bogus_key": 1}})


def test_cluster_reduce_keys_drive_registry_wrap():
    """End-to-end: the keys the validator now allows actually toggle the registry wrap."""
    pytest.importorskip("shap")
    from mlframe.feature_selection.boruta_shap import BorutaShap
    from mlframe.feature_selection.filters.correlated_features import CorrelatedFeaturesSelector
    from mlframe.feature_selection.registry import _instantiate_boruta_shap

    wrapped = _instantiate_boruta_shap(cluster_reduce=True, cluster_corr_threshold=0.85, cluster_min_reduction=0.1)
    bare = _instantiate_boruta_shap(cluster_reduce=False)
    assert isinstance(wrapped, CorrelatedFeaturesSelector), "cluster_reduce=True must yield the CorrelatedFeaturesSelector medoid wrap"
    assert isinstance(bare, BorutaShap) and not isinstance(bare, CorrelatedFeaturesSelector), "cluster_reduce=False must yield bare BorutaShap"
