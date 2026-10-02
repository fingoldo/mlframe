"""Feature-selection config for ``mlframe.training.configs``.

Split out from ``configs.py`` so the sibling config modules that need to reference ``FeatureSelectionConfig`` as a field type (notably
``TrainingConfig`` in ``_training_runtime_configs.py``) can import it without re-entering ``configs.py``; ``configs.py`` re-exports the class.

Each selector is configured by its own strict sub-config (``mlframe.training.fs_params.configs``) generated from the selector's constructor
signature, so an unknown parameter, a wrong type or an unsupported enum value raises when the config is created rather than deep inside a fit.
A selector is enabled by giving its field a config (``mrmr=MRMRConfig()``); ``None`` leaves it off.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

from pydantic import ConfigDict, Field, field_validator, model_validator

from ._configs_base import BaseConfig
from ._inert_fields import InertFieldsWarningMixin

# Keys of the ``rfecv_models_params`` dict ``select_target`` builds; ``RFECVConfig.models`` entries must name one of them.
RFECV_MODEL_NAMES = ("cb_rfecv", "lgb_rfecv", "xgb_rfecv")

# Selector field -> (sub-config class name in ``fs_params.configs``). The classes are imported lazily: building MRMRConfig's 500-odd fields costs
# seconds, which a default (all-selectors-off) config must not pay.
_SELECTOR_CLASSES = {
    "rfecv": "RFECVConfig",
    "mrmr": "MRMRConfig",
    "boruta_shap": "BorutaShapConfig",
    "shap_proxied_fs": "ShapProxiedFSConfig",
    "ace": "ACEConfig",
    "forward_select": "ForwardSelectConfig",
    "greedy_backward_elimination": "GreedyBackwardEliminationConfig",
    "zero_importance_pruning": "ZeroImportancePruningConfig",
    "cascade_select": "CascadeSelectConfig",
}

# Selector field -> (``use_*`` flag, kwargs key) of ``_build_pre_pipelines`` for the selectors it takes as flag + dict.
_PIPELINE_ARGS = {
    "boruta_shap": ("use_boruta_shap", "boruta_shap_kwargs"),
    "shap_proxied_fs": ("use_shap_proxied_fs", "shap_proxied_fs_kwargs"),
    "ace": ("use_ace_fs", "ace_kwargs"),
    "forward_select": ("use_forward_select_fs", "forward_select_kwargs"),
    "greedy_backward_elimination": ("use_greedy_backward_elimination_fs", "greedy_backward_elimination_kwargs"),
    "zero_importance_pruning": ("use_zero_importance_pruning_fs", "zero_importance_pruning_kwargs"),
    "cascade_select": ("use_cascade_select_fs", "cascade_select_kwargs"),
}


def _default_pre_screen() -> Any:
    """A default PreScreenConfig (imported lazily, like the selector sub-configs)."""
    from .fs_params.configs import PreScreenConfig

    return PreScreenConfig()


# Flat names of the previous layout -> where they live now; the first-class ``rfecv_*`` / ``mrmr_*`` levers were constructor parameters all along.
_RENAMED_LEVERS = {
    "rfecv_enable_stability_selection": "rfecv=RFECVConfig(stability_selection=True)",
    "rfecv_enable_permutation_importance": 'rfecv=RFECVConfig(importance_getter="permutation")',
    "rfecv_models": "rfecv=RFECVConfig(models=[...])",
    "rfecv_kwargs": "rfecv=RFECVConfig(<RFECV constructor parameters>)",
    "rfecv_cluster_reduce": "rfecv=RFECVConfig(cluster=ClusterReduceConfig(enable=...))",
    "rfecv_cluster_corr_threshold": "rfecv=RFECVConfig(cluster=ClusterReduceConfig(corr_threshold=...))",
    "rfecv_cluster_min_reduction": "rfecv=RFECVConfig(cluster=ClusterReduceConfig(min_reduction=...))",
    "rfecv_cluster_corr_method": "rfecv=RFECVConfig(cluster=ClusterReduceConfig(corr_method=...))",
    "mrmr_kwargs": "mrmr=MRMRConfig(<MRMR constructor parameters>)",
    "mrmr_identity_cache_scope": "mrmr=MRMRConfig(identity_cache_scope=...)",
    "pre_screen_unsupervised": "pre_screen=PreScreenConfig(enable=...)",
    "pre_screen_variance_threshold": "pre_screen=PreScreenConfig(variance_threshold=...)",
    "pre_screen_null_fraction_threshold": "pre_screen=PreScreenConfig(null_fraction_threshold=...)",
    "use_mrmr_fs": "mrmr=MRMRConfig(...)",
    "use_boruta_shap": "boruta_shap=BorutaShapConfig(...)",
    "boruta_shap_kwargs": "boruta_shap=BorutaShapConfig(<BorutaShap constructor parameters>)",
    "use_shap_proxied_fs": "shap_proxied_fs=ShapProxiedFSConfig(...)",
    "shap_proxied_fs_kwargs": "shap_proxied_fs=ShapProxiedFSConfig(<ShapProxiedFS constructor parameters>)",
    "use_ace_fs": "ace=ACEConfig(...)",
    "ace_kwargs": "ace=ACEConfig(<ACESelector constructor parameters>)",
    "use_forward_select_fs": "forward_select=ForwardSelectConfig(...)",
    "forward_select_kwargs": "forward_select=ForwardSelectConfig(<constructor parameters>)",
    "use_greedy_backward_elimination_fs": "greedy_backward_elimination=GreedyBackwardEliminationConfig(...)",
    "greedy_backward_elimination_kwargs": "greedy_backward_elimination=GreedyBackwardEliminationConfig(<constructor parameters>)",
    "use_zero_importance_pruning_fs": "zero_importance_pruning=ZeroImportancePruningConfig(...)",
    "zero_importance_pruning_kwargs": "zero_importance_pruning=ZeroImportancePruningConfig(<constructor parameters>)",
    "use_cascade_select_fs": "cascade_select=CascadeSelectConfig(...)",
    "cascade_select_kwargs": "cascade_select=CascadeSelectConfig(<constructor parameters>)",
}


class FeatureSelectionConfig(InertFieldsWarningMixin, BaseConfig):
    """Configuration for feature selection.

    Default FS is UNSUPERVISED-ONLY: only the variance==0 / nulls>99% ``pre_screen`` runs; no supervised selector runs unless its field is set.
    A cheap default-on supervised filter (univariate MI top-k) was benched and REJECTED as a default: it wins on linear downstreams and wide /
    noisy data but HURTS noise-robust tree downstreams on low-noise data, so supervised FS stays opt-in.

    Parameters
    ----------
    pre_screen : PreScreenConfig
        Unsupervised pre-screen (zero variance / mostly-null columns) applied once per suite to the train split before any selector.
    rfecv : RFECVConfig, optional
        RFECV selectors to build (``models``), their ``RFECV.__init__`` parameters, and the default-on cluster-medoid wrap
        (``cluster``): features correlated above ``corr_threshold`` are collapsed to one representative before RFECV runs. ``leakage_corr_threshold``
        (default 0.95) sends columns with ``|Pearson(x, y)|`` above it through RFECV's ``leakage_action``; ``None`` disables the check.
    mrmr : MRMRConfig, optional
        mRMR (minimum redundancy, maximum relevance) selector: every ``MRMR.__init__`` parameter plus ``identity_cache_scope``.
    boruta_shap : BorutaShapConfig, optional
        SHAP-driven Boruta wrapper. Off by default: 10-20x the runtime of MRMR / RFECV (a TreeExplainer on a doubled matrix per trial).
    shap_proxied_fs : ShapProxiedFSConfig, optional
        SHAP-coalition-proxy selector (OOF TreeExplainer, subset search, honest re-validation); markedly costlier than MRMR / RFECV.
    ace : ACEConfig, optional
        Artificial Contrasts with Ensembles (Tuv et al. 2009): fits the estimator on ``[X | contrasts]`` once per replicate.
    forward_select : ForwardSelectConfig, optional
        Greedy forward selection (one CV-scored refit per added feature); O(features) refits, so opt-in.
    greedy_backward_elimination : GreedyBackwardEliminationConfig, optional
        Greedy backward elimination from the full set; O(features^2 x folds), so opt-in.
    zero_importance_pruning : ZeroImportancePruningConfig, optional
        Iteratively drops zero-importance features; cheap next to the two above but still several full-frame CV refits.
    cascade_select : CascadeSelectConfig, optional
        Boruta -> forward-select -> RFECV cascade.
    custom_pre_pipelines : dict
        Extra user-supplied pre-pipelines, cloned per model.
    skip_identity_equivalent_pre_pipelines : bool
        A selection pipeline that keeps every column and adds none is not trained as a separate variant (set False to train both).
    unified_cv_policy : bool
        One split policy for every selector that cross-validates or holds out rows internally: temporal suites fold forward in time, grouped
        suites isolate groups, the rest stay i.i.d. (see ``feature_selection.cv_policy``). An explicit selector ``cv`` wins; False keeps each
        selector's own shuffled split.
    use_sample_weights_in_fs : bool
        When True, FS becomes weight-aware and re-runs per weight schema (MRMR.fit / RFECV.fit receive the suite's sample_weight). When False
        (default) FS runs once per target and is reused across weight schemas -- faster, with selected features reflecting uniform weights.
    """

    model_config = ConfigDict(extra="forbid")

    pre_screen: Any = Field(default_factory=_default_pre_screen)
    rfecv: Any = None
    mrmr: Any = None
    boruta_shap: Any = None
    shap_proxied_fs: Any = None
    ace: Any = None
    forward_select: Any = None
    greedy_backward_elimination: Any = None
    zero_importance_pruning: Any = None
    cascade_select: Any = None
    custom_pre_pipelines: Dict[str, Any] = Field(default_factory=dict)
    skip_identity_equivalent_pre_pipelines: bool = True
    unified_cv_policy: bool = True
    use_sample_weights_in_fs: bool = False

    @model_validator(mode="before")
    @classmethod
    def _reject_flat_names_of_the_previous_layout(cls, data: Any) -> Any:
        """A name from the flat layout raises with the spelling that replaces it, instead of the generic unknown-field error."""
        if isinstance(data, dict):
            moved = {k: _RENAMED_LEVERS[k] for k in data if k in _RENAMED_LEVERS}
            for key in data:
                if key not in moved and key.startswith("rfecv_") and key not in cls.model_fields:
                    moved[key] = f"rfecv=RFECVConfig({key[len('rfecv_'):]}=...)"
                elif key not in moved and key.startswith("mrmr_") and key not in cls.model_fields:
                    moved[key] = f"mrmr=MRMRConfig({key[len('mrmr_'):]}=...)"
            if moved:
                listing = "; ".join(f"{old!r} -> {new}" for old, new in moved.items())
                raise ValueError(f"FeatureSelectionConfig: the flat selector fields were replaced by one strict sub-config per selector: {listing}")
        return data

    @field_validator("pre_screen", mode="before")
    @classmethod
    def _coerce_pre_screen(cls, v: Any) -> Any:
        """Build the pre-screen group from a dict, default it when unset."""
        from .fs_params.configs import PreScreenConfig

        if v is None:
            return PreScreenConfig()
        if isinstance(v, PreScreenConfig):
            return v
        if isinstance(v, dict):
            return PreScreenConfig(**v)
        raise TypeError(f"FeatureSelectionConfig.pre_screen must be a PreScreenConfig or a dict, got {type(v).__name__}")

    @field_validator(*_SELECTOR_CLASSES, mode="before")
    @classmethod
    def _coerce_selector(cls, v: Any, info: Any) -> Any:
        """Turn a dict (or ``True`` for "on with defaults") into the selector's strict sub-config; an instance passes through, ``None`` / ``False`` is off."""
        if v is None or v is False:
            return None
        from .fs_params import configs

        target = getattr(configs, _SELECTOR_CLASSES[info.field_name])
        if isinstance(v, target):
            return v
        if v is True:
            return target()
        if isinstance(v, dict):
            return target(**v)
        raise TypeError(f"FeatureSelectionConfig.{info.field_name} must be a {target.__name__}, a dict of its fields, True or None; got {type(v).__name__}")

    def selector_kwargs(self, field_name: str) -> Optional[Dict[str, Any]]:
        """Constructor keyword arguments the caller set for the selector in ``field_name``, or None when that selector is off."""
        sub = getattr(self, field_name)
        return None if sub is None else sub.to_kwargs()

    def pre_pipeline_kwargs(self) -> Dict[str, Any]:
        """Keyword arguments of ``_build_pre_pipelines`` that come from this config (everything but MRMR / RFECV model names and per-run state)."""
        from .fs_params.configs import ClusterReduceConfig

        rfecv = self.rfecv
        cluster = rfecv.cluster if rfecv is not None else ClusterReduceConfig()
        out: Dict[str, Any] = {
            "rfecv_leakage_corr_threshold": rfecv.leakage_corr_threshold if rfecv is not None else 0.95,
            "rfecv_mbh_adaptive_threshold": rfecv.mbh_adaptive_threshold if rfecv is not None else 30,
            "rfecv_cluster_reduce": cluster.enable,
            "rfecv_cluster_corr_threshold": cluster.corr_threshold,
            "rfecv_cluster_min_reduction": cluster.min_reduction,
            "rfecv_cluster_corr_method": cluster.corr_method,
            "rfecv_overrides": rfecv.to_kwargs() if rfecv is not None else None,
            "use_sample_weights_in_fs": self.use_sample_weights_in_fs,
        }
        for name, (flag, kwargs_key) in _PIPELINE_ARGS.items():
            kwargs = self.selector_kwargs(name)
            out[flag] = kwargs is not None
            out[kwargs_key] = kwargs
        return out
