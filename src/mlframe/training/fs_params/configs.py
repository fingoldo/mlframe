"""Strict sub-configs of ``FeatureSelectionConfig``: one per selector, plus the suite-level pre-screen and cluster-reduce groups.

Each selector config extends the model generated from its constructor's signature (``mrmr.py``, ``rfecv.py``, ...), so every
constructor parameter is a typed field and an unknown name, a wrong type or an out-of-range enum value raises when the config is
created. ``to_kwargs`` returns ONLY the fields the caller set: the suite keeps its own defaults for everything else (for example the
shallow-merged MRMR defaults), exactly as when the kwargs were a dict.

The generated modules never import the selector classes, so importing this module costs only pydantic's model build.
"""

from __future__ import annotations

from typing import Any, ClassVar, Dict, Literal, Optional, Tuple

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from .._sparse_params import SparseParamsModel
from .ace import ACEParams
from .boruta_shap import BorutaShapParams
from .cascade_select import CascadeSelectParams
from .forward_select import ForwardSelectParams
from .greedy_backward_elimination import GreedyBackwardEliminationParams
from .mrmr import MRMRParams
from .rfecv import RFECVParams
from .shap_proxied_fs import ShapProxiedFSParams
from .zero_importance_pruning import ZeroImportancePruningParams

#: Keys of the ``rfecv_models_params`` dict the suite builds; ``RFECVConfig.models`` entries must name one of them.
RFECV_MODEL_NAMES = ("cb_rfecv", "lgb_rfecv", "xgb_rfecv")

_STRICT = ConfigDict(extra="forbid", frozen=True, arbitrary_types_allowed=True)


class _Strict(BaseModel):
    """Frozen, unknown-key-rejecting base of the small hand-written groups."""

    model_config = _STRICT


class PreScreenConfig(_Strict):
    """Unsupervised pre-screen applied ONCE per suite to the train split before any selector (train-only fit; val/test see the same drop set).

    Conservative on purpose: aggressive correlation / cardinality filters would risk dropping jointly informative features.
    """

    enable: bool = True
    variance_threshold: float = Field(default=0.0, ge=0.0)  # drop columns whose variance equals this exactly
    null_fraction_threshold: float = Field(default=0.99, gt=0.0, le=1.0)  # drop columns whose null fraction is above this


class ClusterReduceConfig(_Strict):
    """Cluster-medoid pre-reduction wrapped around a selector: correlated features collapse to one representative first.

    A no-op (the bare selector on the full frame) when clustering removes less than ``min_reduction`` of the features.
    """

    enable: bool = True
    corr_threshold: float = Field(default=0.9, gt=0.0, le=1.0)
    min_reduction: float = Field(default=0.05, ge=0.0, lt=1.0)
    corr_method: Literal["pearson", "spearman", "kendall", "su"] = "pearson"

    def registry_kwargs(self) -> Dict[str, Any]:
        """The ``cluster_*`` keys ``registry._instantiate_*`` pops to build the wrapper."""
        return {
            "cluster_reduce": self.enable,
            "cluster_corr_threshold": self.corr_threshold,
            "cluster_min_reduction": self.min_reduction,
            "cluster_corr_method": self.corr_method,
        }


class RFECVConfig(SparseParamsModel, RFECVParams):
    """RFECV selectors the suite builds (``models``), the cluster-medoid wrap, and every ``RFECV.__init__`` parameter as a typed field.

    ``leakage_corr_threshold`` and ``mbh_adaptive_threshold`` keep the suite's historical values (0.95 / 30), not the constructor's, and
    are applied to every RFECV; ``estimator`` is the suite's choice (per ``models``) and is not a field.
    """

    SUITE_ONLY: ClassVar[Tuple[str, ...]] = ("models", "cluster")

    models: Tuple[str, ...] = Field(default=("cb_rfecv",), min_length=1)
    cluster: ClusterReduceConfig = ClusterReduceConfig()
    leakage_corr_threshold: Optional[float] = 0.95
    mbh_adaptive_threshold: int = Field(30, ge=1)

    @field_validator("models", mode="before")
    @classmethod
    def _canonical_models(cls, v: Any) -> Any:
        """Accept a bare backend name (``"cb"``) or a single string, canonicalised to the ``"cb_rfecv"`` keys; unknown names raise."""
        if isinstance(v, str):
            v = (v,)
        out = []
        unknown = []
        for name in v:
            key = name if name in RFECV_MODEL_NAMES else f"{name}_rfecv"
            if key not in RFECV_MODEL_NAMES:
                unknown.append(name)
            elif key not in out:
                out.append(key)
        if unknown:
            raise ValueError(f"RFECVConfig.models: unknown RFECV model(s) {unknown}. Valid names: {list(RFECV_MODEL_NAMES)}")
        return tuple(out)

    @model_validator(mode="after")
    def _check_selection_levers(self) -> "RFECVConfig":
        """``must_include`` and ``must_exclude`` cannot overlap, and no excluded feature may sit in a feature group."""
        include = set(self.must_include or ())
        exclude = set(self.must_exclude or ())
        clash = sorted(include & exclude, key=str)
        if clash:
            raise ValueError(f"RFECVConfig: {clash} are in both must_include and must_exclude")
        grouped = {f for members in (self.feature_groups or {}).values() for f in members}
        banned = sorted(grouped & exclude, key=str)
        if banned:
            raise ValueError(f"RFECVConfig: {banned} are in must_exclude but also members of a feature group")
        return self

    def suite_overrides(self) -> Dict[str, Any]:
        """The two knobs the suite always sets on every RFECV, whether or not the caller wrote them."""
        return {"leakage_corr_threshold": self.leakage_corr_threshold, "mbh_adaptive_threshold": self.mbh_adaptive_threshold}


class MRMRConfig(SparseParamsModel, MRMRParams):
    """MRMR constructor parameters as typed fields, plus the scope of the cross-target identity cache.

    ``identity_cache_scope``: ``"ctx"`` (default, safe) keeps the cache on the suite's TrainingContext so sibling suites cannot poison each
    other's results; ``"process"`` keeps it for the life of the interpreter (CI matrices that reuse identity results opt in).
    """

    SUITE_ONLY: ClassVar[Tuple[str, ...]] = ("identity_cache_scope",)

    identity_cache_scope: Literal["ctx", "process"] = "ctx"

    @model_validator(mode="after")
    def _check_group_levers(self) -> "MRMRConfig":
        """The group-MI knobs only mean something with ``group_aware_mi=True``."""
        explicit = self.model_fields_set
        if {"group_mi_aggregate", "group_mi_min_rows"} & explicit and not self.group_aware_mi:
            raise ValueError("MRMRConfig: group_mi_aggregate / group_mi_min_rows are set but group_aware_mi is False")
        return self


class BorutaShapConfig(SparseParamsModel, BorutaShapParams):
    """BorutaShap constructor parameters plus the cluster-medoid wrap (default on, via ``registry._instantiate_boruta_shap``)."""

    SUITE_ONLY: ClassVar[Tuple[str, ...]] = ("cluster",)

    cluster: ClusterReduceConfig = ClusterReduceConfig()

    def to_kwargs(self) -> Dict[str, Any]:
        """Constructor arguments plus the ``cluster_*`` keys the registry factory consumes."""
        return {**super().to_kwargs(), **self.cluster.registry_kwargs()}


class ShapProxiedFSConfig(SparseParamsModel, ShapProxiedFSParams):
    """ShapProxiedFS constructor parameters (it clusters correlated features itself, so it has no cluster wrap)."""


class ACEConfig(SparseParamsModel, ACEParams):
    """ACESelector constructor parameters."""


class ForwardSelectConfig(SparseParamsModel, ForwardSelectParams):
    """ForwardSelectSelector constructor parameters."""


class GreedyBackwardEliminationConfig(SparseParamsModel, GreedyBackwardEliminationParams):
    """GreedyBackwardEliminationSelector constructor parameters."""


class ZeroImportancePruningConfig(SparseParamsModel, ZeroImportancePruningParams):
    """ZeroImportancePruningSelector constructor parameters."""


class CascadeSelectConfig(SparseParamsModel, CascadeSelectParams):
    """CascadeSelectSelector constructor parameters."""
