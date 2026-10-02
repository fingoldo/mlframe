"""Which selector constructors get a generated strict-parameter model, and how their enum-like parameters are constrained.

Each ``SelectorSpec`` names a selector class; ``_generate`` renders ``<key>.py`` next to this file from that class's signature with
``pyutilz.dev.signature_models``, so the config models cannot drift from the constructors (``tests/training/test_fs_params_in_sync.py``
fails when they do). Enum-like parameters are constrained with the selectors' OWN accepted-value tuples, imported here, so the allowed
values have one source.

This module is only imported by the generator and the sync test: the generated modules do not import the selectors, so building a
``FeatureSelectionConfig`` stays cheap.
"""

from __future__ import annotations

import importlib
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Tuple


def literal_source(values: Tuple[Any, ...]) -> str:
    """``Literal[...]`` source for ``values``; a ``None`` member makes the whole annotation ``Optional``."""
    present = tuple(v for v in values if v is not None)
    text = f"Literal[{', '.join(repr(v) for v in present)}]"
    return f"Optional[{text}]" if None in values else text


@dataclass(frozen=True)
class SelectorSpec:
    """One selector: where its constructor lives, which parameters the suite owns (excluded), and the enum / constraint overrides."""

    key: str
    class_name: str
    target_module: str
    target_attr: str
    exclude: Tuple[str, ...] = ()
    enums: Callable[[], Dict[str, Tuple[Any, ...]]] = field(default=lambda: {})
    overrides: Dict[str, str] = field(default_factory=dict)

    def target(self) -> Any:
        """The selector class (imported lazily; MRMR alone costs seconds)."""
        return getattr(importlib.import_module(self.target_module), self.target_attr)

    def all_overrides(self) -> Dict[str, str]:
        """Enum-derived ``Literal`` annotations plus the hand-written ``overrides`` (the latter win)."""
        out = {name: literal_source(values) for name, values in self.enums().items()}
        out.update(self.overrides)
        return out


def _mrmr_enums() -> Dict[str, Tuple[Any, ...]]:
    """MRMR's accepted-value tuples, keyed by the constructor parameter each one validates (the table in ``_mrmr_validate_strings``)."""
    from mlframe.feature_selection.filters.mrmr import param_constants as c

    return {
        "quantization_method": c._VALID_QUANTIZATION_METHODS,
        "nan_strategy": c._VALID_NAN_STRATEGIES,
        "mrmr_relevance_algo": c._VALID_MRMR_RELEVANCE_ALGOS,
        "mrmr_redundancy_algo": c._VALID_MRMR_REDUNDANCY_ALGOS,
        "fe_unary_preset": c._VALID_FE_UNARY_PRESETS,
        "fe_binary_preset": c._VALID_FE_BINARY_PRESETS,
        "cluster_aggregate_mode": c._VALID_CLUSTER_AGGREGATE_MODES,
        "nbins_strategy": c._VALID_NBINS_STRATEGIES,
        "mi_correction": c._VALID_MI_CORRECTIONS,
        "redundancy_aggregator": c._VALID_REDUNDANCY_AGGREGATORS,
        "stability_selection_method": c._VALID_STABILITY_SELECTION_METHODS,
        "dcd_distance": c._VALID_DCD_DISTANCES,
        "dcd_swap_method": c._VALID_DCD_SWAP_METHODS,
        "additional_rfecv_selection_rule": c._VALID_RFECV_SELECTION_RULES,
        "fe_hybrid_orth_default_scorer": c._VALID_FE_HYBRID_ORTH_DEFAULT_SCORERS,
        "fe_hybrid_orth_basis": c._VALID_FE_HYBRID_ORTH_BASES,
        "fe_hybrid_orth_hsic_kernel": c._VALID_FE_HYBRID_ORTH_HSIC_KERNELS,
        "fe_hybrid_orth_ensemble_aggregator": c._VALID_FE_HYBRID_ORTH_ENSEMBLE_AGGREGATORS,
        "fe_hybrid_orth_cluster_basis_aggregator": c._VALID_FE_HYBRID_ORTH_CLUSTER_BASIS_AGGREGATORS,
        "mi_normalization": ("none", "su"),
        "group_mi_aggregate": ("size", "equal"),
    }


def _rfecv_enums() -> Dict[str, Tuple[Any, ...]]:
    """RFECV's accepted-value tuples."""
    from mlframe.feature_selection.wrappers.rfecv import N_FEATURES_SELECTION_RULES

    return {"n_features_selection_rule": N_FEATURES_SELECTION_RULES}


SPECS: Tuple[SelectorSpec, ...] = (
    SelectorSpec("mrmr", "MRMRParams", "mlframe.feature_selection.filters", "MRMR", enums=_mrmr_enums,
                 overrides={"dcd_tau_cluster": 'Union[float, Literal["auto"]]'}),
    SelectorSpec("rfecv", "RFECVParams", "mlframe.feature_selection.wrappers", "RFECV", exclude=("estimator",), enums=_rfecv_enums),
    SelectorSpec("boruta_shap", "BorutaShapParams", "mlframe.feature_selection.boruta_shap", "BorutaShap"),
    SelectorSpec("shap_proxied_fs", "ShapProxiedFSParams", "mlframe.feature_selection.shap_proxied_fs", "ShapProxiedFS"),
    SelectorSpec("ace", "ACEParams", "mlframe.feature_selection.ace", "ACESelector", enums=lambda: {"importance": ("native", "permutation")}),
    SelectorSpec("forward_select", "ForwardSelectParams", "mlframe.feature_selection.functional_adapters", "ForwardSelectSelector"),
    SelectorSpec("greedy_backward_elimination", "GreedyBackwardEliminationParams", "mlframe.feature_selection.functional_adapters", "GreedyBackwardEliminationSelector"),
    SelectorSpec("zero_importance_pruning", "ZeroImportancePruningParams", "mlframe.feature_selection.functional_adapters", "ZeroImportancePruningSelector"),
    SelectorSpec("cascade_select", "CascadeSelectParams", "mlframe.feature_selection.functional_adapters", "CascadeSelectSelector"),
)
