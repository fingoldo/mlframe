"""Nested config dataclasses for :class:`MRMR`'s constructor (audits/mrmr_audit_2026-07-16/10_config_dataclass_proposal.md).

``MRMR.__init__`` has ~250 flat parameters, grown organically by feature area. This module packages
the six cohesive knob-clusters the audit identified into ``pydantic.BaseModel`` config objects,
mirroring the ``CatFEConfig`` precedent (``cat_fe_state.py``) and ``mlframe.training.feature_handling``'s
pydantic-config pattern -- construction-time validation (a typo'd enum value raises immediately, not
minutes into a fit()) instead of MRMR's ad hoc ``_validate_string_params`` late-validation pass.

**Purely additive, not a breaking migration** (per the proposal's own sequencing note: the flat-kwarg
path stays indefinitely given the enormous blast radius of ~50+ existing call sites). Each config is an
OPTIONAL new constructor kwarg (``MRMR(dcd_config=DCDConfig(enable=False))``); when omitted (the
default, matching every existing caller), ``self.<flat_attr>`` keeps its value exactly as resolved by
the individual flat kwargs, unchanged. When a config IS passed, its fields are copied onto ``self``
AFTER the flat kwargs are stored, so the config wins over the individual flat defaults for that
cluster -- passing both a config AND overriding one of its own flat kwargs is a caller error this
module does not attempt to reconcile (last-applied-wins: config after flats).
"""
from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, ConfigDict, Field


class _MRMRSubConfig(BaseModel):
    """Shared base: forbid unknown fields (catches typos at construction time, matching finding
    S-F5's "typos silently degrade" fix rather than the training package's permissive extra="allow")."""

    model_config = ConfigDict(extra="forbid", frozen=True)


class FastSearchConfig(_MRMRSubConfig):
    """The ``fe_fast_search``-gated sub-knob overrides (``MRMR._FAST_SEARCH_OVERRIDES``).

    Mirrors the class-level override table ``_apply_fast_search_profile`` reads verbatim -- keep this
    dataclass's fields in sync with that table if it grows.
    """

    fe_fast_search: bool = False
    fe_max_steps: int = 1
    fe_pair_prewarp_enable: bool = False
    fe_stability_vote_enable: bool = False
    fe_escalation_underdelivery_enable: bool = False


class StabilitySelectionConfig(_MRMRSubConfig):
    """Stability-selection outer loop (``_stability_outer_fit``): cluster / complementary-pairs."""

    stability_selection_method: Literal["classic", "cluster", "complementary_pairs"] = "classic"
    stability_selection_corr_threshold: float = Field(default=0.8, gt=0.0, le=1.0)
    stability_n_bootstrap: int = Field(default=50, ge=1)
    stability_pi_threshold: float = Field(default=0.6, gt=0.0, lt=1.0)


class SynergyRedundancyConfig(_MRMRSubConfig):
    """Cross-cutting redundancy/synergy research knobs (findings A1-F14 in the ctor signature)."""

    redundancy_aggregator: Optional[Literal["jmim", "auto"]] = None
    bur_lambda: float = Field(default=0.0, ge=0.0)
    relaxmrmr_alpha: float = Field(default=0.0, ge=0.0)
    cmi_perm_stop: bool = False
    cmi_perm_n_permutations: int = Field(default=100, ge=1)
    cmi_perm_alpha: float = Field(default=0.05, gt=0.0, lt=1.0)
    cpt_test: bool = False
    cpt_n_permutations: int = Field(default=200, ge=1)
    pid_synergy_bonus: float = Field(default=0.0, ge=0.0)
    uaed_auto_size: bool = False
    mi_correction: Literal["none", "miller_madow", "chao_shen"] = "none"
    mi_normalization: Literal["none", "su"] = "none"


class GroupAwareConfig(_MRMRSubConfig):
    """Group-aware relevance MI (``I(X;Y|G)``) and the ``groups=`` strictness contract."""

    group_aware_mi: bool = False
    strict_groups: bool = True
    group_mi_aggregate: Literal["size", "equal"] = "size"
    group_mi_min_rows: int = Field(default=20, ge=1)


class DCDConfig(_MRMRSubConfig):
    """Denoised Cluster-Discovery (DCD): correlated-reflection clustering + aggregate scoring."""

    dcd_enable: bool = True
    dcd_distance: str = "su"
    dcd_min_cluster_size: int = Field(default=2, ge=2)
    dcd_max_cluster_size: int = Field(default=12, ge=2)
    dcd_cluster_size_threshold: int = Field(default=4, ge=1)
    dcd_pairwise_cache_max: int = Field(default=50_000, ge=0)
    dcd_swap_method: str = "auto"
    dcd_swap_alpha: float = Field(default=0.05, gt=0.0, lt=1.0)
    dcd_swap_gain_threshold: float = Field(default=0.05, ge=0.0)
    dcd_swap_npermutations: int = Field(default=199, ge=1)
    dcd_tau_calibration_n_pairs: int = Field(default=100, ge=1)
    dcd_tau_calibration_seed: int = 0
    dcd_postoc_compose: bool = False


class HybridOrthScorersConfig(_MRMRSubConfig):
    """Per-scorer enable/param pairs for the hybrid-orth FE family's synergy scorers -- split out of
    ``HybridOrthConfig`` (proposal's own recommendation) since this sub-cluster alone is ~20 fields."""

    ksg_enable: bool = False
    ksg_n_neighbors: int = Field(default=3, ge=1)
    ksg_min_uplift: float = 0.95
    ksg_min_abs_mi_frac: float = Field(default=0.05, ge=0.0)
    copula_enable: bool = False
    copula_n_bins: int = Field(default=20, ge=2)
    dcor_enable: bool = False
    dcor_n_sample: int = Field(default=500, ge=1)
    hsic_enable: bool = False
    hsic_kernel: str = "rbf"
    hsic_n_sample: int = Field(default=500, ge=1)
    jmim_enable: bool = False
    jmim_n_bins: int = Field(default=10, ge=2)
    tc_enable: bool = False
    tc_n_bins: int = Field(default=10, ge=2)
    cmim_enable: bool = False
    cmim_n_bins: int = Field(default=10, ge=2)
    auto_scorer_enable: bool = False
    auto_scorer_n_boot: int = Field(default=5, ge=1)
    ensemble_enable: bool = False
    ensemble_aggregator: str = "mean_rank"
    ensemble_scorers: tuple = ()
    meta_enable: bool = False
    meta_force_scorer: Optional[str] = None
    default_scorer: str = "plug_in"


class HybridOrthConfig(_MRMRSubConfig):
    """The ``fe_hybrid_orth_*`` FE family (single largest cluster, ~70 knobs). Per-scorer knobs are
    nested under ``scorers`` (see ``HybridOrthScorersConfig``); everything else (basis/degree/arity
    generation, adaptive routing, gating) stays flat here."""

    enable: bool = True
    degrees: tuple = (2, 3)
    basis: str = "auto"
    top_k: int = Field(default=5, ge=1)
    extra_bases: tuple = ()
    fourier_freqs: tuple = (1.0, 2.0)
    fourier_powers: tuple = (1, 2)
    spline_knots: int = Field(default=5, ge=1)

    pair_enable: bool = True
    pair_max_degree: int = Field(default=2, ge=1)

    triplet_enable: bool = True
    triplet_max_degree: int = Field(default=1, ge=1)
    triplet_seed_k: int = Field(default=4, ge=1)
    triplet_top_count: int = Field(default=2, ge=1)

    quadruplet_enable: bool = True
    quadruplet_max_degree: int = Field(default=1, ge=1)
    quadruplet_seed_k: int = Field(default=4, ge=1)
    quadruplet_top_count: int = Field(default=2, ge=1)

    adaptive_arity_enable: bool = False
    adaptive_arity_max_arity: int = Field(default=3, ge=1)
    adaptive_arity_max_degree: int = Field(default=1, ge=1)
    adaptive_arity_seed_k: int = Field(default=4, ge=1)
    adaptive_arity_top_count: int = Field(default=3, ge=1)

    lasso_enable: bool = False
    lasso_alpha: float = Field(default=0.01, gt=0.0)
    elasticnet_enable: bool = False
    elasticnet_alpha: float = Field(default=0.01, gt=0.0)

    adaptive_degree_enable: bool = False
    adaptive_degree_range: tuple = (1, 2, 3, 4, 5, 6)
    adaptive_degree_min_uplift: float = 1.05

    conditional_routing_enable: bool = False
    conditional_routing_top_k: int = Field(default=5, ge=1)
    conditional_routing_min_uplift: float = 1.10
    conditional_routing_degrees: tuple = (2, 3)

    diff_basis_enable: bool = False
    diff_basis_corr_threshold: float = Field(default=0.7, gt=0.0, le=1.0)
    diff_basis_degrees: tuple = (1, 2, 3)
    diff_basis_top_k: int = Field(default=3, ge=1)

    cluster_basis_enable: bool = False
    cluster_basis_aggregator: str = "mean_z"
    cluster_basis_degrees: tuple = (2, 3)
    cluster_basis_top_k: int = Field(default=3, ge=1)

    bootstrap_enable: bool = False
    bootstrap_n_boot: int = Field(default=10, ge=1)
    bootstrap_sample_fraction: float = Field(default=0.8, gt=0.0, le=1.0)

    three_gate_enable: bool = False
    three_gate_n_folds: int = Field(default=5, ge=2)
    three_gate_cmi_min: float = Field(default=0.001, ge=0.0)

    scorers: HybridOrthScorersConfig = Field(default_factory=HybridOrthScorersConfig)


# Maps each config's field name -> the flat ``MRMR.__init__`` attribute name it overrides. Built
# once here (not re-derived per instance) so ``_apply_mrmr_config_objects`` stays a straight lookup.
# ``HybridOrthConfig``'s top-level fields prefix with ``fe_hybrid_orth_``; its nested ``scorers``
# fields ALSO prefix with ``fe_hybrid_orth_`` (flattened at apply time, see ``_apply_mrmr_config_objects``).
_FAST_SEARCH_FIELD_MAP = {f: f for f in FastSearchConfig.model_fields}
_STABILITY_FIELD_MAP = {f: f for f in StabilitySelectionConfig.model_fields}
_SYNERGY_FIELD_MAP = {f: f for f in SynergyRedundancyConfig.model_fields}
_GROUP_AWARE_FIELD_MAP = {f: f for f in GroupAwareConfig.model_fields}
_DCD_FIELD_MAP = {f: f for f in DCDConfig.model_fields}
_HYBRID_ORTH_FIELD_MAP = {f: f"fe_hybrid_orth_{f}" for f in HybridOrthConfig.model_fields if f != "scorers"}
_HYBRID_ORTH_SCORERS_FIELD_MAP = {f: f"fe_hybrid_orth_{f}" for f in HybridOrthScorersConfig.model_fields}


def apply_mrmr_config_objects(
    self,
    *,
    fast_search_config: Optional[FastSearchConfig],
    stability_config: Optional[StabilitySelectionConfig],
    synergy_config: Optional[SynergyRedundancyConfig],
    group_aware_config: Optional[GroupAwareConfig],
    dcd_config: Optional[DCDConfig],
    hybrid_orth_config: Optional[HybridOrthConfig],
) -> None:
    """Copy each PASSED (non-``None``) nested config's fields onto ``self``'s matching flat attrs.

    Called at the END of ``MRMR.__init__``, after ``store_params_in_object`` has already set every
    flat attr from its own kwarg -- so a config, when given, is the authoritative source for its
    cluster's flat attrs (overriding whatever the individual flat kwargs resolved to), matching
    ``CatFEConfig``'s existing precedent of "the nested config IS the state, once provided".
    """
    for config, field_map in (
        (fast_search_config, _FAST_SEARCH_FIELD_MAP),
        (stability_config, _STABILITY_FIELD_MAP),
        (synergy_config, _SYNERGY_FIELD_MAP),
        (group_aware_config, _GROUP_AWARE_FIELD_MAP),
        (dcd_config, _DCD_FIELD_MAP),
    ):
        if config is None:
            continue
        for field_name, attr_name in field_map.items():
            setattr(self, attr_name, getattr(config, field_name))
    if hybrid_orth_config is not None:
        for field_name, attr_name in _HYBRID_ORTH_FIELD_MAP.items():
            setattr(self, attr_name, getattr(hybrid_orth_config, field_name))
        for field_name, attr_name in _HYBRID_ORTH_SCORERS_FIELD_MAP.items():
            setattr(self, attr_name, getattr(hybrid_orth_config.scorers, field_name))


__all__ = [
    "FastSearchConfig",
    "StabilitySelectionConfig",
    "SynergyRedundancyConfig",
    "GroupAwareConfig",
    "DCDConfig",
    "HybridOrthScorersConfig",
    "HybridOrthConfig",
    "apply_mrmr_config_objects",
]
