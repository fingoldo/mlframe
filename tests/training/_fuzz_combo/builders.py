"""Shared suite-config builders: single edit point for new fuzz axes.

Each ``*_from_flat`` takes named primitives (no FuzzCombo dependency); the
FuzzCombo-aware wrapper forwards ``combo.*`` attrs. mlframe configs are
imported lazily in-body so this module stays import-light.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

from .combo import FuzzCombo


# ---------------------------------------------------------------------------
# Shared suite-config builders (2026-05-18 refactor)
# ---------------------------------------------------------------------------
#
# Goal: adding a new axis = one edit (here), not N edits across the pytest
# suite call site + the 1M harness. Both call sites consume these builders
# either via a FuzzCombo instance (pytest suite) OR via flat keyword args
# (1M harness which randomises axes via its own _axis_rng).
#
# Pattern: the "_from_flat" function takes named primitives (no FuzzCombo
# dependency); the FuzzCombo-aware wrapper just forwards combo.* attrs.


def build_cat_fe_config_from_flat(
    *,
    use_mrmr_fs: bool,
    cat_fe_enable: bool,
    cat_fe_include_numeric: bool,
):
    """Return a CatFEConfig honoring the cat-FE enable + include_numeric
    axes. None when use_mrmr_fs=False OR when defaults are fine (library
    default already has enable=True, include_numeric=False)."""
    if not use_mrmr_fs:
        return None
    from mlframe.feature_selection.filters.cat_fe_state import CatFEConfig

    if not cat_fe_enable:
        return CatFEConfig(enable=False)
    if cat_fe_include_numeric:
        return CatFEConfig(enable=True, include_numeric=True)
    return None  # library default


def build_mrmr_kwargs_from_flat(
    *,
    use_mrmr_fs: bool,
    interactions_max_order: int = 1,
    fe_max_steps: int = 1,
    cat_fe_config: Any = None,
    fe_npermutations: int = 0,
    fe_ntop_features: int = 0,
    fe_unary_preset: str = "minimal",
    fe_binary_preset: str = "minimal",
    fe_smart_polynom_iters: int = 0,
    fe_smart_polynom_optimization_steps: int = 1000,
    fe_min_polynom_degree: int = 3,
    fe_max_polynom_degree: int = 3,
    # 2026-05-21 -- FE pair-check subsample budget. Threads the new MRMR
    # __init__ knob added in feat(fe+suite) 5223085 alongside the matching
    # fe_smart_polynom subsample. 0 = disabled (legacy full-frame path);
    # >0 AND < len(X) fires the subsample MI sweep with full-n survivor
    # rebuild. Both subsamples share the unified FE subsample upstream
    # (UNIFIED_FE_SUBSAMPLE_N=30_000 as of 2026-06-25); this fuzz builder pins
    # its OWN explicit budget below so its behavior is independent of the
    # upstream default.
    fe_check_pairs_subsample_n: int = 0,
    fe_smart_polynom_subsample_n: int = 0,
    # Suite-side fuzz-speed pins. Callers can override.
    verbose: int = 0,
    max_runtime_mins: int = 1,
    n_workers: int = 1,
    quantization_nbins: int = 5,
    use_simple_mode: bool = True,
    min_nonzero_confidence: float = 0.9,
    max_consec_unconfirmed: int = 2,
    full_npermutations: int = 2,
    # 2026-05-27 friend-graph + cluster-aggregate knobs (mrmr.py __init__).
    build_friend_graph: bool = True,
    friend_graph_prune: bool = False,
    cluster_aggregate_enable: bool = True,
    cluster_aggregate_mode: str = "augment",
    # 2026-05-30 audit-pass-6 Wave 7/8/9 MRMR ctor knobs.
    nbins_strategy: str = "mdlp",
    mi_correction: str = "none",
    redundancy_aggregator: "str | None" = None,
    bur_lambda: float = 0.0,
    cmi_perm_stop: bool = False,
    stability_selection_method: str = "classic",
    mi_normalization: str = "none",
    dcd_enable: bool = False,
    # 2026-05-30 audit-pass-6 LOW-tier (W6 LOW) MRMR Wave 8 scalars.
    # Defaults verified against MRMR.__init__ (filters/mrmr.py:241,249,252,265).
    relaxmrmr_alpha: float = 0.0,
    uaed_auto_size: bool = False,
    cpt_test: bool = False,
    pid_synergy_bonus: float = 0.0,
    # 2026-05-30 audit-pass-7 #2: MRMR ctor knob (mrmr.py:309). Threads as
    # a top-level kwarg into MRMR.__init__.
    baseline_npermutations: int = 2,
    # 2026-05-30 audit-pass-7 #3/#4: per_feature_edges kwargs forwarded
    # via MRMR.nbins_strategy_kwargs (mrmr.py:225 -> _mrmr_fit_impl:341 ->
    # categorize_dataset:nbins_strategy_kwargs -> per_feature_edges.kwargs).
    # Defaults source-verified at _adaptive_nbins.py:511,586.
    low_card_cap: int = 32,
    collapsed_fallback_nbins: int = 5,
    # 2026-05-31 audit-pass-8 #1/#2: top-level MRMR ctor knobs. Names match
    # MRMR.__init__ exactly. Defaults source-verified at filters/mrmr.py:334
    # (cardinality_bias_correction=True) and filters/mrmr.py:326
    # (min_relevance_gain_relative_to_first=0.05).
    cardinality_bias_correction: bool = True,
    min_relevance_gain_relative_to_first: float = 0.05,
    # 2026-05-31 audit-pass-9 (W9) #7: MRMR fe_hybrid_orth master + pair.
    # Defaults source-verified at filters/mrmr.py:656 (enable=False) and
    # filters/mrmr.py:664 (pair_enable=True, meaningful only when master
    # is on). Names match MRMR.__init__ exactly.
    fe_hybrid_orth_enable: bool = False,
    fe_hybrid_orth_pair_enable: bool = True,
    # 2026-05-31 audit-pass-10 (W10) #2/#3/#4/#6: per-stage hybrid-orth
    # tunables. Defaults source-verified at filters/mrmr.py:657-665. Names
    # match MRMR.__init__ exactly. Meaningful only when
    # fe_hybrid_orth_enable=True (and pair_max_degree only meaningful when
    # both master + pair_enable are on); callers should pass source defaults
    # otherwise so the kwargs dict does not shadow downstream defaults.
    fe_hybrid_orth_degrees: tuple = (2, 3),
    fe_hybrid_orth_basis: str = "auto",
    fe_hybrid_orth_top_k: int = 5,
    fe_hybrid_orth_pair_max_degree: int = 2,
    # 2026-05-31 audit-pass-12 (W12). Group B MRMR FE layer master switches +
    # Group C retain_artifacts. Defaults source-verified at HEAD against
    # MRMR.__init__ (filters/mrmr.py:676/691/705/723/725/727/749/751/752/769/
    # 772/774/777/787). Names match MRMR.__init__ exactly. All default-OFF
    # so callers leaving them at the defaults produce the legacy bit-
    # identical kwargs dict.
    fe_hybrid_orth_extra_bases: tuple = (),
    fe_mi_greedy_enable: bool = False,
    fe_kfold_te_enable: bool = False,
    fe_count_encoding_enable: bool = False,
    fe_frequency_encoding_enable: bool = False,
    fe_cat_num_interaction_enable: bool = False,
    fe_missingness_indicator_enable: bool = False,
    fe_missingness_count_enable: bool = False,
    fe_missingness_pattern_enable: bool = False,
    fe_pairwise_ratio_enable: bool = False,
    fe_pairwise_log_ratio_enable: bool = False,
    fe_grouped_delta_enable: bool = False,
    fe_grouped_delta_group_col: "str | None" = None,
    fe_grouped_delta_num_cols: tuple = (),
    fe_lagged_diff_enable: bool = False,
    fe_lagged_diff_time_col: "str | None" = None,
    fe_lagged_diff_value_cols: tuple = (),
    # 2026-05-31 audit-pass-12 (W12) C1: retain_artifacts at mrmr.py:787.
    # When True the fitted MRMR exposes ``export_artifacts()`` which the
    # downstream ShapProxiedFS consumes via the ``precomputed=`` ctor kwarg.
    retain_artifacts: bool = False,
    # 2026-05-31 audit-pass-14 (W14). Defaults source-verified at HEAD
    # against MRMR.__init__:
    #   F14-3 partial_fit_decay = 0.0
    #         partial_fit_min_recompute = 100
    #         partial_fit_window = None
    #         (filters/mrmr.py:845-847)
    #   F14-4 dcd_tau_cluster = 0.7 (filters/mrmr.py:621; "auto" valid
    #         since Layer 47)
    #   F14-5 dcd_distance = "su" (filters/mrmr.py:622)
    #         dcd_swap_method = "auto" (filters/mrmr.py:655)
    # The partial_fit_* params shape future partial_fit() behaviour but
    # do not affect the legacy fit() byte-identical path; forwarded so
    # the ctor branches receive pairwise enumeration. dcd_* params are
    # consumed only when dcd_enable=True (the MRMR-side gate at
    # mrmr.py:589 / Layer 46/47); forwarded verbatim so the canon-collapse
    # layer at FuzzCombo.canonical_key absorbs phantom variation outside
    # the compound gate.
    partial_fit_decay: float = 0.0,
    partial_fit_min_recompute: int = 100,
    partial_fit_window: "int | None" = None,
    dcd_tau_cluster: "float | str" = 0.7,
    dcd_distance: str = "su",
    dcd_swap_method: str = "auto",
    # iter639 audit-pass-15. Layers 62/63/76/85 hybrid-orth scorer family.
    # Defaults source-verified at HEAD against MRMR.__init__
    # (filters/mrmr.py:878 / :897 / :1068 / :1092). All gated downstream of
    # fe_hybrid_orth_enable; canon at FuzzCombo.canonical_key absorbs phantom
    # variation outside the compound gate.
    fe_hybrid_orth_default_scorer: str = "plug_in",
    fe_hybrid_orth_meta_enable: bool = False,
    fe_hybrid_orth_bootstrap_enable: bool = False,
    fe_hybrid_orth_three_gate_enable: bool = False,
    # iter642 audit-pass-15 batch 2. Names match MRMR.__init__ exactly
    # (filters/mrmr.py:1044/784/800/749/845/767).
    fe_hybrid_orth_ensemble_enable: bool = False,
    fe_hybrid_orth_lasso_enable: bool = False,
    fe_hybrid_orth_elasticnet_enable: bool = False,
    fe_hybrid_orth_adaptive_arity_enable: bool = False,
    fe_hybrid_orth_diff_basis_enable: bool = False,
    fe_semi_supervised_enable: bool = False,
    # audit-pass-16 — MRMR Layers 87-91. Defaults match MRMR.__init__ exactly
    # (filters/mrmr.py:1255/1268/1270/1285/1300/1302/1243/1245). All gated
    # downstream of use_mrmr_fs; canon at FuzzCombo.canonical_key absorbs
    # phantom variation outside the gate.
    fe_grouped_agg_enable: bool = False,
    fe_grouped_quantile_enable: bool = False,
    fe_grouped_quantile_target_aware: bool = False,
    fe_cat_pair_enable: bool = False,
    fe_numeric_decompose_enable: bool = False,
    fe_numeric_decompose_digits: tuple = (0, 1, 2),
    fe_local_mi_gate: bool = True,  # audit-pass-17: source default flipped True (L97)
    fe_unified_second_pass_gate: bool = False,
    # audit-pass-17 — Param-Oracle / fe_auto + FE families L92-104. Defaults
    # match MRMR.__init__ (mrmr.py:1478/1426/1296/1375/1387/1403/1407).
    fe_auto: bool = False,
    fe_temporal_agg_enable: bool = False,
    fe_composite_group_agg_enable: bool = False,
    fe_modular_enable: bool = False,
    fe_group_distance_enable: bool = False,
    fe_rare_category_enable: bool = False,
    fe_conditional_residual_enable: bool = False,
    # 2026-06-13 coverage refresh. Names match MRMR.__init__ exactly
    # (mrmr/_mrmr_class.py:2404-2406 / 1306 / 2282 / 2307 / 454 / 512 / 1537).
    embedding_passthrough: bool = True,
    embedding_passthrough_detect_embeddings: bool = True,
    embedding_passthrough_detect_text: bool = True,
    fe_hinge_enable: bool = True,
    fe_conditional_dispersion_enable: bool = True,
    fe_wavelet_enable: bool = True,
    fe_stability_vote_enable: bool = True,
    fe_sufficient_summary_early_stop: bool = True,
    fe_gradient_interaction_enable: bool = False,
    # MRMR FE-family + escalation + hybrid-orth scorer master toggles. Names match MRMR.__init__ verbatim; defaults mirror the source signature.
    fe_rung_schedule_enable: bool = True,
    fe_auto_escalation_enable: bool = True,
    fe_escalation_underdelivery_enable: bool = True,
    fe_synergy_prevalence_rescue_enable: bool = True,
    fe_pair_prewarp_enable: bool = True,
    fe_univariate_basis_enable: bool = True,
    fe_univariate_fourier_enable: bool = True,
    fe_hybrid_orth_triplet_enable: bool = True,
    fe_hybrid_orth_quadruplet_enable: bool = True,
    fe_binned_numeric_agg_enable: bool = True,
    fe_discrete_structural_operators_enable: bool = True,
    fe_pairwise_modular_enable: bool = True,
    fe_integer_lattice_enable: bool = True,
    fe_row_argmax_enable: bool = True,
    fe_conditional_gate_enable: bool = True,
    fe_escalation_feedforward_enable: bool = False,
    fe_gate_med_enable: bool = False,
    fe_pair_perm_null_admission_enable: bool = False,
    fe_ii_routing_enable: bool = False,
    fe_gbm_seeder_enable: bool = False,
    fe_hybrid_orth_adaptive_degree_enable: bool = False,
    fe_hybrid_orth_conditional_routing_enable: bool = False,
    fe_hybrid_orth_cluster_basis_enable: bool = False,
    fe_hybrid_orth_ksg_enable: bool = False,
    fe_hybrid_orth_copula_enable: bool = False,
    fe_hybrid_orth_dcor_enable: bool = False,
    fe_hybrid_orth_hsic_enable: bool = False,
    fe_hybrid_orth_jmim_enable: bool = False,
    fe_hybrid_orth_tc_enable: bool = False,
    fe_hybrid_orth_cmim_enable: bool = False,
    fe_hybrid_orth_auto_scorer_enable: bool = False,
    fe_mi_greedy_cmi_enable: bool = False,
    fe_cat_triple_enable: bool = False,
    fe_rankgauss_enable: bool = False,
) -> Optional[Dict[str, Any]]:
    """Build the mrmr_kwargs dict passed to FeatureSelectionConfig.
    Returns None when use_mrmr_fs=False so the FS step is a no-op.

    Single-edit point: every MRMR knob (existing iter-32.5 axes + any
    future axis) flows through these named params. Both
    test_fuzz_suite.py and _profile_fuzz_1m.py call this so adding a
    new MRMR axis only touches this function (plus the AXES dict +
    FuzzCombo dataclass for the pytest fuzz space).
    """
    if not use_mrmr_fs:
        return None
    kwargs: Dict[str, Any] = {
        "verbose": verbose,
        "max_runtime_mins": max_runtime_mins,
        "n_workers": n_workers,
        "quantization_nbins": quantization_nbins,
        "use_simple_mode": use_simple_mode,
        "min_nonzero_confidence": min_nonzero_confidence,
        "max_consec_unconfirmed": max_consec_unconfirmed,
        "full_npermutations": full_npermutations,
        "interactions_max_order": interactions_max_order,
        "fe_max_steps": fe_max_steps,
        "fe_npermutations": fe_npermutations,
        "fe_ntop_features": fe_ntop_features,
        "fe_unary_preset": fe_unary_preset,
        "fe_binary_preset": fe_binary_preset,
        "fe_smart_polynom_iters": fe_smart_polynom_iters,
        "fe_smart_polynom_optimization_steps": fe_smart_polynom_optimization_steps,
        "fe_min_polynom_degree": fe_min_polynom_degree,
        "fe_max_polynom_degree": fe_max_polynom_degree,
        # Friend-graph + cluster-aggregate knobs flow straight into the MRMR
        # constructor (same names). Defaults mirror mrmr.py so a combo that
        # leaves them at the default produces the same MRMR behaviour as
        # before these axes existed.
        "build_friend_graph": build_friend_graph,
        "friend_graph_prune": friend_graph_prune,
        "cluster_aggregate_enable": cluster_aggregate_enable,
        "cluster_aggregate_mode": cluster_aggregate_mode,
        # 2026-05-30 audit-pass-6 Wave 7/8/9 ctor knobs. Names match
        # MRMR.__init__ exactly (filters/mrmr.py:224-302, 589).
        "nbins_strategy": nbins_strategy,
        "mi_correction": mi_correction,
        "redundancy_aggregator": redundancy_aggregator,
        "bur_lambda": bur_lambda,
        "cmi_perm_stop": cmi_perm_stop,
        "stability_selection_method": stability_selection_method,
        "mi_normalization": mi_normalization,
        "dcd_enable": dcd_enable,
        # 2026-05-30 audit-pass-6 LOW-tier (W6 LOW) MRMR Wave 8 scalars.
        # Names match MRMR.__init__ exactly (filters/mrmr.py:241,249,252,265).
        "relaxmrmr_alpha": relaxmrmr_alpha,
        "uaed_auto_size": uaed_auto_size,
        "cpt_test": cpt_test,
        "pid_synergy_bonus": pid_synergy_bonus,
        # 2026-05-30 audit-pass-7 #2: top-level MRMR ctor knob (mrmr.py:309).
        "baseline_npermutations": baseline_npermutations,
        # 2026-05-31 audit-pass-8 #1/#2: top-level MRMR ctor knobs. Names
        # match MRMR.__init__ exactly (filters/mrmr.py:334, :326).
        "cardinality_bias_correction": cardinality_bias_correction,
        "min_relevance_gain_relative_to_first": min_relevance_gain_relative_to_first,
        # 2026-05-31 audit-pass-9 (W9) #7: fe_hybrid_orth master + pair.
        # Names match MRMR.__init__ exactly (filters/mrmr.py:656, :664).
        "fe_hybrid_orth_enable": fe_hybrid_orth_enable,
        "fe_hybrid_orth_pair_enable": fe_hybrid_orth_pair_enable,
        # 2026-05-31 audit-pass-10 (W10) #2/#3/#4/#6: per-stage hybrid-orth
        # tunables forwarded verbatim. Names match MRMR.__init__ exactly
        # (filters/mrmr.py:657, :658, :663, :665). The canon-collapse layer
        # at FuzzCombo.canonical_key absorbs phantom variation outside the
        # compound gates, so we forward the raw axis values here -- inside
        # the gate the hybrid pipeline is the only code path that reads
        # them, outside the gate they are unread.
        "fe_hybrid_orth_degrees": fe_hybrid_orth_degrees,
        "fe_hybrid_orth_basis": fe_hybrid_orth_basis,
        "fe_hybrid_orth_top_k": fe_hybrid_orth_top_k,
        "fe_hybrid_orth_pair_max_degree": fe_hybrid_orth_pair_max_degree,
        # 2026-05-31 audit-pass-12 (W12). Group B MRMR FE layer master
        # switches + Group C retain_artifacts. Names match MRMR.__init__
        # verbatim (filters/mrmr.py:676/691/705/723/725/727/749/751/752/
        # 769/772/774/777/787). The canon-collapse layer at
        # FuzzCombo.canonical_key absorbs phantom variation outside each
        # axis's documented gate, so we forward the raw axis values --
        # gates inside MRMR.fit gate the actual FE-stage execution on
        # frame contents independently.
        "fe_hybrid_orth_extra_bases": fe_hybrid_orth_extra_bases,
        "fe_mi_greedy_enable": fe_mi_greedy_enable,
        "fe_kfold_te_enable": fe_kfold_te_enable,
        "fe_count_encoding_enable": fe_count_encoding_enable,
        "fe_frequency_encoding_enable": fe_frequency_encoding_enable,
        "fe_cat_num_interaction_enable": fe_cat_num_interaction_enable,
        "fe_missingness_indicator_enable": fe_missingness_indicator_enable,
        "fe_missingness_count_enable": fe_missingness_count_enable,
        "fe_missingness_pattern_enable": fe_missingness_pattern_enable,
        "fe_pairwise_ratio_enable": fe_pairwise_ratio_enable,
        "fe_pairwise_log_ratio_enable": fe_pairwise_log_ratio_enable,
        "fe_grouped_delta_enable": fe_grouped_delta_enable,
        "fe_grouped_delta_group_col": fe_grouped_delta_group_col,
        "fe_grouped_delta_num_cols": fe_grouped_delta_num_cols,
        "fe_lagged_diff_enable": fe_lagged_diff_enable,
        "fe_lagged_diff_time_col": fe_lagged_diff_time_col,
        "fe_lagged_diff_value_cols": fe_lagged_diff_value_cols,
        "retain_artifacts": retain_artifacts,
        # 2026-05-31 audit-pass-14 (W14). Param names match MRMR.__init__
        # exactly (filters/mrmr.py:845-847 / :621 / :622 / :655).
        "partial_fit_decay": partial_fit_decay,
        "partial_fit_min_recompute": partial_fit_min_recompute,
        "partial_fit_window": partial_fit_window,
        "dcd_tau_cluster": dcd_tau_cluster,
        "dcd_distance": dcd_distance,
        "dcd_swap_method": dcd_swap_method,
        # iter639 audit-pass-15. Names match MRMR.__init__ exactly
        # (filters/mrmr.py:878 / :897 / :1068 / :1092). All four are
        # meaningful only inside the fe_hybrid_orth_enable=True gate;
        # canon at FuzzCombo.canonical_key absorbs phantom variation
        # outside the gate so dedup keeps the surviving combos minimal.
        "fe_hybrid_orth_default_scorer": fe_hybrid_orth_default_scorer,
        "fe_hybrid_orth_meta_enable": fe_hybrid_orth_meta_enable,
        "fe_hybrid_orth_bootstrap_enable": fe_hybrid_orth_bootstrap_enable,
        "fe_hybrid_orth_three_gate_enable": fe_hybrid_orth_three_gate_enable,
        # iter642 audit-pass-15 batch 2. Six remaining hybrid-orth sub-
        # features. Names match MRMR.__init__ verbatim.
        "fe_hybrid_orth_ensemble_enable": fe_hybrid_orth_ensemble_enable,
        "fe_hybrid_orth_lasso_enable": fe_hybrid_orth_lasso_enable,
        "fe_hybrid_orth_elasticnet_enable": fe_hybrid_orth_elasticnet_enable,
        "fe_hybrid_orth_adaptive_arity_enable": fe_hybrid_orth_adaptive_arity_enable,
        "fe_hybrid_orth_diff_basis_enable": fe_hybrid_orth_diff_basis_enable,
        "fe_semi_supervised_enable": fe_semi_supervised_enable,
        # audit-pass-16 — MRMR Layers 87-91. Names match MRMR.__init__ exactly.
        "fe_grouped_agg_enable": fe_grouped_agg_enable,
        "fe_grouped_quantile_enable": fe_grouped_quantile_enable,
        "fe_grouped_quantile_target_aware": fe_grouped_quantile_target_aware,
        "fe_cat_pair_enable": fe_cat_pair_enable,
        "fe_numeric_decompose_enable": fe_numeric_decompose_enable,
        "fe_numeric_decompose_digits": fe_numeric_decompose_digits,
        "fe_local_mi_gate": fe_local_mi_gate,
        "fe_unified_second_pass_gate": fe_unified_second_pass_gate,
        # audit-pass-17 — Param-Oracle / fe_auto + FE families L92-104.
        "fe_auto": fe_auto,
        "fe_temporal_agg_enable": fe_temporal_agg_enable,
        "fe_composite_group_agg_enable": fe_composite_group_agg_enable,
        "fe_modular_enable": fe_modular_enable,
        "fe_group_distance_enable": fe_group_distance_enable,
        "fe_rare_category_enable": fe_rare_category_enable,
        "fe_conditional_residual_enable": fe_conditional_residual_enable,
        # 2026-06-13 coverage refresh. Names match MRMR.__init__ verbatim.
        "embedding_passthrough": embedding_passthrough,
        "embedding_passthrough_detect_embeddings": embedding_passthrough_detect_embeddings,
        "embedding_passthrough_detect_text": embedding_passthrough_detect_text,
        "fe_hinge_enable": fe_hinge_enable,
        "fe_conditional_dispersion_enable": fe_conditional_dispersion_enable,
        "fe_wavelet_enable": fe_wavelet_enable,
        "fe_stability_vote_enable": fe_stability_vote_enable,
        "fe_sufficient_summary_early_stop": fe_sufficient_summary_early_stop,
        "fe_gradient_interaction_enable": fe_gradient_interaction_enable,
        # MRMR FE-family + escalation + hybrid-orth scorer master toggles. Names match MRMR.__init__ verbatim.
        "fe_rung_schedule_enable": fe_rung_schedule_enable,
        "fe_auto_escalation_enable": fe_auto_escalation_enable,
        "fe_escalation_underdelivery_enable": fe_escalation_underdelivery_enable,
        "fe_synergy_prevalence_rescue_enable": fe_synergy_prevalence_rescue_enable,
        "fe_pair_prewarp_enable": fe_pair_prewarp_enable,
        "fe_univariate_basis_enable": fe_univariate_basis_enable,
        "fe_univariate_fourier_enable": fe_univariate_fourier_enable,
        "fe_hybrid_orth_triplet_enable": fe_hybrid_orth_triplet_enable,
        "fe_hybrid_orth_quadruplet_enable": fe_hybrid_orth_quadruplet_enable,
        "fe_binned_numeric_agg_enable": fe_binned_numeric_agg_enable,
        "fe_discrete_structural_operators_enable": fe_discrete_structural_operators_enable,
        "fe_pairwise_modular_enable": fe_pairwise_modular_enable,
        "fe_integer_lattice_enable": fe_integer_lattice_enable,
        "fe_row_argmax_enable": fe_row_argmax_enable,
        "fe_conditional_gate_enable": fe_conditional_gate_enable,
        "fe_escalation_feedforward_enable": fe_escalation_feedforward_enable,
        "fe_gate_med_enable": fe_gate_med_enable,
        "fe_pair_perm_null_admission_enable": fe_pair_perm_null_admission_enable,
        "fe_ii_routing_enable": fe_ii_routing_enable,
        "fe_gbm_seeder_enable": fe_gbm_seeder_enable,
        "fe_hybrid_orth_adaptive_degree_enable": fe_hybrid_orth_adaptive_degree_enable,
        "fe_hybrid_orth_conditional_routing_enable": fe_hybrid_orth_conditional_routing_enable,
        "fe_hybrid_orth_cluster_basis_enable": fe_hybrid_orth_cluster_basis_enable,
        "fe_hybrid_orth_ksg_enable": fe_hybrid_orth_ksg_enable,
        "fe_hybrid_orth_copula_enable": fe_hybrid_orth_copula_enable,
        "fe_hybrid_orth_dcor_enable": fe_hybrid_orth_dcor_enable,
        "fe_hybrid_orth_hsic_enable": fe_hybrid_orth_hsic_enable,
        "fe_hybrid_orth_jmim_enable": fe_hybrid_orth_jmim_enable,
        "fe_hybrid_orth_tc_enable": fe_hybrid_orth_tc_enable,
        "fe_hybrid_orth_cmim_enable": fe_hybrid_orth_cmim_enable,
        "fe_hybrid_orth_auto_scorer_enable": fe_hybrid_orth_auto_scorer_enable,
        "fe_mi_greedy_cmi_enable": fe_mi_greedy_cmi_enable,
        "fe_cat_triple_enable": fe_cat_triple_enable,
        "fe_rankgauss_enable": fe_rankgauss_enable,
    }
    # 2026-05-30 audit-pass-7 #3/#4: per_feature_edges.kwargs threaded via
    # MRMR.nbins_strategy_kwargs. Build the dict only when one of these
    # knobs differs from the source default so we don't shadow any existing
    # caller-supplied dict with empty overrides.
    _nbins_kw: Dict[str, Any] = {}
    if low_card_cap != 32:
        _nbins_kw["low_card_cap"] = low_card_cap
    if collapsed_fallback_nbins != 5:
        _nbins_kw["collapsed_fallback_nbins"] = collapsed_fallback_nbins
    if _nbins_kw:
        kwargs["nbins_strategy_kwargs"] = _nbins_kw
    # The MRMR subsample knobs default to UNIFIED_FE_SUBSAMPLE_N upstream; only
    # override when the fuzz axis sets a non-zero budget so existing combos
    # don't accidentally flip the path on.
    if fe_check_pairs_subsample_n > 0:
        kwargs["fe_check_pairs_subsample_n"] = fe_check_pairs_subsample_n
    if fe_smart_polynom_subsample_n > 0:
        kwargs["fe_smart_polynom_subsample_n"] = fe_smart_polynom_subsample_n
    if cat_fe_config is not None:
        kwargs["cat_fe_config"] = cat_fe_config
    return kwargs


def build_mrmr_kwargs(combo: "FuzzCombo") -> Optional[Dict[str, Any]]:
    """FuzzCombo-aware wrapper around build_mrmr_kwargs_from_flat."""
    from .frame_builder import (
        MRMR_FE_GROUP_COL,
        MRMR_FE_GROUP_VAL_COL,
        MRMR_FE_ORDER_COL,
        MRMR_FE_LAG_VAL_COL,
    )

    cat_fe = build_cat_fe_config_from_flat(
        use_mrmr_fs=combo.use_mrmr_fs,
        cat_fe_enable=combo.mrmr_cat_fe_enable_cfg,
        cat_fe_include_numeric=combo.mrmr_cat_fe_include_numeric_cfg,
    )
    # FE subsample only meaningful when an FE entry point actually runs and
    # n_rows exceeds the budget. Couples the new fe_check_pairs_subsample_n_cfg
    # axis to both fe_npermutations / fe_ntop_features (any > 0 fires the FE
    # block) and the smart-polynom subsample (shares the same budget by
    # UNIFIED_FE_SUBSAMPLE_N upstream).
    _subsample_active = (
        combo.use_mrmr_fs
        and combo.fe_check_pairs_subsample_n_cfg > 0
        and combo.n_rows > combo.fe_check_pairs_subsample_n_cfg
        and (combo.mrmr_fe_npermutations_cfg > 0 or combo.mrmr_fe_ntop_features_cfg > 0)
    )
    return build_mrmr_kwargs_from_flat(
        use_mrmr_fs=combo.use_mrmr_fs,
        interactions_max_order=combo.mrmr_interactions_max_order_cfg,
        fe_max_steps=combo.mrmr_fe_max_steps_cfg,
        cat_fe_config=cat_fe,
        fe_npermutations=combo.mrmr_fe_npermutations_cfg,
        fe_ntop_features=combo.mrmr_fe_ntop_features_cfg,
        fe_unary_preset=combo.mrmr_fe_unary_preset_cfg,
        fe_binary_preset=combo.mrmr_fe_binary_preset_cfg,
        fe_smart_polynom_iters=combo.mrmr_fe_smart_polynom_iters_cfg,
        fe_smart_polynom_optimization_steps=combo.mrmr_fe_smart_polynom_steps_cfg,
        fe_min_polynom_degree=combo.mrmr_fe_min_polynom_degree_cfg,
        fe_max_polynom_degree=combo.mrmr_fe_max_polynom_degree_cfg,
        fe_check_pairs_subsample_n=(combo.fe_check_pairs_subsample_n_cfg if _subsample_active else 0),
        fe_smart_polynom_subsample_n=(combo.fe_check_pairs_subsample_n_cfg if _subsample_active else 0),
        build_friend_graph=combo.mrmr_build_friend_graph_cfg,
        friend_graph_prune=combo.mrmr_friend_graph_prune_cfg,
        cluster_aggregate_enable=combo.mrmr_cluster_aggregate_enable_cfg,
        cluster_aggregate_mode=combo.mrmr_cluster_aggregate_mode_cfg,
        # 2026-05-30 audit-pass-6 Wave 7/8/9 ctor knobs.
        nbins_strategy=combo.mrmr_nbins_strategy_cfg,
        mi_correction=combo.mrmr_mi_correction_cfg,
        redundancy_aggregator=combo.mrmr_redundancy_aggregator_cfg,
        bur_lambda=combo.mrmr_bur_lambda_cfg,
        cmi_perm_stop=combo.mrmr_cmi_perm_stop_cfg,
        stability_selection_method=combo.mrmr_stability_selection_method_cfg,
        mi_normalization=combo.mrmr_mi_normalization_cfg,
        dcd_enable=combo.mrmr_dcd_enable_cfg,
        # 2026-05-30 audit-pass-6 LOW-tier (W6 LOW) MRMR Wave 8 scalars.
        relaxmrmr_alpha=combo.mrmr_relaxmrmr_alpha_cfg,
        uaed_auto_size=combo.mrmr_uaed_auto_size_cfg,
        cpt_test=combo.mrmr_cpt_test_cfg,
        pid_synergy_bonus=combo.mrmr_pid_synergy_bonus_cfg,
        # 2026-05-30 audit-pass-7 #2/#3/#4.
        baseline_npermutations=combo.mrmr_baseline_npermutations_cfg,
        low_card_cap=combo.mrmr_low_card_cap_cfg,
        collapsed_fallback_nbins=combo.mrmr_collapsed_fallback_nbins_cfg,
        # 2026-05-31 audit-pass-8 #1/#2.
        cardinality_bias_correction=combo.mrmr_cardinality_bias_correction_cfg,
        min_relevance_gain_relative_to_first=combo.mrmr_min_relevance_gain_relative_to_first_cfg,
        # 2026-05-31 audit-pass-9 (W9) #7: MRMR fe_hybrid_orth master + pair.
        # The canon-collapse layer above already drops these to source defaults
        # when use_mrmr_fs=False (build_mrmr_kwargs returns None for those
        # combos) or when the master is off (pair_enable collapses to default
        # True). Forward the raw axis values so MRMR-on combos exercise the
        # both branches reachable via the pairwise sampler.
        fe_hybrid_orth_enable=combo.mrmr_fe_hybrid_orth_enable_cfg,
        fe_hybrid_orth_pair_enable=combo.mrmr_fe_hybrid_orth_pair_enable_cfg,
        # 2026-05-31 audit-pass-10 (W10) #2/#3/#4/#6: per-stage hybrid-orth
        # tunables. Canon-collapse at FuzzCombo.canonical_key reduces all
        # four to source defaults outside the compound gate; the builder
        # forwards the raw axis values so MRMR-on + master-on combos
        # exercise the both branches reachable via the pairwise sampler.
        fe_hybrid_orth_degrees=combo.mrmr_fe_hybrid_orth_degrees_cfg,
        fe_hybrid_orth_basis=combo.mrmr_fe_hybrid_orth_basis_cfg,
        fe_hybrid_orth_top_k=combo.mrmr_fe_hybrid_orth_top_k_cfg,
        fe_hybrid_orth_pair_max_degree=combo.mrmr_fe_hybrid_orth_pair_max_degree_cfg,
        # 2026-05-31 audit-pass-12 (W12). Map the FuzzCombo axes into the
        # MRMR ctor kwarg names. Group B 4-way axes (cat_aux + ratio_delta_diff)
        # expand into the three / four master switches each.
        fe_hybrid_orth_extra_bases=combo.mrmr_fe_hybrid_orth_extra_bases_cfg,
        fe_mi_greedy_enable=combo.mrmr_fe_mi_greedy_enable_cfg,
        fe_kfold_te_enable=combo.mrmr_fe_kfold_te_enable_cfg,
        # B3: 4-way mrmr_fe_cat_aux_enable_cfg -> 3 master switches.
        fe_count_encoding_enable=(combo.mrmr_fe_cat_aux_enable_cfg == "count"),
        fe_frequency_encoding_enable=(combo.mrmr_fe_cat_aux_enable_cfg == "freq"),
        fe_cat_num_interaction_enable=(combo.mrmr_fe_cat_aux_enable_cfg == "interaction"),
        # B2: 3 sub-axes already 1:1 mapped to the master switches.
        fe_missingness_indicator_enable=combo.mrmr_fe_missingness_indicator_enable_cfg,
        fe_missingness_count_enable=combo.mrmr_fe_missingness_count_enable_cfg,
        fe_missingness_pattern_enable=combo.mrmr_fe_missingness_pattern_enable_cfg,
        # B5: 4-way mrmr_fe_ratio_delta_diff_cfg -> 4 master switches. The frame builder emits a group key (grouped_delta) / order column
        # (lagged_diff) for the matching kind, so all three non-off branches actually run -- the group_col / time_col + their numeric source
        # columns are wired through here so the MRMR FE entry point has the inputs it needs (without them prod no-ops the kind).
        fe_pairwise_ratio_enable=(combo.mrmr_fe_ratio_delta_diff_cfg == "ratio"),
        fe_pairwise_log_ratio_enable=False,  # log_ratio variant deferred (axis covers raw ratio only)
        fe_grouped_delta_enable=(combo.mrmr_fe_ratio_delta_diff_cfg == "grouped_delta"),
        fe_grouped_delta_group_col=(MRMR_FE_GROUP_COL if combo.mrmr_fe_ratio_delta_diff_cfg == "grouped_delta" else None),
        fe_grouped_delta_num_cols=((MRMR_FE_GROUP_VAL_COL,) if combo.mrmr_fe_ratio_delta_diff_cfg == "grouped_delta" else ()),
        fe_lagged_diff_enable=(combo.mrmr_fe_ratio_delta_diff_cfg == "lagged_diff"),
        fe_lagged_diff_time_col=(MRMR_FE_ORDER_COL if combo.mrmr_fe_ratio_delta_diff_cfg == "lagged_diff" else None),
        fe_lagged_diff_value_cols=((MRMR_FE_LAG_VAL_COL,) if combo.mrmr_fe_ratio_delta_diff_cfg == "lagged_diff" else ()),
        # C1: retain_artifacts ON when the artifact-reuse master is on AND
        # both selectors are in the chain. The canonical_key collapse layer
        # already pins the axis to "off" outside the compound gate, but the
        # build_mrmr_kwargs path is reached only when use_mrmr_fs=True so
        # we honour the axis value verbatim here.
        retain_artifacts=(combo.mrmr_shap_proxy_artifact_reuse_cfg == "on" and combo.use_shap_proxied_fs),
        # 2026-05-31 audit-pass-14 (W14). Forward partial_fit + dcd_* axes
        # verbatim; canon-collapse at FuzzCombo.canonical_key reduces them
        # to source defaults outside their compound gates.
        partial_fit_decay=combo.mrmr_partial_fit_decay_cfg,
        partial_fit_min_recompute=combo.mrmr_partial_fit_min_recompute_cfg,
        partial_fit_window=combo.mrmr_partial_fit_window_cfg,
        dcd_tau_cluster=combo.mrmr_dcd_tau_cluster_cfg,
        dcd_distance=combo.mrmr_dcd_distance_cfg,
        dcd_swap_method=combo.mrmr_dcd_swap_method_cfg,
        # iter639 audit-pass-15. Forward hybrid-orth scorer family axes.
        fe_hybrid_orth_default_scorer=combo.mrmr_fe_hybrid_orth_default_scorer_cfg,
        fe_hybrid_orth_meta_enable=combo.mrmr_fe_hybrid_orth_meta_enable_cfg,
        fe_hybrid_orth_bootstrap_enable=combo.mrmr_fe_hybrid_orth_bootstrap_enable_cfg,
        fe_hybrid_orth_three_gate_enable=combo.mrmr_fe_hybrid_orth_three_gate_enable_cfg,
        # iter642 audit-pass-15 batch 2.
        fe_hybrid_orth_ensemble_enable=combo.mrmr_fe_hybrid_orth_ensemble_enable_cfg,
        fe_hybrid_orth_lasso_enable=combo.mrmr_fe_hybrid_orth_lasso_enable_cfg,
        fe_hybrid_orth_elasticnet_enable=combo.mrmr_fe_hybrid_orth_elasticnet_enable_cfg,
        fe_hybrid_orth_adaptive_arity_enable=combo.mrmr_fe_hybrid_orth_adaptive_arity_enable_cfg,
        fe_hybrid_orth_diff_basis_enable=combo.mrmr_fe_hybrid_orth_diff_basis_enable_cfg,
        fe_semi_supervised_enable=combo.mrmr_fe_semi_supervised_enable_cfg,
        # audit-pass-16 — MRMR Layers 87-91.
        fe_grouped_agg_enable=combo.mrmr_fe_grouped_agg_enable_cfg,
        fe_grouped_quantile_enable=combo.mrmr_fe_grouped_quantile_enable_cfg,
        fe_grouped_quantile_target_aware=combo.mrmr_fe_grouped_quantile_target_aware_cfg,
        fe_cat_pair_enable=combo.mrmr_fe_cat_pair_enable_cfg,
        fe_numeric_decompose_enable=combo.mrmr_fe_numeric_decompose_enable_cfg,
        fe_numeric_decompose_digits=combo.mrmr_fe_numeric_decompose_digits_cfg,
        fe_local_mi_gate=combo.mrmr_fe_local_mi_gate_cfg,
        fe_unified_second_pass_gate=combo.mrmr_fe_unified_second_pass_gate_cfg,
        # audit-pass-17 — Param-Oracle / fe_auto + FE families L92-104.
        fe_auto=combo.mrmr_fe_auto_cfg,
        fe_temporal_agg_enable=combo.mrmr_fe_temporal_agg_enable_cfg,
        fe_composite_group_agg_enable=combo.mrmr_fe_composite_group_agg_enable_cfg,
        fe_modular_enable=combo.mrmr_fe_modular_enable_cfg,
        fe_group_distance_enable=combo.mrmr_fe_group_distance_enable_cfg,
        fe_rare_category_enable=combo.mrmr_fe_rare_category_enable_cfg,
        fe_conditional_residual_enable=combo.mrmr_fe_conditional_residual_enable_cfg,
        # 2026-06-13 coverage refresh. Forward verbatim; canon-collapse at
        # FuzzCombo.canonical_key reduces each to its source default outside its
        # documented gate (build_mrmr_kwargs returns None when use_mrmr_fs=False).
        embedding_passthrough=combo.mrmr_embedding_passthrough_cfg,
        embedding_passthrough_detect_embeddings=combo.mrmr_embedding_passthrough_detect_embeddings_cfg,
        embedding_passthrough_detect_text=combo.mrmr_embedding_passthrough_detect_text_cfg,
        fe_hinge_enable=combo.mrmr_fe_hinge_enable_cfg,
        fe_conditional_dispersion_enable=combo.mrmr_fe_conditional_dispersion_enable_cfg,
        fe_wavelet_enable=combo.mrmr_fe_wavelet_enable_cfg,
        fe_stability_vote_enable=combo.mrmr_fe_stability_vote_enable_cfg,
        fe_sufficient_summary_early_stop=combo.mrmr_fe_sufficient_summary_early_stop_cfg,
        fe_gradient_interaction_enable=combo.mrmr_fe_gradient_interaction_enable_cfg,
        # MRMR FE-family + escalation + hybrid-orth scorer master toggles.
        fe_rung_schedule_enable=combo.mrmr_fe_rung_schedule_enable_cfg,
        fe_auto_escalation_enable=combo.mrmr_fe_auto_escalation_enable_cfg,
        fe_escalation_underdelivery_enable=combo.mrmr_fe_escalation_underdelivery_enable_cfg,
        fe_synergy_prevalence_rescue_enable=combo.mrmr_fe_synergy_prevalence_rescue_enable_cfg,
        fe_pair_prewarp_enable=combo.mrmr_fe_pair_prewarp_enable_cfg,
        fe_univariate_basis_enable=combo.mrmr_fe_univariate_basis_enable_cfg,
        fe_univariate_fourier_enable=combo.mrmr_fe_univariate_fourier_enable_cfg,
        fe_hybrid_orth_triplet_enable=combo.mrmr_fe_hybrid_orth_triplet_enable_cfg,
        fe_hybrid_orth_quadruplet_enable=combo.mrmr_fe_hybrid_orth_quadruplet_enable_cfg,
        fe_binned_numeric_agg_enable=combo.mrmr_fe_binned_numeric_agg_enable_cfg,
        fe_discrete_structural_operators_enable=combo.mrmr_fe_discrete_structural_operators_enable_cfg,
        fe_pairwise_modular_enable=combo.mrmr_fe_pairwise_modular_enable_cfg,
        fe_integer_lattice_enable=combo.mrmr_fe_integer_lattice_enable_cfg,
        fe_row_argmax_enable=combo.mrmr_fe_row_argmax_enable_cfg,
        fe_conditional_gate_enable=combo.mrmr_fe_conditional_gate_enable_cfg,
        fe_escalation_feedforward_enable=combo.mrmr_fe_escalation_feedforward_enable_cfg,
        fe_gate_med_enable=combo.mrmr_fe_gate_med_enable_cfg,
        fe_pair_perm_null_admission_enable=combo.mrmr_fe_pair_perm_null_admission_enable_cfg,
        fe_ii_routing_enable=combo.mrmr_fe_ii_routing_enable_cfg,
        fe_gbm_seeder_enable=combo.mrmr_fe_gbm_seeder_enable_cfg,
        fe_hybrid_orth_adaptive_degree_enable=combo.mrmr_fe_hybrid_orth_adaptive_degree_enable_cfg,
        fe_hybrid_orth_conditional_routing_enable=combo.mrmr_fe_hybrid_orth_conditional_routing_enable_cfg,
        fe_hybrid_orth_cluster_basis_enable=combo.mrmr_fe_hybrid_orth_cluster_basis_enable_cfg,
        fe_hybrid_orth_ksg_enable=combo.mrmr_fe_hybrid_orth_ksg_enable_cfg,
        fe_hybrid_orth_copula_enable=combo.mrmr_fe_hybrid_orth_copula_enable_cfg,
        fe_hybrid_orth_dcor_enable=combo.mrmr_fe_hybrid_orth_dcor_enable_cfg,
        fe_hybrid_orth_hsic_enable=combo.mrmr_fe_hybrid_orth_hsic_enable_cfg,
        fe_hybrid_orth_jmim_enable=combo.mrmr_fe_hybrid_orth_jmim_enable_cfg,
        fe_hybrid_orth_tc_enable=combo.mrmr_fe_hybrid_orth_tc_enable_cfg,
        fe_hybrid_orth_cmim_enable=combo.mrmr_fe_hybrid_orth_cmim_enable_cfg,
        fe_hybrid_orth_auto_scorer_enable=combo.mrmr_fe_hybrid_orth_auto_scorer_enable_cfg,
        fe_mi_greedy_cmi_enable=combo.mrmr_fe_mi_greedy_cmi_enable_cfg,
        fe_cat_triple_enable=combo.mrmr_fe_cat_triple_enable_cfg,
        fe_rankgauss_enable=combo.mrmr_fe_rankgauss_enable_cfg,
    )


def build_mlp_kwargs_from_flat(
    *,
    models: tuple[str, ...],
    target_type: str,
    imbalance_ratio: str,
    recurrent_model: "str | None" = None,
    # 2026-05-31 audit-pass-8 #3: PytorchLightningEstimator random_state.
    # Source default None (training/neural/base.py:217).
    random_state: "int | None" = None,
    # 2026-05-31 audit-pass-8 #4: PytorchLightningClassifier class_weight.
    # Source default None (training/neural/base.py:218).
    class_weight: "str | None" = None,
    # 2026-05-31 audit-pass-8 #7: generate_mlp use_layernorm. Source default
    # False (training/neural/flat.py:205; audit-cited :145 was a docstring
    # line, the real signature default lives at :205). Threaded as an
    # MLP-network-builder hparam, NOT a PytorchLightningEstimator __init__
    # arg -- the suite forwards generate_mlp kwargs via hyperparams_config.
    use_layernorm: bool = False,
    # 2026-05-31 audit-pass-8 #8: MLPTorchModel l1_alpha. Source default
    # 0.0 (library default; the BN/LN/GN-excluded L1 branch at
    # _flat_torch_module.py:272-301 only fires when l1_alpha > 0). Threaded
    # as an MLP-hparams field forwarded into the LightningModule.
    l1_alpha: float = 0.0,
    # 2026-05-31 audit-pass-9 (W9). Defaults source-verified at HEAD against
    # PytorchLightningEstimator.__init__ (base.py:264-270) and generate_mlp
    # (flat.py:208-210):
    #   #1 adamw_betas: forwarded into optimizer_kwargs["betas"] which the
    #      tabular-MLP-tuned default at _flat_torch_module.py:499
    #      injects via setdefault when caller did not pass betas.
    #   #2 use_ema: PytorchLightningEstimator __init__ kwarg (False default).
    #   #3 label_smoothing: PytorchLightningEstimator __init__ kwarg
    #      (0.0 default, multiclass-only at base.py:897-907).
    #   #4 focal_loss_gamma: PytorchLightningEstimator __init__ kwarg
    #      (None default, binary-only at base.py:878-884).
    #   #5 use_residual: generate_mlp kwarg (False default; threaded via
    #      mlp_kwargs["network_params"]).
    #   #6 numerical_embedding + kwargs: generate_mlp kwargs (None / None
    #      defaults; threaded via mlp_kwargs["network_params"]).
    adamw_betas: "tuple[float, float]" = (0.9, 0.95),
    use_ema: bool = False,
    label_smoothing: float = 0.0,
    focal_loss_gamma: "float | None" = None,
    use_residual: bool = False,
    numerical_embedding: "str | None" = None,
    numerical_embedding_kwargs_mode: str = "paper_default",
    # 2026-05-31 audit-pass-10 (W10) #1: MLP optimizer selector. "adamw"
    # leaves the LightningModule at its default optimizer
    # (_flat_torch_module.py:86 falls back to torch.optim.AdamW when no
    # override is passed); "muon_hybrid" plumbs MuonAdamWHybrid via
    # mlp_kwargs["model_params"]["optimizer"]. The MuonAdamWHybrid class
    # auto-splits the parameter list into the 2D-hidden group
    # (Newton-Schulz orthogonalized) and the 1D / non-2D group (AdamW)
    # internally; Lightning sees a single Optimizer instance, so no
    # additional configure_optimizers branching is required.
    optimizer: str = "adamw",
    # iter640 audit-pass-15. F-62 Lookahead + F-63 SAM + F-68/69/70 Mixup
    # + F-72 output-only spectral norm. Defaults source-verified at HEAD
    # against MLPTorchModel.__init__ (training/neural/_flat_torch_module.py
    # :43/:46/:48) and generate_mlp (training/neural/flat.py:224). All four
    # are gated on `mlp` in models; canon at FuzzCombo.canonical_key
    # collapses them to source defaults outside the gate.
    use_sam: bool = False,
    use_lookahead: bool = False,
    use_mixup: bool = False,
    spectral_norm_output_only: bool = False,
    # Learnable categorical embeddings (default-on nn.Embedding path; False = legacy CatBoostEncoder target-encoding) + the
    # embed-dim override (None = fastai heuristic, an int forces a fixed width). Both are PytorchLightningEstimator /
    # recurrent-estimator __init__ params (NOT model_params / network_params), so they emit top-level for BOTH mlp + recurrent.
    use_learnable_cat_embeddings: bool = True,
    categorical_embed_dim: "int | None" = None,
) -> Optional[Dict[str, Any]]:
    """Build the mlp_kwargs dict forwarded into PytorchLightningEstimator /
    PytorchLightningClassifier constructors. Returns None when neither MLP
    nor recurrent are in scope so callers can skip the wiring entirely.

    Single-edit point mirroring build_mrmr_kwargs_from_flat / build_shap_proxied
    pattern: every MLP-side knob the fuzz harness exercises maps to its exact
    __init__ parameter name here. Param names verified against
    PytorchLightningEstimator.__init__ (training/neural/base.py:203-219).
    """
    mlp_active = "mlp" in models
    recurrent_active = recurrent_model is not None
    if not (mlp_active or recurrent_active):
        return None
    kwargs: Dict[str, Any] = {}
    # #3 random_state: PytorchLightningEstimator (and the recurrent
    # estimator's wrapper, which inherits the same fit-time seed contract)
    # consume an Optional[int]. Both branches reachable via the fuzz axis;
    # canon collapses to None outside the gate so dedup absorbs phantom
    # variation on combos that wouldn't fire either path.
    if random_state is not None:
        kwargs["random_state"] = random_state
    # Learnable cat embeddings apply to BOTH mlp + recurrent estimators (shared __init__ contract); emit at top level, not
    # inside the mlp_active gate, so recurrent-only combos still hit the OFF path. Emit the False override + non-None dim only
    # when they differ from the source default so the default-True / heuristic-dim path is not shadowed.
    if not use_learnable_cat_embeddings:
        kwargs["use_learnable_cat_embeddings"] = False
    if categorical_embed_dim is not None:
        kwargs["categorical_embed_dim"] = int(categorical_embed_dim)
    # #4 class_weight: only meaningful for the classifier subclass on
    # imbalanced classification targets. The compound gate at the call
    # site (mlp in models AND classification AND rare_5pct/rare_1pct)
    # is mirrored in canonical_key; the builder respects whatever the
    # caller passes and emits the key only when non-None so the source
    # default (None) doesn't shadow downstream caller kwargs.
    if (
        class_weight is not None
        and mlp_active
        and target_type
        in (
            "binary_classification",
            "multiclass_classification",
        )
        and imbalance_ratio in ("rare_5pct", "rare_1pct")
    ):
        kwargs["class_weight"] = class_weight
    # #7 use_layernorm: regression-only meaningful. The audit gate
    # ('mlp' in models AND target_type == "regression") is mirrored in
    # canonical_key. The builder emits the key only when the gate holds
    # AND the caller asked for True so the library default (False)
    # doesn't shadow downstream caller kwargs.
    if use_layernorm and mlp_active and target_type == "regression":
        kwargs["use_layernorm"] = True
    # #8 l1_alpha: exercises the new BN/LN/GN-excluded L1 branch. Only
    # meaningful when MLP is active; canon collapses to 0.0 elsewhere.
    # Emit only when l1_alpha > 0 so the library-default-0.0 path doesn't
    # spuriously shadow downstream caller-supplied kwargs.
    if l1_alpha > 0 and mlp_active:
        kwargs["l1_alpha"] = l1_alpha
    # 2026-05-31 audit-pass-9 (W9). All seven knobs flow only when MLP is
    # actually active; canon collapses every axis to the source default
    # outside its compound gate so dedup absorbs phantom variation. We
    # emit each key only when it differs from the source default so the
    # library-default path is not spuriously shadowed downstream.
    if mlp_active:
        # #1 AdamW betas: forwarded as optimizer_kwargs={"betas": (...)}.
        # The setdefault at _flat_torch_module.py:499 ONLY fires when the
        # caller did not pass betas, so emitting non-default values here
        # exercises the override path; emitting the source default
        # (0.9, 0.95) would be a no-op but we still emit so the wiring
        # surface is asserted on every MLP combo.
        kwargs.setdefault("optimizer_kwargs", {})
        kwargs["optimizer_kwargs"]["betas"] = tuple(adamw_betas)
        # #2 use_ema: PytorchLightningEstimator __init__ kwarg. Only emit
        # when True so the library-default path is not shadowed.
        if use_ema:
            kwargs["use_ema"] = True
        # #3 label_smoothing: multiclass-only. Emit only when >0 AND the
        # multiclass gate holds so the source-default 0.0 path is never
        # shadowed on non-multiclass combos.
        if label_smoothing > 0.0 and target_type == "multiclass_classification":
            kwargs["label_smoothing"] = float(label_smoothing)
        # #4 focal_loss_gamma: binary-only. Emit only when non-None AND
        # the binary gate holds; canon at the call site also restricts
        # to imbalance_ratio in {rare_5pct, rare_1pct} so the focal-
        # loss target (class imbalance) is present.
        if focal_loss_gamma is not None and target_type == "binary_classification":
            kwargs["focal_loss_gamma"] = float(focal_loss_gamma)
        # #5 use_residual: generate_mlp network kwarg. Threaded via
        # mlp_kwargs["network_params"]["use_residual"] -- the trainer
        # merges network_params into mlp_network_params at trainer.py:712.
        # Emit only when True so the library-default-False path is not
        # spuriously shadowed.
        if use_residual:
            kwargs.setdefault("network_params", {})
            kwargs["network_params"]["use_residual"] = True
        # #6 numerical_embedding: generate_mlp kwarg + kwargs literal.
        # Both emit only when an embedding is requested. The kwargs literal
        # expands into the PLR-ctor kwargs dict; "paper_default" leaves
        # the module at its NeurIPS-2024 defaults (no override), while
        # "include_raw_false" overrides include_raw=False so the raw
        # numeric column is dropped from the embedded output.
        if numerical_embedding is not None:
            kwargs.setdefault("network_params", {})
            kwargs["network_params"]["numerical_embedding"] = numerical_embedding
            if numerical_embedding_kwargs_mode == "include_raw_false":
                kwargs["network_params"]["numerical_embedding_kwargs"] = {
                    "include_raw": False,
                }
            # "paper_default" leaves the kwargs dict unset so the module
            # ctor falls through to its library defaults.
        # 2026-05-31 audit-pass-10 (W10) #1: MLP optimizer selector. "adamw"
        # is the library default (no kwargs emission so the LightningModule
        # falls back to torch.optim.AdamW at _flat_torch_module.py:86);
        # "muon_hybrid" wires MuonAdamWHybrid via model_params per the
        # contract docstring at training/neural/_muon_optimizer.py:20.
        # The MuonAdamWHybrid ctor bakes its own betas=(0.9, 0.95) default
        # (_muon_optimizer.py:156) for the internal AdamW sub-optimizer,
        # so the #1 (W9) adamw_betas axis is INEFFECTIVE under this branch
        # (canon-collapse at FuzzCombo level pins betas to (0.9, 0.95) in
        # the muon_hybrid branch).
        if optimizer == "muon_hybrid":
            from mlframe.training.neural._muon_optimizer import MuonAdamWHybrid

            kwargs.setdefault("model_params", {})
            kwargs["model_params"]["optimizer"] = MuonAdamWHybrid
        # iter640 audit-pass-15. F-62/63/68-70/72 MLP options. Each emits
        # under the model_params/network_params child dict only when True
        # so the library-default (False) path stays bit-identical when the
        # axis is off. Names match MLPTorchModel.__init__ / generate_mlp
        # kwargs verbatim.
        if use_sam:
            kwargs.setdefault("model_params", {})
            kwargs["model_params"]["use_sam"] = True
        if use_lookahead:
            kwargs.setdefault("model_params", {})
            kwargs["model_params"]["use_lookahead"] = True
        if use_mixup:
            kwargs.setdefault("model_params", {})
            kwargs["model_params"]["use_mixup"] = True
        if spectral_norm_output_only:
            kwargs.setdefault("network_params", {})
            kwargs["network_params"]["spectral_norm_output_only"] = True
    return kwargs


def build_mlp_kwargs(combo: "FuzzCombo") -> Optional[Dict[str, Any]]:
    """FuzzCombo-aware wrapper around build_mlp_kwargs_from_flat."""
    return build_mlp_kwargs_from_flat(
        models=combo.models,
        target_type=combo.target_type,
        imbalance_ratio=combo.imbalance_ratio,
        recurrent_model=combo.recurrent_model_cfg,
        random_state=combo.mlp_random_state_cfg,
        class_weight=combo.mlp_class_weight_cfg,
        # 2026-05-31 audit-pass-8 #7/#8.
        use_layernorm=combo.mlp_use_layernorm_cfg,
        l1_alpha=combo.mlp_l1_alpha_cfg,
        # 2026-05-31 audit-pass-9 (W9) #1/#2/#3/#4/#5/#6.
        adamw_betas=combo.mlp_adamw_betas_cfg,
        use_ema=combo.mlp_use_ema_cfg,
        label_smoothing=combo.mlp_label_smoothing_cfg,
        focal_loss_gamma=combo.mlp_focal_loss_gamma_cfg,
        use_residual=combo.mlp_use_residual_cfg,
        numerical_embedding=combo.mlp_numerical_embedding_cfg,
        numerical_embedding_kwargs_mode=combo.mlp_numerical_embedding_kwargs_cfg,
        # 2026-05-31 audit-pass-10 (W10) #1.
        optimizer=combo.mlp_optimizer_cfg,
        # iter640 audit-pass-15. F-62/63/68-70/72 MLP options.
        use_sam=combo.mlp_use_sam_cfg,
        use_lookahead=combo.mlp_use_lookahead_cfg,
        use_mixup=combo.mlp_use_mixup_cfg,
        spectral_norm_output_only=combo.mlp_spectral_norm_output_only_cfg,
        use_learnable_cat_embeddings=combo.mlp_use_learnable_cat_embeddings_cfg,
        categorical_embed_dim=combo.mlp_categorical_embed_dim_cfg,
    )


def build_shap_proxied_fs_kwargs_from_flat(
    *,
    use_shap_proxied_fs: bool,
    optimizer: str = "auto",
    revalidate: bool = True,
    trust_guard: bool = True,
    interaction_aware: bool = False,
    cluster_features: "bool | str" = "auto",
    # 2026-05-28 ext axes (active_learning + prefilter_method).
    active_learning: bool = False,
    prefilter_method: str = "auto",
    # 2026-05-28 audit-pass-2 B1-B6 deeper extension axes. Defaults verified
    # against ShapProxiedFS.__init__ (feature_selection/shap_proxied_fs.py:41-89).
    config_jitter: bool = False,
    uncertainty_penalty: float = 0.0,
    within_cluster_refine: bool = True,
    use_bias_corrector: bool = True,
    refine_n_estimators: "int | None" = 100,
    trust_guard_n_estimators: "int | None" = 100,
    # 2026-05-28 audit-pass-3 W3 axes. Defaults verified against
    # ShapProxiedFS.__init__ (feature_selection/shap_proxied_fs.py:69-79).
    cluster_weighting: str = "pca_pc1",
    # iter624 (audit-pass-13 INFORMATIONAL): iter67 SU-pairwise cluster knobs.
    # Defaults verified at shap_proxied_fs.py:228-229.
    cluster_use_precomputed_bins: bool = True,
    cluster_su_threshold: float = 0.5,
    max_interaction_features: int = 16,
    prefilter_top: "int | None" = 2000,
    prefilter_n_estimators: "int | None" = 100,
    # 2026-05-28 audit-pass-5 W5 axes. Defaults verified against
    # ShapProxiedFS.__init__ (feature_selection/shap_proxied_fs.py:62, 78, 89-94).
    trust_guard_stratified_anchors: bool = False,
    trust_guard_uniform_tail_frac: float = 0.2,
    trust_guard_cardinality_dist: str = "zipf",
    trust_guard_zipf_alpha: float = 0.25,
    trust_guard_fidelity_weights: "tuple[float, float]" = (0.6, 0.4),
    trust_guard_metric: str = "proxy_fidelity_score",
    fidelity_floor: float = 0.5,
    oof_shap_n_estimators: "int | None" = 100,
    # 2026-05-30 audit-pass-6 LOW-tier (W6 LOW) ShapProxiedFS iter28-54
    # axes. Defaults verified against ShapProxiedFS.__init__
    # (feature_selection/shap_proxied_fs.py:79-113).
    prefilter_stage1_keep: "int | None" = None,
    prefilter_univariate_batch_size: "int | None" = None,
    shap_prefilter_enabled: bool = True,
    shap_prefilter_safety_factor: int = 4,
    shap_prefilter_min_features: int = 40,
    shap_aware_stage1_keep: bool = True,
    shap_aware_stage1_cushion: int = 8,
    shap_aware_stage1_floor: int = 200,
    refine_ucb_enabled: bool = True,
    refine_ucb_min_eval_size: "int | None" = None,
    refine_ucb_slack: "float | None" = None,
    refine_ucb_stdev_multiplier: float = 1.0,
    revalidation_n_estimators: "int | None" = 100,
    revalidation_ucb_enabled: bool = True,
    revalidation_ucb_min_eval_size: "int | None" = None,
    revalidation_ucb_slack: "float | None" = None,
    revalidation_ucb_stdev_multiplier: "float | None" = None,
    inner_n_jobs_cap: bool = False,
    # 2026-05-31 audit-pass-8 #5: adaptive_prescreen_by_stability. Source
    # default False (feature_selection/shap_proxied_fs.py:208).
    adaptive_prescreen_by_stability: bool = False,
    # 2026-05-31 audit-pass-12 (W12) C1/C2: precomputed cross-selector
    # artifacts dict honoured by ShapProxiedFS.__init__ at
    # shap_proxied_fs.py:258. The fuzz harness threads a sentinel-shaped
    # dict (matching the four ``align_precomputed_to_X`` branches
    # selected by ``align_mode``) when the artifact-reuse master is on;
    # the actual suite consumer substitutes ``mrmr.export_artifacts()``
    # at the call site after MRMR.fit() has run.
    precomputed: "dict | None" = None,
    # 2026-05-31 audit-pass-14 (W14) F14-1: ShapProxiedFS.cluster_backend
    # (shap_proxied_fs.py:258). Source default "auto" since iter75; "su"
    # forces the iter75 path, "pearson" pins the legacy regime.
    cluster_backend: str = "auto",
) -> Optional[Dict[str, Any]]:
    """Build the shap_proxied_fs_kwargs dict passed to
    ``registry.get("ShapProxiedFS").instantiate(**kwargs)`` (which forwards to
    ShapProxiedFS.__init__). Returns None when use_shap_proxied_fs=False so the
    FS step is a no-op (mirrors build_mrmr_kwargs_from_flat).

    Single-edit point: every ShapProxiedFS knob the fuzz harness exercises maps
    to its exact __init__ parameter name here, so adding a new shap-proxied axis
    only touches this function (plus the AXES dict + the dataclass field +
    canonical_key + _build_combo). Param names verified against
    ShapProxiedFS.__init__ (feature_selection/shap_proxied_fs.py).
    """
    if not use_shap_proxied_fs:
        return None
    return {
        "optimizer": optimizer,
        "revalidate": revalidate,
        "trust_guard": trust_guard,
        "interaction_aware": interaction_aware,
        "cluster_features": cluster_features,
        # 2026-05-28 ext axes flow straight into the ShapProxiedFS constructor
        # (same names) -- active_learning toggles the acquisition-loop branch,
        # prefilter_method drives _shap_proxy_prefilter dispatch.
        "active_learning": active_learning,
        "prefilter_method": prefilter_method,
        # 2026-05-28 audit-pass-2 B1-B6 deeper axes (param names match the
        # ShapProxiedFS.__init__ signature verbatim).
        "config_jitter": config_jitter,
        "uncertainty_penalty": uncertainty_penalty,
        "within_cluster_refine": within_cluster_refine,
        "use_bias_corrector": use_bias_corrector,
        "refine_n_estimators": refine_n_estimators,
        "trust_guard_n_estimators": trust_guard_n_estimators,
        # 2026-05-28 audit-pass-3 W3 axes (param names match
        # ShapProxiedFS.__init__ signature verbatim).
        "cluster_weighting": cluster_weighting,
        # iter624 (audit-pass-13 INFORMATIONAL): iter67 ShapProxiedFS SU-
        # pairwise cluster sub-knobs. Param names match ShapProxiedFS
        # __init__ at shap_proxied_fs.py:228-229.
        "cluster_use_precomputed_bins": cluster_use_precomputed_bins,
        "cluster_su_threshold": cluster_su_threshold,
        "max_interaction_features": max_interaction_features,
        "prefilter_top": prefilter_top,
        "prefilter_n_estimators": prefilter_n_estimators,
        # 2026-05-28 audit-pass-5 W5 axes (param names match
        # ShapProxiedFS.__init__ signature verbatim).
        "trust_guard_stratified_anchors": trust_guard_stratified_anchors,
        "trust_guard_uniform_tail_frac": trust_guard_uniform_tail_frac,
        "trust_guard_cardinality_dist": trust_guard_cardinality_dist,
        "trust_guard_zipf_alpha": trust_guard_zipf_alpha,
        "trust_guard_fidelity_weights": trust_guard_fidelity_weights,
        "trust_guard_metric": trust_guard_metric,
        "fidelity_floor": fidelity_floor,
        "oof_shap_n_estimators": oof_shap_n_estimators,
        # 2026-05-30 audit-pass-6 LOW-tier (W6 LOW) ShapProxiedFS knobs.
        # Names match ShapProxiedFS.__init__ verbatim
        # (feature_selection/shap_proxied_fs.py:79-113).
        "prefilter_stage1_keep": prefilter_stage1_keep,
        "prefilter_univariate_batch_size": prefilter_univariate_batch_size,
        "shap_prefilter_enabled": shap_prefilter_enabled,
        "shap_prefilter_safety_factor": shap_prefilter_safety_factor,
        "shap_prefilter_min_features": shap_prefilter_min_features,
        "shap_aware_stage1_keep": shap_aware_stage1_keep,
        "shap_aware_stage1_cushion": shap_aware_stage1_cushion,
        "shap_aware_stage1_floor": shap_aware_stage1_floor,
        "refine_ucb_enabled": refine_ucb_enabled,
        "refine_ucb_min_eval_size": refine_ucb_min_eval_size,
        "refine_ucb_slack": refine_ucb_slack,
        "refine_ucb_stdev_multiplier": refine_ucb_stdev_multiplier,
        "revalidation_n_estimators": revalidation_n_estimators,
        "revalidation_ucb_enabled": revalidation_ucb_enabled,
        "revalidation_ucb_min_eval_size": revalidation_ucb_min_eval_size,
        "revalidation_ucb_slack": revalidation_ucb_slack,
        "revalidation_ucb_stdev_multiplier": revalidation_ucb_stdev_multiplier,
        "inner_n_jobs_cap": inner_n_jobs_cap,
        # 2026-05-31 audit-pass-8 #5: param name matches
        # ShapProxiedFS.__init__ verbatim (shap_proxied_fs.py:208).
        "adaptive_prescreen_by_stability": adaptive_prescreen_by_stability,
        # 2026-05-31 audit-pass-12 (W12) C1/C2: precomputed dict honoured at
        # shap_proxied_fs.py:258. Forwarded verbatim; None preserves the
        # legacy ``recompute-from-scratch`` ctor contract (no behaviour
        # change unless the artifact-reuse master is on).
        "precomputed": precomputed,
        # 2026-05-31 audit-pass-14 (W14) F14-1: cluster_backend forwarded
        # verbatim (param name matches ShapProxiedFS.__init__ at
        # shap_proxied_fs.py:258 exactly).
        "cluster_backend": cluster_backend,
        # 2026-06-04 FS-coverage follow-up -- ShapProxiedFS budget parity with
        # MRMR / RFECV (commit 79779dca control knob). FIXED 5-min wall-clock cap
        # (NOT a fuzz axis), mirroring the rfecv_kwargs/mrmr_kwargs
        # max_runtime_mins=5 budget so a runaway ShapProxiedFS fit is bounded;
        # stop_file is deliberately left at its "stop" default (never set) so a
        # stray stop-flag is never tripped. Param name matches
        # ShapProxiedFS.__init__ (shap_proxied_fs.py) verbatim. NOTE: this builder
        # is currently consumed only by the cross-axis builder tests
        # (test_fuzz_combo_cross_axis_W11C.py) -- ShapProxiedFS is not yet wired
        # into train_mlframe_models_suite (no FeatureSelectionConfig field; it is
        # only reachable via custom_pre_pipelines, which the fuzz suite does not
        # populate for it), so the budget reaches the kwargs dict + builder tests
        # but not a live suite fit until ShapProxiedFS gains a suite entry point.
        "max_runtime_mins": 5,
    }


def _build_precomputed_sentinel_for_align_mode(
    align_mode: str,
) -> Optional[Dict[str, Any]]:
    """2026-05-31 audit-pass-12 (W12) C2 helper. Produce a precomputed-shaped
    sentinel dict that drives ``align_precomputed_to_X`` down each of the
    four documented branches:

      "exact":      feature_names == ["num_0", "num_1", "num_2", "num_3"]
                    -> exact_match branch at
                    _shap_proxy_precomputed.py:168
      "permuted":   feature_names == ["num_3", "num_2", "num_1", "num_0"]
                    -> permutation_match branch at
                    _shap_proxy_precomputed.py:180 (len(X_cols)==len(names))
      "subset":    feature_names == ["num_0","num_1","num_2","num_3","num_4"]
                    -> subset_match branch at :180 (len(X_cols)<len(names))
      "mismatched": feature_names == ["UNKNOWN_A", "UNKNOWN_B"]
                    -> reject + WARN + (None, report) at :216

    The sentinel keeps the suite-level helper exercisable without an
    actual MRMR.fit() call -- the fuzz runner asserts which branch fires
    via ``shap_proxy_report_['precomputed_used']``. Real production
    callers substitute ``mrmr.export_artifacts()`` for the sentinel.
    """
    if align_mode == "exact":
        return {
            "feature_names": ["num_0", "num_1", "num_2", "num_3"],
            "su_to_target": [0.1, 0.2, 0.3, 0.4],
        }
    if align_mode == "permuted":
        return {
            "feature_names": ["num_3", "num_2", "num_1", "num_0"],
            "su_to_target": [0.4, 0.3, 0.2, 0.1],
        }
    if align_mode == "subset":
        return {
            "feature_names": ["num_0", "num_1", "num_2", "num_3", "num_4"],
            "su_to_target": [0.1, 0.2, 0.3, 0.4, 0.5],
        }
    if align_mode == "mismatched":
        return {
            "feature_names": ["UNKNOWN_A", "UNKNOWN_B"],
            "su_to_target": [0.5, 0.5],
        }
    return None


def build_shap_proxied_fs_kwargs(combo: "FuzzCombo") -> Optional[Dict[str, Any]]:
    """FuzzCombo-aware wrapper around build_shap_proxied_fs_kwargs_from_flat."""
    # 2026-05-31 audit-pass-12 (W12) C1/C2: build the precomputed sentinel
    # dict only when both selectors are in the chain AND the artifact-reuse
    # master is on. The canonical_key collapse layer already pins
    # mrmr_shap_proxy_artifact_reuse_cfg = "off" outside that compound gate,
    # so the lookup is safe to gate solely on the master value here.
    _precomputed = None
    if combo.use_mrmr_fs and combo.use_shap_proxied_fs and combo.mrmr_shap_proxy_artifact_reuse_cfg == "on":
        _precomputed = _build_precomputed_sentinel_for_align_mode(
            combo.mrmr_shap_proxy_align_mode_cfg,
        )
    return build_shap_proxied_fs_kwargs_from_flat(
        use_shap_proxied_fs=combo.use_shap_proxied_fs,
        optimizer=combo.shap_proxied_optimizer_cfg,
        revalidate=combo.shap_proxied_revalidate_cfg,
        trust_guard=combo.shap_proxied_trust_guard_cfg,
        interaction_aware=combo.shap_proxied_interaction_aware_cfg,
        cluster_features=combo.shap_proxied_cluster_features_cfg,
        active_learning=combo.shap_proxied_active_learning_cfg,
        prefilter_method=combo.shap_proxied_prefilter_method_cfg,
        config_jitter=combo.shap_proxied_config_jitter_cfg,
        uncertainty_penalty=combo.shap_proxied_uncertainty_penalty_cfg,
        within_cluster_refine=combo.shap_proxied_within_cluster_refine_cfg,
        use_bias_corrector=combo.shap_proxied_use_bias_corrector_cfg,
        refine_n_estimators=combo.shap_proxied_refine_n_estimators_cfg,
        trust_guard_n_estimators=combo.shap_proxied_trust_guard_n_estimators_cfg,
        cluster_weighting=combo.shap_proxied_cluster_weighting_cfg,
        # iter624 (audit-pass-13 INFORMATIONAL): iter67 SU-pairwise cluster knobs.
        cluster_use_precomputed_bins=combo.shap_proxied_cluster_use_precomputed_bins_cfg,
        cluster_su_threshold=combo.shap_proxied_cluster_su_threshold_cfg,
        max_interaction_features=combo.shap_proxied_max_interaction_features_cfg,
        prefilter_top=combo.shap_proxied_prefilter_top_cfg,
        prefilter_n_estimators=combo.shap_proxied_prefilter_n_estimators_cfg,
        trust_guard_stratified_anchors=combo.shap_proxied_trust_guard_stratified_anchors_cfg,
        trust_guard_uniform_tail_frac=combo.shap_proxied_trust_guard_uniform_tail_frac_cfg,
        trust_guard_cardinality_dist=combo.shap_proxied_trust_guard_cardinality_dist_cfg,
        trust_guard_zipf_alpha=combo.shap_proxied_trust_guard_zipf_alpha_cfg,
        trust_guard_fidelity_weights=combo.shap_proxied_trust_guard_fidelity_weights_cfg,
        trust_guard_metric=combo.shap_proxied_trust_guard_metric_cfg,
        fidelity_floor=combo.shap_proxied_fidelity_floor_cfg,
        oof_shap_n_estimators=combo.shap_proxied_oof_shap_n_estimators_cfg,
        # 2026-05-30 audit-pass-6 LOW-tier (W6 LOW) ShapProxiedFS knobs.
        prefilter_stage1_keep=combo.shap_proxied_prefilter_stage1_keep_cfg,
        prefilter_univariate_batch_size=combo.shap_proxied_prefilter_univariate_batch_size_cfg,
        shap_prefilter_enabled=combo.shap_proxied_shap_prefilter_enabled_cfg,
        shap_prefilter_safety_factor=combo.shap_proxied_shap_prefilter_safety_factor_cfg,
        shap_prefilter_min_features=combo.shap_proxied_shap_prefilter_min_features_cfg,
        shap_aware_stage1_keep=combo.shap_proxied_shap_aware_stage1_keep_cfg,
        shap_aware_stage1_cushion=combo.shap_proxied_shap_aware_stage1_cushion_cfg,
        shap_aware_stage1_floor=combo.shap_proxied_shap_aware_stage1_floor_cfg,
        refine_ucb_enabled=combo.shap_proxied_refine_ucb_enabled_cfg,
        refine_ucb_min_eval_size=combo.shap_proxied_refine_ucb_min_eval_size_cfg,
        refine_ucb_slack=combo.shap_proxied_refine_ucb_slack_cfg,
        refine_ucb_stdev_multiplier=combo.shap_proxied_refine_ucb_stdev_multiplier_cfg,
        revalidation_n_estimators=combo.shap_proxied_revalidation_n_estimators_cfg,
        revalidation_ucb_enabled=combo.shap_proxied_revalidation_ucb_enabled_cfg,
        revalidation_ucb_min_eval_size=combo.shap_proxied_revalidation_ucb_min_eval_size_cfg,
        revalidation_ucb_slack=combo.shap_proxied_revalidation_ucb_slack_cfg,
        revalidation_ucb_stdev_multiplier=combo.shap_proxied_revalidation_ucb_stdev_multiplier_cfg,
        inner_n_jobs_cap=combo.shap_proxied_inner_n_jobs_cap_cfg,
        # 2026-05-31 audit-pass-8 #5.
        adaptive_prescreen_by_stability=combo.shap_proxied_adaptive_prescreen_by_stability_cfg,
        # 2026-05-31 audit-pass-12 (W12) C1/C2.
        precomputed=_precomputed,
        # 2026-05-31 audit-pass-14 (W14) F14-1.
        cluster_backend=combo.shap_proxied_cluster_backend_cfg,
    )


def build_composite_discovery_config_from_flat(
    *,
    enabled: bool,
    transforms_mode: Optional[str] = None,
    mi_estimator: str = "bin",
    mi_nbins: int = 16,
    mi_aggregation: str = "mean",
    mi_sample_strategy: str = "random",
    stacked_residual_aggregation: str = "mean",
    discovery_n_jobs: int = 1,
    # 2026-05-22 TVT-MLP audit-followup axes.
    composite_skip_raw_dominates_ratio: float = 0.0,
    composite_skip_ablation_delta_pct: float = 0.0,
    composite_eps_mi_gain: float = -10.0,
    composite_top_k_after_mi: int = 32,
    composite_require_beats_raw_baseline: bool = False,
    composite_per_bin_n_bins: int = 0,
    composite_tiny_screening_mode: str = "per_family",
    composite_include_additive_residual: bool = True,
    # 2026-05-28 deep knobs (4 new axes).
    auto_skip_on_baseline_optimal: bool = False,
    mi_n_neighbors: int = 3,
    auto_base_null_perms: int = 20,
    multi_base_max_k: int = 3,
    # 2026-05-30 audit-pass-6 CV-selector mode (HIGH; mean vs t-LCB et al).
    cv_selector_mode: str = "mean",
    # 2026-05-30 audit-pass-6 LOW-tier (W6 LOW) CV-selector scalars.
    # Defaults verified against CompositeTargetDiscoveryConfig
    # (_composite_target_discovery_config.py:127-130).
    cv_selector_alpha: float = 1.0,
    cv_selector_confidence: float = 0.9,
    cv_selector_quantile_level: float = 0.9,
    cv_persist_fold_scores: bool = False,
    # 2026-05-31 audit-pass-12 (W12) A1: CompositeTargetDiscoveryConfig.
    # multilabel_strategy validator at _composite_target_discovery_config.py:940
    # accepts {"per_target", "skip", "multi_target_regression"}. Field default
    # "per_target" at :773. Forwarded verbatim; canon-collapse to "per_target"
    # outside multilabel/MTR target_types already applied at the FuzzCombo
    # canonical_key layer.
    multilabel_strategy: str = "per_target",
    # 2026-06-03: 11 previously-inert CompositeTargetDiscoveryConfig axes.
    # Names match the dataclass fields exactly; defaults match the dataclass.
    always_build_ct_ensemble_for_raw: bool = True,
    ct_ensemble_dummy_floor_enabled: bool = True,
    ct_ensemble_dummy_floor_tolerance: float = 0.0,
    extreme_ar_group_aware_skip: bool = True,
    extreme_ar_threshold: float = 0.99,
    lag_predict_failsafe_tolerance: float = 0.10,
    oof_holdout_source: str = "external_val",
    oof_holdout_frac: float = 0.2,
    stacking_aware_gate_enabled: bool = False,
    top_m_after_tiny: int = 10,
    use_baseline_diagnostics_hint: bool = True,
):
    """Build a CompositeTargetDiscoveryConfig honoring the discovery
    enable + transforms_mode axes + (iter162) nested MI / stacked /
    parallelism knobs + (2026-05-22) TVT-MLP audit-followup gate axes."""
    from mlframe.training.configs import CompositeTargetDiscoveryConfig

    if not enabled:
        return CompositeTargetDiscoveryConfig(enabled=False)
    if transforms_mode == "unary_only":
        transforms = ["cbrt_y", "log_y", "yeo_johnson_y", "quantile_normal_y"]
    elif transforms_mode == "chain_only":
        transforms = [
            "chain_linres_cbrt",
            "chain_linres_yj",
            "chain_monres_cbrt",
            "chain_monres_yj",
        ]
    elif transforms_mode == "legacy":
        transforms = [
            "diff",
            "ratio",
            "logratio",
            "linear_residual",
            "quantile_residual",
            "monotonic_residual",
        ]
    else:
        transforms = None
    # The additive_residual toggle works on top of any transforms_mode:
    # if the chosen mode would include bivariate residuals, ensure
    # additive_residual is present / absent as requested.
    if transforms is not None and composite_include_additive_residual and "additive_residual" not in transforms and transforms_mode in (None, "legacy"):
        transforms = ["additive_residual", *transforms]
    elif transforms is not None and not composite_include_additive_residual and "additive_residual" in transforms:
        transforms = [t for t in transforms if t != "additive_residual"]
    kw: Dict[str, Any] = {
        "enabled": True,
        "base_candidates": "auto",
        "auto_base_top_k": 3,
        "multi_base_enabled": (multi_base_max_k > 1),
        # 2026-05-28 multi_base_max_k axis (was hardcoded 2). When the axis
        # value is 1 we additionally turn off multi_base_enabled above so the
        # promotion loop short-circuits cleanly.
        "multi_base_max_k": multi_base_max_k,
        # iter162 nested knobs.
        "mi_estimator": mi_estimator,
        "mi_nbins": mi_nbins,
        "mi_aggregation": mi_aggregation,
        "mi_sample_strategy": mi_sample_strategy,
        "stacked_residual_aggregation": stacked_residual_aggregation,
        "discovery_n_jobs": discovery_n_jobs,
        # 2026-05-22 TVT-MLP audit-followup axes.
        "composite_skip_when_raw_dominates_ratio": composite_skip_raw_dominates_ratio,
        "composite_skip_when_ablation_delta_pct": composite_skip_ablation_delta_pct,
        "eps_mi_gain": composite_eps_mi_gain,
        "top_k_after_mi": composite_top_k_after_mi,
        "require_beats_raw_baseline": composite_require_beats_raw_baseline,
        "per_bin_n_bins": composite_per_bin_n_bins,
        "tiny_screening_models": composite_tiny_screening_mode,
        # 2026-06-04 profiling-budget: cap the Phase-B tiny-rerank model size (default 60).
        # Composite discovery has no wall-clock budget of its own, so small tiny models keep
        # the per-combo tiny-rerank (the top composite hotspot) bounded under the per-combo timeout.
        "tiny_model_n_estimators": 5,
        # 2026-05-28 deep knobs (3 of the 4; multi_base_max_k handled above
        # because it also gates multi_base_enabled). These names match the
        # CompositeTargetDiscoveryConfig dataclass fields exactly.
        "auto_skip_on_baseline_optimal": auto_skip_on_baseline_optimal,
        "mi_n_neighbors": mi_n_neighbors,
        "auto_base_null_perms": auto_base_null_perms,
        # 2026-05-30 audit-pass-6 CV-selector mode (HIGH).
        # CompositeTargetDiscoveryConfig.cv_selector_mode at
        # _composite_target_discovery_config.py:117.
        "cv_selector_mode": cv_selector_mode,
        # 2026-05-30 audit-pass-6 LOW-tier (W6 LOW) CV-selector scalars.
        # CompositeTargetDiscoveryConfig fields at
        # _composite_target_discovery_config.py:127-130.
        "cv_selector_alpha": cv_selector_alpha,
        "cv_selector_confidence": cv_selector_confidence,
        "cv_selector_quantile_level": cv_selector_quantile_level,
        "cv_persist_fold_scores": cv_persist_fold_scores,
        # 2026-05-31 audit-pass-12 (W12) A1: multilabel_strategy field at
        # _composite_target_discovery_config.py:773 (validator at :940).
        "multilabel_strategy": multilabel_strategy,
        # 2026-06-03: 11 previously-inert discovery axes (field names verified
        # against _composite_target_discovery_config.py).
        "always_build_ct_ensemble_for_raw": always_build_ct_ensemble_for_raw,
        "ct_ensemble_dummy_floor_enabled": ct_ensemble_dummy_floor_enabled,
        "ct_ensemble_dummy_floor_tolerance": ct_ensemble_dummy_floor_tolerance,
        "extreme_ar_group_aware_skip": extreme_ar_group_aware_skip,
        "extreme_ar_threshold": extreme_ar_threshold,
        "lag_predict_failsafe_tolerance": lag_predict_failsafe_tolerance,
        "oof_holdout_source": oof_holdout_source,
        "oof_holdout_frac": oof_holdout_frac,
        "stacking_aware_gate_enabled": stacking_aware_gate_enabled,
        "top_m_after_tiny": top_m_after_tiny,
        "use_baseline_diagnostics_hint": use_baseline_diagnostics_hint,
    }
    if composite_tiny_screening_mode == "per_family":
        kw["tiny_screening_families"] = ("lightgbm", "linear")
    else:
        kw["tiny_screening_families"] = ("lightgbm",)
    if transforms is not None:
        kw["transforms"] = transforms
    return CompositeTargetDiscoveryConfig(**kw)


def build_composite_discovery_config(combo: "FuzzCombo"):
    """FuzzCombo-aware wrapper."""
    enabled = combo.composite_discovery_enabled_cfg and combo.target_type == "regression"
    return build_composite_discovery_config_from_flat(
        enabled=enabled,
        transforms_mode=combo.composite_transforms_mode_cfg if enabled else None,
        mi_estimator=combo.composite_mi_estimator_cfg,
        mi_nbins=combo.composite_mi_nbins_cfg,
        mi_aggregation=combo.composite_mi_aggregation_cfg,
        mi_sample_strategy=combo.composite_mi_sample_strategy_cfg,
        stacked_residual_aggregation=combo.composite_stacked_residual_aggregation_cfg,
        discovery_n_jobs=combo.composite_discovery_n_jobs_cfg,
        composite_skip_raw_dominates_ratio=combo.composite_skip_raw_dominates_ratio_cfg,
        composite_skip_ablation_delta_pct=combo.composite_skip_ablation_delta_pct_cfg,
        composite_eps_mi_gain=combo.composite_eps_mi_gain_cfg,
        composite_top_k_after_mi=combo.composite_top_k_after_mi_cfg,
        composite_require_beats_raw_baseline=combo.composite_require_beats_raw_baseline_cfg,
        composite_per_bin_n_bins=combo.composite_per_bin_n_bins_cfg,
        composite_tiny_screening_mode=combo.composite_tiny_screening_mode_cfg,
        composite_include_additive_residual=combo.composite_include_additive_residual_cfg,
        # 2026-05-28 deep knobs (4 new axes).
        auto_skip_on_baseline_optimal=combo.composite_auto_skip_on_baseline_optimal_cfg,
        mi_n_neighbors=combo.composite_mi_n_neighbors_cfg,
        auto_base_null_perms=combo.composite_auto_base_null_perms_cfg,
        multi_base_max_k=combo.composite_multi_base_max_k_cfg,
        # 2026-05-30 audit-pass-6 CV-selector mode. When discovery is off
        # the upstream CompositeTargetDiscoveryConfig(enabled=False) early-
        # returns before the cv_selector_mode key is consumed, so passing
        # the axis value unconditionally is safe.
        cv_selector_mode=combo.cv_selector_mode_cfg,
        # 2026-05-30 audit-pass-6 LOW-tier (W6 LOW) CV-selector scalars.
        # Same early-return safety as cv_selector_mode above.
        cv_selector_alpha=combo.cv_selector_alpha_cfg,
        cv_selector_confidence=combo.cv_selector_confidence_cfg,
        cv_selector_quantile_level=combo.cv_selector_quantile_level_cfg,
        cv_persist_fold_scores=combo.cv_persist_fold_scores_cfg,
        # 2026-05-31 audit-pass-12 (W12) A1.
        multilabel_strategy=combo.composite_target_multilabel_strategy_cfg,
        # 2026-06-03: 11 previously-inert discovery axes now forwarded. Safe to
        # pass unconditionally: the enabled=False early-return ignores them.
        always_build_ct_ensemble_for_raw=combo.composite_always_build_ct_ensemble_for_raw_cfg,
        ct_ensemble_dummy_floor_enabled=combo.composite_ct_ensemble_dummy_floor_enabled_cfg,
        ct_ensemble_dummy_floor_tolerance=combo.composite_ct_ensemble_dummy_floor_tolerance_cfg,
        extreme_ar_group_aware_skip=combo.composite_extreme_ar_group_aware_skip_cfg,
        extreme_ar_threshold=combo.composite_extreme_ar_threshold_cfg,
        lag_predict_failsafe_tolerance=combo.composite_lag_predict_failsafe_tolerance_cfg,
        oof_holdout_source=combo.composite_oof_holdout_source_cfg,
        oof_holdout_frac=combo.composite_oof_holdout_frac_cfg,
        stacking_aware_gate_enabled=combo.composite_stacking_aware_gate_enabled_cfg,
        top_m_after_tiny=combo.composite_top_m_after_tiny_cfg,
        use_baseline_diagnostics_hint=combo.composite_use_baseline_diagnostics_hint_cfg,
    )


def build_slice_stable_es_config_from_flat(
    *,
    enabled: bool = False,
    aggregate: str = "mean",
    source: str = "temporal",
    pareto_best_iter_selection: bool = False,
    diagnostic_only: bool = False,
):
    """Build a ``SliceStableESConfig`` honouring the 5 fuzz axes (W6 upgrade
    of the canon-only wiring landed in commit 8d38bf20).

    Field-name mapping (audit suffix `_cfg` -> SOURCE field name on
    ``mlframe.training._training_runtime_configs.SliceStableESConfig``):

      slice_stable_es_enabled_cfg                  -> ``enabled``
      slice_stable_es_aggregate_cfg                -> ``aggregate``
      slice_stable_es_source_cfg                   -> ``source``
      slice_stable_es_pareto_best_iter_selection_cfg
                                                   -> ``pareto_best_iter_selection``
      slice_stable_es_diagnostic_only_cfg          -> ``diagnostic_only``

    All five names verified against
    ``src/mlframe/training/_training_runtime_configs.py:42-95`` (no audit-vs-
    source drift).
    """
    from mlframe.training.configs import SliceStableESConfig

    return SliceStableESConfig(
        enabled=enabled,
        aggregate=aggregate,
        source=source,
        pareto_best_iter_selection=pareto_best_iter_selection,
        diagnostic_only=diagnostic_only,
    )


def build_slice_stable_es_config(combo: "FuzzCombo"):
    """FuzzCombo-aware wrapper around ``build_slice_stable_es_config_from_flat``.

    Threads the 5 fuzz axes through the SliceStableESConfig construction so
    the suite-side inline path exercises the config (current production
    ``train_mlframe_models_suite`` does not accept a slice_stable_es kwarg
    yet; this helper is also consumable by the trainer-direct test_fuzz_combo
    smoke tests and any future suite plumbing).
    """
    return build_slice_stable_es_config_from_flat(
        enabled=combo.slice_stable_es_enabled_cfg,
        aggregate=combo.slice_stable_es_aggregate_cfg,
        source=combo.slice_stable_es_source_cfg,
        pareto_best_iter_selection=combo.slice_stable_es_pareto_best_iter_selection_cfg,
        diagnostic_only=combo.slice_stable_es_diagnostic_only_cfg,
    )
