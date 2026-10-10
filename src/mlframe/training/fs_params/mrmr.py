"""Strict parameters of `mlframe.feature_selection.filters:MRMR`.

GENERATED from the signature; do not edit by hand. A meta-test compares it with the live signature
(`pyutilz.dev.signature_models.signature_drift`). Regenerate: `python -m mlframe.training.fs_params._generate`.
"""

from __future__ import annotations

from typing import Any, Literal, Optional, Union
import collections.abc
import sklearn.model_selection._split
from pydantic import BaseModel, ConfigDict


class MRMRParams(BaseModel):
    """Parameters of `mlframe.feature_selection.filters:MRMR`; an unknown name or a wrong type raises when this is instantiated."""

    model_config = ConfigDict(extra="forbid", frozen=True, arbitrary_types_allowed=True)
    __signature_overrides__ = (
        "additional_rfecv_selection_rule",
        "cluster_aggregate_mode",
        "dcd_distance",
        "dcd_swap_method",
        "dcd_tau_cluster",
        "fe_binary_preset",
        "fe_hybrid_orth_basis",
        "fe_hybrid_orth_cluster_basis_aggregator",
        "fe_hybrid_orth_default_scorer",
        "fe_hybrid_orth_ensemble_aggregator",
        "fe_hybrid_orth_hsic_kernel",
        "fe_unary_preset",
        "group_mi_aggregate",
        "mi_correction",
        "mi_normalization",
        "mrmr_redundancy_algo",
        "mrmr_relevance_algo",
        "nan_strategy",
        "nbins_strategy",
        "quantization_method",
        "redundancy_aggregator",
        "stability_selection_method",
    )
    __signature_skip_default__ = ("quantization_dtype", "dtype", "usability_feature_dtype")

    quantization_method: Literal["quantile", "uniform"] = "quantile"
    quantization_nbins: int = 10
    quantization_dtype: Any = None
    max_categorical_cardinality: int | None = None
    nbins_strategy: Optional[
        Literal[
            "auto",
            "sturges",
            "freedman_diaconis",
            "fd",
            "qs",
            "quantile",
            "uniform",
            "knuth",
            "blocks",
            "mdlp",
            "fayyad_irani",
            "mdlp_validated",
            "fayyad_irani_validated",
            "optimal_joint",
            "cv",
            "mah",
            "mah_sci",
            "sci",
            "marx",
        ]
    ] = "mdlp"
    nbins_strategy_kwargs: dict | None = None
    max_adaptive_nbins: int = 256
    adaptive_nbins_large_n_reg: bool = True
    adaptive_nbins_large_n_reg_threshold: int = 50000
    adaptive_nbins_large_n_reg_nbins: int = 20
    mi_correction: Literal["none", "miller_madow", "chao_shen"] = "none"
    redundancy_aggregator: Optional[Literal["jmim", "auto"]] = None
    bur_lambda: float = 0.0
    relaxmrmr_alpha: float = 0.0
    cmi_perm_stop: bool = False
    cmi_perm_n_permutations: int = 100
    cmi_perm_alpha: float = 0.05
    uaed_auto_size: bool = False
    cpt_test: bool = False
    cpt_n_permutations: int = 200
    stability_selection_method: Literal["classic", "cluster", "complementary_pairs"] = "classic"
    stability_selection_corr_threshold: float = 0.8
    stability_n_bootstrap: int = 50
    stability_pi_threshold: float = 0.6
    pid_synergy_bonus: float = 0.0
    mi_normalization: Literal["none", "su"] = "none"
    nan_strategy: Literal["separate_bin", "fillna_zero", "ffill_bfill", "propagate", "raise"] = "separate_bin"
    factors_names_to_use: collections.abc.Sequence[str] | None = None
    factors_to_use: collections.abc.Sequence[int] | None = None
    mrmr_relevance_algo: Literal["fleuret", "pld"] = "fleuret"
    mrmr_redundancy_algo: Literal["fleuret", "pld_max", "pld_mean"] = "fleuret"
    reduce_gain_on_subelement_chosen: bool = True
    use_simple_mode: bool = False
    run_additional_rfecv_minutes: bool = False
    additional_rfecv_selection_rule: Literal["auto", "argmax", "one_se_min", "one_se_max"] = "one_se_min"
    additional_rfecv_kwargs: dict | None = None
    extra_x_shuffling: bool = True
    dtype: Any = None
    random_seed: int | None = None
    use_gpu: bool = False
    n_workers: int = 1
    min_occupancy: int | None = None
    min_nonzero_confidence: float = 0.99
    full_npermutations: int = 3
    baseline_npermutations: int = 2
    fe_confirm_undersample_rows_per_cell: float = 5.0
    fe_auto_prevalence_debias_chunk_size: int = 50000
    min_relevance_gain: float = 0.0001
    min_relevance_gain_frac: float = 0.001
    min_relevance_gain_mode: str = "relative_to_entropy"
    min_relevance_gain_relative_to_first: float = 0.05
    cardinality_bias_correction: bool = True
    max_consec_unconfirmed: int = 10
    max_runtime_mins: float | None = None
    interactions_min_order: int = 1
    interactions_max_order: int = 1
    interactions_order_reversed: bool = False
    max_veteranes_interactions_order: int = 1
    only_unknown_interactions: bool = False
    fe_fast_search: bool = False
    fe_max_steps: int = 1
    fe_reselect_after_engineering: bool = True
    fe_raw_retention_max_n: int = 20000
    fe_rejection_ledger_cap: Optional[int] = None
    fe_drop_redundant_raw_operands: bool = True
    fe_keep_linearly_usable_raw_operands: Optional[bool] = None
    redundancy_policy: str = "drop"
    fe_raw_redundancy_retain_frac: float = 0.15
    fe_stability_vote_enable: Optional[bool] = None
    fe_stability_vote_k: int | str = 5
    fe_stability_vote_quorum: float = 0.6
    fe_rung_schedule_enable: bool = True
    fe_rung_keep_frac: float | None = None
    fe_rung_rel_floor: float = 0.4
    fe_rung_min_pairs: int = 6
    fe_sufficient_summary_early_stop: bool = True
    fe_sufficient_summary_residual_frac: float = 0.25
    fe_sufficient_summary_maxt_permutations: int = 25
    fe_sufficient_summary_maxt_quantile: float = 0.95
    fe_sufficient_summary_ridge_alpha: float = 0.001
    fe_auto_escalation_enable: bool = True
    fe_escalation_max_pairs: int = 8
    fe_escalation_min_rows: int = 500
    fe_escalation_min_val_corr: float = 0.15
    fe_escalation_poly_degree: int = 6
    fe_escalation_fourier_max_freqs: int = 3
    fe_escalation_max_candidates_per_pair: int = 3
    fe_escalation_pairness_margin: float = 1.15
    fe_escalation_underdelivery_enable: Optional[bool] = None
    fe_escalation_underdelivery_excess_frac: float = 0.05
    fe_escalation_underdelivery_self_ratio: float = 3.0
    fe_synergy_prevalence_rescue_enable: bool = True
    fe_escalation_feedforward_enable: bool = False
    fe_npermutations: int = 3
    fe_ntop_features: int = 0
    fe_max_engineered_operands: int = 8
    fe_unary_preset: Literal["minimal", "medium", "maximal"] = "medium"
    fe_binary_preset: Literal["minimal", "medium", "maximal"] = "minimal"
    fe_max_pair_features: int = 10
    fe_min_nonzero_confidence: float = 0.99
    fe_min_pair_mi: float = 0.001
    fe_min_pair_mi_prevalence: float | str = 1.05
    fe_min_engineered_mi_prevalence: float = 0.9
    fe_mm_debias_prevalence: bool = False
    fe_acceptance: str = "conditional_mi"
    fe_engineered_cmi_retain_frac: float = 0.15
    fe_engineered_cmi_significance_escape_margin: float = 3.0
    fe_engineered_cmi_max_candidates: int = 64
    fe_good_to_best_feature_mi_threshold: float = 0.98
    fe_max_external_validation_factors: int = 0
    fe_max_polynoms: int = 0
    fe_print_best_mis_only: bool = True
    fe_smart_polynom_iters: int = 0
    fe_smart_polynom_optimization_steps: int = 1000
    fe_smart_polynom_subsample_n: int = 30000
    fe_check_pairs_subsample_n: int = 30000
    fe_subsample_stratify: bool | None = True
    fe_min_polynom_degree: int = 1
    fe_max_polynom_degree: int = 6
    fe_min_polynom_coeff: float = -10.0
    fe_max_polynom_coeff: float = 10.0
    fe_hermite_l2_penalty: float = 0.05
    fe_polynomial_basis: str = "chebyshev"
    fe_pair_prewarp_enable: Optional[bool] = None
    fe_pair_prewarp_basis: str = "chebyshev"
    fe_pair_prewarp_max_degree: int = 4
    fe_gate_med_enable: bool = False
    fe_pair_prewarp_uplift_threshold: float = 1.2
    fe_mi_estimator: str = "plugin"
    fe_optimizer: str = "cupy_kernel"
    fe_warm_start: bool = True
    fe_multi_fidelity: bool = True
    verbose: bool | int = 0
    ndigits: int = 5
    parallel_kwargs: dict | None = None
    cv: int | sklearn.model_selection._split.BaseCrossValidator | collections.abc.Iterable | None = 3
    cv_shuffle: bool = False
    random_state: int | None = None
    n_jobs: int = -1
    skip_retraining_on_same_content: bool = True
    max_confirmation_cand_nbins: int | None = None
    fe_fallback_to_all: bool = False
    min_features_fallback: int = 1
    sis_screen_threshold: int = 20000
    sis_dedup_corr_thr: float = 0.92
    cat_fe_config: Any = None
    fit_cache_max: int = 4
    fit_cache_max_mb: Optional[float] = None
    fe_adaptive_threshold_relax: bool = True
    fe_adaptive_relax_factor: float = 0.9
    mrmr_identity_cache_include_y: bool = True
    mrmr_skip_when_prior_was_identity: bool = True
    mrmr_identity_cache_ycorr_threshold: float = 0.5
    strict_groups: bool = True
    group_aware_mi: bool = False
    group_mi_aggregate: Literal["size", "equal"] = "size"
    group_mi_min_rows: int = 20
    build_friend_graph: bool = False
    friend_graph_prune: bool = False
    friend_graph_max_nodes: int = 200
    friend_graph_gpu_backend: Optional[str] = None
    friend_graph_mi_eps: float = 1e-06
    friend_graph_edge_significance: float = 3.0
    friend_graph_garbage_min_degree: int = 3
    friend_graph_unique_ratio: float = 1.0
    friend_graph_unique_max_degree: int = 1
    cluster_aggregate_enable: bool = True
    cluster_aggregate_mode: Literal["augment", "replace"] = "replace"
    cluster_aggregate_methods: tuple = ("mean_z",)
    cluster_aggregate_mi_prevalence: float = 1.0
    cluster_aggregate_min_member_relevance: float = 0.0
    cluster_aggregate_min_cluster_size: int = 3
    cluster_aggregate_max_cluster_size: int = 12
    cluster_aggregate_corr_threshold: float = 0.6
    cluster_aggregate_homogeneity_tau: float = 0.6
    cluster_aggregate_max_candidates: int = 200
    dcd_enable: bool = True
    dcd_tau_cluster: Union[float, Literal["auto"]] = 0.7
    dcd_distance: Literal["su", "vi", "sotoca_pla", "auto"] = "su"
    dcd_super_tau: float = 0.5
    dcd_hierarchy_max_levels: int = 3
    dcd_tau_calibration_n_pairs: int = 100
    dcd_tau_calibration_seed: int = 0
    dcd_cluster_size_threshold: int = 4
    dcd_swap_gain_threshold: float = 0.05
    dcd_swap_method: Literal[
        "auto", "mean_z", "mean_inv_var", "median", "pca_pc1", "factor_score", "pca_pc2", "median_z", "signed_max_abs", "signed_l2_sum"
    ] = "auto"
    dcd_pairwise_cache_max: int = 50000
    dcd_min_cluster_size: int = 2
    dcd_max_cluster_size: int = 12
    dcd_swap_alpha: float = 0.05
    dcd_swap_npermutations: int = 199
    warp_tiebreak_prefer_linear: bool = True
    warp_twin_rank_corr: float = 0.99
    warp_linear_margin: float = 0.05
    dcd_postoc_compose: bool = False
    fe_hybrid_orth_enable: bool = True
    fe_univariate_basis_enable: bool = True
    fe_accuracy_gate: bool = True
    fe_univariate_fourier_enable: bool = True
    fe_univariate_fourier_adaptive: bool = True
    fe_univariate_fourier_adaptive_min_val_corr: float = 0.15
    fe_univariate_fourier_chirp: bool = True
    fe_univariate_fourier_chirp_min_val_corr: float = 0.15
    fe_univariate_fourier_adaptive_max_cols: Optional[int] = 100
    fe_hinge_enable: bool = True
    fe_hinge_top_k: int = 5
    fe_hinge_max_breakpoints: int = 2
    fe_hinge_emit_indicator: bool = False
    fe_hinge_min_heldout_r2_uplift: float = 0.02
    fe_synergy_screen_max_features: int = 250
    fe_synergy_prerank: bool = True
    fe_synergy_exhaustive: str = "auto"
    fe_synergy_exhaustive_max_seconds: float | None = None
    fe_synergy_max_sweep_cost: float = 500000000.0
    fe_synergy_max_pairs: int = 16
    fe_synergy_min_prevalence: float | str = 1.5
    fe_synergy_min_rows: int = 300
    fe_additive_fusion_enable: bool = True
    fe_additive_fusion_floor_margin: float = 1.0
    fe_additive_fusion_max: int = 4
    fe_additive_fusion_ols_r_margin_sd: float = 2.0
    fe_pair_perm_null_admission_enable: bool = False
    fe_pair_perm_null_excess_frac: float = 0.05
    fe_pair_usability_admission_enable: bool = True
    fe_pair_usability_admission_min_corr: float = 0.6
    fe_pair_usability_admission_pairness_margin: float = 1.05
    fe_pair_usability_admission_rank_frac: float = 0.7
    fe_raw_tail_subsume_min_corr: float = 0.85
    fe_pair_usability_prescan_max_pairs: int = 256
    fe_prevalence_rescue_all_pairs: bool = False
    fe_multi_emit_max_per_pair: int = 1
    fe_multi_emit_mi_floor: float = 0.5
    fe_multi_emit_diversity_corr: float = 0.9
    fe_pair_maxt_null_permutations: int = 25
    fe_pair_maxt_null_quantile: float = 0.95
    fe_pair_maxt_min_pairs: int = 30
    fe_ii_routing_enable: bool = False
    fe_ii_routing_null_permutations: int = 25
    fe_ii_routing_null_quantile: float = 0.95
    fe_ii_routing_min_pairs: int = 30
    fe_gbm_seeder_enable: bool = False
    fe_gbm_seeder_min_features: int = 30
    fe_gbm_seeder_top_k_pairs: int = 12
    fe_gbm_seeder_top_k_triples: int = 8
    fe_gbm_seeder_n_estimators: int = 300
    fe_gbm_seeder_max_depth: int = 4
    fe_gbm_seeder_self_gate_margin: float = 0.0
    fe_gbm_seeder_self_gate_reps: int = 5
    fe_gbm_seeder_self_gate_min_z: float = 2.0
    fe_gradient_interaction_enable: bool = False
    fe_triple_maxt_null_permutations: int = 25
    fe_triple_maxt_null_quantile: float = 0.95
    fe_triple_maxt_min_triples: int = 4
    fe_hybrid_orth_degrees: tuple = (2, 3)
    fe_hybrid_orth_basis: Literal["auto", "hermite", "legendre", "chebyshev", "laguerre", "fourier", "rbf", "sigmoid", "pade"] = "auto"
    fe_hybrid_orth_top_k: int = 5
    fe_hybrid_orth_pair_enable: bool = True
    fe_hybrid_orth_pair_max_degree: int = 2
    fe_hybrid_orth_triplet_enable: bool = True
    fe_hybrid_orth_triplet_max_degree: int = 1
    fe_hybrid_orth_triplet_seed_k: int = 4
    fe_hybrid_orth_triplet_top_count: int = 2
    fe_hybrid_orth_quadruplet_enable: bool = True
    fe_hybrid_orth_quadruplet_max_degree: int = 1
    fe_hybrid_orth_quadruplet_seed_k: int = 4
    fe_hybrid_orth_quadruplet_top_count: int = 2
    fe_hybrid_orth_adaptive_arity_enable: bool = False
    fe_hybrid_orth_adaptive_arity_max_arity: int = 3
    fe_hybrid_orth_adaptive_arity_max_degree: int = 1
    fe_hybrid_orth_adaptive_arity_seed_k: int = 4
    fe_hybrid_orth_adaptive_arity_top_count: int = 3
    fe_budget_learning: bool | str = "auto"
    fe_budget_kwargs: Optional[dict] = None
    fe_semi_supervised_enable: bool = False
    fe_hybrid_orth_lasso_enable: bool = False
    fe_hybrid_orth_lasso_alpha: float = 0.01
    fe_hybrid_orth_elasticnet_enable: bool = False
    fe_hybrid_orth_elasticnet_alpha: float = 0.01
    fe_hybrid_orth_elasticnet_l1_ratio: float = 0.5
    fe_hybrid_orth_adaptive_degree_enable: bool = False
    fe_hybrid_orth_adaptive_degree_range: tuple = (1, 2, 3, 4, 5, 6)
    fe_hybrid_orth_adaptive_degree_min_uplift: float = 1.05
    fe_hybrid_orth_conditional_routing_enable: bool = False
    fe_hybrid_orth_conditional_routing_top_k: int = 5
    fe_hybrid_orth_conditional_routing_min_uplift: float = 1.1
    fe_hybrid_orth_conditional_routing_degrees: tuple = (2, 3)
    fe_hybrid_orth_diff_basis_enable: bool = False
    fe_hybrid_orth_diff_basis_corr_threshold: float = 0.7
    fe_hybrid_orth_diff_basis_degrees: tuple = (1, 2, 3)
    fe_hybrid_orth_diff_basis_top_k: int = 3
    fe_hybrid_orth_cluster_basis_enable: bool = False
    fe_hybrid_orth_cluster_basis_aggregator: Literal["mean_z", "median_z", "pc1"] = "mean_z"
    fe_hybrid_orth_cluster_basis_degrees: tuple = (2, 3)
    fe_hybrid_orth_cluster_basis_top_k: int = 3
    fe_hybrid_orth_bootstrap_enable: bool = False
    fe_hybrid_orth_bootstrap_n_boot: int = 10
    fe_hybrid_orth_bootstrap_sample_fraction: float = 0.8
    fe_hybrid_orth_three_gate_enable: bool = False
    fe_hybrid_orth_three_gate_n_folds: int = 5
    fe_hybrid_orth_three_gate_cmi_min: float = 0.001
    fe_hybrid_orth_ksg_enable: bool = False
    fe_hybrid_orth_ksg_n_neighbors: int = 3
    fe_hybrid_orth_ksg_min_uplift: float = 0.95
    fe_hybrid_orth_ksg_min_abs_mi_frac: float = 0.05
    fe_hybrid_orth_copula_enable: bool = False
    fe_hybrid_orth_copula_n_bins: int = 20
    fe_hybrid_orth_dcor_enable: bool = False
    fe_hybrid_orth_dcor_n_sample: int = 500
    fe_hybrid_orth_hsic_enable: bool = False
    fe_hybrid_orth_hsic_kernel: Literal["rbf"] = "rbf"
    fe_hybrid_orth_hsic_n_sample: int = 500
    fe_hybrid_orth_jmim_enable: bool = False
    fe_hybrid_orth_jmim_n_bins: int = 10
    fe_hybrid_orth_tc_enable: bool = False
    fe_hybrid_orth_tc_n_bins: int = 10
    fe_hybrid_orth_cmim_enable: bool = False
    fe_hybrid_orth_cmim_n_bins: int = 10
    fe_hybrid_orth_auto_scorer_enable: bool = False
    fe_hybrid_orth_auto_scorer_n_boot: int = 5
    fe_hybrid_orth_ensemble_enable: bool = False
    fe_hybrid_orth_ensemble_aggregator: Literal["mean_rank", "borda_count", "reciprocal_rank", "mutual_top_k"] = "mean_rank"
    fe_hybrid_orth_ensemble_scorers: tuple = ("plug_in", "ksg", "copula", "dcor", "hsic")
    fe_hybrid_orth_meta_enable: bool = False
    fe_hybrid_orth_meta_force_scorer: Optional[str] = None
    fe_hybrid_orth_default_scorer: Literal[
        "plug_in", "cmim", "jmim", "tc", "ksg", "copula", "dcor", "hsic", "auto", "ensemble", "meta", "lasso", "elasticnet", "auto_oracle"
    ] = "plug_in"
    fe_hybrid_orth_extra_bases: tuple = ()
    fe_hybrid_orth_fourier_freqs: tuple = (1.0, 2.0)
    fe_hybrid_orth_fourier_powers: tuple = (1, 2)
    fe_hybrid_orth_spline_knots: int = 5
    fe_mi_greedy_enable: bool = False
    fe_mi_greedy_top_k: int = 5
    fe_mi_greedy_seed_cols_count: int = 5
    fe_mi_greedy_include_unary: bool = True
    fe_mi_greedy_include_binary: bool = True
    fe_mi_greedy_cmi_enable: bool = False
    fe_mi_greedy_cmi_top_k: int = 5
    fe_mi_greedy_cmi_seed_cols_count: int = 4
    fe_mi_greedy_cmi_min_gain: float = 0.005
    fe_kfold_te_enable: bool = True
    fe_kfold_te_cols: tuple = ()
    fe_kfold_te_folds: int = 5
    fe_kfold_te_smoothing: float = 10.0
    fe_kfold_te_stats: tuple = ("mean", "std", "skew", "kurt")
    fe_binned_numeric_agg_enable: bool = True
    fe_binned_numeric_agg_stats: tuple = ("mean", "std", "skew", "kurt")
    fe_binned_numeric_agg_nbins: int = 10
    fe_binned_numeric_agg_max_pairs: int = 64
    fe_binned_numeric_agg_redundancy_gate: bool = True
    fe_binned_numeric_agg_min_cmi_gain: float = 0.005
    fe_count_encoding_enable: bool = False
    fe_count_encoding_cols: tuple = ()
    fe_frequency_encoding_enable: bool = False
    fe_frequency_encoding_cols: tuple = ()
    fe_cat_num_interaction_enable: bool = False
    fe_cat_num_interaction_cat_cols: tuple = ()
    fe_cat_num_interaction_num_cols: tuple = ()
    fe_cat_num_interaction_folds: int = 5
    fe_cat_num_interaction_smoothing: float = 10.0
    fe_missingness_indicator_enable: bool = False
    fe_missingness_indicator_cols: tuple = ()
    fe_missingness_count_enable: bool = False
    fe_missingness_pattern_enable: bool = False
    fe_missingness_pattern_top_k: int = 5
    fe_pairwise_ratio_enable: bool = False
    fe_pairwise_ratio_cols: tuple = ()
    fe_pairwise_ratio_eps: float = 1e-09
    fe_pairwise_log_ratio_enable: bool = False
    fe_pairwise_log_ratio_cols: tuple = ()
    fe_grouped_delta_enable: bool = False
    fe_grouped_delta_group_col: str | None = None
    fe_grouped_delta_num_cols: tuple = ()
    fe_lagged_diff_enable: bool = False
    fe_lagged_diff_time_col: str | None = None
    fe_lagged_diff_value_cols: tuple = ()
    fe_lagged_diff_periods: tuple = (1, 2)
    fe_local_mi_gate: bool = True
    fe_local_mi_gate_top_k: int = 20
    fe_unified_second_pass_gate: bool = False
    fe_unified_second_pass_max_keep: int | None = None
    fe_unified_second_pass_min_gain: float = 0.005
    fe_grouped_agg_enable: bool = False
    fe_grouped_agg_stats: tuple = ("mean", "std", "min", "max", "nunique", "skew", "median")
    fe_grouped_agg_group_cols: tuple = ()
    fe_grouped_agg_num_cols: tuple = ()
    fe_grouped_agg_top_k: int = 10
    fe_composite_group_agg_enable: bool = False
    fe_composite_group_agg_key_sets: tuple = ()
    fe_composite_group_agg_max_arity: int = 2
    fe_composite_group_agg_stats: tuple = ("mean", "std", "count")
    fe_composite_group_agg_num_cols: tuple = ()
    fe_composite_group_agg_top_k: int = 10
    fe_grouped_quantile_enable: bool = False
    fe_grouped_quantile_quantiles: tuple = (0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95)
    fe_grouped_quantile_target_aware: bool = False
    fe_grouped_quantile_n_bins: int = 5
    fe_grouped_quantile_top_k: int = 8
    fe_grouped_quantile_group_cols: tuple = ()
    fe_grouped_quantile_num_cols: tuple = ()
    fe_cat_pair_enable: bool = False
    fe_cat_pair_min_interaction_info: float = 0.001
    fe_cat_pair_cat_cols: tuple = ()
    fe_cat_pair_top_k: int = 5
    fe_cat_triple_enable: bool = False
    fe_cat_triple_min_interaction_info: float = 0.001
    fe_cat_triple_cat_cols: tuple = ()
    fe_cat_triple_beam_width: int = 3
    fe_cat_triple_top_k: int = 3
    fe_numeric_decompose_enable: bool = False
    fe_numeric_decompose_precisions: tuple = (1, 0.1, 0.01, 0.001)
    fe_numeric_decompose_digits: tuple = (0, 1, 2)
    fe_numeric_decompose_n_boot: int = 10
    fe_numeric_decompose_top_k: int = 5
    fe_modular_enable: bool = False
    fe_modular_periods: tuple = (7, 12, 24, 30, 365)
    fe_modular_top_k: int = 6
    fe_discrete_structural_operators_enable: bool = True
    fe_pairwise_modular_enable: bool = True
    fe_pairwise_modular_top_k: int = 4
    fe_pairwise_modular_max_int_cols: int = 30
    fe_pairwise_modular_max_triple_cols: int = 20
    fe_integer_lattice_enable: bool = True
    fe_integer_lattice_top_k: int = 4
    fe_integer_lattice_max_int_cols: int = 30
    fe_row_argmax_enable: bool = True
    fe_row_argmax_top_k: int = 4
    fe_row_argmax_max_cols: int = 30
    fe_conditional_gate_enable: bool = True
    fe_conditional_gate_top_k: int = 4
    fe_conditional_gate_max_cols: int = 200
    fe_conditional_gate_k_gate: int = 8
    fe_conditional_gate_k_operand: int = 10
    fe_group_distance_enable: bool = False
    fe_group_distance_top_k: int = 6
    fe_group_distance_group_cols: tuple = ()
    fe_group_distance_num_cols: tuple = ()
    fe_rare_category_enable: bool = False
    fe_rare_category_cols: tuple = ()
    fe_rare_category_threshold: float = 0.01
    fe_rare_category_top_k: int = 10
    fe_conditional_residual_enable: bool = False
    fe_conditional_residual_cols: tuple = ()
    fe_conditional_residual_n_bins: int = 10
    fe_conditional_residual_top_k: int = 10
    fe_conditional_residual_max_pair_cols: int = 6
    fe_conditional_dispersion_enable: bool = True
    fe_conditional_dispersion_cols: tuple = ()
    fe_conditional_dispersion_n_bins: int = 10
    fe_conditional_dispersion_top_k: int = 10
    fe_conditional_dispersion_max_pair_cols: int = 6
    fe_conditional_quantile_rank_enable: bool = False
    fe_conditional_quantile_rank_cols: tuple = ()
    fe_conditional_quantile_rank_n_bins: int = 10
    fe_conditional_quantile_rank_top_k: int = 10
    fe_conditional_quantile_rank_max_pair_cols: int = 6
    fe_ordinal_pattern_enable: bool = False
    fe_ordinal_pattern_cols: tuple = ()
    fe_ordinal_pattern_k: int = 3
    fe_ordinal_pattern_max_cols_for_tuples: int = 5
    fe_ordinal_pattern_n_folds: int = 5
    fe_ordinal_pattern_smoothing: float = 10.0
    fe_ordinal_pattern_top_k: int = 5
    fe_random_fourier_enable: bool = False
    fe_random_fourier_cols: tuple = ()
    fe_random_fourier_m: int = 64
    fe_random_fourier_bandwidth: Optional[float] = None
    fe_random_fourier_max_cols_for_block: int = 8
    fe_random_fourier_top_k: int = 8
    fe_sir_direction_enable: bool = False
    fe_sir_direction_cols: tuple = ()
    fe_sir_direction_n_slices: int = 10
    fe_sir_direction_n_directions: int = 2
    fe_sir_direction_max_cols_for_block: int = 8
    fe_sir_direction_top_k: int = 2
    fe_lof_enable: bool = False
    fe_lof_cols: tuple = ()
    fe_lof_k: int = 20
    fe_lof_max_ref: int = 2000
    fe_lof_max_cols_for_block: int = 8
    fe_lof_top_k: int = 1
    fe_mahalanobis_density_enable: bool = False
    fe_mahalanobis_density_cols: tuple = ()
    fe_mahalanobis_density_max_cols_for_block: int = 20
    fe_mahalanobis_density_top_k: int = 1
    fe_offset_product_enable: bool = True
    fe_offset_product_cols: tuple = ()
    fe_offset_product_max_pair_cols: int = 6
    fe_offset_product_top_k: int = 3
    fe_offset_product_scan_rows: int = 100000
    fe_wavelet_enable: bool = True
    fe_wavelet_max_cols: Optional[int] = 100
    fe_wavelet_cols: tuple = ()
    fe_wavelet_max_scale: int = 3
    fe_wavelet_max_legs: int = 6
    fe_wavelet_top_k: int = 8
    fe_rankgauss_enable: bool = False
    fe_rankgauss_cols: tuple = ()
    fe_rankgauss_top_k: int = 10
    fe_temporal_agg_enable: bool = False
    fe_temporal_agg_entity_cols: tuple = ()
    fe_temporal_agg_value_cols: tuple = ()
    fe_temporal_agg_time_col: str | None = None
    fe_temporal_agg_stats: tuple = ("mean", "std", "count")
    fe_temporal_agg_windows: tuple = ()
    fe_temporal_agg_lags: tuple = (1,)
    fe_temporal_agg_top_k: int = 10
    retain_artifacts: bool = False
    retain_bins: bool = True
    partial_fit_decay: float = 0.0
    partial_fit_min_recompute: int = 100
    partial_fit_window: int | None = None
    fe_auto: bool = False
    stop_file: str = "stop"
    cache_dir: str | None = None
    embedding_passthrough: bool = True
    embedding_passthrough_detect_embeddings: bool = True
    embedding_passthrough_detect_text: bool = True
    usability_aware_lists: bool = False
    usability_w_linear: float = 0.85
    usability_w_universal: float = 0.5
    usability_feature_dtype: Any = None
    usability_max_base_features: int = 16
    usability_pool_kwargs: dict | None = None
    usability_greedy_kwargs: dict | None = None
    multioutput_strategy: Optional[str] = "union"
    fast_search_config: Any = None
    stability_config: Any = None
    synergy_config: Any = None
    group_aware_config: Any = None
    dcd_config: Any = None
    hybrid_orth_config: Any = None
