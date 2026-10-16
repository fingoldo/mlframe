"""A module-level function with no production call site enforces nothing.

A guard, a validator, a cleanup hook that nothing calls looks exactly like one that does: it is
defined, it is tested, it appears in review. The only difference is that whatever it enforces is not
enforced, and that difference shows up as the bug it was written to prevent rather than as anything
pointing here. A test calling it is not a production call site, and neither is an `__all__` entry or
a doctest -- that is how this class of dead control hides.

Baselined rather than gated: this repo has a large existing set, and the number that matters is
whether a NEW one appears. The baseline can only shrink -- an entry that gains a caller is drained
on the next refresh.

`ignore` takes bare names for the shapes this check cannot judge and must not guess at: entry points
the interpreter or a framework calls, and the lazy-module protocol.
"""

from __future__ import annotations

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
BASELINE = Path(__file__).resolve().parent / "_uncalled_functions_baseline.json"

#: Called by the interpreter, a framework or the import system rather than by name.
#: `__getattr__`/`__dir__` are PEP 562 lazy-module hooks; `main` is a console entry point;
#: `upgrade`/`downgrade` are alembic's.
_INTERPRETER_INVOKED = {"__getattr__", "__dir__", "main", "upgrade", "downgrade"}

#: Documented public API: users call these, nothing inside the package has to. Each is exported (``__all__`` or a package
#: re-export) and covered by tests of its own.
_PUBLIC_API = {
    "ewma",
    "trimmed_mean",
    "overlap",
    "mrr",  # core / metrics utilities
    "feature_matrix",
    "to_polars",
    "indicator",
    "inject_matched_probes",
    "probe_false_discovery_rate",  # mlframe.data
    "replay_report",
    "spec_hash_for",
    "assert_oracle_agrees_with_dit",
    "reference_crosscheck",  # benchmark-bed tooling
    "create_aggregated_features",
    "optimize_pipeline_by_gridsearch",
    "get_full_classifier_name",
    "predict_mlframe_models_suite",
    "serving_base_from_columns",
    "blend",  # blend is exported as pseudo_bma_blend
    "bootstrap_metric_clustered",  # single-metric form, re-exported from mlframe.evaluation.bootstrap
    "track_package",  # extension hook: a downstream package opts its numba kernels into the dependency-aware cache stamp
    "create_mlflow_run_label",
    "get_or_create_mlflow_run",
    "embed_website_to_mlflow",  # user-facing helpers of mlframe.integrations.mlflow, each pinned by tests/integrations/test_mlflow.py
    "reset_throttle_counts",  # documented in mlframe.utils.log_throttle.__all__; the test-isolation reset for every throttle key, used by 30+ tests
    "clip_to_quantiles_hard",  # documented preset for choosing a target transform (1%/99% quantile clip)
    "clip_to_quantiles_winsor_quantile",  # documented preset for choosing a target transform (soft winsorise)
    "set_wavedec_dispatch_threshold",  # documented tuning knob for the wavelet single/batched dispatch; a user sets it
    "reset_openmp_smoke",  # compatibility alias for the old hnswlib API; a no-op kept so old imports keep working
    "operand_uplift_null_bar",  # exported through the module's __all__ (feature_engineering_pairs operand floor); public surface
    "set_dtw_dispatch_threshold",  # documented tuning knob for the DTW GPU/CPU crossover; a user sets it
    "get_metric_direction",  # public accessor for a metric's registered direction
    "list_registered",  # public introspection of the metric registry
}

#: Re-exported from a package ``__init__`` (``from x import f`` or ``__all__``): the package's public surface for library
#: users, so no production call site is expected. Classified as public API; new names must be added here with the export.
_REEXPORTED_PUBLIC_API = {
    "get_stage1_survivors",  # listed in the _shap_proxy_prefilter module __all__: public accessor for the stage-1 survivor mask
    "train_postcalibrators",  # re-exported by the calibration.post facade (from ._post_train_calibrators import ...)
    "_baseline_inputs_hash",
    "_detect_fourier_freq_for_col",
    "_ewma_compute_batched",
    "_frac_diff_inverse_compute_batched",
    "_knuth_log_posterior",
    "_ksg_mi_1d",
    "_polars_df_has_null_in_categorical",
    "_pre_pipeline_cache_clear",
    "_pre_pipeline_cache_set",
    "_quantile_edges_2d_njit_partition",
    "_residualise",
    "_resolve_model_spec",
    "acquire_provider",
    "add_anchor_extrapolation_features",
    "add_fast_rolling_stats",
    "add_ohlcv_ratios_rlags",
    "add_ohlcv_ta_indicators",
    "add_sentinel_missing_count_feature",
    "additive_interaction_diagnostic",
    "adversarial_rebin_categorical",
    "adversarial_stochastic_blend",
    "adversarial_validation_feature_audit",
    "align_feature_direction",
    "anchor_density_features",
    "anchor_ewm_features",
    "anchor_quadratic_extrapolation_features",
    "anchor_residual_rmse_features",
    "apply_calendar_anomaly_flag",
    "apply_cat_cooccurrence_svd",
    "apply_decision_threshold",
    "apply_feature_direction",
    "apply_gaussian_power_transform",
    "apply_group_bias_correction",
    "apply_group_zero_sum_constraint_multi",
    "apply_logical_constraints",
    "apply_outlier_policy",
    "apply_prediction_band_correction",
    "assembled_column_names",
    "assert_estimators_disjoint",
    "auc_ci",
    "audit_target_over_time",
    "augment_temporal_drift",
    "augment_to_match_test_distribution",
    "auto_aware_gate",
    "backtest_override",
    "bayesian_alpha_fit",
    "bayesian_alpha_fit_bootstrap",
    "benchmark_polyeval_cpu_backends",
    "binned_unique_count",
    "boolean_pair_interactions",
    "borda_fuse",
    "build_meta_stack_ensemble",
    "build_panel_sequence_tensor",
    "cascade_select_stable",
    "cat_cooccurrence_svd_with_recipes",
    "cb_native_only",
    "check_cpu_flag",
    "check_pairwise_score_correlation",
    "clear_numeric_code_cache",
    "cmi_or_csu",
    "cohen_kappa_binary",
    "collapse_predictions_by_group",
    "collapse_rare_categories",
    "collect_and_score",
    "compare_selectors",
    "compose_fuzzy_membership_figure",
    "compose_spectral_embedding_figure",
    "composite_model_card",
    "composite_report",
    "compute_active_virtual_features",
    "compute_adaptive_bandwidth_attention",
    "compute_adasyn_smote_features",
    "compute_adversarial_flip_features",
    "compute_anchor_attention",
    "compute_anomaly_score_features",
    "compute_apriori_itemsets_features",
    "compute_autoencoder_features",
    "compute_aux_mlp_features",
    "compute_auxiliary_feature_prediction_features",
    "compute_band_conditional_anchor_features",
    "compute_baseline_disagreement_balanced_features",
    "compute_baseline_disagreement_smote_features",
    "compute_baseline_disagreement_v2_features",
    "compute_baseline_surprise_features",
    "compute_bgm_clustered_smote_features",
    "compute_bgmm_density_ratio_features",
    "compute_bgmm_dual_class_features",
    "compute_bgmm_multiscale_features",
    "compute_bgmm_quantile_bands_features",
    "compute_bgmm_virtual_features",
    "compute_bidir_residual_band_features",
    "compute_boosted_attention",
    "compute_boosting_leaf_features",
    "compute_borderline_smote_features",
    "compute_class_balanced_hard_row_features",
    "compute_class_conditional_anchor_attention",
    "compute_class_distance_features",
    "compute_class_mahalanobis_features",
    "compute_cluster_smote_features",
    "compute_conformal_coverage_failure_features",
    "compute_conformal_locally_adaptive_features",
    "compute_counterfactual_substitution_features",
    "compute_cross_feature_reconstruction_features",
    "compute_cutmix_features",
    "compute_decision_region_depth_features",
    "compute_denoising_autoencoder_features",
    "compute_density_ratio_features",
    "compute_density_weighted_smote_features",
    "compute_diffusion_noise_features",
    "compute_disagreement_band_features",
    "compute_distributional_moments_features",
    "compute_fca_closed_concepts_features",
    "compute_fisher_weighted_residual_features",
    "compute_focal_lgb_features",
    "compute_geodesic_kgraph_features",
    "compute_gradient_direction_agreement_features",
    "compute_hard_row_attention_features",
    "compute_ib_baseline_codes_features",
    "compute_inducing_attention_features",
    "compute_jackknife_endpoint_stability_features",
    "compute_ks_shift_features",
    "compute_lda_projection_features",
    "compute_local_classifier_features",
    "compute_local_curvature_features",
    "compute_local_density_gradient_features",
    "compute_local_intrinsic_dim_features",
    "compute_local_lift_features",
    "compute_local_linear_attention",
    "compute_mdl_binning_pairwise_features",
    "compute_mixup_boundary_features",
    "compute_multi_baseline_hard_row_features",
    "compute_multi_temp_band_attention_features",
    "compute_multi_temp_cbhr_features",
    "compute_multi_temp_residual_band_features",
    "compute_multi_temperature_attention",
    "compute_multi_threshold_ordinal_features",
    "compute_multiscale_rate_features",
    "compute_multiscale_smote_features",
    "compute_nca_projection_features",
    "compute_neighbor_aggregate_features",
    "compute_numaggs_parallel",
    "compute_pairwise_kl_features",
    "compute_per_class_spectral_attention",
    "compute_per_column_rff",
    "compute_performer_attention_features",
    "compute_persistence_diagram_features",
    "compute_positional_encoding",
    "compute_pred_augmented_attention",
    "compute_prediction_band_attention_features",
    "compute_predictive_info_delta_features",
    "compute_pseudo_smote_features",
    "compute_pure_pos_smote_features",
    "compute_quantile_band_attention_features",
    "compute_quantile_spread_fan_features",
    "compute_residual_attention",
    "compute_residual_band_attention_features",
    "compute_residual_stratified_distance_features",
    "compute_rf_proximity_attention",
    "compute_rff_features",
    "compute_robustness_budget_features",
    "compute_row_level_then_average_predictions",
    "compute_sign_residual_baseline_features",
    "compute_signed_residual_band_features",
    "compute_smote_distance_features",
    "compute_spectral_attention",
    "compute_stacked_quantile_neighbours",
    "compute_stacked_row_attention",
    "compute_target_kmeans_codebook_features",
    "compute_target_quantile_attention",
    "compute_test_likeness",
    "compute_tree_path_boolean_features",
    "compute_variance_baseline_features",
    "config_from_dict",
    "control_difference_augment",
    "create_experiment",
    "create_ohlcv_wholemarket_features",
    "create_redundant_continuous_factor",
    "create_windowed_features",
    "cross_validate_asymmetric_rescale",
    "cumulative_gains_curve",
    "custom_collate_fn",
    "cusum_features",
    "data_banzhaf",
    "delong_test",
    "describe_available_panels",
    "detect_correlated_label_groups",
    "detect_expanding_window_feature_leakage",
    "detect_pair_symmetry",
    "detect_sentinel_values",
    "dfa_alpha2_quadratic",
    "diagnose_error_accumulation",
    "discover_and_wrap",
    "discover_and_wrap_classification",
    "discover_logical_constraints",
    "discover_structure",
    "discretize_sklearn",
    "distribution_matching_subset_search",
    "dro_reweight_fit",
    "drop_noninformative_vs_reference",
    "drop_rare_features",
    "drop_raw_after_embedding",
    "dual_optimizer_weight_blend",
    "embedding_only",
    "enrich_ensemble_preds_with_numaggs",
    "ensemble_probabilistic_predictions_streaming",
    "entity_diff_features",
    "entity_inter_event_features",
    "evaluate_model",
    "ewma_multi_alpha_features",
    "ewma_residual",
    "export_serving_spec",
    "extract_sequences_chunked",
    "f_beta_score",
    "fast_adjusted_r2_score",
    "fast_cv_rmse",
    "fast_epsilon_band_accuracy",
    "fast_explained_variance",
    "fast_mape_mean",
    "fast_mase",
    "fast_mean_bias_error",
    "fast_mrae",
    "fast_multiclass_confusion_metrics_block",
    "fast_nash_sutcliffe",
    "fast_pearson_corr",
    "fast_percent_better",
    "fast_regression_metrics_block",
    "fast_rel_mae",
    "fast_smape",
    "fast_tweedie_deviance",
    "fast_wmape",
    "find_change_points",
    "fit_group_bias_correction",
    "fit_operand_prewarp",
    "fit_with_unlabeled",
    "flag_subgroup_only_feature_overfit_risk",
    "flatten_masked_multilabel",
    "format_unified_target_verdict_table",
    "frac_diff",
    "friedman_h_statistic",
    "fuzzy_entity_group_features",
    "fuzzy_partition_encode",
    "g_mean_binary",
    "gains_table",
    "gaussian_power_transform_search",
    "general_acf",
    "generate_modelling_data",
    "generate_pareto_artifact",
    "geometric_weight_blend",
    "get_experiment_routes",
    "get_experiments",
    "get_handler_spec_for_axis",
    "get_render_failure_stats",
    "get_sapp_dataset",
    "get_simple_stats_names",
    "gini_from_auc",
    "gmm_bic_membership_features",
    "graph_neighbor_aggregate",
    "graph_spectral_feature_names",
    "graph_spectral_features",
    "graph_structural_features",
    "greedy_backward_ensemble_elimination",
    "group_causal_attention_mask",
    "heterogeneous_relevance_vote",
    "hgb_max_features_cap",
    "hill_climb_ensemble",
    "holiday_calendar_features",
    "holiday_name_target_encode_cross_locale",
    "imputation_sensitivity_check",
    "impute_with_missing_indicator",
    "interval_quality_summary",
    "inverse_distance_weighted_aggregate",
    "is_catboost_model",
    "is_linear_model",
    "is_within_noise_band",
    "kalman_filter_posterior_1d",
    "kalman_smoother_posterior_1d",
    "knn_aggregate",
    "knn_gradient_features",
    "knn_graph_edges",
    "knn_label_dispersion_features",
    "knn_shapley_multi_target_regression",
    "knn_shapley_multilabel",
    "knn_within_bucket_aggregate",
    "knockoff_importance",
    "last_k_distinct_states_with_durations",
    "latent_parameter_recovery_features",
    "lift_curve",
    "link_prediction_features",
    "list_transforms",
    "load_mlframe_suite",
    "load_serving_spec",
    "local_density_features",
    "local_linear_detrend",
    "lock_differences",
    "magnitude_sample_weight",
    "make_composite_regressor",
    "make_ensemble_outlier_scores",
    "make_per_column_specs",
    "make_purged_cv",
    "masked_multilabel_logloss_objective",
    "melt_to_long_gbm_features",
    "merge_perticker_and_wholemarket_features",
    "mi_or_su",
    "mi_or_su_from_classes",
    "model_card_verdict",
    "monotonic_deviation_stability_filter",
    "multi_decomposition_feature_bank",
    "multi_scale_hurst",
    "multi_window_aggregate",
    "multifractal_dfa",
    "nadaraya_watson_smooth",
    "native_xgboost_gpu_shap_contribs",
    "nested_ma_decompose",
    "nested_ma_decompose_chain",
    "null_importance_filter",
    "odds_ratio_combine",
    "optimal_ordinal_cutpoints",
    "optimise_pair_multimode",
    "optimize_composite",
    "optimize_group_blend_weight",
    "optimize_persistence_floor",
    "orth_scoring_memo_scope",
    "outlier_cap_or_missing",
    "overfit_verdict",
    "pair_vi",
    "particle_filter_posterior",
    "per_fold_majority_accept",
    "per_group_behavioral_stability",
    "per_group_cum_reduce",
    "per_group_nadaraya_watson_smooth",
    "per_group_nth",
    "per_group_recency_weighted_mean",
    "per_group_recency_weighted_mode",
    "per_group_rolling_reduce",
    "plugin_mi_classif_dispatch",
    "plugin_mi_classif_fast",
    "polars_dynamic_window_aggregate",
    "positions_within_group",
    "postprocess_candidates",
    "predict_quantile_ensemble",
    "predicted_group_aggregate_feature",
    "predictions_time_instability",
    "predictor_consensus_entropy",
    "predictor_consensus_trimmed_stats",
    "predictor_disagreement_features",
    "predictor_max_pairwise_distance",
    "predictor_outlier_signature",
    "predictor_quantile_spread",
    "predictor_top2_mode_gap",
    "predictor_weighted_consensus",
    "prewarm",
    "prewarm_fs_cupy_kernels",
    "process_nans",
    "process_nulls",
    "provider_status",
    "pseudo_bma_weights",
    "quantile_normalize_per_group",
    "radius_aggregate",
    "randomize_as_of_lag",
    "rank_percentile_transform",
    "rank_subgroup_feature_overfit_risk",
    "ranked_probability_score",
    "read_experiment",
    "read_glove_embeddings",
    "read_route",
    "recency_weighted_rolling_mean",
    "recency_weighted_rolling_std",
    "recommend_transform_candidates",
    "regime_conditioned_median_fill",
    "regime_headroom_map",
    "register_model_axis_support",
    "remediate_drifting_features",
    "render_regime_headroom_map",
    "reset_capability_cache",
    "reset_render_failure_stats",
    "residency_audit",
    "rff_apply_state",
    "ridge_coefficient_prefilter",
    "rolling_dfa_alpha",
    "rolling_dominant_freq_idx",
    "rolling_extrema_density",
    "rolling_hf_lf_ratio",
    "rolling_higuchi_fd",
    "rolling_hurst",
    "rolling_integral_above_baseline",
    "rolling_longest_monotone_run",
    "rolling_mean_abs_d2",
    "rolling_n_peaks",
    "rolling_n_troughs",
    "rolling_periodicity_score",
    "rolling_quantile_spread",
    "rolling_shannon_entropy_binned",
    "rolling_spectral_bandwidth",
    "rolling_spectral_centroid",
    "rolling_spectral_entropy",
    "rolling_spectral_flatness",
    "rolling_spectral_flux",
    "rolling_spectral_rolloff",
    "rolling_target_correlation_tracker",
    "rolling_total_variation",
    "rolling_zero_crossings",
    "route_spec_column_by_group",
    "row_wise_extremality_index",
    "rows_until_next_anchor",
    "rrf_ensemble",
    "rrf_fuse",
    "run_efs",
    "run_grid",
    "run_pysr_fe",
    "safely_compute_mps",
    "scan_temporal_leak",
    "seed_averaged_fold_scores",
    "select_column_transforms",
    "select_features_fdr",
    "select_features_noise_floor",
    "select_outlier_threshold",
    "sequence2vec_entity_features",
    "shapley_blend",
    "shared_attribute_edges",
    "should_promote",
    "show_mps_regions",
    "showcase_pycaret_datasets",
    "sibling_group_cold_start_fill",
    "sort_df_by_time_column",
    "specificity_npv_fpr_fnr",
    "spiegelhalter_z",
    "stability_select_specs",
    "stack_relational_features",
    "stochastic_bandit_selection",
    "stochastic_bandit_selection_ensemble",
    "stream_key",
    "support_mask_from_selector",
    "tfidf_only",
    "tfidf_svd_entity_embedding",
    "tmc_shapley",
    "top_k_accuracy",
    "train_automl_models_suite",
    "train_test_support_screen",
    "training_sample_weight_from_valuation",
    "unanimous_permutation_prune",
    "update_routes_audiences",
    "use_dcd",
    "use_group_mi",
    "variance_gated_pairwise_diff",
    "varying_size_top_k_subsets",
    "wait_prewarm",
    "windowed_edge_aggregate_diff",
    "winkler_score_per_group",
    "winkler_score_per_row",
    # exported through the defining module's __all__ (triage 2026-10-16): public surface, no in-repo caller expected
    "_load_model_cache_clear",
    "_marginal_mm_mi",
    "_mark_kaleido_persistent_burned",
    "_optin_keep_frac",
    "adversarial_frame",
    "analyse_and_clean_features",
    "apply_features_cleaning",
    "apply_lof_block",
    "apply_mahalanobis_density",
    "apply_random_fourier_block",
    "apply_sir_direction",
    "audit_target_maturity",
    "batch_mi_with_noise_gate_cupy_v1",
    "bernstein_design",
    "best_on_calibration_mi",
    "bin_smooth",
    "braun_blanquet",
    "bur_term",
    "cache_artefact",
    "categorical_series",
    "chao_shen_entropy",
    "chao_shen_mi",
    "check_feature_direction_stability",
    "clean_mlframe_model",
    "clear_cmi_xc_resident_cache",
    "clusterize",
    "cmim_score",
    "composite_group_agg_with_recipes",
    "compute_gp_smoothed_features",
    "compute_inverse_frequency_class_weights",
    "compute_naive_outlier_score",
    "compute_outlier_detector_score",
    "compute_simple_stats_numba_arr",
    "constant_series",
    "create_ts_features_parallel",
    "crps_from_quantiles",
    "cv_stability_check",
    "delta_weights",
    "detect_conditional_gate",
    "detect_integer_lattice",
    "detect_pairwise_modular",
    "detect_regime_changepoints",
    "detect_row_argmax",
    "dice",
    "discover_logical_constraints_soft",
    "dpmine_mi",
    "easy_ensemble_fit_predict",
    "edges_quantile",
    "ensure_composite_collinear_tuning",
    "ensure_composite_corr_tuning",
    "ensure_per_member_tuning",
    "fit_basis_mi",
    "fit_composite_similarity",
    "flatten_masked_multilabel_class_weights",
    "fragment_df_on_ram_usage_increase",
    "gap_fraction",
    "gegenbauer_design",
    "generate_cat_pair_crosses",
    "generate_cat_triple_crosses",
    "generate_market_price",
    "geometric_median",
    "get_ts_window_name",
    "grouped_agg_with_recipes",
    "high_card_text_column",
    "high_card_text_series",
    "hull_ma_deviation",
    "hull_moving_average_multi",
    "hybrid_orth_mi_elasticnet_fe_with_recipes",
    "hybrid_orth_mi_lasso_fe_with_recipes",
    "hybrid_orth_mi_meta_fe",
    "inf_heavy_float_series",
    "jacobi_design",
    "jmim_score",
    "joint_synergy_mi",
    "ksg_mi_dispatch",
    "kulczynski",
    "lgb_dataset_reuse_capable",
    "lof_scores",
    "log_methods",
    "mah_mi",
    "mahalanobis_density_feature",
    "map_at_k",
    "median_safe",
    "minde_mi",
    "ndcg_at_k",
    "ochiai",
    "ordinal_pattern_lexicographic_rank",
    "parse_diff_basis_col_name",
    "per_cell_stats_bincount",
    "plot_alpha_stability",
    "plot_interval_coverage",
    "plot_interval_width_vs_x",
    "plot_linear_fit",
    "plot_mi_gain_with_ci",
    "plot_per_family_disagreement",
    "plot_per_fold_tiny_rmse",
    "plot_predictions_vs_actual",
    "plot_qq",
    "plot_reliability_diagram",
    "post_calibrate_model",
    "prepare_df_for_xgboost",
    "prod_like_frame",
    "prod_like_frame_small",
    "proportions_significantly_different",
    "proximity_to_distance",
    "purged_oof_holdout",
    "quantile_safe",
    "quantile_summary",
    "random_fourier_features",
    "rank_subpopulation_drift_severity",
    "recency_step_weight",
    "recency_weights",
    "reconstruct_pseudo_group_ids",
    "reject_outliers",
    "renyi_alpha_cmi",
    "required_n_for_proportion",
    "reset_cache",
    "rf_outlier_measure",
    "rf_proximity_matrix",
    "robust_mean_mestimator",
    "scan_conditional_gate",
    "scan_integer_lattice_pairs",
    "scan_row_argmax",
    "score_pair_mi",
    "select_best_iteration_by_aggregate_cv",
    "select_true_last_standardized",
    "sequence2vec_transform_new_entities",
    "sir_direction_features",
    "smoothed_target_encode_column",
    "sparse_null_series",
    "standalone_diagnostics",
    "suggest_non_outlying_data_indices",
    "to_pandas_or_array",
    "total_correlation",
    "triple_interaction_information",
    "tversky",
    "upper_tail_coefficient",
    "usable_text_features",
    "visualize_ml_metric_by_time",
    "wilson_interval",
    "winsorized_mean",
    "wrap_with_logging",
    "xgb_dmatrix_reuse_capable",
    "compute_mutual_info_regression",  # exported through the defining module __all__ (owner decision: export)
    "parse_routing_col_name",  # exported through the defining module __all__ (owner decision: export)
    "npnbArrayMinMax",  # re-exported by mlframe.core (its package namespace already exposes it)
    "topk_by_partition",  # re-exported by mlframe.core (its package namespace already exposes it)
    "arrayCountingArgSort",  # re-exported by mlframe.core (its package namespace already exposes it)
    "arrayCountingArgSortAndUniqueValues",  # re-exported by mlframe.core (its package namespace already exposes it)
    "arrayCountingSort",  # re-exported by mlframe.core (its package namespace already exposes it)
    "get_dist_percentage_span_for_sd",  # re-exported by mlframe.core (its package namespace already exposes it)
    "box_cox_plus_c",  # re-exported by mlframe.estimators (its package namespace already exposes it)
    "identity",  # re-exported by mlframe.estimators (its package namespace already exposes it)
    "inv_box_cox_plus_c",  # re-exported by mlframe.estimators (its package namespace already exposes it)
    "inv_log_plus_c",  # re-exported by mlframe.estimators (its package namespace already exposes it)
    "log_plus_c",  # re-exported by mlframe.estimators (its package namespace already exposes it)
    "qubed",  # re-exported by mlframe.estimators (its package namespace already exposes it)
    "evaluate_estimators",  # re-exported by mlframe.evaluation (its package namespace already exposes it)
    "plot_pr_curve",  # re-exported by mlframe.evaluation (its package namespace already exposes it)
    "categorize_1d_array",  # re-exported by mlframe.feature_selection.filters (its package namespace already exposes it)
    "compute_shap_on_cv",  # re-exported by mlframe.inference (its package namespace already exposes it)
    "get_models_raw_predictions",  # re-exported by mlframe.inference (its package namespace already exposes it)
    "read_trained_models",  # re-exported by mlframe.inference (its package namespace already exposes it)
    "fast_aucs_per_group",  # re-exported by mlframe.metrics (its package namespace already exposes it)
    "make_brier_precision_scorer",  # re-exported by mlframe.metrics (its package namespace already exposes it)
    "numba_warmup",  # re-exported by mlframe.metrics (its package namespace already exposes it)
    "js_divergence",  # re-exported by mlframe.metrics (its package namespace already exposes it)
    "kl_divergence",  # re-exported by mlframe.metrics (its package namespace already exposes it)
    "ks_distribution_distance",  # re-exported by mlframe.metrics (its package namespace already exposes it)
    "population_stability_index",  # re-exported by mlframe.metrics (its package namespace already exposes it)
    "wasserstein_1d",  # re-exported by mlframe.metrics (its package namespace already exposes it)
    "compute_batch_rmse",  # re-exported by mlframe.metrics (its package namespace already exposes it)
    "set_gpu_thresholds",  # re-exported by mlframe.metrics (its package namespace already exposes it)
    "fast_multilabel_classification_metrics_block",  # re-exported by mlframe.metrics (its package namespace already exposes it)
    "dcg_at_k",  # re-exported by mlframe.metrics (its package namespace already exposes it)
    "expected_reciprocal_rank",  # re-exported by mlframe.metrics (its package namespace already exposes it)
    "hit_at_k",  # re-exported by mlframe.metrics (its package namespace already exposes it)
    "precision_at_k",  # re-exported by mlframe.metrics (its package namespace already exposes it)
    "ProbaScoreProxy",  # re-exported by mlframe.metrics (its package namespace already exposes it)
    "optimize_finite_onedimensional_search_space",  # re-exported by mlframe.models (its package namespace already exposes it)
    "mc_dropout_predict",  # re-exported by mlframe.training (its package namespace already exposes it)
    "predictive_entropy",  # re-exported by mlframe.training (its package namespace already exposes it)
    "duan_log_smearing_factor",  # re-exported by mlframe.training (its package namespace already exposes it)
    "smearing_predict",  # re-exported by mlframe.training (its package namespace already exposes it)
    "tta_predict",  # re-exported by mlframe.training (its package namespace already exposes it)
    "tta_predict_spread",  # re-exported by mlframe.training (its package namespace already exposes it)
    "mean_election",  # exported by mlframe.votenrank through its lazy export map (leaderboard._rules)
    "plurality_election",  # exported by mlframe.votenrank through its lazy export map (leaderboard._rules)
    "borda_election",  # exported by mlframe.votenrank through its lazy export map (leaderboard._rules)
    "dowdall_election",  # exported by mlframe.votenrank through its lazy export map (leaderboard._rules)
    "copeland_election",  # exported by mlframe.votenrank through its lazy export map (leaderboard._rules)
    "condorcet_election",  # exported by mlframe.votenrank through its lazy export map (leaderboard._rules)
    "baldwin_election",  # exported by mlframe.votenrank through its lazy export map (leaderboard._rules)
    "minimax_election",  # exported by mlframe.votenrank through its lazy export map (leaderboard._rules)
    "optimality_gap_election",  # exported by mlframe.votenrank through its lazy export map (leaderboard._rules)
    "threshold_election",  # exported by mlframe.votenrank through its lazy export map (leaderboard._rules)
    "spearmanr_batched",  # exported by mlframe.metrics (direct-use public surface)
    "spearmanr_batched_dispatch",  # exported by mlframe.metrics (direct-use public surface)
    "set_spearmanr_dispatch_threshold",  # exported by mlframe.metrics (direct-use public surface)
    "bootstrap_auc_distribution_parallel",  # exported by mlframe.metrics (direct-use public surface)
    "composite_value_report",  # exported by mlframe.training.composite (__all__)
    "mondrian_ood_summary",  # exported by mlframe.training.composite (__all__)
    "concordance_index",  # exported by mlframe.training.composite (__all__)
    "check_and_rediscover",  # exported by mlframe.training.composite (__all__)
    "detect_cat_columns",  # exported by mlframe.training.composite (__all__)
    "ksg_mi_pair",  # exported by mlframe.feature_selection.filters (__all__)
    "ksg_mi_with_significance",  # exported by mlframe.feature_selection.filters (__all__)
    "nsb_mi",  # exported by mlframe.feature_selection.filters (__all__)
    "optimal_bin_edges",  # exported by mlframe.feature_selection.filters (__all__)
    "apply_bin_edges",  # exported by mlframe.feature_selection.filters (__all__)
    "compose_pair_fe",  # exported by mlframe.feature_selection.filters (__all__)
    "validate_pair_fe_cv",  # exported by mlframe.feature_selection.filters (__all__)
    "numpy_rng_scope",  # exported by its package (__all__ / lazy export map)
    "curated_fe_pipelines",  # exported by its package (__all__ / lazy export map)
    "recommended_curated_fe_names",  # exported by its package (__all__ / lazy export map)
    "wavedec_dispatch",  # exported by its package (__all__ / lazy export map)
    "wavelet_denoise",  # exported by its package (__all__ / lazy export map)
    "benjamini_hochberg_reject",  # exported lazily through mlframe.training
    "list_registered_specs",  # exported lazily through mlframe.training
    "unregister_metric",  # exported lazily through mlframe.training
    "calibration_adjusted_score",  # exported lazily through mlframe.training
    "recalibration_rmse_gain",  # exported lazily through mlframe.training
    "rank_transforms_by_waic",  # exported lazily through mlframe.training
    "score_trivial_baselines",  # exported through mlframe.feature_selection.filters
    "select_usability_aware_features",  # exported through mlframe.feature_selection.filters
}

#: Offline synthetic-data generators (calibration probability samplers): called by benchmarks and tests to build
#: controlled inputs, never by the shipped pipeline. The gate scans production files only, so a benchmark call would not
#: clear the entry; the reason is recorded here instead.
#: Cache-reset helpers that tests call to isolate a memoised availability probe (documented in each docstring).
_TEST_UTILITIES = {
    "reset_catboost_available_cache",
    "reset_cluster_su_gpu_probe",
    "reset_gpu_model_available_cache",
    "is_numba_active",  # fast-path probe that tests assert on; not a production call site
    "clear_mah_y_binning_cache",  # test-isolation reset of the MAH binning cache
    "clear_feature_matrix_cache",  # test-isolation reset of the error-analysis feature-matrix cache
    "clear_shared_error_prep",  # test-isolation reset of the shared error-prep cache
    "_clear_gate_cache",  # test-isolation reset of the ensemble gate cache
    "reset_gpu_argsort_probe",  # test-isolation reset of the GPU argsort probe
    "reset_fe_family_wall",  # test-isolation reset of the FE family wall-clock accumulator
    "_reset_fe_gpu_pool_limit_flag",  # test-isolation reset of the GPU pool-limit flag
    "_reset_for_tests",  # test-isolation reset of the kernel-tuning state
    "_xgb_cache_clear",  # test-isolation reset of the xgboost shim cache
    "_lib_versions_cache_clear",  # test-isolation reset of the library-version cache
    "cache_bytes",  # recorded-size readout for tests and diagnostics of the CatBoost pool budget
    "is_triton_available",  # Triton availability probe for callers and tests
    "reset_gpu_probe",  # test hook that re-probes GPU availability after monkeypatching
    "reset_cmi_forder_cache",  # test/free-memory reset of cached F-order copies
    "reset_gpu_metrics_probe",  # test hook that re-probes GPU availability for metrics
    "_lgb_cache_clear",  # test-isolation reset of the LightGBM dataset cache
    "_is_windows",  # single chokepoint for OS detection, monkeypatched by tests
    "_ranknet_pair_cache_clear",  # test-only reset of the RankNet pair cache
    "set_default_cache",  # test hook that replaces the process-wide cache singleton
}

_OFFLINE_GENERATORS = {
    "generate_probs_from_outcomes",
    "generate_similar_probs",
    "generate_similar_probs_by_ranking",
    "generate_similar_probs_logit_space",
    "generate_similar_probs_random_walk",
    "assess_prediction_band_stability",
    "optimize_persistence_floor_per_class",  # benchmark-only calibration helpers
    "ListAllSkLearnClassifiers",  # interactive discovery print, run by hand
    "MakeSureBlasAndLaPackAreInstalled",  # interactive BLAS/LAPACK diagnostic, run by hand
    "show_sys_ram_usage",  # interactive RAM diagnostic, run by hand
    "_probability_only",  # difficulty-sweep exploration helper for the synthetic generator, run by hand
    "get_tracker_table",  # loads experiment-impact-tracker cost data for offline analysis
    "agg_pipeline_metric",  # hand-run analysis of CV results across pipelines
    "compute_ml_metrics",  # hand-run scoring helper for CV folds, outside the shipped pipeline
    "create_dummy_lagged_predictions",  # synthetic data generator for hand-run demos
    "visualize_prediction_vs_truth",  # hand-run plotting helper for pipeline results
    "plot_beautified_lift",  # hand-run plotting helper for lift curves
    "plot_roc_curve",  # hand-run plotting helper for ROC curves
    "analyze_xgboost_model",  # hand-run diagnostic on a fitted xgboost model
    "explain_top_feature_importances",  # hand-run explanation helper for importances
    "rank_features_by_split_gain",  # hand-run ranking helper for split gains in a report
    "crows_pipeline",  # fairness experiment pipeline for the votenrank study, run by hand
    "stereo_pipeline",  # fairness experiment pipeline for the votenrank study, run by hand
    "winobias_pipeline",  # fairness experiment pipeline for the votenrank study, run by hand
    "naive_gpt2_score",  # fairness baseline score for the votenrank study, run by hand
    "naive_t5_score",  # fairness baseline score for the votenrank study, run by hand
    "naive_masking_score",  # fairness baseline score for the votenrank study, run by hand
    "count_and_plot",  # stability experiment plotting helper, run by hand
    "create_exp_pic",  # stability experiment plotting helper, run by hand
    "evaluate_grouped",  # hand-run demo that reports per-group accuracy for a sample model
    "load_data",  # example-dataset loader for trying the package out by hand
    "get_fe_family_max_concurrency",  # diagnostic reader of the peak FE family concurrency, read by hand
    "phase_ram_snapshot",  # per-phase RAM delta readout for training diagnostics
    "record_phase",  # manual timing backfill for training diagnostics
    "agreement_rate",  # votenrank study metric, run by hand
    "best_unary_transform",  # baseline comparator for the FE benchmarks, run by hand
    "score_triplet_baselines",  # baseline comparator for the triplet FE benchmarks, run by hand
    "replay_cv_results",  # visualisation of a stored CV dump, run by hand
    "example_highd",  # demo that a sliced-MI estimate is near zero on independent samples, run by hand
    "compare_ensembles",  # ensemble comparison plot for hand-run analysis
    "preprocess_glue",  # cleaning of a scraped GLUE leaderboard, run by hand
    "preprocess_sglue",  # cleaning of a scraped SuperGLUE leaderboard, run by hand
    "preprocess_value",  # cleaning of a scraped VALUE leaderboard, run by hand
    "compute_iia",  # votenrank Monte-Carlo IIA study, run by hand
    "kendall_tau",  # votenrank study correlation against the AM ranking, run by hand
}

#: Kernel versions kept deliberately when a faster one replaced them (the repository keeps every version for A/B and
#: fallback): the typed-list DWT pair, the DFS MDLP twin of the BFS default, the CUDA DTW diagonal step, the numba column-
#: stats kernel the benchmark compares against.
_RETAINED_KERNELS = {
    "_topn_fixed_r",
    "_topn_fixed_r_parallel",  # keep-all-kernels rule: serial and chunk-parallel top-N kernels kept side by side (A/B), see _shap_proxy_search docstring
    "_wavedec_numba_typedlist",
    "_waverec_numba_typedlist",
    "_mdlp_recurse_validated",
    "_numba_cuda_diagonal_step",
    "_col_stats_float_numba_kernel",
    "_count_le_2d_njit",
    "_count_le_2d_njit_parallel",  # bit-identity reference twins of the row-major count kernels (keep-all-kernels rule); parity pinned in tests/feature_selection/discretization/test_multiselect_quantile_edges.py
    "device_permuted_y_batch",  # measured-rejected device permutation generator (slower than the host kernel at 99 x 2000); kept as the bit-identical parity reference, see its module docstring
    "_quantile_edges_2d_radix_njit",  # reference twin of the tiled radix quantile kernel (keep-all-kernels rule); parity pinned in tests/feature_selection/discretization/test_multiselect_quantile_edges.py
    "_lnc_correction",  # deprecated first-pass LNC kept behind a flag for A/B parity; no switch currently reaches it
    "_get_mi_mm_from_values_kernel",  # superseded by the length-aware _nek kernel; kept as the original, nothing switches to it
    "_resample_desc_order_counting",  # retained building block: the resampler now uses _fused_resample_auc, and this counting sort stays as the documented fallback primitive
    "score_triplet_cross_basis_by_mi_uplift",  # alternative triplet scorer (MI uplift over the best source); the live path uses score_triplet_cross_basis_by_cmi and no switch reaches this one; exercised by tests/feature_selection/mrmr/biz_val/test_biz_value_mrmr_scorers/test_triplet_cross_basis.py
    "_plugin_mi_from_counts",  # plug-in MI helper of the retained uplift scorer above; no live caller
    "_discrete_score_numpy_loop",  # original per-class reference for the discrete path, kept as the fallback; no switch reaches it
    "_ks_statistic_numpy",  # reference KS kept callable as the fallback; no switch reaches it
    "_project_single_head_njit",  # parallel single-head projection kept for the no-BLAS fallback and parity; no switch reaches it
    "arrayCountingArgSortThreaded",  # parallel counterpart of the counting argsort; the production path uses the serial form and no switch reaches this one
    "arrayCountingArgSortAndUniqueValuesThreaded",  # parallel counterpart of the counting argsort with unique values; no switch reaches it
    "arrayMinMaxParallel",  # parallel counterpart of the min/max scan; no switch reaches it
    "naive_mean_var_two_pass_seq",  # naive A/B variant kept for the numerical-stability bench; the bench chose the default path and no switch reaches this one
    "naive_moments_two_pass_seq",  # naive A/B variant kept for the numerical-stability bench; the bench chose the default path and no switch reaches this one
    "_fused_regression_welford_par",  # parallel fused Welford regression kernel kept as an alternative; no switch reaches it
    "compute_nunique_mode_quantiles_numba",  # marked not recommended in its own docstring; the numpy variant is the live path
    "kahan_dot_seq",  # compensated-sum alternative kept for the numerical-stability bench
    "kahan_sum_seq",  # compensated-sum alternative kept for the numerical-stability bench
    "kahan_two_pass_var_seq",  # compensated two-pass variance kept for the numerical-stability bench
    "welford_mean_var_seq",  # single-pass Welford kept for the numerical-stability bench
    "welford_moments_seq",  # single-pass Welford moments kept for the numerical-stability bench
    "smote_synthesize_rowloop",  # row-loop SMOTE kept as the pre-vectorisation reference
    "_rff_analytic_mixed_partial_energy_loop",  # per-pair reference kernel kept for the exact RFF energy check
    "_combine_factorize_serial_njit",  # fully-serial reference of the combine-factorize kernel, kept as the baseline
    "_pack_for_bitmap_numpy",  # numpy reference packer kept for the bit-identity bench and test
    "_fused_regression_welford_seq",  # sequential fused Welford regression kernel kept as the alternative to the parallel one
    "dtw_cupy_full",  # pre-CPX-P0-2 full-matrix DTW kept as an alternative backend; no switch reaches it
}

#: Called, but the pinned py-ci-shared (v1.17.0) cannot see it: a lazy import with an ``except ImportError: f = None``
#: fallback reads as a local shadow there. Fixed upstream in 9b48122; drop this entry when the pin moves past it.
_SCANNER_LAG = {"_fill_bf_batch_njit"}

#: Looked up by name in ``sys.modules`` (``_fit_scoped_release._RELEASERS``) so a never-imported module is not imported to be cleared.
#: The recipe replay helpers are named by string in ``engineered_recipes/_recipe_dispatch.py`` (``_routed(module, name)`` resolves them
#: lazily with importlib), so no call expression exists for the scanner to see.
_NAME_DISPATCHED = {
    "clear_gpu_operand_table_caches",
    "_apply_cat_num_residual",
    "_apply_cluster_aggregate",
    "_apply_composite_group_agg_recipe",
    "_apply_conditional_dispersion_recipe",
    "_apply_conditional_quantile_rank_recipe",
    "_apply_conditional_residual_recipe",
    "_apply_count_encoded",
    "_apply_factorize",
    "_apply_frequency_encoded",
    "_apply_group_distance_recipe",
    "_apply_grouped_agg_recipe",
    "_apply_hermite_pair",
    "_apply_hinge_basis",
    "_apply_kfold_target_encoded",
    "_apply_lof_recipe",
    "_apply_mahalanobis_density_recipe",
    "_apply_mi_greedy_transform",
    "_apply_ordinal_pattern_te_recipe",
    "_apply_orth_cluster_basis",
    "_apply_orth_diff_basis",
    "_apply_orth_fourier",
    "_apply_orth_pair_cross",
    "_apply_orth_quadruplet_cross",
    "_apply_orth_spline",
    "_apply_orth_triplet_cross",
    "_apply_orth_univariate",
    "_apply_orth_wavelet",
    "_apply_random_fourier_recipe",
    "_apply_rankgauss_recipe",
    "_apply_rare_category_recipe",
    "_apply_sir_direction_recipe",
    "_apply_target_encoding",
    "_apply_unary_binary",
    "apply_offset_product_recipe",
    "apply_oof_cell2d_recipe",
    "apply_oof_warp1d_recipe",
}

#: Replaced by a faster or more efficient version and kept as the reference the replacement is checked against. Each entry
#: names its replacement; a superseded function with no parity test is a gap, not a reference.
_SUPERSEDED = {
    "subset_redundancy",  # scalar form of subset_redundancy_many; parity pinned in tests/feature_selection/shap_proxied/test_shap_proxy_calibrate.py
    "subset_uncertainty",  # scalar form of subset_uncertainty_many; parity pinned in tests/feature_selection/shap_proxied/test_shap_proxy_objective.py
    "_pack_bins_for_kernel",  # per-column Python loop replaced by _compute_marginals_packed; parity pinned in tests/feature_selection/shap_proxied/test_shap_proxy_cluster_su_fused_setup.py
    "hyvarinen_score",  # deprecated alias of mutual_information_score; parity + DeprecationWarning pinned in tests/calibration/test_mutual_information_rename.py
    "_oof_column_gather",  # stacks the per-fold tables first; _oof_column_from_table does the same gather on the table already stacked and is what the builder calls
    "_oof_column_foldloop",  # parity oracle for _oof_column_gather (tests/feature_selection/fe/gpu/test_binagg_oof_gather_parity.py)
    "_pair_combo_mi_cupy_loop",  # per-combo cupy twin of _pair_combo_mi_cupy; parity pinned in tests/feature_selection/gpu/test_usability_pool_cupy_batched_twin.py
    "_propose_fourier",  # predecessor of _propose_fourier_both_warps; parity pinned in tests/feature_selection/fe/adaptive/test_propose_fourier_parity.py
    "mdlp_inner_cuts",  # reference form of the honest-null cut search; the null kernels use the presorted twin, equality pinned in tests/feature_selection/discretization/test_mdlp_presorted_cuts_parity.py
    "score_triplet_cross_basis_by_mi_uplift",  # predecessor of score_triplet_cross_basis_by_cmi (the production scorer); kept public, pinned in tests/feature_selection/mrmr/biz_val/test_biz_value_mrmr_scorers/test_triplet_cross_basis.py
    "mi_classif_streamed_fill",  # zero-copy twin of mi_classif_streamed (the production path); kept as its reference, parity pinned in tests/feature_selection/filters/test_streamed_gpu_mi.py::test_streamed_fill_matches_copy_path
    "_yj_forward_scalar",  # scalar reference for _yj_forward (numpy and numba paths); parity pinned in tests/training/composite/transforms/test_yj_forward_scalar_parity.py
    "_entropy_binary",  # binary form of _entropy_from_counts; parity pinned in tests/feature_engineering/transformer/test_entropy_binary_parity.py
    "_flags_to_combo_key",  # flag-set form of the oracle key _combo_key; parity pinned in tests/feature_selection/filters/test_flags_combo_key_parity.py
    "_is_minimise_metric",  # now delegates to the metrics registry, the live direction source; parity pinned in tests/training/core/test_metric_direction_parity.py
    "_default_metric_is_auc",  # the unset-metric check; parity pinned in tests/training/core/test_metric_direction_parity.py
    "_per_query_groups",  # index-array form of the live group slices in metrics.ranking._iter_group_slices; parity pinned in tests/reporting/test_per_query_groups_parity.py
    "_pointwise_log_ratio",  # the log term the I_ccs kernel computes inline; the helper re-sum matches it, pinned in tests/feature_selection/filters/test_pointwise_log_ratio_parity.py
    "_get_fused_mi_term",  # the cp.fuse per-cell term; the resident MI now runs the fused MI-from-codes RawKernel (0cb9186a0). Formula pinned to numpy in tests/feature_selection/gpu/test_gpu_fused_mi_term.py
    "grok_mutual_information_old",  # naive per-cell form of grok_mutual_information; parity pinned in tests/feature_selection/test_grok_mi_old_parity.py
    "_yj_inverse_scalar",  # scalar reference for _yj_inverse; parity pinned in tests/training/composite/transforms/test_yj_inverse_scalar_parity.py
}

#: Valid code path a library user enables by explicit choice; slower than the default, so it is not called in production.
#: Each entry names the switch that reaches it. Empty until a function is classified this way.
_OPT_IN_SLOW: set[str] = {
    "per_fold_stability_select",  # ShapProxiedFS once per outer fold (n_folds fits); opt-in via the function call, ~3-7x the default fit cost
    "proposal_generator",  # cheap-proxy candidate seeds for an external honest search; opt-in via the function call
}

#: PLUGIN_HOOK: an extension point a downstream package calls by name; no in-repo caller by design.
_PLUGIN_HOOKS = {
    "forget_pool_bytes",  # downstream evictor calls it after evicting outside evict_for_budget
}

#: FRAMEWORK_HOOK: invoked by name by a framework (a joblib pool warm-up, a callback); the caller is outside the repo or the framework.
_FRAMEWORK_HOOKS: set[str] = {
    "_pool_warmup_noop",  # joblib pool warm-up callable, handed to the worker pool by reference
}

#: PLANNED: a documented feature whose caller is not written yet; each entry names the feature and the owner decides wire or drop.
_PLANNED: set[str] = {
    "_suppress_numeric_warnings",  # worker-thread warning filter for the numeric module, planned for its callers
    "batch_weighted_mean",  # batch wrapper over weighted_mean_1d, planned for batched neighbour aggregation
    "batch_weighted_std",  # batch wrapper over weighted_std_1d, planned for batched neighbour aggregation
    "batch_weighted_vector_mean",  # batch per-query vector mean, planned for batched neighbour aggregation
    "fill_grid_1d",  # GPU launch-geometry helper for flat kernels, planned to replace hand-written grid sizes
    "parallel_mi_with_null",  # null-mean-accumulating MI twin, planned for the null-calibrated screen
    "parallel_mi_besag_clifford_with_null",  # null-mean-accumulating Besag-Clifford twin, planned for the null-calibrated screen
    "apply_gpu_unary_batched",  # batched GPU unary apply, planned for the resident FE path
    "apply_gpu_binary_batched",  # batched GPU binary apply, planned for the resident FE path
    "sanitize_name_list",  # remaps cat_features names through the sanitiser, planned for the training input path
    "_residual_variance_ratio",  # OLS residual-variance criterion described in the structural hints, planned for the hint decision
    "_transform_via",  # applies a fitted pre-pipeline or passes X through, planned for the ensemble predict path
    "carve_calib_conformal",  # dispatches the structure-appropriate conformal calibration carver, planned for the conformal split
    "disable_native_es_for_slice_stable",  # strips native early stopping under slice-stable ES, planned for per-slice boosters
    "translate_sklearn_mlp_overrides_to_recurrent_config_kwargs",  # translates sklearn MLP overrides to the recurrent config, planned for the recurrent trainer
    "wrap_with_lookahead",  # conditional Lookahead wrapper, planned for the neural optimiser setup
    "_group_features_mi_njit",  # fused per-group MI in one njit dispatch, planned for the ranking feature selection
    "rerank_candidates",  # re-ranks SHAP-proxy candidates by the corrector's predicted loss, planned for the corrector step
    "compute_permutation_importances",  # user-facing permutation importance helper not yet exported, planned as public API
    "weighted_std_1d",  # library helper with no in-repo caller; owner decides to export or drop
    "smote_synthesize_vectorized",  # library helper with no in-repo caller; owner decides to export or drop
    "_column_tie_fraction",  # library helper with no in-repo caller; owner decides to export or drop
    "ensure_batch_mi_noise_gate_tuning",  # library helper with no in-repo caller; owner decides to export or drop
    "_global_stat",  # library helper with no in-repo caller; owner decides to export or drop
    "_per_cell_raw_moments_njit",  # library helper with no in-repo caller; owner decides to export or drop
    "_per_cell_stats_gpu",  # library helper with no in-repo caller; owner decides to export or drop
    "_compute_pair_ii_mm",  # library helper with no in-repo caller; owner decides to export or drop
    "_should_apply_mm_for_pair",  # library helper with no in-repo caller; owner decides to export or drop
    "fe_columns",  # library helper with no in-repo caller; owner decides to export or drop
    "pair_is_tail_concentrated",  # library helper with no in-repo caller; owner decides to export or drop
    "mi_direct_gpu_batched_streamed",  # library helper with no in-repo caller; owner decides to export or drop
    "_gpu_evaluate_basis_column",  # library helper with no in-repo caller; owner decides to export or drop
    "clear_resident_codes_handoff",  # library helper with no in-repo caller; owner decides to export or drop
    "fe_gpu_resident_enabled",  # library helper with no in-repo caller; owner decides to export or drop
    "ensure_fe_gpu_binning_tuning",  # library helper with no in-repo caller; owner decides to export or drop
    "ensure_fe_gpu_pairs_mi_tuning",  # library helper with no in-repo caller; owner decides to export or drop
    "gpu_resident_pair_recipes",  # library helper with no in-repo caller; owner decides to export or drop
    "grand_fused_pair_mi",  # library helper with no in-repo caller; owner decides to export or drop
    "rank_bin_codes_gpu_resident",  # library helper with no in-repo caller; owner decides to export or drop
    "_segmented_sse",  # library helper with no in-repo caller; owner decides to export or drop
    "suppress_numba_warnings",  # library helper with no in-repo caller; owner decides to export or drop
    "_greedy_merge_bins",  # library helper with no in-repo caller; owner decides to export or drop
    "mdlp_bin_edges_validated",  # library helper with no in-repo caller; owner decides to export or drop
    "_ent_from_counts",  # library helper with no in-repo caller; owner decides to export or drop
    "_nnz_from_counts",  # library helper with no in-repo caller; owner decides to export or drop
    "optimize_all_pairs_numba_kernel",  # library helper with no in-repo caller; owner decides to export or drop
    "pool_table_use_resident",  # library helper with no in-repo caller; owner decides to export or drop
    "_quantile_codes_1d_njit",  # library helper with no in-repo caller; owner decides to export or drop
    "plugin_mi_classif_batch_fast",  # library helper with no in-repo caller; owner decides to export or drop
    "get_binningprocess_featureselectors",  # library helper with no in-repo caller; owner decides to export or drop
    "_kruskal_wallis_h",  # library helper with no in-repo caller; owner decides to export or drop
    "_mann_whitney_u_z",  # library helper with no in-repo caller; owner decides to export or drop
    "dtw_dispatch",  # library helper with no in-repo caller; owner decides to export or drop
    "_initialize_mutable_defaults",  # library helper with no in-repo caller; owner decides to export or drop
    "make_dmatrix",  # library helper with no in-repo caller; owner decides to export or drop
    "make_lgb_dataset",  # library helper with no in-repo caller; owner decides to export or drop
    "make_pool",  # library helper with no in-repo caller; owner decides to export or drop
    "_discovery_cache_bytes_total",  # library helper with no in-repo caller; owner decides to export or drop
    "_rolling_quantile_radius",  # library helper with no in-repo caller; owner decides to export or drop
    "attach_grouped_causal_bases",  # library helper with no in-repo caller; owner decides to export or drop
    "screen_base_pool",  # library helper with no in-repo caller; owner decides to export or drop
    "_monotone_fraction",  # library helper with no in-repo caller; owner decides to export or drop
    "_registered_transform_names",  # library helper with no in-repo caller; owner decides to export or drop
    "_linear_residual_fit_closed",  # library helper with no in-repo caller; owner decides to export or drop
    "unpack_ctx_configs_to_locals",  # library helper with no in-repo caller; owner decides to export or drop
    "recover_composite_y_scale_metrics",  # library helper with no in-repo caller; owner decides to export or drop
    "_models_need_pandas_cat_prep",  # library helper with no in-repo caller; owner decides to export or drop
    "_run_one_weight_iteration",  # library helper with no in-repo caller; owner decides to export or drop
    "_excess_kurtosis",  # library helper with no in-repo caller; owner decides to export or drop
    "_skewness",  # library helper with no in-repo caller; owner decides to export or drop
    "compare_targets_performance",  # library helper with no in-repo caller; owner decides to export or drop
}

#: Optional entry points a user calls when they want them; no parity test is required, unlike _SUPERSEDED or _OPT_IN_SLOW.
_OPTIN = {
    "mi_classif_streamed",  # streamed GPU MI for columns that do not fit resident; chosen per fit on large inputs
    "with_mix_base_column",  # target-mix base column the user adds explicitly to a frame
    "precompute_all",  # pre-split warm-up entry point; calls the two stubs below
    "precompute_composite_target_specs",  # stub: raises NotImplementedError until the split-time implementation lands
    "precompute_dummy_baselines",  # stub: raises NotImplementedError until the split-time implementation lands
    "_require_plotly",  # plotly import gate for the viz extra; metrics plotting moved to the reporting renderers, kept as the documented guard for direct metrics.core users
    "_split_arrays",  # thin wrapper over group_values_by_label (which the charts call directly), kept as the error-analysis module's local entry point
    "_wrap_text",  # thin plotly-flavoured wrapper over wrap_title_lines (the renderers call the measuring twin), kept for fixed-width <br> folding
    "_get_cached_plot_idx",  # LRU-cached scatter subsample index; charts now use subsample_preserving_extremes, kept for fixed-seed uniform subsampling
    "clear_pinned_d2h",  # releases page-locked host memory on request; the buffer is reused across fits for PCIe speed, so no automatic clear
}


def _production_files() -> list[Path]:
    """Shipped modules only. Benchmarks (``_benchmarks/`` and ``_bench_*`` helpers) keep deliberately-uncalled baselines."""
    return sorted(p for p in (REPO_ROOT / "src").rglob("*.py") if "_benchmarks" not in p.as_posix() and not p.name.startswith("_bench_"))


def test_no_new_function_without_a_caller(request):
    """Kills: a new guard or validator that nothing ever calls."""
    from py_ci_shared.uncalled_functions import assert_no_new_uncalled_function

    files = _production_files()
    assert len(files) > 1000, f"only {len(files)} production files found -- the scan lost its subject"

    assert_no_new_uncalled_function(
        files,
        REPO_ROOT,
        BASELINE,
        ignore=_INTERPRETER_INVOKED
        | _PUBLIC_API
        | _REEXPORTED_PUBLIC_API
        | _OFFLINE_GENERATORS
        | _TEST_UTILITIES
        | _RETAINED_KERNELS
        | _SCANNER_LAG
        | _NAME_DISPATCHED
        | _SUPERSEDED
        | _PLUGIN_HOOKS
        | _PLANNED
        | _FRAMEWORK_HOOKS
        | _OPT_IN_SLOW
        | _OPTIN,
    )
