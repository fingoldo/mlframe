"""The FuzzCombo dataclass: canonicalisation + identity helpers.

See the package __init__ docstring for the canonicalisation contract.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class FuzzCombo:
    models: tuple[str, ...]
    input_type: str
    n_rows: int
    cat_feature_count: int
    null_fraction_cats: float
    use_mrmr_fs: bool
    weight_schemas: tuple[str, ...]
    target_type: str
    auto_detect_cats: bool
    align_polars_categorical_dicts: bool
    seed: int
    # New axes 2026-04-24 — have defaults so existing pinned sensor combos
    # and stored ``_fuzz_results.jsonl`` rows keep deserialising cleanly.
    prefer_polarsds: bool = True
    use_text_features: bool = True
    honor_user_dtype: bool = False
    target_carrier: str = "numpy"
    text_col_count: int = 0
    embedding_col_count: int = 0
    # 2026-04-24 combo extension from test_suite_coverage_gaps analysis
    outlier_detection: "str | None" = None
    use_ensembles: bool = False
    continue_on_model_failure: bool = False
    iterations: int = 3
    prefer_calibrated_classifiers: bool = False
    inject_degenerate_cols: bool = False
    inject_inf_nan: bool = False
    with_datetime_col: bool = False
    inject_zero_col: bool = False
    fairness_col: "str | None" = None
    custom_prep: "str | None" = None
    input_storage: str = "memory"
    # 2026-04-24 round 2 — config-field axes
    fillna_value_cfg: "float | None" = None
    scaler_name_cfg: "str | None" = "standard"
    categorical_encoding_cfg: str = "ordinal"
    skip_categorical_encoding_cfg: bool = False
    val_placement_cfg: str = "forward"
    test_size_cfg: float = 0.1
    trainset_aging_limit_cfg: "float | None" = None
    cat_text_card_threshold_cfg: int = 300
    early_stopping_rounds_cfg: "int | None" = None
    use_robust_eval_metric_cfg: bool = True
    # Fix G — adversarial axes
    inject_label_leak: bool = False
    inject_rank_deficient: bool = False
    inject_all_nan_col: bool = False
    # R3 — drift, imbalance, weird-cat axes
    inject_test_drift: "str | None" = None
    imbalance_ratio: str = "balanced"
    weird_cat_content: "str | None" = None
    # 2026-04-24 Phase H — multilabel dispatch axis. Only meaningful when
    # target_type == multilabel_classification.
    multilabel_strategy_cfg: str = "auto"
    # 2026-05-04 — LTR ensembling method. Only meaningful when
    # target_type == learning_to_rank.
    ranking_ensemble_method: str = "rrf"
    # 2026-04-26 batch 1 — additional config-field axes
    fix_infinities_cfg: bool = True
    ensure_float32_cfg: bool = True
    remove_constant_columns_cfg: bool = True
    imputer_strategy_cfg: "str | None" = "mean"
    shuffle_val_cfg: bool = False
    shuffle_test_cfg: bool = False
    wholeday_splitting_cfg: bool = True
    val_sequential_fraction_cfg: float = 0.5
    # 2026-04-26 batch 3 — multilabel dispatch axes
    multilabel_n_chains_cfg: int = 3
    multilabel_chain_order_cfg: str = "random"
    multilabel_cv_cfg: int = 5
    # 2026-04-26 batch 4 — PreprocessingExtensionsConfig axes
    prep_ext_scaler_cfg: "str | None" = None
    prep_ext_kbins_cfg: "int | None" = None
    prep_ext_polynomial_degree_cfg: "int | None" = None
    prep_ext_dim_reducer_cfg: "str | None" = None
    prep_ext_nonlinear_cfg: "str | None" = None
    # PySR symbolic regression (gated for cls / large-n / text / embedding combos in canonical_key).
    prep_ext_pysr_enabled_cfg: bool = False
    # MRMR NaN handling strategy. No-op when use_mrmr_fs is False (axis canonicalised away).
    mrmr_nan_strategy_cfg: str = "separate_bin"
    # 2026-04-26 batch 5 — RFECV
    rfecv_estimator_cfg: "str | None" = None
    # 2026-04-26 batch 6 — recurrent
    recurrent_model_cfg: "str | None" = None
    # 2026-04-28 batch 4 followup — confidence analysis
    include_confidence_analysis_cfg: bool = False
    # 2026-05-11 Wave 15 — MRMR-internal knobs
    mrmr_interactions_max_order_cfg: int = 1
    mrmr_fe_max_steps_cfg: int = 1
    mrmr_cat_fe_enable_cfg: bool = True
    # 2026-05-11 Wave 21 — assorted high-value config-toggle axes
    dummy_baselines_enabled_cfg: bool = True
    baseline_diagnostics_enabled_cfg: bool = True
    use_groups_cfg: bool = True
    apply_outlier_to_val_cfg: bool = True
    multilabel_allow_uncalibrated_cfg: bool = False
    report_residual_audit_cfg: bool = True
    ltr_assume_comparable_scales_cfg: bool = False
    # 2026-05-18 — composite-target discovery (Packs J + K) fuzz axes.
    # Defaulted False / None so existing pinned sensor combos and
    # archived ``_fuzz_results.jsonl`` rows keep deserialising.
    composite_discovery_enabled_cfg: bool = False
    composite_transforms_mode_cfg: "str | None" = None
    # 2026-05-18 — MRMR feature-engineering FE-search knobs. All
    # default to library defaults so existing pinned sensor combos
    # and archived ``_fuzz_results.jsonl`` rows keep deserialising
    # without behaviour change.
    mrmr_fe_npermutations_cfg: int = 0
    mrmr_fe_ntop_features_cfg: int = 0
    mrmr_fe_unary_preset_cfg: str = "minimal"
    mrmr_fe_binary_preset_cfg: str = "minimal"
    mrmr_fe_smart_polynom_iters_cfg: int = 0
    mrmr_fe_smart_polynom_steps_cfg: int = 10
    mrmr_fe_min_polynom_degree_cfg: int = 3
    mrmr_fe_max_polynom_degree_cfg: int = 3
    mrmr_cat_fe_include_numeric_cfg: bool = False
    # 2026-05-19 -- PreprocessingExtensionsConfig polynomial-auto-tune
    # axes (iter-69 byte cap + iter-340 polynomial_max_features cap).
    # Defaults mirror the live configs so existing pinned sensor combos
    # deserialise unchanged.
    prep_ext_polynomial_max_features_cfg: "int | None" = 10_000
    prep_ext_polynomial_interaction_only_cfg: bool = True
    prep_ext_memory_safety_max_bytes_cfg: "int | None" = 500_000_000
    # 2026-05-19 -- composite-discovery stacked-residual axes.
    composite_use_stacked_discovery_cfg: bool = False
    composite_use_stacked_discovery_residual_cfg: bool = False
    # 2026-05-22 — six gate-flip axes from the TVT-MLP-collapse cascade.
    # Defaults are the POST-FIX values; pre-fix variants live in AXES for
    # regression coverage. All canonicalise to post-fix when composite
    # discovery is OFF.
    composite_skip_raw_dominates_ratio_cfg: float = 0.0
    composite_skip_ablation_delta_pct_cfg: float = 0.0
    composite_eps_mi_gain_cfg: float = -10.0
    composite_top_k_after_mi_cfg: int = 32
    composite_require_beats_raw_baseline_cfg: bool = False
    composite_per_bin_n_bins_cfg: int = 0
    composite_tiny_screening_mode_cfg: str = "per_family"
    composite_include_additive_residual_cfg: bool = True
    mlp_activation_cfg: str = "ReLU"
    composite_skip_wrap_pass_predict_cfg: bool = True
    # 2026-05-21 -- mini-HPT (target + feature distribution analyzer) toggle.
    # Default True mirrors the suite signature default; archived
    # _fuzz_results.jsonl rows that pre-date this axis deserialise as True.
    enable_target_distribution_analyzer_cfg: bool = True
    # 2026-05-21 -- MRMR FE pair-check subsample knob. Default 0 (disabled)
    # mirrors the legacy axis behaviour; archived rows pre-dating this axis
    # deserialise as 0 (no subsample).
    fe_check_pairs_subsample_n_cfg: int = 0
    # 2026-05-21 iter150 -- multi-target / multi-target-type axis. See AXES
    # docstring for value semantics. Default None preserves legacy single-
    # target fuzz behaviour for combos archived before this axis landed.
    extra_targets: "str | None" = None
    # 2026-05-21 iter151 -- P0/P1/P2 audit fill-in. All fields default
    # to the pre-iter151 implicit values so combos archived before this
    # batch deserialise without behaviour change.
    enable_quantile_regression_cfg: bool = False
    linear_alpha_cfg: float = 1.0
    linear_solver_cfg: str = "lbfgs"
    enable_feature_handling_config_cfg: bool = False
    enable_precomputed_cfg: bool = False
    test_sequential_fraction_cfg: "float | None" = None
    calib_size_cfg: "float | None" = None
    use_boruta_shap_cfg: bool = False
    boruta_importance_measure_cfg: str = "gini"
    # 2026-06-03 FS-coverage audit -- BorutaShap.__init__ knobs (defaults match
    # BorutaShap signature so default combos are unchanged).
    boruta_optimistic_cfg: bool = True
    boruta_train_or_test_cfg: str = "train"
    boruta_premerge_clusters_cfg: bool = False
    # 2026-06-04 FS-coverage follow-up -- BorutaShap.__init__ early_stop_* knobs
    # (defaults match the BorutaShap signature so default combos are unchanged).
    boruta_early_stop_tentative_cfg: bool = False
    boruta_early_stop_patience_cfg: int = 20
    boruta_early_stop_margin_cfg: float = 0.15
    use_sample_weights_in_fs_cfg: bool = False
    fallback_to_sklearn_cfg: bool = True
    prefer_gpu_configs_cfg: bool = True
    prefer_cpu_for_lightgbm_cfg: bool = True
    mrmr_identity_cache_scope_cfg: str = "ctx"
    skip_identity_equivalent_pre_pipelines_cfg: bool = True
    rfecv_leakage_corr_threshold_cfg: float = 0.95
    rfecv_mbh_adaptive_threshold_cfg: int = 30
    # 2026-06-03 FS-coverage audit -- RFECV.__init__ knobs (defaults match the
    # RFECV signature string-enum defaults so default combos are unchanged).
    rfecv_votes_aggregation_cfg: str = "Borda"
    rfecv_search_method_cfg: str = "ModelBasedHeuristic"
    # 2026-05-22 iter162 -- nested-config / depth-2 audit fill-in.
    # All default to the field's library default so combos archived
    # pre-iter162 deserialise without behaviour change.
    fhc_cache_eviction_strategy_cfg: str = "size_weighted"
    fhc_cache_allow_pickle_cfg: bool = False
    fhc_cache_ram_fraction_cfg: float = 0.3
    fhc_text_definite_text_mean_chars_cfg: int = 100
    fhc_text_min_alphabet_entropy_cfg: float = 4.5
    fhc_repro_deterministic_torch_cfg: bool = False
    fhc_auto_locale_detection_cfg: str = "fallback_only"
    enable_viz_rendering_cfg: bool = False
    reporting_prob_histogram_yscale_cfg: str = "auto"
    reporting_title_metrics_template_cfg: str = "ICE BR_DECOMP ECE CMAEW LL ROC_AUC PR_AUC"
    reporting_matplotlib_rcparams_cfg: "str | None" = None
    reporting_multiclass_panels_cfg: str = "CONFUSION PR_F1 ROC CALIB_GRID PROB_DIST TOP_K_ACC"
    confidence_model_kwargs_cfg: str = "default"
    composite_mi_estimator_cfg: str = "bin"
    composite_mi_nbins_cfg: int = 16
    composite_mi_aggregation_cfg: str = "mean"
    composite_mi_sample_strategy_cfg: str = "random"
    composite_stacked_residual_aggregation_cfg: str = "mean"
    composite_discovery_n_jobs_cfg: int = 1
    quantile_crossing_fix_cfg: str = "sort"
    quantile_coverage_pairs_cfg: str = "default"
    quantile_wrapper_n_jobs_cfg: Any = "auto"
    mlp_predict_batch_size_cfg: "int | None" = None
    ltr_cb_loss_fn_cfg: str = "YetiRankPairwise"
    ltr_lgb_objective_cfg: str = "lambdarank"
    ltr_rrf_k_cfg: int = 60
    recurrent_precision_cfg: str = "32-true"
    recurrent_sequence_preprocessing_cfg: str = "none"
    # 2026-05-22 iter170 -- wave-3 depth-3+ audit. All default to the
    # library default so combos archived pre-iter170 deserialise unchanged.
    lgb_feature_fraction_cfg: float = 1.0
    lgb_num_leaves_cfg: int = 31
    xgb_max_depth_cfg: int = 6
    xgb_colsample_bynode_cfg: float = 1.0
    cb_border_count_cfg: int = 254
    hgb_max_leaf_nodes_cfg: int = 31
    rfecv_cv_n_splits_cfg: int = 2
    robust_q_low_cfg: float = 0.01
    robust_q_high_cfg: float = 0.99
    tfidf_max_features_cfg: int = 5000
    kbins_encode_cfg: str = "ordinal"
    nonlinear_n_components_cfg: int = 100
    pysr_operator_preset_cfg: str = "standard"
    confidence_ensemble_quantile_cfg: float = 0.1
    cat_text_card_threshold_pct_cfg: float = 0.001
    rfecv_n_features_selection_rule_cfg: str = "auto"
    rfecv_stability_selection_cfg: bool = False
    rfecv_leakage_action_cfg: str = "warn"
    mrmr_fe_adaptive_threshold_relax_cfg: bool = True
    mrmr_use_simple_mode_cfg: bool = False
    mrmr_identity_cache_include_y_cfg: bool = True
    # 2026-05-27 MRMR friend-graph + cluster-aggregate (defaults mirror mrmr.py __init__)
    mrmr_build_friend_graph_cfg: bool = True
    mrmr_friend_graph_prune_cfg: bool = False
    mrmr_cluster_aggregate_enable_cfg: bool = True
    mrmr_cluster_aggregate_mode_cfg: str = "augment"
    # 2026-05-28 ShapProxiedFS axes (defaults mirror ShapProxiedFS.__init__)
    use_shap_proxied_fs: bool = False
    shap_proxied_optimizer_cfg: str = "auto"
    shap_proxied_revalidate_cfg: bool = True
    shap_proxied_trust_guard_cfg: bool = True
    shap_proxied_interaction_aware_cfg: bool = False
    shap_proxied_cluster_features_cfg: "bool | str" = "auto"
    # 2026-05-28 ShapProxiedFS extension axes (defaults mirror ShapProxiedFS.__init__)
    shap_proxied_active_learning_cfg: bool = False
    shap_proxied_prefilter_method_cfg: str = "auto"
    # 2026-05-28 ShapProxiedFS deeper extension axes (B1-B6 audit-pass-2).
    # Defaults verified against feature_selection/shap_proxied_fs.py:41-89.
    shap_proxied_config_jitter_cfg: bool = False
    shap_proxied_uncertainty_penalty_cfg: float = 0.0
    shap_proxied_within_cluster_refine_cfg: bool = True
    shap_proxied_use_bias_corrector_cfg: bool = True
    shap_proxied_refine_n_estimators_cfg: "int | None" = 100
    shap_proxied_trust_guard_n_estimators_cfg: "int | None" = 100
    # 2026-05-28 ShapProxiedFS audit-pass-3 axes (W3). Defaults mirror
    # ShapProxiedFS.__init__ (feature_selection/shap_proxied_fs.py:69-79).
    shap_proxied_cluster_weighting_cfg: str = "pca_pc1"
    # iter624: iter67 cluster_use_precomputed_bins + cluster_su_threshold.
    # Defaults verified at shap_proxied_fs.py:228-229.
    shap_proxied_cluster_use_precomputed_bins_cfg: bool = True
    shap_proxied_cluster_su_threshold_cfg: float = 0.5
    shap_proxied_max_interaction_features_cfg: int = 16
    shap_proxied_prefilter_top_cfg: "int | None" = 2000
    shap_proxied_prefilter_n_estimators_cfg: "int | None" = 100
    # 2026-05-28 ShapProxiedFS audit-pass-5 axes (W5). Defaults verified against
    # ShapProxiedFS.__init__ (feature_selection/shap_proxied_fs.py:62, 78, 89-94).
    shap_proxied_trust_guard_stratified_anchors_cfg: bool = False
    shap_proxied_trust_guard_uniform_tail_frac_cfg: float = 0.2
    shap_proxied_trust_guard_cardinality_dist_cfg: str = "zipf"
    shap_proxied_trust_guard_zipf_alpha_cfg: float = 0.25
    shap_proxied_trust_guard_fidelity_weights_cfg: "tuple[float, float]" = (0.6, 0.4)
    shap_proxied_trust_guard_metric_cfg: str = "proxy_fidelity_score"
    shap_proxied_fidelity_floor_cfg: float = 0.5
    shap_proxied_oof_shap_n_estimators_cfg: "int | None" = 100
    # 2026-05-28 audit-pass-2 PART A: 4 LOW-tier coverage-gap axes.
    # Defaults mirror EnsemblingConfig / TrainingBehaviorConfig /
    # PreprocessingExtensionsConfig in src/mlframe/training/_model_configs.py
    # + _preprocessing_configs.py.
    ensembling_degenerate_class_ratio_cfg: float = 0.01
    target_temporal_audit_granularity_cfg: str = "auto"
    prep_ext_dim_n_components_cfg: int = 50
    # 2026-05-28 TextDetectionConfig.text_min_cardinality (default 300)
    fhc_text_min_cardinality_cfg: int = 300
    # 2026-05-28 CompositeTargetDiscoveryConfig deep knobs (defaults mirror the dataclass)
    composite_auto_skip_on_baseline_optimal_cfg: bool = False
    composite_mi_n_neighbors_cfg: int = 3
    composite_auto_base_null_perms_cfg: int = 20
    composite_multi_base_max_k_cfg: int = 3
    # 2026-05-28 TrainingBehaviorConfig.extreme_ar_group_aware_skip_models (enum: default/include_linear/empty)
    extreme_ar_group_aware_skip_models_cfg: str = "default_neural"
    # 2026-05-28 FeatureSelectionConfig.pre_screen_null_fraction_threshold (default 0.99)
    fs_pre_screen_null_fraction_threshold_cfg: float = 0.99
    # 2026-05-28 LinearModelConfig.l1_ratio (ElasticNet mix; default 0.5 in src)
    linear_l1_ratio_cfg: float = 0.5
    # 2026-05-28 RecurrentConfig.hidden_size (library default 128)
    recurrent_hidden_size_cfg: int = 128
    catfe_fwer_correction_cfg: str = "none"
    catfe_perm_budget_strategy_cfg: str = "bandit_ucb1"
    catfe_permutation_null_cfg: str = "joint_independence"
    catfe_bootstrap_ci_n_replicates_cfg: int = 0
    catfe_use_miller_madow_cfg: "bool | None" = None
    catfe_refine_passes_cfg: int = 0
    catfe_enable_streaming_cache_cfg: bool = False
    catfe_unknown_strategy_cfg: str = "clip"
    composite_screening_cfg: str = "hybrid"
    composite_tiny_model_num_leaves_cfg: int = 15
    composite_tiny_model_learning_rate_cfg: float = 0.1
    composite_raw_baseline_tolerance_cfg: float = 1.02
    composite_use_wilcoxon_gate_cfg: bool = False
    composite_detect_alpha_drift_cfg: bool = True
    composite_reject_on_alpha_drift_cfg: bool = False
    reporting_figsize_cfg: str = "default"
    reporting_plot_dpi_cfg: "int | None" = None
    reporting_quantile_panels_cfg: str = "default"
    reporting_ltr_panels_cfg: str = "default"
    reporting_plotly_template_cfg: "str | None" = None
    reporting_matplotlib_style_cfg: "str | None" = None
    baseline_quick_model_n_estimators_cfg: int = 200
    baseline_quick_model_num_leaves_cfg: int = 31
    baseline_quick_model_learning_rate_cfg: float = 0.05
    baseline_sample_n_cfg: int = 50_000
    baseline_high_potential_min_dominance_pct_cfg: float = 5.0
    baseline_best_model_min_lift_cfg: float = 1.5
    dummy_stratified_n_repeats_cfg: int = 20
    dummy_paired_bootstrap_n_resamples_cfg: int = 1000
    ltr_mlp_loss_fn_cfg: str = "ranknet"
    ltr_eval_at_cfg: str = "default"
    multilabel_force_native_xgb_cfg: bool = False
    fhc_pricing_cap_usd_cfg: "float | None" = None
    fhc_pricing_warn_above_usd_cfg: float = 1.0
    fhc_logging_verbose_cfg: bool = False
    fhc_repro_langdetect_seed_cfg: int = 0
    fhc_repro_pinned_svd_solver_params_cfg: bool = True
    fhc_repro_forbid_nonatomic_fs_cfg: bool = False
    fhc_repro_deterministic_eviction_cfg: bool = False
    fhc_cache_prefetch_enabled_cfg: bool = True
    fhc_cache_prefetch_vram_safety_factor_cfg: float = 2.0
    fhc_memory_pressure_watermark_pct_cfg: int = 85
    fhc_text_min_mean_tokens_cfg: float = 4.0
    fhc_text_min_unique_ratio_cfg: float = 0.95
    fhc_text_respect_explicit_cat_dtype_cfg: bool = True
    recurrent_input_mode_cfg: str = "hybrid"
    recurrent_num_workers_cfg: int = 0
    # 2026-05-23 iter180 -- DEPTH-4 booster sub-params + FHC persistence
    # + multilabel list-typed fields.
    lgb_boosting_type_cfg: str = "gbdt"
    lgb_dart_drop_rate_cfg: float = 0.1
    lgb_goss_top_rate_cfg: float = 0.2
    xgb_tree_method_cfg: str = "auto"
    xgb_hist_max_bin_cfg: int = 256
    cb_bootstrap_type_cfg: str = "Bayesian"
    cb_bayesian_bagging_temperature_cfg: float = 1.0
    cb_bernoulli_subsample_cfg: float = 0.8
    cb_grow_policy_cfg: str = "SymmetricTree"
    cb_lossguide_max_leaves_cfg: int = 31
    fhc_cache_persistence_cfg: str = "auto"
    multilabel_per_label_thresholds_cfg: "str | None" = None
    multilabel_chain_seeds_cfg: "str | None" = None
    # F1 -- enable_crash_reporting (suite-level kwarg; Windows-only meaningful).
    enable_crash_reporting_cfg: bool = False
    # 2026-05-26 iter291 new-functionality axes (post-2-day commit wave).
    # Defaults match the post-fix library defaults so existing pinned sensor
    # combos + archived ``_fuzz_results.jsonl`` rows keep deserialising.
    bucket_stratify_cfg: bool = True
    composite_cardinality_cap_cfg: int = 200
    honest_estimator_diagnostics_cfg: bool = True
    cross_target_ensemble_strategy_cfg: str = "nnls_stack"
    add_cyclical_date_features_cfg: bool = False
    add_extended_date_features_cfg: bool = False
    use_nnls_weights_in_blends_cfg: bool = True
    enable_prediction_envelope_clip_cfg: bool = True
    # 2026-05-27 iter332 audit-driven new-functionality axes.
    ensembling_force_legacy_cfg: bool = False
    ensembling_quantile_budget_bytes_cfg: int = 500 * 1024 * 1024
    ensembling_flag_degenerate_conf_subset_cfg: bool = True
    mlp_extreme_ar_group_aware_skip_cfg: bool = False
    mlp_extreme_ar_threshold_cfg: float = 0.99
    mlp_drop_per_group_constants_cfg: bool = False
    composite_always_build_ct_ensemble_for_raw_cfg: bool = True
    composite_ct_ensemble_dummy_floor_enabled_cfg: bool = True
    composite_extreme_ar_group_aware_skip_cfg: bool = True
    composite_oof_holdout_source_cfg: str = "external_val"
    composite_stacking_aware_gate_enabled_cfg: bool = False
    composite_use_baseline_diagnostics_hint_cfg: bool = True
    fs_pre_screen_unsupervised_cfg: bool = True
    fs_pre_screen_variance_threshold_cfg: float = 0.0
    baseline_init_score_top_k_cfg: int = 1
    # 2026-05-27 iter350 audit batch 2 axes.
    use_ap12_calibrated_probs_in_ensemble_cfg: bool = True
    mlp_extreme_ar_weight_decay_factor_cfg: float = 100.0
    feature_drift_auto_apply_neural_overrides_cfg: bool = False
    target_temporal_audit_column_cfg: "str | None" = None
    composite_lag_predict_failsafe_tolerance_cfg: float = 0.10
    composite_extreme_ar_threshold_cfg: float = 0.99
    composite_ct_ensemble_dummy_floor_tolerance_cfg: float = 0.0
    composite_oof_holdout_frac_cfg: float = 0.2
    composite_top_m_after_tiny_cfg: int = 10
    prep_ext_tfidf_keep_sparse_cfg: bool = True
    recurrent_use_attention_cfg: bool = True
    ltr_xgb_objective_cfg: str = "rank:ndcg"
    baseline_init_score_apply_target_types_cfg: str = "regression_only"
    # 2026-05-28 audit-pass-4 SAFE-subset (W4). Defaults source-verified:
    #   - calibration_policy_auto_pick / n_bootstrap / candidates: mirror
    #     CalibrationConfig at src/mlframe/calibration/policy.py:464-469.
    #     (audit said False/200/tuple; source says True/1000/None.)
    #   - pipeline_cache_ram_budget_fraction: _model_configs.py:641.
    #   - reporting_compute_trainset_metrics: _reporting_configs.py:96.
    #     (audit said True; source says False.)
    #   - reporting_mase_seasonality: _reporting_configs.py:140; int not None.
    #     (audit said None|sequence; source says int=1.)
    #   - recurrent_use_stratified_sampler: _recurrent_config.py:90.
    #     (audit said False or True; source says True.)
    #   - behavior_model_file_hash_suffix: _model_configs.py:547; bool not str.
    #     (audit said str|None default None; source says bool=True.)
    calibration_policy_auto_pick_cfg: bool = True
    calibration_n_bootstrap_cfg: int = 1000
    calibration_candidates_cfg: "tuple[str, ...] | None" = None
    pipeline_cache_ram_budget_fraction_cfg: float = 0.4
    reporting_compute_trainset_metrics_cfg: bool = False
    reporting_mase_seasonality_cfg: int = 1
    recurrent_use_stratified_sampler_cfg: bool = True
    behavior_model_file_hash_suffix_cfg: bool = True
    # 2026-05-30 audit-pass-6 (W6). Defaults source-verified against
    # SliceStableESConfig (_training_runtime_configs.py:42-95),
    # TrainingBehaviorConfig.early_stop_on_worsening (_model_configs.py:505),
    # MRMR Wave 7/8/9 ctor args (filters/mrmr.py:224-302, 589), and
    # CompositeTargetDiscoveryConfig.cv_selector_mode
    # (_composite_target_discovery_config.py:117).
    slice_stable_es_enabled_cfg: bool = False
    slice_stable_es_aggregate_cfg: str = "mean"
    slice_stable_es_source_cfg: str = "temporal"
    slice_stable_es_pareto_best_iter_selection_cfg: bool = False
    slice_stable_es_diagnostic_only_cfg: bool = False
    early_stop_on_worsening_cfg: bool = True
    mrmr_nbins_strategy_cfg: str = "mdlp"
    mrmr_mi_correction_cfg: str = "none"
    mrmr_redundancy_aggregator_cfg: "str | None" = None
    mrmr_bur_lambda_cfg: float = 0.0
    mrmr_cmi_perm_stop_cfg: bool = False
    mrmr_stability_selection_method_cfg: str = "classic"
    mrmr_mi_normalization_cfg: str = "none"
    # 2026-05-30 audit-pass-7 #1: source default flipped False -> True at
    # mrmr.py:596 (commit e4562791). Fuzz dataclass default mirrors the
    # production default so default-config combos exercise the DCD branch;
    # the AXES pair stays (False, True) so the legacy branch is still
    # sampled.
    mrmr_dcd_enable_cfg: bool = True
    # 2026-05-30 audit-pass-7 #2/#3/#4: defaults source-verified against
    # mrmr.py:309 (baseline_npermutations=2) and _adaptive_nbins.py:511,586
    # (low_card_cap=32, collapsed_fallback_nbins=5).
    mrmr_baseline_npermutations_cfg: int = 2
    mrmr_low_card_cap_cfg: int = 32
    mrmr_collapsed_fallback_nbins_cfg: int = 5
    cv_selector_mode_cfg: str = "mean"
    # S27 close-out: TrainingBehaviorConfig.auto_wrap_partial_fit_es real
    # ctor param. False = leave wrap ON (source default); True = force OFF.
    auto_wrap_partial_fit_es_force_off_cfg: bool = False
    # 2026-05-30 audit-pass-6 LOW-tier deferred batch (W6 LOW). 28 axes,
    # S27 dropped (no source ctor param). Defaults source-verified against
    # ShapProxiedFS.__init__ (feature_selection/shap_proxied_fs.py:79-113),
    # TrainingBehaviorConfig.early_stop_on_worsening_{coeff,min_iters}
    # (_model_configs.py:506-507), MRMR Wave 8 sibling knobs
    # (filters/mrmr.py:241,249,252,265), and CompositeTargetDiscoveryConfig
    # cv_selector_{alpha,confidence,quantile_level} + cv_persist_fold_scores
    # (_composite_target_discovery_config.py:127-130).
    shap_proxied_prefilter_stage1_keep_cfg: "int | None" = None
    shap_proxied_prefilter_univariate_batch_size_cfg: "int | None" = None
    shap_proxied_shap_prefilter_enabled_cfg: bool = True
    shap_proxied_shap_prefilter_safety_factor_cfg: int = 4
    shap_proxied_shap_prefilter_min_features_cfg: int = 40
    shap_proxied_shap_aware_stage1_keep_cfg: bool = True
    shap_proxied_shap_aware_stage1_cushion_cfg: int = 2
    shap_proxied_shap_aware_stage1_floor_cfg: int = 200
    shap_proxied_refine_ucb_enabled_cfg: bool = True
    shap_proxied_refine_ucb_min_eval_size_cfg: "int | None" = None
    shap_proxied_refine_ucb_slack_cfg: "float | None" = None
    shap_proxied_refine_ucb_stdev_multiplier_cfg: float = 1.0
    shap_proxied_revalidation_n_estimators_cfg: "int | None" = 100
    shap_proxied_revalidation_ucb_enabled_cfg: bool = True
    shap_proxied_revalidation_ucb_min_eval_size_cfg: "int | None" = None
    shap_proxied_revalidation_ucb_slack_cfg: "float | None" = None
    shap_proxied_revalidation_ucb_stdev_multiplier_cfg: "float | None" = None
    shap_proxied_inner_n_jobs_cap_cfg: bool = False
    early_stop_on_worsening_coeff_cfg: int = 5
    early_stop_on_worsening_min_iters_cfg: int = 5
    mrmr_relaxmrmr_alpha_cfg: float = 0.0
    mrmr_uaed_auto_size_cfg: bool = False
    mrmr_cpt_test_cfg: bool = False
    mrmr_pid_synergy_bonus_cfg: float = 0.0
    cv_selector_alpha_cfg: float = 1.0
    cv_selector_confidence_cfg: float = 0.9
    cv_selector_quantile_level_cfg: float = 0.9
    cv_persist_fold_scores_cfg: bool = False
    # 2026-05-31 audit-pass-8 HIGH (#1-#4). Defaults source-verified at HEAD:
    #   #1 mrmr_cardinality_bias_correction_cfg: True
    #      (src/mlframe/feature_selection/filters/mrmr.py:334)
    #   #2 mrmr_min_relevance_gain_relative_to_first_cfg: 0.05
    #      (src/mlframe/feature_selection/filters/mrmr.py:326)
    #   #3 mlp_random_state_cfg: None
    #      (src/mlframe/training/neural/base.py:217)
    #   #4 mlp_class_weight_cfg: None
    #      (src/mlframe/training/neural/base.py:218)
    mrmr_cardinality_bias_correction_cfg: bool = True
    mrmr_min_relevance_gain_relative_to_first_cfg: float = 0.05
    mlp_random_state_cfg: "int | None" = None
    mlp_class_weight_cfg: "str | None" = None
    # 2026-05-31 audit-pass-8 MED + LOW->MED (#5/#7/#8/#9/#10). Defaults
    # source-verified at HEAD:
    #   #5 shap_proxied_adaptive_prescreen_by_stability_cfg: False
    #      (src/mlframe/feature_selection/shap_proxied_fs.py:208)
    #   #7 mlp_use_layernorm_cfg: False
    #      (src/mlframe/training/neural/flat.py:205; doc-cite drift -- the
    #       audit said line 145 which is part of the ResidualBlock docstring,
    #       the real ``generate_mlp`` signature default lives at :205)
    #   #8 mlp_l1_alpha_cfg: 0.0 (no fuzz exposure pre-iter613; default is
    #      whatever the suite/builder forwards. Library default for the
    #      hparam is 0.0; the BN/LN/GN exclusion branch at
    #      _flat_torch_module.py:272-301 only fires when l1_alpha > 0)
    #   #9 mlp_inject_zero_sample_weight_batch_cfg: False (the
    #      ``_warned_zero_weight_batch`` once-per-fit WARN at
    #      _flat_torch_module.py:233-256 is dead code in fuzz today; True
    #      arms the frame-builder side to spike weights so the branch fires)
    #   #10 inject_xor_synergy_pair_cfg: False (fuzz frames emit no
    #       guaranteed XOR-synergy pair today; True arms the frame-builder
    #       side so the _force_cond branch at
    #       feature_selection/filters/evaluation.py:596 surfaces a pure-
    #       synergy survivor in mrmr_gains_)
    shap_proxied_adaptive_prescreen_by_stability_cfg: bool = False
    mlp_use_layernorm_cfg: bool = False
    mlp_l1_alpha_cfg: float = 0.0
    mlp_inject_zero_sample_weight_batch_cfg: bool = False
    inject_xor_synergy_pair_cfg: bool = False
    # 2026-05-31 audit-pass-9 (W9). Defaults source-verified at HEAD:
    #   #1 mlp_adamw_betas_cfg = (0.9, 0.95)
    #      (src/mlframe/training/neural/_flat_torch_module.py:499)
    #   #2 mlp_use_ema_cfg = False
    #      (src/mlframe/training/neural/base.py:266)
    #   #3 mlp_label_smoothing_cfg = 0.0
    #      (src/mlframe/training/neural/base.py:268)
    #   #4 mlp_focal_loss_gamma_cfg = None
    #      (src/mlframe/training/neural/base.py:269)
    #   #5 mlp_use_residual_cfg = False
    #      (src/mlframe/training/neural/flat.py:208)
    #   #6 mlp_numerical_embedding_cfg = None
    #      mlp_numerical_embedding_kwargs_cfg = "paper_default"
    #      (src/mlframe/training/neural/flat.py:209-210)
    #   #7 mrmr_fe_hybrid_orth_enable_cfg = False (mrmr.py:656)
    #      mrmr_fe_hybrid_orth_pair_enable_cfg = True (mrmr.py:664;
    #         meaningful only when master is on)
    mlp_adamw_betas_cfg: "tuple[float, float]" = (0.9, 0.95)
    mlp_use_ema_cfg: bool = False
    mlp_label_smoothing_cfg: float = 0.0
    mlp_focal_loss_gamma_cfg: "float | None" = None
    mlp_use_residual_cfg: bool = False
    mlp_numerical_embedding_cfg: "str | None" = None
    mlp_numerical_embedding_kwargs_cfg: str = "paper_default"
    mrmr_fe_hybrid_orth_enable_cfg: bool = False
    mrmr_fe_hybrid_orth_pair_enable_cfg: bool = True
    # 2026-05-31 audit-pass-10 (W10). Defaults source-verified at HEAD:
    #   #1 mlp_optimizer_cfg = "adamw"
    #      (training/neural/_flat_torch_module.py:86 -- ``optimizer = optimizer
    #      or torch.optim.AdamW`` falls back to AdamW when caller did not
    #      override). MuonAdamWHybrid class lives at
    #      training/neural/_muon_optimizer.py:123; wiring contract per
    #      docstring at _muon_optimizer.py:20: model_params["optimizer"] =
    #      MuonAdamWHybrid.
    #   #2 mrmr_fe_hybrid_orth_degrees_cfg = (2, 3)
    #      (feature_selection/filters/mrmr.py:657)
    #   #3 mrmr_fe_hybrid_orth_basis_cfg = "auto"
    #      (feature_selection/filters/mrmr.py:658)
    #   #4 mrmr_fe_hybrid_orth_top_k_cfg = 5
    #      (feature_selection/filters/mrmr.py:663)
    #   #6 mrmr_fe_hybrid_orth_pair_max_degree_cfg = 2
    #      (feature_selection/filters/mrmr.py:665)
    mlp_optimizer_cfg: str = "adamw"
    mrmr_fe_hybrid_orth_degrees_cfg: "tuple[int, ...]" = (2, 3)
    mrmr_fe_hybrid_orth_basis_cfg: str = "auto"
    mrmr_fe_hybrid_orth_top_k_cfg: int = 5
    mrmr_fe_hybrid_orth_pair_max_degree_cfg: int = 2
    # 2026-05-31 audit-pass-12 (W12). Defaults source-verified at HEAD:
    #   Group A (F-34 MTR dispatch):
    #     A1 composite_target_multilabel_strategy_cfg = "per_target"
    #        (src/mlframe/training/_composite_target_discovery_config.py:773
    #         + validator at :940 accepts
    #         {"per_target", "skip", "multi_target_regression"})
    #     A2 enable_ct_ensemble_cfg = True (existing suite-side default; new
    #        axis surfaces the
    #        ``_phase_composite_post_xt_ensemble._build_cross_target_ensemble_for_target``
    #        early-return WARN gate when target_type=multi_target_regression)
    #     A3 mtr_eval_metric_cfg = None (canon-only marker for the new
    #        metrics_registry MTR entries: ``rmse_macro/_micro/_max``,
    #        ``mae_macro/_max``, ``r2_macro/_min`` -- see
    #        src/mlframe/training/metrics_registry.py
    #        ``_register_builtin_multi_target_regression``)
    #   Group B (MRMR FE layers):
    #     B1 mrmr_fe_kfold_te_enable_cfg = False
    #        (feature_selection/filters/mrmr.py:705)
    #     B2 mrmr_fe_missingness_indicator_enable_cfg = False (mrmr.py:749)
    #        mrmr_fe_missingness_count_enable_cfg     = False (mrmr.py:751)
    #        mrmr_fe_missingness_pattern_enable_cfg   = False (mrmr.py:752)
    #     B3 mrmr_fe_cat_aux_enable_cfg = "off" (single 4-way axis that maps
    #        to the three master switches at mrmr.py:723/725/727)
    #     B4 mrmr_fe_hybrid_orth_extra_bases_cfg = ()
    #        (mrmr.py:676; only meaningful under master fe_hybrid_orth)
    #     B5 mrmr_fe_ratio_delta_diff_cfg = "off" (single 4-way axis covering
    #        mrmr.py:769/772/774/777 master switches)
    #     B6 mrmr_fe_mi_greedy_enable_cfg = False (mrmr.py:691)
    #   Group C (MRMR + ShapProxiedFS artifact-reuse pipeline):
    #     C1 mrmr_shap_proxy_artifact_reuse_cfg = "off"
    #        (couples MRMR.retain_artifacts at mrmr.py:787 with
    #         ShapProxiedFS.precomputed at shap_proxied_fs.py:258)
    #        mrmr_shap_proxy_align_mode_cfg = "exact"
    #        (covers _shap_proxy_precomputed.align_precomputed_to_X branches:
    #         exact / permutation / subset / mismatched at :168, :180, :216)
    composite_target_multilabel_strategy_cfg: str = "per_target"
    enable_ct_ensemble_cfg: bool = True
    mtr_eval_metric_cfg: "str | None" = None
    mrmr_fe_kfold_te_enable_cfg: bool = False
    mrmr_fe_missingness_indicator_enable_cfg: bool = False
    mrmr_fe_missingness_count_enable_cfg: bool = False
    mrmr_fe_missingness_pattern_enable_cfg: bool = False
    mrmr_fe_cat_aux_enable_cfg: str = "off"
    mrmr_fe_hybrid_orth_extra_bases_cfg: tuple = ()
    mrmr_fe_ratio_delta_diff_cfg: str = "off"
    mrmr_fe_mi_greedy_enable_cfg: bool = False
    mrmr_shap_proxy_artifact_reuse_cfg: str = "off"
    mrmr_shap_proxy_align_mode_cfg: str = "exact"
    # 2026-05-31 audit-pass-14 (W14). Defaults source-verified at HEAD:
    #   F14-1 shap_proxied_cluster_backend_cfg = "auto"
    #         (src/mlframe/feature_selection/shap_proxied_fs.py:258)
    #   F14-3 mrmr_partial_fit_decay_cfg = 0.0
    #         mrmr_partial_fit_min_recompute_cfg = 100
    #         mrmr_partial_fit_window_cfg = None
    #         (src/mlframe/feature_selection/filters/mrmr.py:845-847)
    #   F14-4 mrmr_dcd_tau_cluster_cfg = 0.7
    #         (src/mlframe/feature_selection/filters/mrmr.py:621)
    #   F14-5 mrmr_dcd_distance_cfg = "su"  (mrmr.py:622)
    #         mrmr_dcd_swap_method_cfg = "auto"  (mrmr.py:655)
    shap_proxied_cluster_backend_cfg: str = "auto"
    mrmr_partial_fit_decay_cfg: float = 0.0
    mrmr_partial_fit_min_recompute_cfg: int = 100
    mrmr_partial_fit_window_cfg: "int | None" = None
    mrmr_dcd_tau_cluster_cfg: "float | str" = 0.7
    mrmr_dcd_distance_cfg: str = "su"
    mrmr_dcd_swap_method_cfg: str = "auto"
    # iter639 audit-pass-15 — MRMR hybrid-orth scorer family (Layers 62, 63,
    # 76, 85) + MLP optimizer wrappers (F-62 Lookahead, F-63 SAM, F-68/69/70
    # Mixup) + F-72 output-only spectral norm. Defaults source-verified at
    # HEAD against feature_selection/filters/mrmr.py and
    # training/neural/_flat_torch_module.py / training/neural/flat.py.
    mrmr_fe_hybrid_orth_default_scorer_cfg: str = "plug_in"
    mrmr_fe_hybrid_orth_meta_enable_cfg: bool = False
    mrmr_fe_hybrid_orth_bootstrap_enable_cfg: bool = False
    mrmr_fe_hybrid_orth_three_gate_enable_cfg: bool = False
    mlp_use_sam_cfg: bool = False
    mlp_use_lookahead_cfg: bool = False
    mlp_use_mixup_cfg: bool = False
    mlp_spectral_norm_output_only_cfg: bool = False
    # iter642 audit-pass-15 batch 2 — 6 remaining MRMR hybrid-orth sub-
    # features. Defaults source-verified at HEAD against MRMR.__init__
    # (mrmr.py:1044/784/800/749/845/767).
    mrmr_fe_hybrid_orth_ensemble_enable_cfg: bool = False
    mrmr_fe_hybrid_orth_lasso_enable_cfg: bool = False
    mrmr_fe_hybrid_orth_elasticnet_enable_cfg: bool = False
    mrmr_fe_hybrid_orth_adaptive_arity_enable_cfg: bool = False
    mrmr_fe_hybrid_orth_diff_basis_enable_cfg: bool = False
    mrmr_fe_semi_supervised_enable_cfg: bool = False
    # audit-pass-16 — MRMR Layers 87-91. Defaults source-verified at HEAD
    # against MRMR.__init__ (mrmr.py:1255/1268/1270/1285/1300/1302/1243/1245).
    mrmr_fe_grouped_agg_enable_cfg: bool = False
    mrmr_fe_grouped_quantile_enable_cfg: bool = False
    mrmr_fe_grouped_quantile_target_aware_cfg: bool = False
    mrmr_fe_cat_pair_enable_cfg: bool = False
    mrmr_fe_numeric_decompose_enable_cfg: bool = False
    mrmr_fe_numeric_decompose_digits_cfg: tuple = (0, 1, 2)
    mrmr_fe_local_mi_gate_cfg: bool = True  # audit-pass-17: source default flipped True (L97)
    mrmr_fe_unified_second_pass_gate_cfg: bool = False
    # audit-pass-17 — Param-Oracle / fe_auto + FE families L92-104.
    mrmr_fe_auto_cfg: bool = False
    mrmr_fe_temporal_agg_enable_cfg: bool = False
    mrmr_fe_composite_group_agg_enable_cfg: bool = False
    mrmr_fe_modular_enable_cfg: bool = False
    mrmr_fe_group_distance_enable_cfg: bool = False
    mrmr_fe_rare_category_enable_cfg: bool = False
    mrmr_fe_conditional_residual_enable_cfg: bool = False
    # 2026-06-13 coverage refresh -- embedding passthrough + 5 default-ON / 1
    # default-OFF MRMR FE families. Defaults mirror MRMR.__init__ source.
    mrmr_embedding_passthrough_cfg: bool = True
    mrmr_embedding_passthrough_detect_embeddings_cfg: bool = True
    mrmr_embedding_passthrough_detect_text_cfg: bool = True
    mrmr_fe_hinge_enable_cfg: bool = True
    mrmr_fe_conditional_dispersion_enable_cfg: bool = True
    mrmr_fe_wavelet_enable_cfg: bool = True
    mrmr_fe_stability_vote_enable_cfg: bool = True
    mrmr_fe_sufficient_summary_early_stop_cfg: bool = True
    mrmr_fe_gradient_interaction_enable_cfg: bool = False
    # Learnable categorical embeddings default-on (nn.Embedding); this axis also samples the legacy CatBoostEncoder OFF path + a fixed embed dim vs the fastai heuristic (None).
    mlp_use_learnable_cat_embeddings_cfg: bool = True
    mlp_categorical_embed_dim_cfg: "int | None" = None

    def canonical_key(self) -> tuple:
        """Hashable tuple used for dedup. Canonicalizes semantically
        equivalent combos so e.g. ``align_polars_categorical_dicts=True`` with
        pandas input collapses to the False variant."""
        align = self.align_polars_categorical_dicts
        # 2026-05-12 Wave 30: polars input only reaches models that
        # support it natively. For non-polars-native models (LGB,
        # linear, MLP), canonicalise to pandas so the downstream
        # sklearn pipeline encoder (CatBoostEncoder) / strategies
        # receive pandas DataFrames they can consume. Without this, a
        # polars DataFrame hits CatBoostEncoder.fit and raises
        # ``ValueError: Unexpected input type: <class 'polars...'>``
        # (surfaced c0002/c0004 fuzz failures).
        _input_type = self.input_type
        if _input_type != "pandas":
            _any_polars_native = any(
                m in self.models for m in ("cb", "xgb", "hgb", "mlp", "lstm", "gru", "transformer")
            )
            if not _any_polars_native:
                _input_type = "pandas"
        null_frac = self.null_fraction_cats if self.cat_feature_count > 0 else 0.0
        # fairness_col is meaningful only if that column exists → None
        # when cat_feature_count == 0 (no cat_0 to reference).
        fairness = self.fairness_col if self.cat_feature_count > 0 else None
        # custom_prep=pca2 makes sense only on a clean all-numeric
        # frame — IncrementalPCA can't consume:
        #   * string/cat/text/embedding columns (no pre-encoding before
        #     custom_pre_pipeline in the mlframe pipeline),
        #   * NaN values (sklearn IncrementalPCA explicitly rejects NaN;
        #     the error message even suggests HistGradientBoosting as
        #     the alternative),
        #   * all-null / all-const columns (degenerate for PCA's
        #     variance computation).
        # Canonicalise custom_prep → None for any of those axes so
        # the pairwise sampler doesn't waste combos on a guaranteed-fail
        # configuration. Users who want PCA on real data must
        # pre-process upstream.
        pca_incompatible = (
            self.cat_feature_count > 0
            or self.text_col_count > 0
            or self.embedding_col_count > 0
            or self.inject_inf_nan          # injects np.nan → PCA rejects
            or self.inject_degenerate_cols  # adds all-null column → PCA rejects
        )
        custom_prep = self.custom_prep if not pca_incompatible else None
        use_ensembles = self.use_ensembles
        target_carrier = self.target_carrier
        if self.target_type == "multilabel_classification":
            # Multilabel FTE intentionally unpacks list cells to (N, K)
            # ndarray. Native list-Series targets are a different
            # contract, so keep that axis for a dedicated future sensor.
            target_carrier = "numpy"
        _has_neural = ("mlp" in self.models) or (self.recurrent_model_cfg is not None) or any(m in self.models for m in ("lstm", "gru", "transformer"))
        return (
            tuple(sorted(self.models)),
            _input_type,
            self.n_rows,
            self.cat_feature_count,
            null_frac,
            self.use_mrmr_fs,
            tuple(sorted(self.weight_schemas)),
            # 2026-05-31 audit-pass-9 #8: multi_target_regression collapses
            # to "regression" when no native-MTR model is in the subset (no
            # native multi-target backend -- the suite would either crash
            # or silently fall back to sklearn MultiOutputRegressor wrap,
            # which is the regression code path under a thin adapter).
            # Native-MTR backends per docs/multi_target_regression_design.md:
            # CatBoost (MultiRMSE) + MLP (F-24 (N, K) auto-detect).
            (
                self.target_type
                if (
                    self.target_type != "multi_target_regression"
                    or any(m in self.models for m in ("mlp", "cb"))
                )
                else "regression"
            ),
            target_carrier,
            self.auto_detect_cats,
            align,
            self.prefer_polarsds,
            self.use_text_features,
            self.honor_user_dtype,
            # text_col_count: passthrough. The historical canonicalisation
            # here zeroed text columns on small-n CB + heavy NaN combos
            # (fuzz c0056 / c0070) where CB's default
            # ``occurrence_lower_bound=50`` cannot build a TF-IDF
            # dictionary on tiny inner-CV folds. That production hang
            # is now fixed in ``training/helpers.compute_cb_text_processing``
            # (called from ``trainer._train_model_with_fallback`` and
            # the RFECV inner-fold path in ``feature_selection/wrappers``)
            # which scales ``occurrence_lower_bound`` proportionally to
            # the actual fit-time row count. The fuzz axis stays fully
            # exercised.
            self.text_col_count,
            self.embedding_col_count,
            # 2026-04-24 combo-extension axes
            # OneClassSVM has O(n²) fit cost — collapse to None on the
            # large-row tier (n>=1200) so it doesn't dominate runtime.
            # IsolationForest and LOF stay enabled across all sizes.
            None if (self.outlier_detection == "ocsvm" and self.n_rows >= 1200) else self.outlier_detection,
            use_ensembles,
            self.continue_on_model_failure,
            self.iterations,
            # ``prefer_calibrated_classifiers=True`` + multilabel raises
            # ``NotImplementedError`` at trainer.py:_validate_multilabel_calibration_compat
            # (CalibratedClassifierCV is single-output only). Canonicalise
            # to False for multilabel combos so dedup collapses the
            # known-incompatible variant. Surfaced 2026-04-28 default seed
            # c0060 (lgb_xgb / multilabel + prefer_calibrated=True) -
            # previously masked because the polars ``pl.List`` -> pandas
            # roundtrip presented as 1-D, and the calibration guard
            # didn't fire; with the multilabel target normalised the
            # guard correctly rejects, so the canon owns the dedup.
            False if (self.prefer_calibrated_classifiers and self.target_type == "multilabel_classification") else self.prefer_calibrated_classifiers,
            # CB+multilabel+degenerate canon RETIRED 2026-04-27 (batch 2).
            # Production fix: explicit cat_features list now drops columns
            # whose dtype is numeric (via the existing dtype-aware filter
            # in feature_selection/wrappers.py + the new num-degenerate
            # guard in trainer._train_model_with_fallback). The fuzz axis
            # exercises the full inject_degenerate_cols × CB × multilabel
            # cross-product again.
            self.inject_degenerate_cols,
            self.inject_inf_nan,
            self.with_datetime_col,
            self.inject_zero_col,
            fairness,
            custom_prep,
            self.input_storage,
            # 2026-04-24 round 2 — config field axes
            self.fillna_value_cfg,
            self.scaler_name_cfg,
            self.categorical_encoding_cfg,
            self.skip_categorical_encoding_cfg,
            self.val_placement_cfg,
            self.test_size_cfg,
            self.trainset_aging_limit_cfg,
            self.cat_text_card_threshold_cfg,
            self.early_stopping_rounds_cfg,
            self.use_robust_eval_metric_cfg,
            # Fix G — adversarial axes
            self.inject_label_leak,
            self.inject_rank_deficient,
            self.inject_all_nan_col,
            # R3 — drift, imbalance, weird-cat
            # inject_test_drift canonicalises to None when n_rows is too
            # small to meaningfully distinguish train from test slices.
            self.inject_test_drift if self.n_rows >= 300 else None,
            # imbalance_ratio canonicalisation: meaningful only on
            # binary classification. Extreme imbalance on small frames
            # causes random val/test splits to drop one class entirely
            # ("CatBoostError: Target contains only one unique value",
            # 2026-04-24 c0062/c0085). Clamp by expected minority count
            # per 10% slice — need ≥2 minority rows in each split's
            # worst-case unlucky draw, i.e. frac * slice_size ≥ ~4 → need
            # frac * n * 0.1 ≥ 4 → frac ≥ 40/n. At n=1200, rare_1pct
            # (0.01) = 12 total minority, slice=1.2 — unreliable → clamp
            # to rare_5pct. rare_5pct survives at n≥800.
            self._canonical_imbalance(),
            # weird_cat_content relevant only if there are cat columns.
            self.weird_cat_content if self.cat_feature_count > 0 else None,
            # Phase H: multilabel_strategy_cfg only meaningful for multilabel.
            # `chain` dispatch breaks two assumptions when the inputs aren't
            # already classifier-friendly numeric pandas:
            #   * sklearn ClassifierChain wraps the inner estimator and
            #     forwards the raw frame — polars Enum / Utf8 + raw cat
            #     columns reach HGB/LGB which can't handle strings (Cluster
            #     "could not convert string to float: 'A'").
            #   * the linear model in multilabel uses LinearRegression which
            #     produces (N, K) predictions; report_regression_model_perf
            #     then crashes plotting 2-D preds vs targets.
            # Downgrade chain → wrapper for combos that would hit either,
            # so chain is exercised only on safe (pandas, no-cats, no-linear)
            # multilabel combos.
            self._canonical_multilabel_strategy(),
            # 2026-04-26 batch 1 — config-field axes
            # fix_infinities=False is intentional Inf pass-through (per
            # preprocessing.py:154-156). Combining it with inject_inf_nan
            # (which puts np.inf into num_0) feeds raw Inf to XGB/HGB and
            # crashes them — that's a user-misconfiguration, not a bug.
            # Canonicalise away so the dedup pass collapses these combos
            # into the safe variant. The fix_infinities=False axis is still
            # exercised on clean-data combos.
            self.fix_infinities_cfg if not self.inject_inf_nan else True,
            self.ensure_float32_cfg,
            # remove_constant_columns axis is live on all combos: the prod robust scaler now guards a
            # zero-IQR column (skip / no-scale) instead of dividing by quantile(None)-quantile(None), so
            # degenerate / all-NaN columns no longer force constant-removal to avoid a scaler crash.
            self.remove_constant_columns_cfg,
            # imputer_strategy is meaningful only if nulls / NaNs exist.
            # Without missing values the imputer never fires → all variants
            # collapse to the default to avoid wasting combos on a no-op axis.
            self._canonical_imputer_strategy(),
            self.shuffle_val_cfg,
            self.shuffle_test_cfg,
            # wholeday_splitting requires a datetime column to take effect.
            self.wholeday_splitting_cfg if self.with_datetime_col else True,
            # val_sequential_fraction=1.0 + val_placement=backward + non-shuffled
            # splits + degenerate-col injection occasionally land an empty val
            # window after constant-col removal trims rows (c0102: shape=(0,29)
            # reaches SimpleImputer). Pre-existing splitter edge case — not
            # batch-4-introduced. Canonicalise the val_sequential_fraction
            # value down to 0.5 in the trigger window so dedup absorbs the
            # zero-row variant; the underlying splitter bug needs a separate
            # fix.
            (
                0.5
                if (
                    self.val_sequential_fraction_cfg == 1.0
                    and self.val_placement_cfg == "backward"
                    and not self.shuffle_val_cfg
                    and self.inject_degenerate_cols
                )
                else self.val_sequential_fraction_cfg
            ),
            # multilabel_n_chains/chain_order/cv only matter when
            # target_type=multilabel AND strategy=chain. For all other
            # combos these are no-ops, so canonicalise to defaults to
            # avoid wasting combo budget on identical-behaviour entries.
            self.multilabel_n_chains_cfg if self._is_chain_dispatch() else 3,
            self.multilabel_chain_order_cfg if self._is_chain_dispatch() else "random",
            self.multilabel_cv_cfg if self._is_chain_dispatch() else 5,
            # PreprocessingExtensionsConfig axes — none of these tolerate
            # raw NaN/Inf or all-null columns; canonicalise to None when
            # NaN-injecting axes are on so the dedup pass collapses the
            # known-bad combinations. Categorical-bearing inputs are also
            # canonicalised away because the sklearn-bridge requires
            # already-encoded numeric input.
            self._canonical_prep_ext("scaler"),
            self._canonical_prep_ext("kbins"),
            self._canonical_prep_ext("polynomial_degree"),
            self._canonical_prep_ext("dim_reducer"),
            self._canonical_prep_ext("nonlinear"),
            # RFECV: only meaningful when the underlying model is in the
            # combo's mlframe_models list and n_rows is small enough that
            # the iterative re-fit doesn't blow runtime budget.
            self._canonical_rfecv_estimator(),
            # Recurrent: needs sequence inputs alongside the tabular df.
            # The synthetic builder emits sequences only on the small-row
            # tier (training time scales linearly per epoch with n).
            self._canonical_recurrent_model(),
            # 2026-05-11 Wave 15 — MRMR-internal knobs collapse to default
            # when MRMR is disabled. cat-FE-enable also collapses when
            # there are 0/1 cat features (cat-FE requires >=2). interactions
            # >= 2 collapses when cat_feature_count == 0 (cat-FE is the
            # primary trigger for the Wave-14-fixed kway-engineered-cols
            # path; pure numeric MRMR at order>=2 exercises a different
            # branch but is still useful to fuzz, so keep that variant).
            self.mrmr_interactions_max_order_cfg if self.use_mrmr_fs else 1,
            self.mrmr_fe_max_steps_cfg if self.use_mrmr_fs else 1,
            self.mrmr_cat_fe_enable_cfg if (
                self.use_mrmr_fs and self.cat_feature_count >= 2
            ) else True,
            # 2026-05-11 Wave 21 — config-toggle axes with relevance gates
            self.dummy_baselines_enabled_cfg,
            self.baseline_diagnostics_enabled_cfg,
            # use_groups only matters when wholeday_splitting is on AND there's
            # a datetime column to derive groups from.
            self.use_groups_cfg if (
                self.with_datetime_col and self.wholeday_splitting_cfg
            ) else True,
            # apply_outlier_to_val only matters when outlier_detection is set.
            self.apply_outlier_to_val_cfg if self.outlier_detection is not None else True,
            # multilabel_allow_uncalibrated only meaningful for multilabel.
            self.multilabel_allow_uncalibrated_cfg if (
                self.target_type == "multilabel_classification"
            ) else False,
            # report_residual_audit only matters for regression target.
            self.report_residual_audit_cfg if self.target_type == "regression" else True,
            # ltr_assume_comparable_scales only meaningful for LTR.
            self.ltr_assume_comparable_scales_cfg if (
                self.target_type == "learning_to_rank"
            ) else False,
            # composite-discovery only fires on regression targets and
            # only when at least 2 numeric base candidates exist. For
            # non-regression / no-numeric-bases combos collapse to the
            # disabled variant so dedup absorbs identical-behaviour
            # combos.
            self.composite_discovery_enabled_cfg if (
                self.target_type == "regression"
                # require enough columns for a base candidate pool (the
                # frame builder emits ``num_0..num_3`` for n_rows >= 300
                # so this is always satisfied when target=regression;
                # the gate is here for future shrinkage).
            ) else False,
            # composite_transforms_mode is a no-op when discovery is off.
            self.composite_transforms_mode_cfg if (
                self.composite_discovery_enabled_cfg
                and self.target_type == "regression"
            ) else None,
            # MRMR FE knobs canonicalise to defaults when
            # ``use_mrmr_fs=False`` so dedup collapses identical
            # behaviour combos. The library defaults (npermutations=0,
            # ntop_features=0, unary/binary_preset="minimal",
            # smart_polynom_iters=0, etc.) keep the FE step OFF and
            # the polynomial search disabled; any non-default value
            # activates a code path the prior fuzz axis space did
            # not exercise.
            self.mrmr_fe_npermutations_cfg if self.use_mrmr_fs else 0,
            self.mrmr_fe_ntop_features_cfg if self.use_mrmr_fs else 0,
            self.mrmr_fe_unary_preset_cfg if self.use_mrmr_fs else "minimal",
            self.mrmr_fe_binary_preset_cfg if self.use_mrmr_fs else "minimal",
            self.mrmr_fe_smart_polynom_iters_cfg if self.use_mrmr_fs else 0,
            # smart_polynom_steps is a no-op when smart_polynom_iters=0.
            self.mrmr_fe_smart_polynom_steps_cfg if (
                self.use_mrmr_fs and self.mrmr_fe_smart_polynom_iters_cfg > 0
            ) else 10,
            # min/max polynom_degree only matter when smart_polynom is on.
            self.mrmr_fe_min_polynom_degree_cfg if (
                self.use_mrmr_fs and self.mrmr_fe_smart_polynom_iters_cfg > 0
            ) else 3,
            self.mrmr_fe_max_polynom_degree_cfg if (
                self.use_mrmr_fs and self.mrmr_fe_smart_polynom_iters_cfg > 0
            ) else 3,
            # CatFEConfig.include_numeric only matters when cat-FE is
            # enabled AND MRMR is on AND there are categorical columns
            # to mix discretized numerics with.
            self.mrmr_cat_fe_include_numeric_cfg if (
                self.use_mrmr_fs
                and self.mrmr_cat_fe_enable_cfg
                and self.cat_feature_count > 0
            ) else False,
            # 2026-05-19 -- polynomial auto-tune axes are no-ops when
            # prep_ext_polynomial_degree_cfg is None (the whole polynomial
            # step is OFF). Canonicalise to library defaults so dedup
            # absorbs combos that differ only on inactive knobs.
            self.prep_ext_polynomial_max_features_cfg if (
                self.prep_ext_polynomial_degree_cfg is not None
            ) else 10_000,
            self.prep_ext_polynomial_interaction_only_cfg if (
                self.prep_ext_polynomial_degree_cfg is not None
            ) else True,
            # memory_safety_max_bytes guards every sklearn-bridge stage
            # output, not just polynomial; keep meaningful whenever ANY
            # prep-ext stage is on. Canonicalise to the default cap when
            # none of scaler / kbins / polynomial / dim_reducer /
            # nonlinear is active.
            self.prep_ext_memory_safety_max_bytes_cfg if (
                self.prep_ext_scaler_cfg is not None
                or self.prep_ext_kbins_cfg is not None
                or self.prep_ext_polynomial_degree_cfg is not None
                or self.prep_ext_dim_reducer_cfg is not None
                or self.prep_ext_nonlinear_cfg is not None
            ) else 500_000_000,
            # Composite stacked-discovery knobs no-op unless composite
            # discovery is enabled AND the target is regression (the
            # discovery is regression-only). Canonicalise to False so
            # dedup collapses identical-behaviour combos.
            self.composite_use_stacked_discovery_cfg if (
                self.composite_discovery_enabled_cfg
                and self.target_type == "regression"
            ) else False,
            self.composite_use_stacked_discovery_residual_cfg if (
                self.composite_discovery_enabled_cfg
                and self.target_type == "regression"
            ) else False,
            # skip_wrap_pass_predict toggles per-component re-wrap at
            # predict; only meaningful when composite discovery actually
            # produced wrappers (regression + enabled).
            self.composite_skip_wrap_pass_predict_cfg if (
                self.composite_discovery_enabled_cfg
                and self.target_type == "regression"
            ) else True,
            # 2026-05-22 -- TVT-MLP audit-followup gate axes. All only
            # meaningful when composite discovery is on; canonicalise
            # to the post-fix default when off.
            self.composite_skip_raw_dominates_ratio_cfg if (
                self.composite_discovery_enabled_cfg
                and self.target_type == "regression"
            ) else 0.0,
            self.composite_skip_ablation_delta_pct_cfg if (
                self.composite_discovery_enabled_cfg
                and self.target_type == "regression"
            ) else 0.0,
            self.composite_eps_mi_gain_cfg if (
                self.composite_discovery_enabled_cfg
                and self.target_type == "regression"
            ) else -10.0,
            self.composite_top_k_after_mi_cfg if (
                self.composite_discovery_enabled_cfg
                and self.target_type == "regression"
            ) else 32,
            self.composite_require_beats_raw_baseline_cfg if (
                self.composite_discovery_enabled_cfg
                and self.target_type == "regression"
            ) else False,
            self.composite_per_bin_n_bins_cfg if (
                self.composite_discovery_enabled_cfg
                and self.target_type == "regression"
            ) else 0,
            self.composite_tiny_screening_mode_cfg if (
                self.composite_discovery_enabled_cfg
                and self.target_type == "regression"
            ) else "per_family",
            self.composite_include_additive_residual_cfg if (
                self.composite_discovery_enabled_cfg
                and self.target_type == "regression"
            ) else True,
            # mlp_activation_cfg only meaningful when 'mlp' is in models
            # AND target_type is regression / classification head;
            # canonicalise to "ReLU" otherwise. NOTE: axis not yet
            # plumbed into the suite call (see comment in AXES dict).
            self.mlp_activation_cfg if (
                "mlp" in self.models
                and self.target_type in ("regression", "binary_classification",
                                          "multiclass_classification")
            ) else "ReLU",
            # 2026-05-21 -- mini-HPT (target + feature distribution analyzer)
            # toggle. Axis is meaningful on every target type since both
            # detectors run unconditionally when enabled.
            self.enable_target_distribution_analyzer_cfg,
            # 2026-05-21 -- FE check-pairs subsample knob is meaningful only
            # when the FE inner pass actually runs. Canonicalise to 0 when
            # use_mrmr_fs is False OR both FE entry points (npermutations +
            # ntop_features) are 0 -- the survivor-rebuild branch can't be
            # exercised in those cases, so dedup-collapsing 50_000 -> 0
            # avoids spending pairwise budget on identical-behaviour combos.
            # Subsample also needs n_rows > subsample_n: canon-to-0 at
            # n_rows=1000 (always <= 50_000 budget), keep at n_rows=200_000.
            self.fe_check_pairs_subsample_n_cfg if (
                self.use_mrmr_fs
                and (self.mrmr_fe_npermutations_cfg > 0 or self.mrmr_fe_ntop_features_cfg > 0)
                and self.n_rows > self.fe_check_pairs_subsample_n_cfg
            ) else 0,
            # 2026-05-21 iter150 -- extra_targets canon:
            #  - multilabel / LTR: collapse to None (multilabel is already 2-D
            #    within ONE target; LTR has its own ranker dispatch path that
            #    bypasses the target_by_type loop).
            #  - mixed_reg_bin / mixed_reg_bin_2each require regression primary:
            #    collapse to None otherwise (the secondary type would conflict
            #    with the primary target inference logic in the suite).
            (None if self.target_type in ("multilabel_classification", "learning_to_rank")
             else (None if (self.extra_targets in ("mixed_reg_bin", "mixed_reg_bin_2each")
                            and self.target_type != "regression")
                   else self.extra_targets)),
            # 2026-05-21 iter151 -- P0/P1/P2 canons. Each axis collapses to
            # its dataclass default when the feature it gates is inactive,
            # so semantically-no-op combos dedup down to one representative.
            # P0-1: quantile only meaningful on regression primary.
            (self.enable_quantile_regression_cfg if self.target_type == "regression" else False),
            # P0-2: linear axes only meaningful when "linear" in models.
            (self.linear_alpha_cfg if "linear" in self.models else 1.0),
            (self.linear_solver_cfg if "linear" in self.models else "lbfgs"),
            # P0-3 / P0-4: enable flags carry through; FHC/precomputed have
            # no preconditions on other combo axes so no further canon.
            self.enable_feature_handling_config_cfg,
            self.enable_precomputed_cfg,
            # P1-5: test_sequential_fraction is a time-axis split; only
            # meaningful when with_datetime_col is True (otherwise the
            # splitter has no time signal to sort by).
            (self.test_sequential_fraction_cfg if self.with_datetime_col else None),
            # P1-6: calib_size always meaningful; passthrough.
            self.calib_size_cfg,
            # P1-7: use_boruta_shap independent.
            self.use_boruta_shap_cfg,
            # importance driver only meaningful when BorutaShap is on; otherwise
            # canonicalise to the default so it doesn't split dedup buckets.
            (self.boruta_importance_measure_cfg if self.use_boruta_shap_cfg else "gini"),
            # 2026-06-03 FS-coverage audit -- BorutaShap.__init__ knobs only
            # meaningful when BorutaShap is on; collapse to the BorutaShap
            # signature default otherwise so dedup buckets don't split.
            (self.boruta_optimistic_cfg if self.use_boruta_shap_cfg else True),
            (self.boruta_train_or_test_cfg if self.use_boruta_shap_cfg else "train"),
            (self.boruta_premerge_clusters_cfg if self.use_boruta_shap_cfg else False),
            # 2026-06-04 FS-coverage follow-up -- BorutaShap.__init__ early_stop_*
            # knobs. The master toggle is only meaningful when BorutaShap is on;
            # patience/margin only bite when the master toggle is also on (they
            # are read solely inside the early_stop branch of
            # boruta_shap/_fit_explain), so collapse them to the BorutaShap
            # signature defaults outside that compound gate -- otherwise they'd
            # split dedup buckets for runs that behave identically.
            (self.boruta_early_stop_tentative_cfg if self.use_boruta_shap_cfg else False),
            (
                self.boruta_early_stop_patience_cfg
                if (self.use_boruta_shap_cfg and self.boruta_early_stop_tentative_cfg)
                else 20
            ),
            (
                self.boruta_early_stop_margin_cfg
                if (self.use_boruta_shap_cfg and self.boruta_early_stop_tentative_cfg)
                else 0.15
            ),
            # 2026-06-03 FS-coverage audit -- RFECV.__init__ knobs only
            # meaningful when an RFECV selector is in the pre-pipeline chain
            # (rfecv_estimator_cfg is not None); collapse to RFECV signature
            # defaults otherwise.
            (self.rfecv_votes_aggregation_cfg if self.rfecv_estimator_cfg is not None else "Borda"),
            (self.rfecv_search_method_cfg if self.rfecv_estimator_cfg is not None else "ModelBasedHeuristic"),
            # P1-8: use_sample_weights_in_fs only meaningful when any FS is
            # enabled (MRMR / RFECV / Boruta) AND weights schema includes
            # something non-uniform (otherwise FS receives all-1s weights
            # and the branch is a no-op).
            (self.use_sample_weights_in_fs_cfg if (
                (self.use_mrmr_fs or self.rfecv_estimator_cfg is not None
                 or self.use_boruta_shap_cfg)
                and any(s != "uniform" for s in self.weight_schemas)
            ) else False),
            # P1-9: fallback_to_sklearn only meaningful when polars-ds
            # preferred (prefer_polarsds=True). Otherwise the path is dead.
            (self.fallback_to_sklearn_cfg if self.prefer_polarsds else True),
            # P1-10a/b: device toggles always meaningful; passthrough.
            self.prefer_gpu_configs_cfg,
            self.prefer_cpu_for_lightgbm_cfg,
            # P2-16: mrmr_identity_cache_scope only meaningful when MRMR
            # is enabled.
            (self.mrmr_identity_cache_scope_cfg if self.use_mrmr_fs else "ctx"),
            # P2-17: skip_identity dedup-skip only meaningful when any FS
            # is active AND there's an ensembling path that could benefit
            # from non-deduped pipelines.
            (self.skip_identity_equivalent_pre_pipelines_cfg if (
                self.use_mrmr_fs or self.rfecv_estimator_cfg is not None
                or self.use_boruta_shap_cfg
            ) else True),
            # P2-18a/b: RFECV thresholds only meaningful when RFECV is on.
            (self.rfecv_leakage_corr_threshold_cfg if self.rfecv_estimator_cfg is not None else 0.95),
            (self.rfecv_mbh_adaptive_threshold_cfg if self.rfecv_estimator_cfg is not None else 30),
            # 2026-05-22 iter162 nested-config canons.
            # FHC sub-configs only meaningful when enable_feature_handling_config_cfg=True.
            (self.fhc_cache_eviction_strategy_cfg if self.enable_feature_handling_config_cfg else "size_weighted"),
            (self.fhc_cache_allow_pickle_cfg if self.enable_feature_handling_config_cfg else False),
            (self.fhc_cache_ram_fraction_cfg if self.enable_feature_handling_config_cfg else 0.3),
            (self.fhc_text_definite_text_mean_chars_cfg if self.enable_feature_handling_config_cfg else 100),
            (self.fhc_text_min_alphabet_entropy_cfg if self.enable_feature_handling_config_cfg else 4.5),
            (self.fhc_repro_deterministic_torch_cfg if self.enable_feature_handling_config_cfg else False),
            (self.fhc_auto_locale_detection_cfg if self.enable_feature_handling_config_cfg else "fallback_only"),
            # ReportingConfig nested -- always meaningful (suite always reports).
            self.reporting_prob_histogram_yscale_cfg,
            self.reporting_title_metrics_template_cfg,
            self.reporting_matplotlib_rcparams_cfg,
            # multiclass_panels only meaningful for multiclass / multilabel.
            (self.reporting_multiclass_panels_cfg if self.target_type in ("multiclass_classification", "multilabel_classification") else "CONFUSION PR_F1 ROC CALIB_GRID PROB_DIST TOP_K_ACC"),
            # ConfidenceAnalysisConfig.model_kwargs only when confidence_analysis enabled.
            (self.confidence_model_kwargs_cfg if self.include_confidence_analysis_cfg else "default"),
            # CompositeTargetDiscoveryConfig nested only when composite enabled (regression only).
            (self.composite_mi_estimator_cfg if self.composite_discovery_enabled_cfg and self.target_type == "regression" else "bin"),
            (self.composite_mi_nbins_cfg if self.composite_discovery_enabled_cfg and self.target_type == "regression" else 16),
            (self.composite_mi_aggregation_cfg if self.composite_discovery_enabled_cfg and self.target_type == "regression" else "mean"),
            (self.composite_mi_sample_strategy_cfg if self.composite_discovery_enabled_cfg and self.target_type == "regression" else "random"),
            # stacked_residual_aggregation only when composite_use_stacked_discovery_residual=True.
            (self.composite_stacked_residual_aggregation_cfg if (
                self.composite_discovery_enabled_cfg
                and self.target_type == "regression"
                and self.composite_use_stacked_discovery_residual_cfg
            ) else "mean"),
            (self.composite_discovery_n_jobs_cfg if self.composite_discovery_enabled_cfg and self.target_type == "regression" else 1),
            # QuantileRegressionConfig nested only when enable_quantile and regression primary.
            (self.quantile_crossing_fix_cfg if self.enable_quantile_regression_cfg and self.target_type == "regression" else "sort"),
            (self.quantile_coverage_pairs_cfg if self.enable_quantile_regression_cfg and self.target_type == "regression" else "default"),
            (self.quantile_wrapper_n_jobs_cfg if self.enable_quantile_regression_cfg and self.target_type == "regression" else "auto"),
            # MLP predict batch size only when "mlp" in models.
            (self.mlp_predict_batch_size_cfg if "mlp" in self.models else None),
            # LearningToRankConfig nested only for LTR.
            (self.ltr_cb_loss_fn_cfg if self.target_type == "learning_to_rank" else "YetiRankPairwise"),
            (self.ltr_lgb_objective_cfg if self.target_type == "learning_to_rank" else "lambdarank"),
            (self.ltr_rrf_k_cfg if self.target_type == "learning_to_rank" and self.ranking_ensemble_method == "rrf" else 60),
            # RecurrentConfig nested only when a recurrent model is requested.
            (self.recurrent_precision_cfg if self._canonical_recurrent_model() is not None else "32-true"),
            (self.recurrent_sequence_preprocessing_cfg if self._canonical_recurrent_model() is not None else "none"),
            # F5 -- PySR cannot consume non-finite values; when inject_inf_nan is on, the True/False variants are behaviour-identical (both crash on the first inf/nan row) so collapse the True variant to False to spare pairwise budget. Also collapse when the upstream cat-feature gate already disabled the bridge (cats + non-encoded path) since PySR runs inside the sklearn bridge.
            self._canonical_pysr_enabled(),
            # F1 -- enable_crash_reporting is a Windows-only Faulthandler dump hook; on non-Windows hosts the True variant is a no-op so collapse to False. On Windows the axis stays meaningful.
            self._canonical_enable_crash_reporting(),
            # 2026-05-26 iter291 -- new-functionality canons.
            # bucket_stratify is a regression-target split-time toggle. For
            # non-regression primaries it's a no-op gate, so collapse to the
            # default True so the dedup pass coalesces identical-behaviour
            # combos. Note: multi-target combos with a regression primary
            # keep the axis live (the split applies to the primary).
            (self.bucket_stratify_cfg if self.target_type == "regression" else True),
            # composite_cardinality_cap is meaningful only when bucket_stratify
            # is True AND there is a cat column whose cardinality could exceed
            # the cap (n_rows tier 200k routinely produces high-card splits).
            # Canonicalise to the default 200 otherwise so the cap variant
            # collapses cleanly.
            (
                self.composite_cardinality_cap_cfg
                if (self.target_type == "regression" and self.bucket_stratify_cfg
                    and (self.cat_feature_count > 0 or self.n_rows >= 50_000))
                else 200
            ),
            # honest_estimator_diagnostics is a reporting-side aggregator; always
            # meaningful regardless of target type (the per-target aggregator
            # walks every produced (target, model) pair).
            self.honest_estimator_diagnostics_cfg,
            # cross_target_ensemble_strategy only meaningful when composite
            # discovery is on AND target is regression. Canon to "nnls_stack"
            # default otherwise.
            (
                self.cross_target_ensemble_strategy_cfg
                if (self.composite_discovery_enabled_cfg
                    and self.target_type == "regression")
                else "nnls_stack"
            ),
            # cyclical / extended date features only meaningful when a datetime
            # column is present. Canon to False otherwise (the feature engine
            # has no datetime to consume).
            (self.add_cyclical_date_features_cfg if self.with_datetime_col else False),
            (self.add_extended_date_features_cfg if self.with_datetime_col else False),
            # NNLS weights in blends only meaningful when ensembling is on.
            (self.use_nnls_weights_in_blends_cfg if self.use_ensembles else True),
            # Prediction-envelope clip is regression-only. Canon to True
            # (the post-fix default) for non-regression primaries so the
            # False variant doesn't waste pairwise budget on no-op gates.
            (self.enable_prediction_envelope_clip_cfg if self.target_type == "regression" else True),
            # 2026-05-27 iter332 audit-driven canons.
            # Ensembling knobs only meaningful when use_ensembles is True.
            (self.ensembling_force_legacy_cfg if self.use_ensembles else False),
            (self.ensembling_quantile_budget_bytes_cfg if self.use_ensembles else 500 * 1024 * 1024),
            # flag_degenerate_conf_subset is binary-classification-only;
            # collapse to default for other target types.
            (
                self.ensembling_flag_degenerate_conf_subset_cfg
                if (self.use_ensembles and self.target_type == "binary_classification")
                else True
            ),
            # MLP knobs only meaningful when MLP is in scope.
            (self.mlp_extreme_ar_group_aware_skip_cfg if "mlp" in self.models else False),
            (self.mlp_extreme_ar_threshold_cfg if "mlp" in self.models else 0.99),
            (self.mlp_drop_per_group_constants_cfg if "mlp" in self.models else False),
            # Composite-target knobs only meaningful when composite discovery
            # is enabled AND target is regression (composite is regression-only).
            (
                self.composite_always_build_ct_ensemble_for_raw_cfg
                if (self.composite_discovery_enabled_cfg and self.target_type == "regression")
                else True
            ),
            (
                self.composite_ct_ensemble_dummy_floor_enabled_cfg
                if (self.composite_discovery_enabled_cfg and self.target_type == "regression")
                else True
            ),
            (
                self.composite_extreme_ar_group_aware_skip_cfg
                if (self.composite_discovery_enabled_cfg and self.target_type == "regression")
                else True
            ),
            (
                self.composite_oof_holdout_source_cfg
                if (self.composite_discovery_enabled_cfg and self.target_type == "regression")
                else "external_val"
            ),
            (
                self.composite_stacking_aware_gate_enabled_cfg
                if (self.composite_discovery_enabled_cfg and self.target_type == "regression")
                else False
            ),
            (
                self.composite_use_baseline_diagnostics_hint_cfg
                if (self.composite_discovery_enabled_cfg and self.target_type == "regression")
                else True
            ),
            # FS pre-screen only meaningful when MRMR / RFECV / Boruta active.
            (
                self.fs_pre_screen_unsupervised_cfg
                if (self.use_mrmr_fs or self.rfecv_estimator_cfg is not None
                    or self.use_boruta_shap_cfg)
                else True
            ),
            (
                self.fs_pre_screen_variance_threshold_cfg
                if (self.use_mrmr_fs or self.rfecv_estimator_cfg is not None
                    or self.use_boruta_shap_cfg)
                else 0.0
            ),
            # BaselineDiagnostics init_score top_k only meaningful when
            # baseline_diagnostics is enabled.
            (self.baseline_init_score_top_k_cfg if self.baseline_diagnostics_enabled_cfg else 1),
            # 2026-05-27 iter350 audit batch 2 canons.
            # AP12 calibrated probs in ensemble only meaningful when ensembling on.
            (self.use_ap12_calibrated_probs_in_ensemble_cfg if self.use_ensembles else True),
            # MLP weight_decay factor only meaningful when MLP is in scope.
            (self.mlp_extreme_ar_weight_decay_factor_cfg if "mlp" in self.models else 100.0),
            # Feature-drift auto-apply only meaningful when feature_drift report runs.
            self.feature_drift_auto_apply_neural_overrides_cfg,
            # Temporal audit column only meaningful when a datetime column exists.
            (self.target_temporal_audit_column_cfg if self.with_datetime_col else None),
            # Composite knobs only meaningful when composite discovery on AND regression.
            (
                self.composite_lag_predict_failsafe_tolerance_cfg
                if (self.composite_discovery_enabled_cfg and self.target_type == "regression")
                else 0.10
            ),
            (
                self.composite_extreme_ar_threshold_cfg
                if (self.composite_discovery_enabled_cfg and self.target_type == "regression")
                else 0.99
            ),
            (
                self.composite_ct_ensemble_dummy_floor_tolerance_cfg
                if (self.composite_discovery_enabled_cfg and self.target_type == "regression")
                else 0.0
            ),
            (
                self.composite_oof_holdout_frac_cfg
                if (self.composite_discovery_enabled_cfg and self.target_type == "regression")
                else 0.2
            ),
            (
                self.composite_top_m_after_tiny_cfg
                if (self.composite_discovery_enabled_cfg and self.target_type == "regression")
                else 10
            ),
            # tfidf_keep_sparse only meaningful when prep_ext has text + tfidf path.
            (
                self.prep_ext_tfidf_keep_sparse_cfg
                if self.text_col_count > 0
                else True
            ),
            # Recurrent use_attention only meaningful when a recurrent model is requested.
            (self.recurrent_use_attention_cfg if self._canonical_recurrent_model() is not None else True),
            # LTR xgb_objective only meaningful for LTR + xgb.
            (
                self.ltr_xgb_objective_cfg
                if (self.target_type == "learning_to_rank" and "xgb" in self.models)
                else "rank:ndcg"
            ),
            # BD init_score target types only meaningful when BD enabled.
            (
                self.baseline_init_score_apply_target_types_cfg
                if self.baseline_diagnostics_enabled_cfg
                else "regression_only"
            ),
            # 2026-05-27 MRMR friend-graph + cluster-aggregate (recent mrmr.py
            # features). All collapse to the mrmr.py defaults when MRMR is off
            # so non-FS combos don't gain phantom variation. friend_graph_prune
            # only bites when the graph is actually built; cluster_aggregate_mode
            # only when aggregation is enabled.
            (self.mrmr_build_friend_graph_cfg if self.use_mrmr_fs else True),
            (
                self.mrmr_friend_graph_prune_cfg
                if (self.use_mrmr_fs and self.mrmr_build_friend_graph_cfg)
                else False
            ),
            (self.mrmr_cluster_aggregate_enable_cfg if self.use_mrmr_fs else True),
            (
                self.mrmr_cluster_aggregate_mode_cfg
                if (self.use_mrmr_fs and self.mrmr_cluster_aggregate_enable_cfg)
                else "augment"
            ),
            # 2026-05-28 ShapProxiedFS axes. All sub-knobs collapse to the
            # ShapProxiedFS.__init__ defaults when use_shap_proxied_fs is off so
            # non-shap-proxied combos don't gain phantom variation.
            self.use_shap_proxied_fs,
            (self.shap_proxied_optimizer_cfg if self.use_shap_proxied_fs else "auto"),
            (self.shap_proxied_revalidate_cfg if self.use_shap_proxied_fs else True),
            (self.shap_proxied_trust_guard_cfg if self.use_shap_proxied_fs else True),
            (self.shap_proxied_interaction_aware_cfg if self.use_shap_proxied_fs else False),
            (self.shap_proxied_cluster_features_cfg if self.use_shap_proxied_fs else "auto"),
            # 2026-05-28 ShapProxiedFS extension axes (active_learning +
            # prefilter_method). Collapse to ShapProxiedFS __init__ defaults
            # (False / "auto") when use_shap_proxied_fs is off.
            (self.shap_proxied_active_learning_cfg if self.use_shap_proxied_fs else False),
            (self.shap_proxied_prefilter_method_cfg if self.use_shap_proxied_fs else "auto"),
            # 2026-05-28 ShapProxiedFS deeper extension axes (B1-B6 audit-pass-2).
            # All collapse to ShapProxiedFS.__init__ defaults when the selector
            # is off so non-shap combos don't gain phantom variation.
            (self.shap_proxied_config_jitter_cfg if self.use_shap_proxied_fs else False),
            (self.shap_proxied_uncertainty_penalty_cfg if self.use_shap_proxied_fs else 0.0),
            # within_cluster_refine ALSO depends on cluster_features being on
            # (literal False switches off the cluster pass entirely, so the
            # refine flag becomes a no-op); canon True there so the toggle
            # doesn't fork canonical keys it can't actually affect.
            (
                self.shap_proxied_within_cluster_refine_cfg
                if (self.use_shap_proxied_fs
                    and self.shap_proxied_cluster_features_cfg is not False)
                else True
            ),
            (self.shap_proxied_use_bias_corrector_cfg if self.use_shap_proxied_fs else True),
            (self.shap_proxied_refine_n_estimators_cfg if self.use_shap_proxied_fs else 100),
            (
                self.shap_proxied_trust_guard_n_estimators_cfg
                if self.use_shap_proxied_fs
                else 100
            ),
            # 2026-05-28 ShapProxiedFS audit-pass-3 axes (W3). All collapse to
            # ShapProxiedFS.__init__ defaults when the selector is off so
            # non-shap combos don't gain phantom variation. cluster_weighting
            # ALSO depends on cluster_features != False (the literal False
            # disables clustering entirely, making the weighting head a
            # no-op); canon to "pca_pc1" there so the toggle doesn't fork
            # canonical keys it can't actually affect. max_interaction_features
            # ALSO depends on interaction_aware=True (the cap is consumed only
            # by the interaction-tensor build); canon to 16 there so the value
            # doesn't fork canonical keys when interactions are off.
            (
                self.shap_proxied_cluster_weighting_cfg
                if (self.use_shap_proxied_fs
                    and self.shap_proxied_cluster_features_cfg is not False)
                else "pca_pc1"
            ),
            # iter624 canon: cluster_use_precomputed_bins / cluster_su_threshold
            # are consumed only when ShapProxiedFS clustering is active. Outside
            # that compound gate (use_shap_proxied_fs=True AND cluster_features
            # is not False) the toggle is a no-op; canon to source defaults
            # so it doesn't fork canonical keys it can't affect.
            (
                self.shap_proxied_cluster_use_precomputed_bins_cfg
                if (self.use_shap_proxied_fs
                    and self.shap_proxied_cluster_features_cfg is not False)
                else True
            ),
            (
                self.shap_proxied_cluster_su_threshold_cfg
                if (self.use_shap_proxied_fs
                    and self.shap_proxied_cluster_features_cfg is not False)
                else 0.5
            ),
            (
                self.shap_proxied_max_interaction_features_cfg
                if (self.use_shap_proxied_fs and self.shap_proxied_interaction_aware_cfg)
                else 16
            ),
            (self.shap_proxied_prefilter_top_cfg if self.use_shap_proxied_fs else 2000),
            (
                self.shap_proxied_prefilter_n_estimators_cfg
                if self.use_shap_proxied_fs
                else 100
            ),
            # 2026-05-28 ShapProxiedFS audit-pass-5 axes (W5). All collapse to
            # ShapProxiedFS.__init__ defaults when the selector is off so non-shap
            # combos don't gain phantom variation. Secondary gates per
            # AUDIT_PASS_5: stratified_anchors + uniform_tail_frac additionally
            # require trust_guard=True AND prefilter_method in ("two_stage",
            # "univariate") to cache an F-score vector (shap_proxied_fs.py:530-
            # 538); uniform_tail_frac further only matters when stratified_anchors
            # is on (otherwise no anchor weights to sample uniform-vs-weighted
            # against); zipf_alpha only matters when cardinality_dist=="zipf"
            # (alpha is unused for the uniform branch).
            (
                self.shap_proxied_trust_guard_stratified_anchors_cfg
                if (self.use_shap_proxied_fs
                    and self.shap_proxied_trust_guard_cfg
                    and self.shap_proxied_prefilter_method_cfg
                    in ("two_stage", "univariate"))
                else False
            ),
            (
                self.shap_proxied_trust_guard_uniform_tail_frac_cfg
                if (self.use_shap_proxied_fs
                    and self.shap_proxied_trust_guard_cfg
                    and self.shap_proxied_prefilter_method_cfg
                    in ("two_stage", "univariate")
                    and self.shap_proxied_trust_guard_stratified_anchors_cfg)
                else 0.2
            ),
            (
                self.shap_proxied_trust_guard_cardinality_dist_cfg
                if self.use_shap_proxied_fs
                else "zipf"
            ),
            (
                self.shap_proxied_trust_guard_zipf_alpha_cfg
                if (self.use_shap_proxied_fs
                    and self.shap_proxied_trust_guard_cardinality_dist_cfg == "zipf")
                else 0.25
            ),
            (
                self.shap_proxied_trust_guard_fidelity_weights_cfg
                if self.use_shap_proxied_fs
                else (0.6, 0.4)
            ),
            (
                self.shap_proxied_trust_guard_metric_cfg
                if self.use_shap_proxied_fs
                else "proxy_fidelity_score"
            ),
            (
                self.shap_proxied_fidelity_floor_cfg
                if self.use_shap_proxied_fs
                else 0.5
            ),
            (
                self.shap_proxied_oof_shap_n_estimators_cfg
                if self.use_shap_proxied_fs
                else 100
            ),
            # 2026-05-28 audit-pass-2 PART A canons.
            # EnsemblingConfig.degenerate_class_ratio only meaningful on
            # classification + ensembling (the degenerate-subset gate is
            # binary/multilabel-only); mirror the existing
            # flag_degenerate_conf_subset_cfg gating.
            (
                self.ensembling_degenerate_class_ratio_cfg
                if (self.use_ensembles
                    and self.target_type in ("binary_classification",
                                              "multilabel_classification"))
                else 0.01
            ),
            # target_temporal_audit_granularity only meaningful when the
            # temporal-audit phase actually runs (target_temporal_audit_column
            # resolved to a real datetime via with_datetime_col gating).
            (
                self.target_temporal_audit_granularity_cfg
                if (self.with_datetime_col
                    and self.target_temporal_audit_column_cfg == "ts_col")
                else "auto"
            ),
            # PreprocessingExtensionsConfig.dim_n_components only meaningful
            # when a dim-reducer is actually picked (PCA or TruncatedSVD).
            (
                self.prep_ext_dim_n_components_cfg
                if self.prep_ext_dim_reducer_cfg in ("PCA", "TruncatedSVD")
                else 50
            ),
            # 2026-05-28 TextDetectionConfig.text_min_cardinality. Collapse to
            # library default (300) when FHC is not enabled -- axis is a no-op
            # when the FHC sub-config never gets built.
            (self.fhc_text_min_cardinality_cfg if self.enable_feature_handling_config_cfg else 300),
            # 2026-05-28 CompositeTargetDiscoveryConfig deep knobs. Each collapses
            # to its library default when composite discovery is off OR the gating
            # sub-axis is off. composite_auto_skip_on_baseline_optimal_cfg ALSO
            # requires baseline_diagnostics_enabled_cfg; composite_mi_n_neighbors_cfg
            # ALSO requires composite_mi_estimator_cfg='knn'.
            (
                self.composite_auto_skip_on_baseline_optimal_cfg
                if (
                    self.composite_discovery_enabled_cfg
                    and self.target_type == "regression"
                    and self.baseline_diagnostics_enabled_cfg
                )
                else False
            ),
            (
                self.composite_mi_n_neighbors_cfg
                if (
                    self.composite_discovery_enabled_cfg
                    and self.target_type == "regression"
                    and self.composite_mi_estimator_cfg == "knn"
                )
                else 3
            ),
            (
                self.composite_auto_base_null_perms_cfg
                if (self.composite_discovery_enabled_cfg and self.target_type == "regression")
                else 20
            ),
            (
                self.composite_multi_base_max_k_cfg
                if (self.composite_discovery_enabled_cfg and self.target_type == "regression")
                else 3
            ),
            # 2026-05-28 TrainingBehaviorConfig.extreme_ar_group_aware_skip_models.
            # Only meaningful when MLP-extreme-AR skip is on AND mlp is in the
            # combo's models tuple (the skip-list axis only bites then).
            (
                self.extreme_ar_group_aware_skip_models_cfg
                if (self.mlp_extreme_ar_group_aware_skip_cfg and "mlp" in self.models)
                else "default_neural"
            ),
            # 2026-05-28 FeatureSelectionConfig.pre_screen_null_fraction_threshold.
            # Sibling of fs_pre_screen_variance_threshold_cfg: only fires when
            # an FS method (MRMR / RFECV / BorutaShap) actually invokes the
            # pre-screen path. Mirror the existing variance-threshold gating
            # exactly so the two siblings collapse on the SAME condition.
            (
                self.fs_pre_screen_null_fraction_threshold_cfg
                if (self.use_mrmr_fs or self.rfecv_estimator_cfg is not None
                    or self.use_boruta_shap_cfg)
                else 0.99
            ),
            # 2026-05-28 LinearModelConfig.l1_ratio. Only meaningful when
            # 'linear' is in the models AND the solver is 'saga' (sklearn
            # raises l1_ratio>0 with lbfgs/liblinear). Canon to 0.0 (Ridge-
            # equivalent) when those conditions don't hold so disabled
            # combos collapse to a single canonical key.
            (
                self.linear_l1_ratio_cfg
                if ("linear" in self.models and self.linear_solver_cfg == "saga")
                else 0.0
            ),
            # 2026-05-28 RecurrentConfig.hidden_size. Only meaningful when
            # a canonical recurrent model is picked. Canon to library default
            # (128) otherwise.
            (
                self.recurrent_hidden_size_cfg
                if self._canonical_recurrent_model() is not None
                else 128
            ),
            # 2026-05-28 audit-pass-4 SAFE-subset W4 canons. All collapse to
            # source defaults when the gating condition is off so non-gated
            # combos don't gain phantom variation.
            # CalibrationConfig.policy_auto_pick only meaningful for
            # classification targets (the calibrator palette is binary-only
            # in the auto-pick path).
            (
                self.calibration_policy_auto_pick_cfg
                if self.target_type in ("binary_classification",
                                          "multilabel_classification")
                else True
            ),
            # CalibrationConfig.n_bootstrap consumed only when
            # policy_auto_pick is on AND target is classification.
            (
                self.calibration_n_bootstrap_cfg
                if (self.target_type in ("binary_classification",
                                          "multilabel_classification")
                    and self.calibration_policy_auto_pick_cfg)
                else 1000
            ),
            # CalibrationConfig.candidates: same gate as n_bootstrap.
            (
                self.calibration_candidates_cfg
                if (self.target_type in ("binary_classification",
                                          "multilabel_classification")
                    and self.calibration_policy_auto_pick_cfg)
                else None
            ),
            # TrainingBehaviorConfig.pipeline_cache_ram_budget_fraction: the
            # PipelineCache is always live in the suite, so no secondary gate.
            self.pipeline_cache_ram_budget_fraction_cfg,
            # ReportingConfig.compute_trainset_metrics: every suite emits a
            # report, so no secondary gate.
            self.reporting_compute_trainset_metrics_cfg,
            # ReportingConfig.mase_seasonality: MASE is a regression-only
            # metric (Hyndman-Koehler 2006); canon to library default 1 for
            # non-regression targets.
            (
                self.reporting_mase_seasonality_cfg
                if self.target_type == "regression"
                else 1
            ),
            # RecurrentConfig.use_stratified_sampler: weighted sampling is
            # consumed only when a recurrent model is picked AND the target
            # is classification (stratified sampler is class-balance based).
            (
                self.recurrent_use_stratified_sampler_cfg
                if (self._canonical_recurrent_model() is not None
                    and self.target_type in ("binary_classification",
                                               "multilabel_classification"))
                else True
            ),
            # TrainingBehaviorConfig.model_file_hash_suffix: the per-model
            # hash-suffix decision fires on every model save, so no gate.
            self.behavior_model_file_hash_suffix_cfg,
            # 2026-05-30 audit-pass-6 (W6) canons.
            # SliceStableES axes: master enable flag is independent (always
            # meaningful); the 4 sub-knobs collapse to SliceStableESConfig
            # source defaults when the master is OFF so dedup absorbs combos
            # that differ only on disabled-branch knobs.
            self.slice_stable_es_enabled_cfg,
            (
                self.slice_stable_es_aggregate_cfg
                if self.slice_stable_es_enabled_cfg
                else "mean"
            ),
            (
                self.slice_stable_es_source_cfg
                if self.slice_stable_es_enabled_cfg
                else "temporal"
            ),
            (
                self.slice_stable_es_pareto_best_iter_selection_cfg
                if self.slice_stable_es_enabled_cfg
                else False
            ),
            (
                self.slice_stable_es_diagnostic_only_cfg
                if self.slice_stable_es_enabled_cfg
                else False
            ),
            # Curve-shape ES detector: meaningful on every model that runs
            # iterative fits with a val metric (boosters + linear partial-fit
            # ES wrapper). No secondary gate -- the detector is unconditionally
            # constructed when the booster path runs.
            self.early_stop_on_worsening_cfg,
            # MRMR Wave 7/8/9 axes: all collapse to MRMR.__init__ defaults
            # when use_mrmr_fs is False so dedup absorbs identical-behaviour
            # combos that differ only on inactive MRMR knobs.
            (self.mrmr_nbins_strategy_cfg if self.use_mrmr_fs else "mdlp"),
            (self.mrmr_mi_correction_cfg if self.use_mrmr_fs else "none"),
            (self.mrmr_redundancy_aggregator_cfg if self.use_mrmr_fs else None),
            (self.mrmr_bur_lambda_cfg if self.use_mrmr_fs else 0.0),
            (self.mrmr_cmi_perm_stop_cfg if self.use_mrmr_fs else False),
            (
                self.mrmr_stability_selection_method_cfg
                if self.use_mrmr_fs
                else "classic"
            ),
            (self.mrmr_mi_normalization_cfg if self.use_mrmr_fs else "none"),
            # audit-pass-7 #1: collapse fallback matches mrmr.py:596 default
            # (True). Pre-flip the fallback was False; the source default
            # change makes True the canonical "no MRMR" baseline.
            (self.mrmr_dcd_enable_cfg if self.use_mrmr_fs else True),
            # audit-pass-7 #2: baseline_npermutations only meaningful when
            # MRMR fires (gates evaluate_candidate baseline-screen quorum).
            (self.mrmr_baseline_npermutations_cfg if self.use_mrmr_fs else 2),
            # audit-pass-7 #3: low_card_cap threaded via nbins_strategy_kwargs
            # to per_feature_edges; only fires under MRMR.
            (self.mrmr_low_card_cap_cfg if self.use_mrmr_fs else 32),
            # audit-pass-7 #4: collapsed_fallback_nbins only fires when a
            # supervised binning method collapses a column. Compound gate:
            # MRMR ON AND nbins_strategy in {mdlp, fayyad_irani}. Other
            # methods (quantile/qs/sturges/freedman_diaconis/...) never
            # trigger the fallback so the axis is canon-only there.
            (
                self.mrmr_collapsed_fallback_nbins_cfg
                if (
                    self.use_mrmr_fs
                    and self.mrmr_nbins_strategy_cfg in ("mdlp", "fayyad_irani")
                )
                else 5
            ),
            # CV-selector mode: only meaningful when composite discovery is
            # enabled AND target is regression (the discovery is regression-
            # only). Mirrors the existing composite_* canon pattern.
            (
                self.cv_selector_mode_cfg
                if (self.composite_discovery_enabled_cfg
                    and self.target_type == "regression")
                else "mean"
            ),
            # S27 close-out: auto_wrap_partial_fit_es_force_off (no gate --
            # the wrap path is unconditionally exercised whenever a linear-
            # family model + X_val/y_val are present, which is independent of
            # the model list axis; collapse-less canon is safe).
            self.auto_wrap_partial_fit_es_force_off_cfg,
            # 2026-05-30 audit-pass-6 LOW-tier deferred batch (W6 LOW) canons.
            # ShapProxiedFS Stage-A (S1-S8): collapse to shap_proxied_fs.py:79-87
            # defaults when use_shap_proxied_fs=False.
            (
                self.shap_proxied_prefilter_stage1_keep_cfg
                if self.use_shap_proxied_fs
                else None
            ),
            (
                self.shap_proxied_prefilter_univariate_batch_size_cfg
                if self.use_shap_proxied_fs
                else None
            ),
            (
                self.shap_proxied_shap_prefilter_enabled_cfg
                if self.use_shap_proxied_fs
                else True
            ),
            (
                self.shap_proxied_shap_prefilter_safety_factor_cfg
                if self.use_shap_proxied_fs
                else 4
            ),
            (
                self.shap_proxied_shap_prefilter_min_features_cfg
                if self.use_shap_proxied_fs
                else 40
            ),
            (
                self.shap_proxied_shap_aware_stage1_keep_cfg
                if self.use_shap_proxied_fs
                else True
            ),
            (
                # 2026-05-31 audit-pass-14 F14-2: source default flipped 8 -> 2
                # in iter76 (shap_proxied_fs.py:249); fallback follows prod.
                self.shap_proxied_shap_aware_stage1_cushion_cfg
                if self.use_shap_proxied_fs
                else 2
            ),
            (
                self.shap_proxied_shap_aware_stage1_floor_cfg
                if self.use_shap_proxied_fs
                else 200
            ),
            # ShapProxiedFS Refine UCB (S9-S12): gate on use_shap_proxied_fs
            # AND shap_proxied_within_cluster_refine_cfg (the within-cluster
            # refine branch is the only consumer of these knobs).
            (
                self.shap_proxied_refine_ucb_enabled_cfg
                if (self.use_shap_proxied_fs
                    and self.shap_proxied_within_cluster_refine_cfg)
                else True
            ),
            (
                self.shap_proxied_refine_ucb_min_eval_size_cfg
                if (self.use_shap_proxied_fs
                    and self.shap_proxied_within_cluster_refine_cfg)
                else None
            ),
            (
                self.shap_proxied_refine_ucb_slack_cfg
                if (self.use_shap_proxied_fs
                    and self.shap_proxied_within_cluster_refine_cfg)
                else None
            ),
            (
                self.shap_proxied_refine_ucb_stdev_multiplier_cfg
                if (self.use_shap_proxied_fs
                    and self.shap_proxied_within_cluster_refine_cfg)
                else 1.0
            ),
            # ShapProxiedFS Revalidation (S13-S17): gate on use_shap_proxied_fs
            # AND shap_proxied_revalidate_cfg.
            (
                self.shap_proxied_revalidation_n_estimators_cfg
                if (self.use_shap_proxied_fs
                    and self.shap_proxied_revalidate_cfg)
                else 100
            ),
            (
                self.shap_proxied_revalidation_ucb_enabled_cfg
                if (self.use_shap_proxied_fs
                    and self.shap_proxied_revalidate_cfg)
                else True
            ),
            (
                self.shap_proxied_revalidation_ucb_min_eval_size_cfg
                if (self.use_shap_proxied_fs
                    and self.shap_proxied_revalidate_cfg)
                else None
            ),
            (
                self.shap_proxied_revalidation_ucb_slack_cfg
                if (self.use_shap_proxied_fs
                    and self.shap_proxied_revalidate_cfg)
                else None
            ),
            (
                self.shap_proxied_revalidation_ucb_stdev_multiplier_cfg
                if (self.use_shap_proxied_fs
                    and self.shap_proxied_revalidate_cfg)
                else None
            ),
            # ShapProxiedFS threading (S18): only meaningful when the
            # selector runs at all.
            (
                self.shap_proxied_inner_n_jobs_cap_cfg
                if self.use_shap_proxied_fs
                else False
            ),
            # Curve-shape ES scalars (S25, S26): only meaningful when the
            # worsening detector is enabled.
            (
                self.early_stop_on_worsening_coeff_cfg
                if self.early_stop_on_worsening_cfg
                else 5
            ),
            (
                self.early_stop_on_worsening_min_iters_cfg
                if self.early_stop_on_worsening_cfg
                else 5
            ),
            # MRMR Wave 8 LOW scalars (S32, S34, S35, S37): collapse to
            # MRMR.__init__ defaults when use_mrmr_fs=False.
            (self.mrmr_relaxmrmr_alpha_cfg if self.use_mrmr_fs else 0.0),
            (self.mrmr_uaed_auto_size_cfg if self.use_mrmr_fs else False),
            (self.mrmr_cpt_test_cfg if self.use_mrmr_fs else False),
            (self.mrmr_pid_synergy_bonus_cfg if self.use_mrmr_fs else 0.0),
            # CV-selector LOW knobs (S41-S44): collapse to discovery defaults
            # when composite discovery is OFF (mirrors cv_selector_mode_cfg).
            (
                self.cv_selector_alpha_cfg
                if (self.composite_discovery_enabled_cfg
                    and self.target_type == "regression")
                else 1.0
            ),
            (
                self.cv_selector_confidence_cfg
                if (self.composite_discovery_enabled_cfg
                    and self.target_type == "regression")
                else 0.9
            ),
            (
                self.cv_selector_quantile_level_cfg
                if (self.composite_discovery_enabled_cfg
                    and self.target_type == "regression")
                else 0.9
            ),
            (
                self.cv_persist_fold_scores_cfg
                if (self.composite_discovery_enabled_cfg
                    and self.target_type == "regression")
                else False
            ),
            # 2026-05-31 audit-pass-8 HIGH (#1-#4) canon-collapse rules.
            # #1 cardinality_bias_correction only meaningful when MRMR fires;
            # collapse to source default True (mrmr.py:334) under
            # use_mrmr_fs=False so non-MRMR combos share a canonical key with
            # the post-flip baseline.
            (
                self.mrmr_cardinality_bias_correction_cfg
                if self.use_mrmr_fs
                else True
            ),
            # #2 min_relevance_gain_relative_to_first only fires inside
            # _screen_predictors; collapse to source default 0.05
            # (mrmr.py:326) when MRMR is off.
            (
                self.mrmr_min_relevance_gain_relative_to_first_cfg
                if self.use_mrmr_fs
                else 0.05
            ),
            # #3 mlp_random_state only meaningful when a neural-family
            # estimator runs. Audit gate: 'mlp' in models OR recurrent
            # is active. Collapse to source default None outside that gate.
            (
                self.mlp_random_state_cfg
                if ("mlp" in self.models or self.recurrent_model_cfg is not None)
                else None
            ),
            # #4 mlp_class_weight only meaningful when MLP runs against an
            # imbalanced classification target. Compound gate: 'mlp' in
            # models AND target_type in {binary_classification,
            # multiclass_classification} AND imbalance_ratio in
            # {rare_5pct, rare_1pct}. Collapse to source default None
            # outside that gate.
            (
                self.mlp_class_weight_cfg
                if (
                    "mlp" in self.models
                    and self.target_type in (
                        "binary_classification",
                        "multiclass_classification",
                    )
                    and self.imbalance_ratio in ("rare_5pct", "rare_1pct")
                )
                else None
            ),
            # 2026-05-31 audit-pass-8 MED + LOW->MED (#5/#7/#8/#9/#10) canon-
            # collapse rules.
            # #5 adaptive_prescreen_by_stability only fires inside
            # ShapProxiedFS.compute_shap_matrix; collapse to source default
            # False (shap_proxied_fs.py:208) when use_shap_proxied_fs=False.
            (
                self.shap_proxied_adaptive_prescreen_by_stability_cfg
                if self.use_shap_proxied_fs
                else False
            ),
            # #7 use_layernorm flip is regression-only (LN-on-classifier-
            # logits is pathological); collapse to False outside the gate
            # 'mlp' in models AND target_type == regression.
            (
                self.mlp_use_layernorm_cfg
                if ("mlp" in self.models and self.target_type == "regression")
                else False
            ),
            # #8 l1_alpha BN/LN/GN-excluded branch only fires for MLP. Collapse
            # to 0.0 (no-op) outside 'mlp' in models.
            (
                self.mlp_l1_alpha_cfg
                if "mlp" in self.models
                else 0.0
            ),
            # #9 zero-weight-batch injection requires recency / non-uniform
            # weights AND MLP active so the WARN branch can actually fire.
            (
                self.mlp_inject_zero_sample_weight_batch_cfg
                if (
                    "mlp" in self.models
                    and self.weight_schemas != ("uniform",)
                )
                else False
            ),
            # #10 XOR synergy pair only meaningful when MRMR fleuret-mode
            # conditional-MI gate can fire (use_mrmr_fs=True AND
            # mrmr_interactions_max_order_cfg >= 2). Collapse to False
            # outside that gate so dedup absorbs phantom variation.
            (
                self.inject_xor_synergy_pair_cfg
                if (
                    self.use_mrmr_fs
                    and self.mrmr_interactions_max_order_cfg >= 2
                )
                else False
            ),
            # 2026-05-31 audit-pass-9 (W9). Eight new MLP / MRMR / target-type
            # axes; each collapses to its source default outside the documented
            # gate so dedup absorbs phantom variation.
            #
            # #1 mlp_adamw_betas only meaningful when 'mlp' in models (suite
            # always picks AdamW for the MLP path). Collapse to source default
            # (0.9, 0.95) (_flat_torch_module.py:499) otherwise.
            (
                self.mlp_adamw_betas_cfg
                if "mlp" in self.models
                else (0.9, 0.95)
            ),
            # #2 mlp_use_ema: collapse to False outside 'mlp' in models AND
            # outside the use_swa mutual-exclusion gate. base.py:767 raises
            # ValueError when both flags are True, so canon also collapses
            # use_ema=True to False whenever any use_swa axis indicator is
            # on -- the fuzz suite does not expose a use_swa axis today, so
            # the gate reduces to 'mlp' in models. When a use_swa axis lands,
            # add ``and not self.mlp_use_swa_cfg`` to the gate.
            (
                self.mlp_use_ema_cfg
                if "mlp" in self.models
                else False
            ),
            # #3 mlp_label_smoothing: gated at base.py:897-907 to multiclass.
            # Collapse to source default 0.0 outside ('mlp' AND multiclass).
            (
                self.mlp_label_smoothing_cfg
                if (
                    "mlp" in self.models
                    and self.target_type == "multiclass_classification"
                )
                else 0.0
            ),
            # #4 mlp_focal_loss_gamma: gated at base.py:878-884 to binary
            # classification; the focal-loss target is class imbalance, so we
            # restrict variation to combos where the imbalance axis actually
            # produces a rare positive class -- otherwise BCEWithLogitsLoss
            # and focal loss are indistinguishable on a balanced binary
            # target and dedup should collapse them. Outside the compound
            # gate the axis collapses to None (BCE unchanged).
            (
                self.mlp_focal_loss_gamma_cfg
                if (
                    "mlp" in self.models
                    and self.target_type == "binary_classification"
                    and self.imbalance_ratio in ("rare_5pct", "rare_1pct")
                )
                else None
            ),
            # #5 mlp_use_residual: ResidualLinearBlock wrapper at flat.py:465.
            # Collapse to source default False outside 'mlp' in models. The
            # spectral_norm interaction at flat.py:472-478 only emits a WARN
            # (no semantic flip) so we keep both branches reachable when MLP
            # is in scope; once a spectral_norm axis is added the canon will
            # refine to gate on spectral_norm=False as well.
            (
                self.mlp_use_residual_cfg
                if "mlp" in self.models
                else False
            ),
            # #6 mlp_numerical_embedding + kwargs literal. Both collapse to
            # source default (None / "paper_default") outside 'mlp' in
            # models. The kwargs axis additionally collapses to
            # "paper_default" when the embedding axis is None, since the
            # kwargs dict is only consumed when the embedding branch fires.
            (
                self.mlp_numerical_embedding_cfg
                if "mlp" in self.models
                else None
            ),
            (
                self.mlp_numerical_embedding_kwargs_cfg
                if (
                    "mlp" in self.models
                    and self.mlp_numerical_embedding_cfg is not None
                )
                else "paper_default"
            ),
            # #7 mrmr_fe_hybrid_orth master + pair. Master collapses to False
            # outside use_mrmr_fs (mrmr.py:656). pair_enable collapses to
            # source default True (mrmr.py:664) outside (use_mrmr_fs AND
            # master==True) -- when the master is off the pair stage is
            # skipped entirely and the True/False variants are behaviour-
            # identical.
            (
                self.mrmr_fe_hybrid_orth_enable_cfg
                if self.use_mrmr_fs
                else False
            ),
            (
                self.mrmr_fe_hybrid_orth_pair_enable_cfg
                if (
                    self.use_mrmr_fs
                    and self.mrmr_fe_hybrid_orth_enable_cfg
                )
                else True
            ),
            # #8 multi_target_regression canon-collapse is applied at the
            # PRIMARY target_type slot earlier in the tuple (see comment
            # there); F-23 inject_inf_nan-vs-MLP mirror is applied at the
            # PRIMARY inject_inf_nan slot earlier (see comment there). Both
            # are NOT duplicated here.
            # ------------------------------------------------------------
            # 2026-05-31 audit-pass-10 (W10). Five new axes; each collapses
            # to its source default outside the documented gate.
            #
            # #1 mlp_optimizer: collapses to "adamw" outside 'mlp' in
            # models. _flat_torch_module.py:86 falls back to AdamW when
            # caller does not supply the optimizer; both variants reduce
            # to the same code path on non-MLP combos.
            (
                self.mlp_optimizer_cfg
                if "mlp" in self.models
                else "adamw"
            ),
            # #2 mrmr_fe_hybrid_orth_degrees: collapses to source default
            # (2, 3) outside (use_mrmr_fs AND master==True) -- the hybrid
            # FE pipeline only fires under that compound gate, so the
            # tuple variants are behaviour-identical otherwise.
            (
                self.mrmr_fe_hybrid_orth_degrees_cfg
                if (
                    self.use_mrmr_fs
                    and self.mrmr_fe_hybrid_orth_enable_cfg
                )
                else (2, 3)
            ),
            # #3 mrmr_fe_hybrid_orth_basis: collapses to "auto" outside
            # (use_mrmr_fs AND master==True). Same gate as #2; the basis
            # routing only runs when the hybrid pipeline runs.
            (
                self.mrmr_fe_hybrid_orth_basis_cfg
                if (
                    self.use_mrmr_fs
                    and self.mrmr_fe_hybrid_orth_enable_cfg
                )
                else "auto"
            ),
            # #4 mrmr_fe_hybrid_orth_top_k: collapses to 5 outside
            # (use_mrmr_fs AND master==True). Same gate as #2.
            (
                self.mrmr_fe_hybrid_orth_top_k_cfg
                if (
                    self.use_mrmr_fs
                    and self.mrmr_fe_hybrid_orth_enable_cfg
                )
                else 5
            ),
            # #6 mrmr_fe_hybrid_orth_pair_max_degree: compound gate --
            # collapses to 2 outside (use_mrmr_fs AND master==True AND
            # pair_enable==True). pair_max_degree only governs the
            # pair-cross stage at _mrmr_fit_impl.py:292 which is gated
            # behind ``_h_pair_enable`` (mrmr.py:664).
            (
                self.mrmr_fe_hybrid_orth_pair_max_degree_cfg
                if (
                    self.use_mrmr_fs
                    and self.mrmr_fe_hybrid_orth_enable_cfg
                    and self.mrmr_fe_hybrid_orth_pair_enable_cfg
                )
                else 2
            ),
            # ------------------------------------------------------------
            # 2026-05-31 audit-pass-12 (W12). 12 new axes; each collapses to
            # its source default outside the documented gate so dedup absorbs
            # phantom variation.
            #
            # Group A -- F-34 MTR suite-side dispatch.
            # A1 composite_target_multilabel_strategy: only consumed by
            # CompositeTargetDiscoveryConfig.multilabel_strategy when the
            # target_type is multilabel or MTR (validator accepts the
            # value but downstream branches only read the field on those
            # target types). Canon collapses to "per_target" (the source
            # default at :773) otherwise.
            (
                self.composite_target_multilabel_strategy_cfg
                if self.target_type in (
                    "multilabel_classification", "multi_target_regression",
                )
                else "per_target"
            ),
            # A2 enable_ct_ensemble: the early-return WARN at
            # _phase_composite_post_xt_ensemble fires only when target_type
            # == multi_target_regression (D2 in commit d48245de). Outside
            # that target, the True/False variants are behaviour-identical
            # (the CT ensemble path runs normally regardless of this flag
            # today; the suite-internal gate ignores it). Canon collapses
            # to True (the suite-side default) outside the MTR target.
            (
                self.enable_ct_ensemble_cfg
                if self.target_type == "multi_target_regression"
                else True
            ),
            # A3 mtr_eval_metric: the new metrics_registry MTR entries
            # (rmse_macro/_micro/_max, mae_macro/_max, r2_macro/_min) are
            # reachable only on multi_target_regression targets. Canon
            # collapses to None outside that target so dedup absorbs combos
            # that could not consume the metric value.
            (
                self.mtr_eval_metric_cfg
                if self.target_type == "multi_target_regression"
                else None
            ),
            # Group B -- MRMR FE layer master switches.
            # B1 fe_kfold_te: K-fold target encoding requires at least one
            # categorical column to encode. Canon collapses to False outside
            # (use_mrmr_fs AND cat_feature_count >= 1) so dedup absorbs
            # no-op combos.
            (
                self.mrmr_fe_kfold_te_enable_cfg
                if (self.use_mrmr_fs and self.cat_feature_count >= 1)
                else False
            ),
            # B2 missingness-aware FE (indicator / count / pattern). Auto-
            # detect at mrmr.py:740 only picks columns with NaN rate in
            # [1%, 99%]. Canon collapses to False outside (use_mrmr_fs AND
            # any NaN source in the frame): inject_inf_nan, inject_all_nan_col,
            # or cat columns with null_fraction_cats > 0.
            (
                self.mrmr_fe_missingness_indicator_enable_cfg
                if (
                    self.use_mrmr_fs
                    and (
                        self.inject_inf_nan
                        or self.inject_all_nan_col
                        or (self.cat_feature_count > 0 and self.null_fraction_cats > 0)
                    )
                )
                else False
            ),
            (
                self.mrmr_fe_missingness_count_enable_cfg
                if (
                    self.use_mrmr_fs
                    and (
                        self.inject_inf_nan
                        or self.inject_all_nan_col
                        or (self.cat_feature_count > 0 and self.null_fraction_cats > 0)
                    )
                )
                else False
            ),
            (
                self.mrmr_fe_missingness_pattern_enable_cfg
                if (
                    self.use_mrmr_fs
                    and (
                        self.inject_inf_nan
                        or self.inject_all_nan_col
                        or (self.cat_feature_count > 0 and self.null_fraction_cats > 0)
                    )
                )
                else False
            ),
            # B3 fe_cat_aux (count / freq / cat-num interaction): all three
            # need a categorical column. Canon collapses to "off" outside
            # (use_mrmr_fs AND cat_feature_count >= 1). For "interaction"
            # the auto-detection at mrmr.py:728 needs at least one numeric
            # column too -- the synthetic builder always emits num_0..num_3
            # so the gate reduces to the cat-column check.
            (
                self.mrmr_fe_cat_aux_enable_cfg
                if (self.use_mrmr_fs and self.cat_feature_count >= 1)
                else "off"
            ),
            # B4 fe_hybrid_orth_extra_bases: only consumed when the master
            # hybrid_orth pipeline runs. Canon collapses to () outside
            # (use_mrmr_fs AND mrmr_fe_hybrid_orth_enable_cfg) -- same
            # compound gate as the W10 hybrid-orth tunables.
            (
                self.mrmr_fe_hybrid_orth_extra_bases_cfg
                if (
                    self.use_mrmr_fs
                    and self.mrmr_fe_hybrid_orth_enable_cfg
                )
                else ()
            ),
            # B5 fe_ratio_delta_diff: each kind has its own gate.
            #   * "ratio" / log_ratio: needs >=2 numeric columns -- always
            #     satisfied by the synthetic builder (num_0..num_3).
            #   * "grouped_delta": needs fe_grouped_delta_group_col, which
            #     the fuzz frame builder does NOT supply today. Collapses
            #     to "off".
            #   * "lagged_diff": needs fe_lagged_diff_time_col, ditto.
            #     Collapses to "off".
            # The aggregate canon: outside use_mrmr_fs OR when the chosen
            # kind has no supporting frame data, collapse to "off".
            (
                self.mrmr_fe_ratio_delta_diff_cfg
                if (
                    self.use_mrmr_fs
                    and self.mrmr_fe_ratio_delta_diff_cfg in ("off", "ratio")
                )
                else "off"
            ),
            # B6 fe_mi_greedy: sibling to hybrid_orth; gate is use_mrmr_fs
            # only.
            (
                self.mrmr_fe_mi_greedy_enable_cfg
                if self.use_mrmr_fs
                else False
            ),
            # Group C -- MRMR + ShapProxiedFS artifact-reuse pipeline.
            # C1 master switch. Both selectors must be in the chain. Canon
            # collapses to "off" outside (use_mrmr_fs AND use_shap_proxied_fs)
            # since the handoff cannot fire without both endpoints.
            (
                self.mrmr_shap_proxy_artifact_reuse_cfg
                if (self.use_mrmr_fs and self.use_shap_proxied_fs)
                else "off"
            ),
            # C2 align_mode: only relevant when artifact_reuse is "on" --
            # otherwise no precomputed dict is passed and the
            # align_precomputed_to_X function is never invoked. Canon
            # collapses to "exact" outside the artifact-reuse-on gate.
            (
                self.mrmr_shap_proxy_align_mode_cfg
                if (
                    self.use_mrmr_fs
                    and self.use_shap_proxied_fs
                    and self.mrmr_shap_proxy_artifact_reuse_cfg == "on"
                )
                else "exact"
            ),
            # 2026-05-31 audit-pass-14 (W14). Canon-collapse the 6 new axes
            # outside their documented gates so the dedup pass absorbs
            # phantom variation. Defaults mirror prod (verified at HEAD).
            # F14-1: cluster_backend collapses to "auto" when ShapProxiedFS
            # is off (the toggle is unread).
            (
                self.shap_proxied_cluster_backend_cfg
                if self.use_shap_proxied_fs
                else "auto"
            ),
            # F14-3: partial_fit_* ctor params shape future partial_fit()
            # behaviour; the legacy fit() byte-identical path ignores them.
            # Canon to source defaults when MRMR is off.
            (
                self.mrmr_partial_fit_decay_cfg
                if self.use_mrmr_fs
                else 0.0
            ),
            (
                self.mrmr_partial_fit_min_recompute_cfg
                if self.use_mrmr_fs
                else 100
            ),
            (
                self.mrmr_partial_fit_window_cfg
                if self.use_mrmr_fs
                else None
            ),
            # F14-4: dcd_tau_cluster only fires when DCD is on (the
            # auto-calibration path lives behind dcd_enable). Canon to
            # 0.7 outside the compound gate.
            (
                self.mrmr_dcd_tau_cluster_cfg
                if (self.use_mrmr_fs and self.mrmr_dcd_enable_cfg)
                else 0.7
            ),
            # F14-5: dcd_distance + dcd_swap_method both gated on DCD.
            # Canon to "su" / "auto" source defaults outside the gate.
            (
                self.mrmr_dcd_distance_cfg
                if (self.use_mrmr_fs and self.mrmr_dcd_enable_cfg)
                else "su"
            ),
            (
                self.mrmr_dcd_swap_method_cfg
                if (self.use_mrmr_fs and self.mrmr_dcd_enable_cfg)
                else "auto"
            ),
            # iter639 audit-pass-15. Layers 62/63/76/85 hybrid-orth scorer
            # family. All four are meaningful only inside the
            # fe_hybrid_orth_enable=True compound gate; canon to source
            # defaults outside so dedup absorbs phantom variation that
            # would otherwise multiply combo count 24x without exercising
            # any new code path.
            (
                self.mrmr_fe_hybrid_orth_default_scorer_cfg
                if (self.use_mrmr_fs and self.mrmr_fe_hybrid_orth_enable_cfg)
                else "plug_in"
            ),
            (
                self.mrmr_fe_hybrid_orth_meta_enable_cfg
                if (self.use_mrmr_fs and self.mrmr_fe_hybrid_orth_enable_cfg)
                else False
            ),
            (
                self.mrmr_fe_hybrid_orth_bootstrap_enable_cfg
                if (self.use_mrmr_fs and self.mrmr_fe_hybrid_orth_enable_cfg)
                else False
            ),
            (
                self.mrmr_fe_hybrid_orth_three_gate_enable_cfg
                if (self.use_mrmr_fs and self.mrmr_fe_hybrid_orth_enable_cfg)
                else False
            ),
            # iter642 audit-pass-15 batch 2. Six remaining hybrid-orth sub-
            # features all collapse to False outside the master gate.
            (
                self.mrmr_fe_hybrid_orth_ensemble_enable_cfg
                if (self.use_mrmr_fs and self.mrmr_fe_hybrid_orth_enable_cfg)
                else False
            ),
            (
                self.mrmr_fe_hybrid_orth_lasso_enable_cfg
                if (self.use_mrmr_fs and self.mrmr_fe_hybrid_orth_enable_cfg)
                else False
            ),
            (
                self.mrmr_fe_hybrid_orth_elasticnet_enable_cfg
                if (self.use_mrmr_fs and self.mrmr_fe_hybrid_orth_enable_cfg)
                else False
            ),
            (
                self.mrmr_fe_hybrid_orth_adaptive_arity_enable_cfg
                if (self.use_mrmr_fs and self.mrmr_fe_hybrid_orth_enable_cfg)
                else False
            ),
            (
                self.mrmr_fe_hybrid_orth_diff_basis_enable_cfg
                if (self.use_mrmr_fs and self.mrmr_fe_hybrid_orth_enable_cfg)
                else False
            ),
            (
                self.mrmr_fe_semi_supervised_enable_cfg
                if (self.use_mrmr_fs and self.mrmr_fe_hybrid_orth_enable_cfg)
                else False
            ),
            # audit-pass-16. MRMR Layers 87-91 FE mechanisms gate on
            # use_mrmr_fs (independent of hybrid-orth). Master switches
            # collapse to False outside use_mrmr_fs; the two sub-knobs
            # (quantile target_aware, decompose digits) collapse outside
            # their own master gate too.
            self.mrmr_fe_grouped_agg_enable_cfg if self.use_mrmr_fs else False,
            self.mrmr_fe_grouped_quantile_enable_cfg if self.use_mrmr_fs else False,
            (
                self.mrmr_fe_grouped_quantile_target_aware_cfg
                if (self.use_mrmr_fs and self.mrmr_fe_grouped_quantile_enable_cfg)
                else False
            ),
            self.mrmr_fe_cat_pair_enable_cfg if self.use_mrmr_fs else False,
            self.mrmr_fe_numeric_decompose_enable_cfg if self.use_mrmr_fs else False,
            (
                self.mrmr_fe_numeric_decompose_digits_cfg
                if (self.use_mrmr_fs and self.mrmr_fe_numeric_decompose_enable_cfg)
                else (0, 1, 2)
            ),
            # audit-pass-17: local_mi_gate source default is now True (L97);
            # collapse to True (the source default) outside use_mrmr_fs.
            self.mrmr_fe_local_mi_gate_cfg if self.use_mrmr_fs else True,
            self.mrmr_fe_unified_second_pass_gate_cfg if self.use_mrmr_fs else False,
            # audit-pass-17. Param-Oracle / fe_auto + FE families L92-104, all
            # gated on use_mrmr_fs. fe_auto=True turns the per-flag FE switches
            # ON internally for the fit (the individual flags are then dead),
            # so it is forwarded as its own axis; the per-flag axes above stay
            # canon-collapsed by their own gates.
            self.mrmr_fe_auto_cfg if self.use_mrmr_fs else False,
            self.mrmr_fe_temporal_agg_enable_cfg if self.use_mrmr_fs else False,
            self.mrmr_fe_composite_group_agg_enable_cfg if self.use_mrmr_fs else False,
            self.mrmr_fe_modular_enable_cfg if self.use_mrmr_fs else False,
            self.mrmr_fe_group_distance_enable_cfg if self.use_mrmr_fs else False,
            self.mrmr_fe_rare_category_enable_cfg if self.use_mrmr_fs else False,
            self.mrmr_fe_conditional_residual_enable_cfg if self.use_mrmr_fs else False,
            # 2026-06-13 coverage refresh. embedding_passthrough master gates on
            # use_mrmr_fs AND a passthrough-eligible column being present
            # (embedding OR free-text); with neither there is nothing to route
            # around, so the axis collapses to the source default True. The two
            # detect-* sub-knobs additionally require the master to be True.
            (
                self.mrmr_embedding_passthrough_cfg
                if (self.use_mrmr_fs and (self.embedding_col_count > 0 or self.text_col_count > 0))
                else True
            ),
            (
                self.mrmr_embedding_passthrough_detect_embeddings_cfg
                if (self.use_mrmr_fs and self.embedding_col_count > 0 and self.mrmr_embedding_passthrough_cfg)
                else True
            ),
            (
                self.mrmr_embedding_passthrough_detect_text_cfg
                if (self.use_mrmr_fs and self.text_col_count > 0 and self.mrmr_embedding_passthrough_cfg)
                else True
            ),
            # Five default-ON FE families: their OFF branch is the new coverage.
            # All gate on use_mrmr_fs; collapse to the source default True outside.
            self.mrmr_fe_hinge_enable_cfg if self.use_mrmr_fs else True,
            self.mrmr_fe_conditional_dispersion_enable_cfg if self.use_mrmr_fs else True,
            self.mrmr_fe_wavelet_enable_cfg if self.use_mrmr_fs else True,
            self.mrmr_fe_stability_vote_enable_cfg if self.use_mrmr_fs else True,
            self.mrmr_fe_sufficient_summary_early_stop_cfg if self.use_mrmr_fs else True,
            # Gradient-interaction seeder feeds the interaction stage: gate on
            # use_mrmr_fs AND interactions_max_order>=2; canon to source-default
            # False outside (the seeder is a no-op when no interactions run).
            (
                self.mrmr_fe_gradient_interaction_enable_cfg
                if (self.use_mrmr_fs and self.mrmr_interactions_max_order_cfg >= 2)
                else False
            ),
            # iter640 audit-pass-15. F-62/63/68-70/72 MLP options. Gated
            # on 'mlp' in models; canon to False outside so non-MLP combos
            # collapse to one variant per knob.
            self.mlp_use_sam_cfg if "mlp" in self.models else False,
            self.mlp_use_lookahead_cfg if "mlp" in self.models else False,
            self.mlp_use_mixup_cfg if "mlp" in self.models else False,
            (
                self.mlp_spectral_norm_output_only_cfg
                if "mlp" in self.models
                else False
            ),
            # Cat learnable-embeddings: only meaningful with raw cats + a neural model; collapse to default elsewhere so dedup
            # doesn't waste combos on a no-op axis.
            self.mlp_use_learnable_cat_embeddings_cfg if (self.cat_feature_count > 0 and _has_neural) else True,
            self.mlp_categorical_embed_dim_cfg if (self.cat_feature_count > 0 and _has_neural and self.mlp_use_learnable_cat_embeddings_cfg) else None,
            # Chart/report RENDERING toggle. Canon to False on the large n_rows tier: rendering ~5 figures per combo against 200k-row scores is
            # the memory blow-up the runner originally hardcoded show/save off to avoid; the small tier carries the rendering coverage cheaply.
            self.enable_viz_rendering_cfg if self.n_rows <= 1000 else False,
        )

    def _canonical_recurrent_model(self) -> "str | None":
        rec = self.recurrent_model_cfg
        if rec is None:
            return None
        # Recurrent training is the slowest model on the bench (PyTorch
        # Lightning loop). Cap at the small-row tier so a fuzz combo
        # doesn't push past per-test timeout. Was n_rows > 600 against
        # the legacy (300, 600, 1000, 5000) tiers; the 2026-05-21 axis
        # bump (1000, 200_000) makes 1000 the small tier so recurrent
        # only fires there (200k would explode timeout).
        if self.n_rows > 1000:
            return None
        # The recurrent classifier path expects a 1-D label; multilabel
        # (N, K) targets aren't supported by RecurrentTorchModel today.
        if self.target_type == "multilabel_classification":
            return None
        # Text / embedding columns can't be packed into the auxiliary
        # feature matrix (RecurrentDataset.aux_features expects float32),
        # and there's no TF-IDF preflight on the recurrent path.
        if self.text_col_count > 0 or self.embedding_col_count > 0:
            return None
        # Categorical features need encoding before reaching the
        # recurrent path's StandardScaler (np.asarray on string columns
        # raises). Require either zero cats or polars-ds encoding active.
        if self.cat_feature_count > 0 and not (
            self.prefer_polarsds
            and self.categorical_encoding_cfg in ("ordinal", "onehot")
            and not self.skip_categorical_encoding_cfg
        ):
            return None
        # Recurrent + MRMR + small n + heavy aging/OD trim collapses the
        # tabular train to ≤ ~100 rows, MRMR then drops every feature, and
        # the companion mlframe model (e.g. CB) hits Pool() with empty
        # labels (c0079). Disable recurrent in this combination — the
        # tabular feature-selection axis is exercised separately.
        if self.use_mrmr_fs:
            return None
        return rec

    def _canonical_pysr_enabled(self) -> bool:
        """F5 -- PySR symbolic regression cannot consume inf / NaN values. Whenever ``inject_inf_nan=True`` the PySR path crashes on the first non-finite row regardless of the True/False axis value, so canonicalise to the disabled variant to spare pairwise-coverage budget."""
        if not self.prep_ext_pysr_enabled_cfg:
            return False
        if self.inject_inf_nan or self.inject_all_nan_col:
            return False
        return True

    def _canonical_enable_crash_reporting(self) -> bool:
        """F1 -- crash_reporting is a Windows-only Faulthandler dump hook (``mlframe.training.crash_reporting``). On non-Windows hosts the True variant is a no-op so collapse to False; on Windows it stays meaningful."""
        import platform as _platform
        if _platform.system() != "Windows":
            return False
        return bool(self.enable_crash_reporting_cfg)

    def _canonical_rfecv_estimator(self) -> "str | None":
        rfe = self.rfecv_estimator_cfg
        if rfe is None:
            return None
        # rfecv name is "<base>_rfecv" — strip suffix to look up base model.
        base = rfe.rsplit("_rfecv", 1)[0]
        if base not in self.models:
            return None
        # RFECV iterates 10+ refits; cap at the small-row tier. Was
        # n_rows > 1200 against legacy tiers; the 2026-05-21 axis bump
        # (1000, 200_000) makes 1000 the small tier so RFECV only fires
        # there (200k * 10 refits would dominate the suite wall).
        if self.n_rows > 1000:
            return None
        # sklearn RFECV.fit (via check_classification_targets) raises
        # "Supported target types are: ('binary', 'multiclass'). Got
        # 'multilabel-indicator'". Multilabel is not in scope for RFECV.
        if self.target_type == "multilabel_classification":
            return None
        # Rare imbalance (rare_5pct / rare_1pct on small n) lets RFECV's
        # internal CV folds land on single-class y, raising "Invalid
        # classes inferred from unique values of y. Expected: [0], got
        # [1]". Disable RFECV unless the target distribution is balanced.
        if (
            self.target_type == "binary_classification"
            and self.imbalance_ratio != "balanced"
        ):
            return None
        return rfe

    def _canonical_prep_ext(self, name: str) -> "Any":
        """Collapse PreprocessingExtensions axes to None when the surrounding
        combo would feed the sklearn bridge data it cannot consume:
          * raw NaN / Inf in numeric columns (sklearn transformers raise);
          * an all-NaN column (PCA/scalers reject any NaN);
          * raw categorical columns reaching the bridge — happens whenever
            cats exist AND any of: polars-ds pipeline is off (no encoding
            step at all), categorical encoding is explicitly skipped, or
            categorical encoding is None.
        For the dim-reducer specifically, also require enough features
        (heuristic: ≥ 8 cat columns onehot-expanded so n_features beats
        the default dim_n_components=50) — otherwise sklearn raises
        "n_components must be <= n_features".
        """
        attr = f"prep_ext_{name}_cfg"
        value = getattr(self, attr)
        if value is None:
            return None
        if self.inject_inf_nan or self.inject_all_nan_col:
            return None
        if self.cat_feature_count > 0:
            cats_will_be_encoded = (
                self.prefer_polarsds
                and self.categorical_encoding_cfg in ("ordinal", "onehot")
                and not self.skip_categorical_encoding_cfg
            )
            if not cats_will_be_encoded:
                return None
        # Text columns flow through as raw strings unless tfidf_columns is
        # explicitly set; embedding columns are pl.List(Float32) which
        # numpy.asarray turns into ragged object arrays. Both crash any
        # downstream sklearn transformer (StandardScaler / Polynomial / PCA).
        # Disable prep_ext entirely for combos with text or embedding.
        if self.text_col_count > 0 or self.embedding_col_count > 0:
            return None
        # Latent NaN sources unrelated to the inject_* axes — the polars-ds
        # imputer chain doesn't always run (e.g. when the caller sets
        # imputer_strategy=None or scaler=None). The bridge then sees NaN
        # values that crash PolynomialFeatures / KBinsDiscretizer / PCA
        # (c0116, c0047). Require the imputer + scaler to be active so the
        # bridge always sees clean input. Also disable when the datetime
        # path is on (datetime extraction may introduce NaT/NaN that the
        # polars-ds imputer doesn't handle for newly-derived columns).
        if (
            self.imputer_strategy_cfg is None
            or self.scaler_name_cfg is None
            or self.with_datetime_col
            # inject_zero_col adds a constant-zero numeric column. The
            # polars-ds RobustScaler computes (X - median) / IQR; for
            # zero-variance the IQR=0, the divide returns NaN, and the
            # bridge then sees NaN that crashes PolynomialFeatures /
            # KBinsDiscretizer (c0116). Disable prep_ext for this combo.
            or self.inject_zero_col
        ):
            return None
        # dim_reducer requires n_features >= dim_n_components (default 50).
        # The synthetic builder emits 4 numeric + cat_feature_count*~5 (for
        # onehot) features. Conservative threshold: cat_feature_count >= 8
        # with onehot (~40+ extra cols) OR ordinal (cats stay as 1 col each)
        # is risky → require onehot + cats >= 8 for dim_reducer.
        if name == "dim_reducer":
            if not (
                self.cat_feature_count >= 8
                and self.categorical_encoding_cfg == "onehot"
            ):
                return None
        return value

    def _canonical_text_col_count(self) -> int:
        """Mirror the text_col_count canonicalisation in canonical_key.

        Currently a passthrough — historical canonicalisation rules
        (which dropped text cols on small-n CB + heavy NaN, and on
        cb_rfecv + NaN) were masking real production hangs in CB's
        text-feature path. Those are now fixed in production code:
        ``trainer.py`` skips text features whenever the effective
        train rows are below CB's occurrence_lower_bound floor.
        """
        return self.text_col_count

    def _is_chain_dispatch(self) -> bool:
        return (
            self.target_type == "multilabel_classification"
            and self._canonical_multilabel_strategy() == "chain"
        )

    def _canonical_multilabel_strategy(self) -> str:
        if self.target_type != "multilabel_classification":
            return "auto"
        if self.multilabel_strategy_cfg != "chain":
            return self.multilabel_strategy_cfg
        # Chain dispatch is only safe with classifier-friendly numeric input:
        # sklearn ClassifierChain wraps the inner estimator and bypasses the
        # polars-native fastpath, so categorical/string columns reach HGB/LGB
        # raw. The linear model also routes (N,K) preds through the regression
        # report and crashes on multilabel-shaped predictions. Both downgrades
        # are correctness fixes, not capability removals — chain is still
        # exercised for the combos that satisfy the prerequisites.
        if self.cat_feature_count > 0:
            return "wrapper"
        if "linear" in self.models:
            return "wrapper"
        return "chain"

    def _canonical_imputer_strategy(self) -> "str | None":
        """imputer_strategy variants are no-ops when there's nothing to impute.

        Returns the chosen strategy only when the frame is expected to
        contain nulls / NaNs — otherwise canonicalises to the default
        ('mean') so the dedup pass collapses the no-op variants."""
        has_nulls = (
            self.inject_inf_nan
            or self.inject_all_nan_col
            or (self.cat_feature_count > 0 and self.null_fraction_cats > 0)
            or self.inject_degenerate_cols  # adds an all-null column
        )
        return self.imputer_strategy_cfg if has_nulls else "mean"

    def _canonical_imbalance(self) -> str:
        if "classification" not in self.target_type:
            return "balanced"
        # multiclass and multilabel: imbalance fuzz is its own can of worms
        # (per-class balancing for multiclass, per-label for multilabel —
        # neither is currently supported by the synthetic builder).
        # Collapse to balanced for these target types.
        if self.target_type in ("multiclass_classification", "multilabel_classification"):
            return "balanced"
        imb = self.imbalance_ratio
        frac = {"rare_5pct": 0.05, "rare_1pct": 0.01, "balanced": 0.5}.get(imb, 0.5)
        # Effective minority-source N after train-side trimming. When the
        # combo also enables trainset_aging_limit (drops first/last K% of
        # train) + outlier_detection (filters ~5%), the surviving train
        # may have ~half the row count and a corresponding fraction of
        # minority rows. The earlier "n*0.1*frac >= 4" guard considered
        # only the val/test slice size — train-thinning broke c0048 at
        # n=5000 (1923 rows post-trim → 19 expected positives → unlucky 0).
        effective_n = float(self.n_rows)
        if self.trainset_aging_limit_cfg is not None:
            effective_n *= self.trainset_aging_limit_cfg
        if self.outlier_detection is not None:
            effective_n *= 0.95  # OD typically drops ~5% (contamination=0.05)
        # Each split gets ~0.1×n rows. Require frac × 0.1 × n_eff ≥ 4
        # (~4 minority rows expected in the smallest slice).
        if frac * 0.1 * effective_n < 4:
            # Try the next-safer rarity level.
            if imb == "rare_1pct":
                return "rare_5pct" if 0.05 * 0.1 * effective_n >= 4 else "balanced"
            if imb == "rare_5pct":
                return "balanced"
        return imb

    def short_id(self) -> str:
        h = hashlib.blake2s(repr(self.canonical_key()).encode(), digest_size=4).hexdigest()
        return f"c{self.seed:04d}_{h}"

    def pytest_id(self) -> str:
        # Include a human-readable prefix so failing IDs are diagnostic.
        tag = "_".join(sorted(self.models))
        short_input = self.input_type.replace("polars_", "pl_")
        return f"{self.short_id()}-{tag}-{short_input}-n{self.n_rows}"

    def to_json(self) -> dict:
        return {
            "short_id": self.short_id(),
            "models": list(self.models),
            "input_type": self.input_type,
            "n_rows": self.n_rows,
            "cat_feature_count": self.cat_feature_count,
            "null_fraction_cats": self.null_fraction_cats,
            "use_mrmr_fs": self.use_mrmr_fs,
            "weight_schemas": list(self.weight_schemas),
            "target_type": self.target_type,
            "target_carrier": self.target_carrier,
            "auto_detect_cats": self.auto_detect_cats,
            "align_polars_categorical_dicts": self.align_polars_categorical_dicts,
            "seed": self.seed,
            "prefer_polarsds": self.prefer_polarsds,
            "use_text_features": self.use_text_features,
            "honor_user_dtype": self.honor_user_dtype,
            "text_col_count": self.text_col_count,
            "embedding_col_count": self.embedding_col_count,
            # 2026-04-24 combo-extension axes
            "outlier_detection": self.outlier_detection,
            "use_ensembles": self.use_ensembles,
            "continue_on_model_failure": self.continue_on_model_failure,
            "iterations": self.iterations,
            "prefer_calibrated_classifiers": self.prefer_calibrated_classifiers,
            "inject_degenerate_cols": self.inject_degenerate_cols,
            "inject_inf_nan": self.inject_inf_nan,
            "with_datetime_col": self.with_datetime_col,
            "inject_zero_col": self.inject_zero_col,
            "fairness_col": self.fairness_col,
            "custom_prep": self.custom_prep,
            "input_storage": self.input_storage,
            # 2026-04-24 round 2
            "fillna_value_cfg": self.fillna_value_cfg,
            "scaler_name_cfg": self.scaler_name_cfg,
            "categorical_encoding_cfg": self.categorical_encoding_cfg,
            "skip_categorical_encoding_cfg": self.skip_categorical_encoding_cfg,
            "val_placement_cfg": self.val_placement_cfg,
            "test_size_cfg": self.test_size_cfg,
            "trainset_aging_limit_cfg": self.trainset_aging_limit_cfg,
            "cat_text_card_threshold_cfg": self.cat_text_card_threshold_cfg,
            "early_stopping_rounds_cfg": self.early_stopping_rounds_cfg,
            "use_robust_eval_metric_cfg": self.use_robust_eval_metric_cfg,
            # Fix G
            "inject_label_leak": self.inject_label_leak,
            "inject_rank_deficient": self.inject_rank_deficient,
            "inject_all_nan_col": self.inject_all_nan_col,
            # R3
            "inject_test_drift": self.inject_test_drift,
            "imbalance_ratio": self.imbalance_ratio,
            "weird_cat_content": self.weird_cat_content,
            "multilabel_strategy_cfg": self.multilabel_strategy_cfg,
            # 2026-04-26 batch 1
            "fix_infinities_cfg": self.fix_infinities_cfg,
            "ensure_float32_cfg": self.ensure_float32_cfg,
            "remove_constant_columns_cfg": self.remove_constant_columns_cfg,
            "imputer_strategy_cfg": self.imputer_strategy_cfg,
            "shuffle_val_cfg": self.shuffle_val_cfg,
            "shuffle_test_cfg": self.shuffle_test_cfg,
            "wholeday_splitting_cfg": self.wholeday_splitting_cfg,
            "val_sequential_fraction_cfg": self.val_sequential_fraction_cfg,
            # batch 3 — multilabel dispatch
            "multilabel_n_chains_cfg": self.multilabel_n_chains_cfg,
            "multilabel_chain_order_cfg": self.multilabel_chain_order_cfg,
            "multilabel_cv_cfg": self.multilabel_cv_cfg,
            # batch 4 — PreprocessingExtensionsConfig
            "prep_ext_scaler_cfg": self.prep_ext_scaler_cfg,
            "prep_ext_kbins_cfg": self.prep_ext_kbins_cfg,
            "prep_ext_polynomial_degree_cfg": self.prep_ext_polynomial_degree_cfg,
            "prep_ext_dim_reducer_cfg": self.prep_ext_dim_reducer_cfg,
            "prep_ext_nonlinear_cfg": self.prep_ext_nonlinear_cfg,
            "prep_ext_pysr_enabled_cfg": self.prep_ext_pysr_enabled_cfg,
            "mrmr_nan_strategy_cfg": self.mrmr_nan_strategy_cfg,
            # batch 5
            "rfecv_estimator_cfg": self.rfecv_estimator_cfg,
            # batch 6
            "recurrent_model_cfg": self.recurrent_model_cfg,
            # 2026-04-28 batch 4 followup
            "include_confidence_analysis_cfg": self.include_confidence_analysis_cfg,
        }


# ---------------------------------------------------------------------------
# Known-xfail rules (single source of truth for auto-xfail)
# ---------------------------------------------------------------------------


# _rule_linear_polars_gating_bug REMOVED 2026-04-22 (Fix 11):
# core.py:3085 now computes polars_pipeline_applied per-strategy:
#   polars_pipeline_applied AND strategy.supports_polars
#                            AND NOT strategy.requires_encoding
# Linear (supports_polars=False, requires_encoding=True) always gets
# skip_preprocessing=False, so its pre_pipeline runs the encoder fully.
# Permanent regression guard: test_polars_full_combo_with_linear in
# test_integration_prod_like_polars.py (xfail removed) +
# test_sensor_linear_polars_gating_bug in test_fuzz_regression_sensors.py.


# _rule_mrmr_plus_linear_multi_pandas REMOVED 2026-04-23: new-seed fuzz
# showed 4/6 XPASS from this rule — the MRMR+linear+pandas combos that
# once failed on feature-name mismatch now pass (composite of all
# 2026-04-22/23 fixes — per-model pipeline cloning, _is_fitted
# Pipeline-aware check, MRMR in-place drop, Fix 10 polars-native MRMR,
# Fix 11 per-strategy polars_pipeline_applied, Fix 12 dt==class dispatch).
# Rule is now misleading; permanent regression guard: the integration
# test suite's test_polars_full_combo_with_linear (already un-xfailed).

# _rule_cb_nan_in_cat_features_mrmr REMOVED 2026-04-23: fixed in trainer.py
# `_polars_nullable_categorical_cols` — the candidate list for the fill_null
# pre-fit pass now includes pl.Utf8 / pl.String dtypes (previously only
# Categorical / Enum). Raw Utf8 cat columns with nulls are now filled
# before CB sees them.


# _rule_mrmr_plus_xgb_lgb_polars_utf8_small REMOVED 2026-04-23: same root
# cause as _rule_cb_sparse_text_small — categorize_dataset incorrectly
# routed polars Categorical / Utf8 columns through the numeric branch
# because `dt in set_of_dtype_classes` uses hash equality, and
# `pl.Categorical` instance hash differs from class hash. Permanent
# regression guard: test_sensor_categorize_dataset_recognizes_polars_cat_dtypes.


# _rule_cb_sparse_text_small REMOVED 2026-04-23: the underlying failures
# were a symptom of the categorize_dataset dt-in-set hash bug
# (filters.py:2660, `dt in {pl.Utf8, pl.Categorical, ...}` returns False
# for Categorical instances because class-vs-instance hash differs).
# Fixed there with explicit `==` checks per dtype. Both c0048 and c0098
# now PASS. Permanent regression guard:
# test_sensor_categorize_dataset_recognizes_polars_cat_dtypes.


# REMOVED 2026-04-22: _rule_polars_schema_dispatch_bug
#
# Root cause was: _build_tier_dfs in core.py cached tier-DFs keyed only on
# strategy.feature_tier() — which collides between Polars and pandas inputs
# when a non-polars-native strategy (Linear) runs before a polars-native
# strategy (XGB) in the same multi-model suite. Linear stashed pandas
# tier-DFs under tier=(False,False); XGB retrieved the same key and got
# pandas back; XGBoostStrategy.prepare_polars_dataframe then tried
# df.schema.items() on a pandas frame and raised AttributeError.
#
# Fix: cache key now = (tier, kind) where kind ∈ {"pl","pd"} sampled from
# the first non-None input. See test_sensor_tier_cache_polars_pandas_collision
# in test_fuzz_regression_sensors.py — permanent regression guard.


# _rule_mrmr_single_linear_pandas REMOVED 2026-04-22: MRMR.transform now
# uses getattr(self, 'support_', None) so a fit() that exits without
# setting support_ (e.g. early-exit on low-MI synthetic data) degrades
# to pass-through instead of raising. Regression guard:
# test_sensor_mrmr_transform_handles_missing_support_ in test_fuzz_regression_sensors.py.


# _rule_multilabel_full_pipeline_deferred REMOVED 2026-04-25 — full
# multilabel integration landed in Session 6. All 42 multilabel combos
# in the fuzz suite pass end-to-end after target_type plumbing into
# get_training_configs, MultiOutputClassifier wrapping for
# HGB/XGB/LGB/Linear, multilabel-aware report path, MRMR target injection
# fix, and supervised-encoder target collapse. See CHANGELOG.md
# "Session 6: multilabel full-pipeline integration" entry.

