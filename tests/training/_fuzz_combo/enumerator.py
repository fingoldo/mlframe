"""Deterministic pairwise/triple-greedy combo enumerator + dedup."""
from __future__ import annotations

import random
from itertools import combinations as iter_combinations
from typing import Any

from .axes import AXES, MODELS
from .combo import FuzzCombo


# ---------------------------------------------------------------------------
# Enumerator: pairwise-greedy, deterministic, deduplicated
# ---------------------------------------------------------------------------


def _powerset_nonempty(items: tuple[str, ...]) -> list[tuple[str, ...]]:
    out: list[tuple[str, ...]] = []
    for r in range(1, len(items) + 1):
        for sub in iter_combinations(items, r):
            out.append(tuple(sorted(sub)))
    return out


# iter373 (2026-05-27): set of model flavours that ship a native ranker
# implementation. Used to reject fuzz combos where target_type=LTR is paired
# with a subset that contains NO ranker (e.g. ``models=('linear',)`` or
# ``models=('hgb',)``) -- the runtime raises NotImplementedError on those.
# c0120 was such a combo, surfaced 2026-05-27 by the random-pick /loop;
# without the filter the enumerator emits ~5 unrunnable combos per 150 (any
# LTR subset that happens to be hgb / linear / sklearn only).
_LTR_NATIVE_RANKERS: frozenset[str] = frozenset({"cb", "xgb", "lgb", "mlp"})


def _combo_is_runnable(models: tuple[str, ...], target_type: str) -> bool:
    """Cheap structural check: does this (models, target_type) pair have a
    chance of completing training? Returns False for LTR combos whose model
    subset contains NO native ranker flavour. Other target types are always
    runnable from this axis-pair perspective (other axes do their own
    canonicalisation downstream)."""
    if target_type == "learning_to_rank":
        return any(m in _LTR_NATIVE_RANKERS for m in models)
    return True


def _sample_axes(rng: random.Random) -> dict[str, Any]:
    return {name: rng.choice(values) for name, values in AXES.items()}


def _build_combo(models: tuple[str, ...], axes: dict[str, Any], seed: int) -> FuzzCombo:
    return FuzzCombo(
        models=tuple(sorted(models)),
        input_type=axes["input_type"],
        n_rows=axes["n_rows"],
        cat_feature_count=axes["cat_feature_count"],
        null_fraction_cats=axes["null_fraction_cats"],
        use_mrmr_fs=axes["use_mrmr_fs"],
        weight_schemas=axes["weight_schemas"],
        target_type=axes["target_type"],
        auto_detect_cats=axes["auto_detect_cats"],
        align_polars_categorical_dicts=axes["align_polars_categorical_dicts"],
        seed=seed,
        prefer_polarsds=axes.get("prefer_polarsds", True),
        use_text_features=axes.get("use_text_features", True),
        honor_user_dtype=axes.get("honor_user_dtype", False),
        target_carrier=axes.get("target_carrier", "numpy"),
        text_col_count=axes.get("text_col_count", 0),
        embedding_col_count=axes.get("embedding_col_count", 0),
        # 2026-04-24 combo-extension axes
        outlier_detection=axes.get("outlier_detection"),
        use_ensembles=axes.get("use_ensembles", False),
        continue_on_model_failure=axes.get("continue_on_model_failure", False),
        iterations=axes.get("iterations", 3),
        prefer_calibrated_classifiers=axes.get("prefer_calibrated_classifiers", False),
        inject_degenerate_cols=axes.get("inject_degenerate_cols", False),
        inject_inf_nan=axes.get("inject_inf_nan", False),
        with_datetime_col=axes.get("with_datetime_col", False),
        inject_zero_col=axes.get("inject_zero_col", False),
        fairness_col=axes.get("fairness_col"),
        custom_prep=axes.get("custom_prep"),
        input_storage=axes.get("input_storage", "memory"),
        # 2026-04-24 round 2
        fillna_value_cfg=axes.get("fillna_value_cfg"),
        scaler_name_cfg=axes.get("scaler_name_cfg", "standard"),
        categorical_encoding_cfg=axes.get("categorical_encoding_cfg", "ordinal"),
        skip_categorical_encoding_cfg=axes.get("skip_categorical_encoding_cfg", False),
        val_placement_cfg=axes.get("val_placement_cfg", "forward"),
        test_size_cfg=axes.get("test_size_cfg", 0.1),
        trainset_aging_limit_cfg=axes.get("trainset_aging_limit_cfg"),
        cat_text_card_threshold_cfg=axes.get("cat_text_card_threshold_cfg", 300),
        early_stopping_rounds_cfg=axes.get("early_stopping_rounds_cfg"),
        use_robust_eval_metric_cfg=axes.get("use_robust_eval_metric_cfg", True),
        # Fix G
        inject_label_leak=axes.get("inject_label_leak", False),
        inject_rank_deficient=axes.get("inject_rank_deficient", False),
        inject_all_nan_col=axes.get("inject_all_nan_col", False),
        # R3
        inject_test_drift=axes.get("inject_test_drift"),
        imbalance_ratio=axes.get("imbalance_ratio", "balanced"),
        weird_cat_content=axes.get("weird_cat_content"),
        multilabel_strategy_cfg=axes.get("multilabel_strategy_cfg", "auto"),
        # 2026-04-26 batch 1
        fix_infinities_cfg=axes.get("fix_infinities_cfg", True),
        ensure_float32_cfg=axes.get("ensure_float32_cfg", True),
        remove_constant_columns_cfg=axes.get("remove_constant_columns_cfg", True),
        imputer_strategy_cfg=axes.get("imputer_strategy_cfg", "mean"),
        shuffle_val_cfg=axes.get("shuffle_val_cfg", False),
        shuffle_test_cfg=axes.get("shuffle_test_cfg", False),
        wholeday_splitting_cfg=axes.get("wholeday_splitting_cfg", True),
        val_sequential_fraction_cfg=axes.get("val_sequential_fraction_cfg", 0.5),
        # batch 3 — multilabel dispatch
        multilabel_n_chains_cfg=axes.get("multilabel_n_chains_cfg", 3),
        multilabel_chain_order_cfg=axes.get("multilabel_chain_order_cfg", "random"),
        multilabel_cv_cfg=axes.get("multilabel_cv_cfg", 5),
        # batch 4 — PreprocessingExtensionsConfig
        prep_ext_scaler_cfg=axes.get("prep_ext_scaler_cfg"),
        prep_ext_kbins_cfg=axes.get("prep_ext_kbins_cfg"),
        prep_ext_polynomial_degree_cfg=axes.get("prep_ext_polynomial_degree_cfg"),
        prep_ext_dim_reducer_cfg=axes.get("prep_ext_dim_reducer_cfg"),
        prep_ext_nonlinear_cfg=axes.get("prep_ext_nonlinear_cfg"),
        prep_ext_pysr_enabled_cfg=axes.get("prep_ext_pysr_enabled_cfg", False),
        mrmr_nan_strategy_cfg=axes.get("mrmr_nan_strategy_cfg", "separate_bin"),
        # batch 5
        rfecv_estimator_cfg=axes.get("rfecv_estimator_cfg"),
        # batch 6
        recurrent_model_cfg=axes.get("recurrent_model_cfg"),
        # 2026-04-28 batch 4 followup
        include_confidence_analysis_cfg=axes.get("include_confidence_analysis_cfg", False),
        # 2026-05-11 Wave 15 -- MRMR-internal knobs
        mrmr_interactions_max_order_cfg=axes.get("mrmr_interactions_max_order_cfg", 1),
        mrmr_fe_max_steps_cfg=axes.get("mrmr_fe_max_steps_cfg", 1),
        mrmr_cat_fe_enable_cfg=axes.get("mrmr_cat_fe_enable_cfg", True),
        # 2026-05-11 Wave 21 -- assorted config-toggle axes
        dummy_baselines_enabled_cfg=axes.get("dummy_baselines_enabled_cfg", True),
        baseline_diagnostics_enabled_cfg=axes.get("baseline_diagnostics_enabled_cfg", True),
        use_groups_cfg=axes.get("use_groups_cfg", True),
        apply_outlier_to_val_cfg=axes.get("apply_outlier_to_val_cfg", True),
        multilabel_allow_uncalibrated_cfg=axes.get("multilabel_allow_uncalibrated_cfg", False),
        report_residual_audit_cfg=axes.get("report_residual_audit_cfg", True),
        ltr_assume_comparable_scales_cfg=axes.get("ltr_assume_comparable_scales_cfg", False),
        composite_discovery_enabled_cfg=axes.get("composite_discovery_enabled_cfg", False),
        composite_transforms_mode_cfg=axes.get("composite_transforms_mode_cfg", None),
        mrmr_fe_npermutations_cfg=axes.get("mrmr_fe_npermutations_cfg", 0),
        mrmr_fe_ntop_features_cfg=axes.get("mrmr_fe_ntop_features_cfg", 0),
        mrmr_fe_unary_preset_cfg=axes.get("mrmr_fe_unary_preset_cfg", "minimal"),
        mrmr_fe_binary_preset_cfg=axes.get("mrmr_fe_binary_preset_cfg", "minimal"),
        mrmr_fe_smart_polynom_iters_cfg=axes.get("mrmr_fe_smart_polynom_iters_cfg", 0),
        mrmr_fe_smart_polynom_steps_cfg=axes.get("mrmr_fe_smart_polynom_steps_cfg", 10),
        mrmr_fe_min_polynom_degree_cfg=axes.get("mrmr_fe_min_polynom_degree_cfg", 3),
        mrmr_fe_max_polynom_degree_cfg=axes.get("mrmr_fe_max_polynom_degree_cfg", 3),
        mrmr_cat_fe_include_numeric_cfg=axes.get("mrmr_cat_fe_include_numeric_cfg", False),
        # 2026-05-19 -- PreprocessingExtensionsConfig polynomial-auto-tune
        # axes. Previously declared in AXES + dataclass but not threaded
        # through _build_combo, so randomised values silently fell back to
        # dataclass defaults. Wired 2026-05-21.
        prep_ext_polynomial_max_features_cfg=axes.get(
            "prep_ext_polynomial_max_features_cfg", 10_000
        ),
        prep_ext_polynomial_interaction_only_cfg=axes.get(
            "prep_ext_polynomial_interaction_only_cfg", True
        ),
        prep_ext_memory_safety_max_bytes_cfg=axes.get(
            "prep_ext_memory_safety_max_bytes_cfg", 500_000_000
        ),
        # 2026-05-19 -- composite-discovery stacked-residual axes. Same
        # un-wired bug as above; defaulting these collapsed every combo to
        # the False / True dataclass defaults regardless of the axis value.
        composite_use_stacked_discovery_cfg=axes.get(
            "composite_use_stacked_discovery_cfg", False
        ),
        composite_use_stacked_discovery_residual_cfg=axes.get(
            "composite_use_stacked_discovery_residual_cfg", False
        ),
        # 2026-05-22 -- six gate-flip axes from the TVT-MLP-collapse
        # cascade. Defaults match the post-fix values; AXES holds the
        # pre-fix variant for regression coverage.
        composite_skip_raw_dominates_ratio_cfg=axes.get(
            "composite_skip_raw_dominates_ratio_cfg", 0.0
        ),
        composite_skip_ablation_delta_pct_cfg=axes.get(
            "composite_skip_ablation_delta_pct_cfg", 0.0
        ),
        composite_eps_mi_gain_cfg=axes.get(
            "composite_eps_mi_gain_cfg", -10.0
        ),
        composite_top_k_after_mi_cfg=axes.get(
            "composite_top_k_after_mi_cfg", 32
        ),
        composite_require_beats_raw_baseline_cfg=axes.get(
            "composite_require_beats_raw_baseline_cfg", False
        ),
        composite_per_bin_n_bins_cfg=axes.get(
            "composite_per_bin_n_bins_cfg", 0
        ),
        composite_tiny_screening_mode_cfg=axes.get(
            "composite_tiny_screening_mode_cfg", "per_family"
        ),
        composite_include_additive_residual_cfg=axes.get(
            "composite_include_additive_residual_cfg", True
        ),
        mlp_activation_cfg=axes.get("mlp_activation_cfg", "ReLU"),
        composite_skip_wrap_pass_predict_cfg=axes.get(
            "composite_skip_wrap_pass_predict_cfg", True
        ),
        # 2026-05-21 -- mini-HPT (target + feature distribution analyzer)
        # + MRMR FE pair-check subsample knob.
        enable_target_distribution_analyzer_cfg=axes.get(
            "enable_target_distribution_analyzer_cfg", True
        ),
        fe_check_pairs_subsample_n_cfg=axes.get(
            "fe_check_pairs_subsample_n_cfg", 0
        ),
        # 2026-05-21 iter150 -- multi-target / multi-target-type axis.
        extra_targets=axes.get("extra_targets", None),
        # 2026-05-21 iter151 -- P0/P1/P2 audit fill-in.
        enable_quantile_regression_cfg=axes.get("enable_quantile_regression_cfg", False),
        linear_alpha_cfg=axes.get("linear_alpha_cfg", 1.0),
        linear_solver_cfg=axes.get("linear_solver_cfg", "lbfgs"),
        enable_feature_handling_config_cfg=axes.get("enable_feature_handling_config_cfg", False),
        enable_precomputed_cfg=axes.get("enable_precomputed_cfg", False),
        test_sequential_fraction_cfg=axes.get("test_sequential_fraction_cfg", None),
        calib_size_cfg=axes.get("calib_size_cfg", None),
        use_boruta_shap_cfg=axes.get("use_boruta_shap_cfg", False),
        boruta_importance_measure_cfg=axes.get("boruta_importance_measure_cfg", "gini"),
        # 2026-06-03 FS-coverage audit -- BorutaShap.__init__ knobs.
        boruta_optimistic_cfg=axes.get("boruta_optimistic_cfg", True),
        boruta_train_or_test_cfg=axes.get("boruta_train_or_test_cfg", "train"),
        boruta_premerge_clusters_cfg=axes.get("boruta_premerge_clusters_cfg", False),
        # 2026-06-04 FS-coverage follow-up -- BorutaShap.__init__ early_stop_* knobs.
        boruta_early_stop_tentative_cfg=axes.get("boruta_early_stop_tentative_cfg", False),
        boruta_early_stop_patience_cfg=axes.get("boruta_early_stop_patience_cfg", 20),
        boruta_early_stop_margin_cfg=axes.get("boruta_early_stop_margin_cfg", 0.15),
        # 2026-06-03 FS-coverage audit -- RFECV.__init__ knobs.
        rfecv_votes_aggregation_cfg=axes.get("rfecv_votes_aggregation_cfg", "Borda"),
        rfecv_search_method_cfg=axes.get("rfecv_search_method_cfg", "ModelBasedHeuristic"),
        # 2026-06-03: these 5 were in AXES + FuzzCombo + canonical_key + suite-wired
        # but were never applied here in _build_combo, so every combo carried the
        # dataclass default and the axis was silently inert (never fuzzed). Wiring
        # them through makes the sampled value take effect; defaults match the
        # dataclass so default-valued combos are unchanged.
        fs_pre_screen_unsupervised_cfg=axes.get("fs_pre_screen_unsupervised_cfg", True),
        fs_pre_screen_variance_threshold_cfg=axes.get("fs_pre_screen_variance_threshold_cfg", 0.0),
        ranking_ensemble_method=axes.get("ranking_ensemble_method", "rrf"),
        target_temporal_audit_column_cfg=axes.get("target_temporal_audit_column_cfg", None),
        mlp_extreme_ar_group_aware_skip_cfg=axes.get("mlp_extreme_ar_group_aware_skip_cfg", False),
        # 2026-06-03: 11 CompositeTargetDiscoveryConfig axes that were in AXES +
        # canonical_key but neither applied here nor passed by the discovery
        # builder, so they were silently inert. Wired through here + in
        # build_composite_discovery_config(_from_flat); defaults match the
        # CompositeTargetDiscoveryConfig dataclass.
        composite_always_build_ct_ensemble_for_raw_cfg=axes.get("composite_always_build_ct_ensemble_for_raw_cfg", True),
        composite_ct_ensemble_dummy_floor_enabled_cfg=axes.get("composite_ct_ensemble_dummy_floor_enabled_cfg", True),
        composite_ct_ensemble_dummy_floor_tolerance_cfg=axes.get("composite_ct_ensemble_dummy_floor_tolerance_cfg", 0.0),
        composite_extreme_ar_group_aware_skip_cfg=axes.get("composite_extreme_ar_group_aware_skip_cfg", True),
        composite_extreme_ar_threshold_cfg=axes.get("composite_extreme_ar_threshold_cfg", 0.99),
        composite_lag_predict_failsafe_tolerance_cfg=axes.get("composite_lag_predict_failsafe_tolerance_cfg", 0.10),
        composite_oof_holdout_source_cfg=axes.get("composite_oof_holdout_source_cfg", "external_val"),
        composite_oof_holdout_frac_cfg=axes.get("composite_oof_holdout_frac_cfg", 0.2),
        composite_stacking_aware_gate_enabled_cfg=axes.get("composite_stacking_aware_gate_enabled_cfg", False),
        composite_top_m_after_tiny_cfg=axes.get("composite_top_m_after_tiny_cfg", 10),
        composite_use_baseline_diagnostics_hint_cfg=axes.get("composite_use_baseline_diagnostics_hint_cfg", True),
        # TrainingSplitConfig.composite_cardinality_cap (Field default 200, ge=2);
        # was inert (not applied here). Wired into the split_config build below.
        composite_cardinality_cap_cfg=axes.get("composite_cardinality_cap_cfg", 200),
        use_sample_weights_in_fs_cfg=axes.get("use_sample_weights_in_fs_cfg", False),
        fallback_to_sklearn_cfg=axes.get("fallback_to_sklearn_cfg", True),
        prefer_gpu_configs_cfg=axes.get("prefer_gpu_configs_cfg", True),
        prefer_cpu_for_lightgbm_cfg=axes.get("prefer_cpu_for_lightgbm_cfg", True),
        mrmr_identity_cache_scope_cfg=axes.get("mrmr_identity_cache_scope_cfg", "ctx"),
        skip_identity_equivalent_pre_pipelines_cfg=axes.get(
            "skip_identity_equivalent_pre_pipelines_cfg", True
        ),
        rfecv_leakage_corr_threshold_cfg=axes.get("rfecv_leakage_corr_threshold_cfg", 0.95),
        rfecv_mbh_adaptive_threshold_cfg=axes.get("rfecv_mbh_adaptive_threshold_cfg", 30),
        # 2026-05-22 iter162 -- nested-config / depth-2 audit fill-in.
        fhc_cache_eviction_strategy_cfg=axes.get("fhc_cache_eviction_strategy_cfg", "size_weighted"),
        fhc_cache_allow_pickle_cfg=axes.get("fhc_cache_allow_pickle_cfg", False),
        fhc_cache_ram_fraction_cfg=axes.get("fhc_cache_ram_fraction_cfg", 0.3),
        fhc_text_definite_text_mean_chars_cfg=axes.get("fhc_text_definite_text_mean_chars_cfg", 100),
        fhc_text_min_alphabet_entropy_cfg=axes.get("fhc_text_min_alphabet_entropy_cfg", 4.5),
        fhc_repro_deterministic_torch_cfg=axes.get("fhc_repro_deterministic_torch_cfg", False),
        fhc_auto_locale_detection_cfg=axes.get("fhc_auto_locale_detection_cfg", "fallback_only"),
        enable_viz_rendering_cfg=axes.get("enable_viz_rendering_cfg", False),
        reporting_prob_histogram_yscale_cfg=axes.get("reporting_prob_histogram_yscale_cfg", "auto"),
        reporting_title_metrics_template_cfg=axes.get(
            "reporting_title_metrics_template_cfg",
            "ICE BR_DECOMP ECE CMAEW LL ROC_AUC PR_AUC",
        ),
        reporting_matplotlib_rcparams_cfg=axes.get("reporting_matplotlib_rcparams_cfg", None),
        reporting_multiclass_panels_cfg=axes.get(
            "reporting_multiclass_panels_cfg",
            "CONFUSION PR_F1 ROC CALIB_GRID PROB_DIST TOP_K_ACC",
        ),
        confidence_model_kwargs_cfg=axes.get("confidence_model_kwargs_cfg", "default"),
        composite_mi_estimator_cfg=axes.get("composite_mi_estimator_cfg", "bin"),
        composite_mi_nbins_cfg=axes.get("composite_mi_nbins_cfg", 16),
        composite_mi_aggregation_cfg=axes.get("composite_mi_aggregation_cfg", "mean"),
        composite_mi_sample_strategy_cfg=axes.get("composite_mi_sample_strategy_cfg", "random"),
        composite_stacked_residual_aggregation_cfg=axes.get(
            "composite_stacked_residual_aggregation_cfg", "mean"
        ),
        composite_discovery_n_jobs_cfg=axes.get("composite_discovery_n_jobs_cfg", 1),
        quantile_crossing_fix_cfg=axes.get("quantile_crossing_fix_cfg", "sort"),
        quantile_coverage_pairs_cfg=axes.get("quantile_coverage_pairs_cfg", "default"),
        quantile_wrapper_n_jobs_cfg=axes.get("quantile_wrapper_n_jobs_cfg", "auto"),
        mlp_predict_batch_size_cfg=axes.get("mlp_predict_batch_size_cfg", None),
        ltr_cb_loss_fn_cfg=axes.get("ltr_cb_loss_fn_cfg", "YetiRankPairwise"),
        ltr_lgb_objective_cfg=axes.get("ltr_lgb_objective_cfg", "lambdarank"),
        ltr_rrf_k_cfg=axes.get("ltr_rrf_k_cfg", 60),
        recurrent_precision_cfg=axes.get("recurrent_precision_cfg", "32-true"),
        recurrent_sequence_preprocessing_cfg=axes.get("recurrent_sequence_preprocessing_cfg", "none"),
        # 2026-05-22 iter170 -- wave-3 depth-3+ audit.
        lgb_feature_fraction_cfg=axes.get("lgb_feature_fraction_cfg", 1.0),
        lgb_num_leaves_cfg=axes.get("lgb_num_leaves_cfg", 31),
        xgb_max_depth_cfg=axes.get("xgb_max_depth_cfg", 6),
        xgb_colsample_bynode_cfg=axes.get("xgb_colsample_bynode_cfg", 1.0),
        cb_border_count_cfg=axes.get("cb_border_count_cfg", 254),
        hgb_max_leaf_nodes_cfg=axes.get("hgb_max_leaf_nodes_cfg", 31),
        rfecv_cv_n_splits_cfg=axes.get("rfecv_cv_n_splits_cfg", 2),
        robust_q_low_cfg=axes.get("robust_q_low_cfg", 0.01),
        robust_q_high_cfg=axes.get("robust_q_high_cfg", 0.99),
        tfidf_max_features_cfg=axes.get("tfidf_max_features_cfg", 5000),
        kbins_encode_cfg=axes.get("kbins_encode_cfg", "ordinal"),
        nonlinear_n_components_cfg=axes.get("nonlinear_n_components_cfg", 100),
        pysr_operator_preset_cfg=axes.get("pysr_operator_preset_cfg", "standard"),
        confidence_ensemble_quantile_cfg=axes.get("confidence_ensemble_quantile_cfg", 0.1),
        cat_text_card_threshold_pct_cfg=axes.get("cat_text_card_threshold_pct_cfg", 0.001),
        rfecv_n_features_selection_rule_cfg=axes.get("rfecv_n_features_selection_rule_cfg", "auto"),
        rfecv_stability_selection_cfg=axes.get("rfecv_stability_selection_cfg", False),
        rfecv_leakage_action_cfg=axes.get("rfecv_leakage_action_cfg", "warn"),
        mrmr_fe_adaptive_threshold_relax_cfg=axes.get("mrmr_fe_adaptive_threshold_relax_cfg", True),
        mrmr_use_simple_mode_cfg=axes.get("mrmr_use_simple_mode_cfg", False),
        mrmr_identity_cache_include_y_cfg=axes.get("mrmr_identity_cache_include_y_cfg", True),
        mrmr_build_friend_graph_cfg=axes.get("mrmr_build_friend_graph_cfg", True),
        mrmr_friend_graph_prune_cfg=axes.get("mrmr_friend_graph_prune_cfg", False),
        mrmr_cluster_aggregate_enable_cfg=axes.get("mrmr_cluster_aggregate_enable_cfg", True),
        mrmr_cluster_aggregate_mode_cfg=axes.get("mrmr_cluster_aggregate_mode_cfg", "augment"),
        use_shap_proxied_fs=axes.get("use_shap_proxied_fs", False),
        shap_proxied_optimizer_cfg=axes.get("shap_proxied_optimizer_cfg", "auto"),
        shap_proxied_revalidate_cfg=axes.get("shap_proxied_revalidate_cfg", True),
        shap_proxied_trust_guard_cfg=axes.get("shap_proxied_trust_guard_cfg", True),
        shap_proxied_interaction_aware_cfg=axes.get("shap_proxied_interaction_aware_cfg", False),
        shap_proxied_cluster_features_cfg=axes.get("shap_proxied_cluster_features_cfg", "auto"),
        # 2026-05-28 new fuzz axes (ShapProxiedFS ext, FHC text card, composite deep,
        # extreme-AR skip list, FS null-frac, linear l1 ratio, recurrent hidden_size).
        shap_proxied_active_learning_cfg=axes.get("shap_proxied_active_learning_cfg", False),
        shap_proxied_prefilter_method_cfg=axes.get("shap_proxied_prefilter_method_cfg", "auto"),
        # 2026-05-28 audit-pass-2 B1-B6: ShapProxiedFS deeper extension axes.
        shap_proxied_config_jitter_cfg=axes.get("shap_proxied_config_jitter_cfg", False),
        shap_proxied_uncertainty_penalty_cfg=axes.get("shap_proxied_uncertainty_penalty_cfg", 0.0),
        shap_proxied_within_cluster_refine_cfg=axes.get(
            "shap_proxied_within_cluster_refine_cfg", True
        ),
        shap_proxied_use_bias_corrector_cfg=axes.get(
            "shap_proxied_use_bias_corrector_cfg", True
        ),
        shap_proxied_refine_n_estimators_cfg=axes.get(
            "shap_proxied_refine_n_estimators_cfg", 100
        ),
        shap_proxied_trust_guard_n_estimators_cfg=axes.get(
            "shap_proxied_trust_guard_n_estimators_cfg", 100
        ),
        # 2026-05-28 ShapProxiedFS audit-pass-3 axes (W3).
        shap_proxied_cluster_weighting_cfg=axes.get(
            "shap_proxied_cluster_weighting_cfg", "pca_pc1"
        ),
        # iter624 (audit-pass-13 INFORMATIONAL): iter67 SU-pairwise cluster.
        shap_proxied_cluster_use_precomputed_bins_cfg=axes.get(
            "shap_proxied_cluster_use_precomputed_bins_cfg", True
        ),
        shap_proxied_cluster_su_threshold_cfg=axes.get(
            "shap_proxied_cluster_su_threshold_cfg", 0.5
        ),
        shap_proxied_max_interaction_features_cfg=axes.get(
            "shap_proxied_max_interaction_features_cfg", 16
        ),
        shap_proxied_prefilter_top_cfg=axes.get(
            "shap_proxied_prefilter_top_cfg", 2000
        ),
        shap_proxied_prefilter_n_estimators_cfg=axes.get(
            "shap_proxied_prefilter_n_estimators_cfg", 100
        ),
        # 2026-05-28 ShapProxiedFS audit-pass-5 axes (W5).
        shap_proxied_trust_guard_stratified_anchors_cfg=axes.get(
            "shap_proxied_trust_guard_stratified_anchors_cfg", False
        ),
        shap_proxied_trust_guard_uniform_tail_frac_cfg=axes.get(
            "shap_proxied_trust_guard_uniform_tail_frac_cfg", 0.2
        ),
        shap_proxied_trust_guard_cardinality_dist_cfg=axes.get(
            "shap_proxied_trust_guard_cardinality_dist_cfg", "zipf"
        ),
        shap_proxied_trust_guard_zipf_alpha_cfg=axes.get(
            "shap_proxied_trust_guard_zipf_alpha_cfg", 0.25
        ),
        shap_proxied_trust_guard_fidelity_weights_cfg=axes.get(
            "shap_proxied_trust_guard_fidelity_weights_cfg", (0.6, 0.4)
        ),
        shap_proxied_trust_guard_metric_cfg=axes.get(
            "shap_proxied_trust_guard_metric_cfg", "proxy_fidelity_score"
        ),
        shap_proxied_fidelity_floor_cfg=axes.get(
            "shap_proxied_fidelity_floor_cfg", 0.5
        ),
        shap_proxied_oof_shap_n_estimators_cfg=axes.get(
            "shap_proxied_oof_shap_n_estimators_cfg", 100
        ),
        # 2026-05-28 audit-pass-2 PART A coverage-gap axes.
        ensembling_degenerate_class_ratio_cfg=axes.get(
            "ensembling_degenerate_class_ratio_cfg", 0.01
        ),
        target_temporal_audit_granularity_cfg=axes.get(
            "target_temporal_audit_granularity_cfg", "auto"
        ),
        prep_ext_dim_n_components_cfg=axes.get("prep_ext_dim_n_components_cfg", 50),
        fhc_text_min_cardinality_cfg=axes.get("fhc_text_min_cardinality_cfg", 300),
        composite_auto_skip_on_baseline_optimal_cfg=axes.get(
            "composite_auto_skip_on_baseline_optimal_cfg", False
        ),
        composite_mi_n_neighbors_cfg=axes.get("composite_mi_n_neighbors_cfg", 3),
        composite_auto_base_null_perms_cfg=axes.get("composite_auto_base_null_perms_cfg", 20),
        composite_multi_base_max_k_cfg=axes.get("composite_multi_base_max_k_cfg", 3),
        extreme_ar_group_aware_skip_models_cfg=axes.get(
            "extreme_ar_group_aware_skip_models_cfg", "default_neural"
        ),
        fs_pre_screen_null_fraction_threshold_cfg=axes.get(
            "fs_pre_screen_null_fraction_threshold_cfg", 0.99
        ),
        linear_l1_ratio_cfg=axes.get("linear_l1_ratio_cfg", 0.5),
        recurrent_hidden_size_cfg=axes.get("recurrent_hidden_size_cfg", 128),
        catfe_fwer_correction_cfg=axes.get("catfe_fwer_correction_cfg", "none"),
        catfe_perm_budget_strategy_cfg=axes.get("catfe_perm_budget_strategy_cfg", "bandit_ucb1"),
        catfe_permutation_null_cfg=axes.get("catfe_permutation_null_cfg", "joint_independence"),
        catfe_bootstrap_ci_n_replicates_cfg=axes.get("catfe_bootstrap_ci_n_replicates_cfg", 0),
        catfe_use_miller_madow_cfg=axes.get("catfe_use_miller_madow_cfg", None),
        catfe_refine_passes_cfg=axes.get("catfe_refine_passes_cfg", 0),
        catfe_enable_streaming_cache_cfg=axes.get("catfe_enable_streaming_cache_cfg", False),
        catfe_unknown_strategy_cfg=axes.get("catfe_unknown_strategy_cfg", "clip"),
        composite_screening_cfg=axes.get("composite_screening_cfg", "hybrid"),
        composite_tiny_model_num_leaves_cfg=axes.get("composite_tiny_model_num_leaves_cfg", 15),
        composite_tiny_model_learning_rate_cfg=axes.get("composite_tiny_model_learning_rate_cfg", 0.1),
        composite_raw_baseline_tolerance_cfg=axes.get("composite_raw_baseline_tolerance_cfg", 1.02),
        composite_use_wilcoxon_gate_cfg=axes.get("composite_use_wilcoxon_gate_cfg", False),
        composite_detect_alpha_drift_cfg=axes.get("composite_detect_alpha_drift_cfg", True),
        composite_reject_on_alpha_drift_cfg=axes.get("composite_reject_on_alpha_drift_cfg", False),
        reporting_figsize_cfg=axes.get("reporting_figsize_cfg", "default"),
        reporting_plot_dpi_cfg=axes.get("reporting_plot_dpi_cfg", None),
        reporting_quantile_panels_cfg=axes.get("reporting_quantile_panels_cfg", "default"),
        reporting_ltr_panels_cfg=axes.get("reporting_ltr_panels_cfg", "default"),
        reporting_plotly_template_cfg=axes.get("reporting_plotly_template_cfg", None),
        reporting_matplotlib_style_cfg=axes.get("reporting_matplotlib_style_cfg", None),
        baseline_quick_model_n_estimators_cfg=axes.get("baseline_quick_model_n_estimators_cfg", 200),
        baseline_quick_model_num_leaves_cfg=axes.get("baseline_quick_model_num_leaves_cfg", 31),
        baseline_quick_model_learning_rate_cfg=axes.get("baseline_quick_model_learning_rate_cfg", 0.05),
        baseline_sample_n_cfg=axes.get("baseline_sample_n_cfg", 50_000),
        baseline_high_potential_min_dominance_pct_cfg=axes.get("baseline_high_potential_min_dominance_pct_cfg", 5.0),
        baseline_best_model_min_lift_cfg=axes.get("baseline_best_model_min_lift_cfg", 1.5),
        dummy_stratified_n_repeats_cfg=axes.get("dummy_stratified_n_repeats_cfg", 20),
        dummy_paired_bootstrap_n_resamples_cfg=axes.get("dummy_paired_bootstrap_n_resamples_cfg", 1000),
        ltr_mlp_loss_fn_cfg=axes.get("ltr_mlp_loss_fn_cfg", "ranknet"),
        ltr_eval_at_cfg=axes.get("ltr_eval_at_cfg", "default"),
        multilabel_force_native_xgb_cfg=axes.get("multilabel_force_native_xgb_cfg", False),
        fhc_pricing_cap_usd_cfg=axes.get("fhc_pricing_cap_usd_cfg", None),
        fhc_pricing_warn_above_usd_cfg=axes.get("fhc_pricing_warn_above_usd_cfg", 1.0),
        fhc_logging_verbose_cfg=axes.get("fhc_logging_verbose_cfg", False),
        fhc_repro_langdetect_seed_cfg=axes.get("fhc_repro_langdetect_seed_cfg", 0),
        fhc_repro_pinned_svd_solver_params_cfg=axes.get("fhc_repro_pinned_svd_solver_params_cfg", True),
        fhc_repro_forbid_nonatomic_fs_cfg=axes.get("fhc_repro_forbid_nonatomic_fs_cfg", False),
        fhc_repro_deterministic_eviction_cfg=axes.get("fhc_repro_deterministic_eviction_cfg", False),
        fhc_cache_prefetch_enabled_cfg=axes.get("fhc_cache_prefetch_enabled_cfg", True),
        fhc_cache_prefetch_vram_safety_factor_cfg=axes.get("fhc_cache_prefetch_vram_safety_factor_cfg", 2.0),
        fhc_memory_pressure_watermark_pct_cfg=axes.get("fhc_memory_pressure_watermark_pct_cfg", 85),
        fhc_text_min_mean_tokens_cfg=axes.get("fhc_text_min_mean_tokens_cfg", 4.0),
        fhc_text_min_unique_ratio_cfg=axes.get("fhc_text_min_unique_ratio_cfg", 0.95),
        fhc_text_respect_explicit_cat_dtype_cfg=axes.get("fhc_text_respect_explicit_cat_dtype_cfg", True),
        recurrent_input_mode_cfg=axes.get("recurrent_input_mode_cfg", "hybrid"),
        recurrent_num_workers_cfg=axes.get("recurrent_num_workers_cfg", 0),
        # 2026-05-23 iter180 -- DEPTH-4.
        lgb_boosting_type_cfg=axes.get("lgb_boosting_type_cfg", "gbdt"),
        lgb_dart_drop_rate_cfg=axes.get("lgb_dart_drop_rate_cfg", 0.1),
        lgb_goss_top_rate_cfg=axes.get("lgb_goss_top_rate_cfg", 0.2),
        xgb_tree_method_cfg=axes.get("xgb_tree_method_cfg", "auto"),
        xgb_hist_max_bin_cfg=axes.get("xgb_hist_max_bin_cfg", 256),
        cb_bootstrap_type_cfg=axes.get("cb_bootstrap_type_cfg", "Bayesian"),
        cb_bayesian_bagging_temperature_cfg=axes.get("cb_bayesian_bagging_temperature_cfg", 1.0),
        cb_bernoulli_subsample_cfg=axes.get("cb_bernoulli_subsample_cfg", 0.8),
        cb_grow_policy_cfg=axes.get("cb_grow_policy_cfg", "SymmetricTree"),
        cb_lossguide_max_leaves_cfg=axes.get("cb_lossguide_max_leaves_cfg", 31),
        fhc_cache_persistence_cfg=axes.get("fhc_cache_persistence_cfg", "auto"),
        multilabel_per_label_thresholds_cfg=axes.get("multilabel_per_label_thresholds_cfg", None),
        multilabel_chain_seeds_cfg=axes.get("multilabel_chain_seeds_cfg", None),
        # F1 -- enable_crash_reporting suite-level kwarg (Windows-only meaningful).
        enable_crash_reporting_cfg=axes.get("enable_crash_reporting_cfg", False),
        # 2026-05-28 audit-pass-4 SAFE-subset (W4): 8 axes.
        calibration_policy_auto_pick_cfg=axes.get("calibration_policy_auto_pick_cfg", True),
        calibration_n_bootstrap_cfg=axes.get("calibration_n_bootstrap_cfg", 1000),
        calibration_candidates_cfg=axes.get("calibration_candidates_cfg", None),
        pipeline_cache_ram_budget_fraction_cfg=axes.get(
            "pipeline_cache_ram_budget_fraction_cfg", 0.4
        ),
        reporting_compute_trainset_metrics_cfg=axes.get(
            "reporting_compute_trainset_metrics_cfg", False
        ),
        reporting_mase_seasonality_cfg=axes.get("reporting_mase_seasonality_cfg", 1),
        recurrent_use_stratified_sampler_cfg=axes.get(
            "recurrent_use_stratified_sampler_cfg", True
        ),
        behavior_model_file_hash_suffix_cfg=axes.get(
            "behavior_model_file_hash_suffix_cfg", True
        ),
        # 2026-05-30 audit-pass-6 (W6).
        slice_stable_es_enabled_cfg=axes.get("slice_stable_es_enabled_cfg", False),
        slice_stable_es_aggregate_cfg=axes.get("slice_stable_es_aggregate_cfg", "mean"),
        slice_stable_es_source_cfg=axes.get("slice_stable_es_source_cfg", "temporal"),
        slice_stable_es_pareto_best_iter_selection_cfg=axes.get(
            "slice_stable_es_pareto_best_iter_selection_cfg", False
        ),
        slice_stable_es_diagnostic_only_cfg=axes.get(
            "slice_stable_es_diagnostic_only_cfg", False
        ),
        early_stop_on_worsening_cfg=axes.get("early_stop_on_worsening_cfg", True),
        mrmr_nbins_strategy_cfg=axes.get("mrmr_nbins_strategy_cfg", "mdlp"),
        mrmr_mi_correction_cfg=axes.get("mrmr_mi_correction_cfg", "none"),
        mrmr_redundancy_aggregator_cfg=axes.get("mrmr_redundancy_aggregator_cfg", None),
        mrmr_bur_lambda_cfg=axes.get("mrmr_bur_lambda_cfg", 0.0),
        mrmr_cmi_perm_stop_cfg=axes.get("mrmr_cmi_perm_stop_cfg", False),
        mrmr_stability_selection_method_cfg=axes.get(
            "mrmr_stability_selection_method_cfg", "classic"
        ),
        mrmr_mi_normalization_cfg=axes.get("mrmr_mi_normalization_cfg", "none"),
        mrmr_dcd_enable_cfg=axes.get("mrmr_dcd_enable_cfg", True),
        # 2026-05-30 audit-pass-7 #2/#3/#4 -- defaults verified at
        # mrmr.py:309 and _adaptive_nbins.py:511,586.
        mrmr_baseline_npermutations_cfg=axes.get("mrmr_baseline_npermutations_cfg", 2),
        mrmr_low_card_cap_cfg=axes.get("mrmr_low_card_cap_cfg", 32),
        mrmr_collapsed_fallback_nbins_cfg=axes.get("mrmr_collapsed_fallback_nbins_cfg", 5),
        cv_selector_mode_cfg=axes.get("cv_selector_mode_cfg", "mean"),
        auto_wrap_partial_fit_es_force_off_cfg=axes.get(
            "auto_wrap_partial_fit_es_force_off_cfg", False
        ),
        # 2026-05-30 audit-pass-6 LOW-tier deferred batch (W6 LOW).
        shap_proxied_prefilter_stage1_keep_cfg=axes.get(
            "shap_proxied_prefilter_stage1_keep_cfg", None
        ),
        shap_proxied_prefilter_univariate_batch_size_cfg=axes.get(
            "shap_proxied_prefilter_univariate_batch_size_cfg", None
        ),
        shap_proxied_shap_prefilter_enabled_cfg=axes.get(
            "shap_proxied_shap_prefilter_enabled_cfg", True
        ),
        shap_proxied_shap_prefilter_safety_factor_cfg=axes.get(
            "shap_proxied_shap_prefilter_safety_factor_cfg", 4
        ),
        shap_proxied_shap_prefilter_min_features_cfg=axes.get(
            "shap_proxied_shap_prefilter_min_features_cfg", 40
        ),
        shap_proxied_shap_aware_stage1_keep_cfg=axes.get(
            "shap_proxied_shap_aware_stage1_keep_cfg", True
        ),
        shap_proxied_shap_aware_stage1_cushion_cfg=axes.get(
            "shap_proxied_shap_aware_stage1_cushion_cfg", 2
        ),
        shap_proxied_shap_aware_stage1_floor_cfg=axes.get(
            "shap_proxied_shap_aware_stage1_floor_cfg", 200
        ),
        shap_proxied_refine_ucb_enabled_cfg=axes.get(
            "shap_proxied_refine_ucb_enabled_cfg", True
        ),
        shap_proxied_refine_ucb_min_eval_size_cfg=axes.get(
            "shap_proxied_refine_ucb_min_eval_size_cfg", None
        ),
        shap_proxied_refine_ucb_slack_cfg=axes.get(
            "shap_proxied_refine_ucb_slack_cfg", None
        ),
        shap_proxied_refine_ucb_stdev_multiplier_cfg=axes.get(
            "shap_proxied_refine_ucb_stdev_multiplier_cfg", 1.0
        ),
        shap_proxied_revalidation_n_estimators_cfg=axes.get(
            "shap_proxied_revalidation_n_estimators_cfg", 100
        ),
        shap_proxied_revalidation_ucb_enabled_cfg=axes.get(
            "shap_proxied_revalidation_ucb_enabled_cfg", True
        ),
        shap_proxied_revalidation_ucb_min_eval_size_cfg=axes.get(
            "shap_proxied_revalidation_ucb_min_eval_size_cfg", None
        ),
        shap_proxied_revalidation_ucb_slack_cfg=axes.get(
            "shap_proxied_revalidation_ucb_slack_cfg", None
        ),
        shap_proxied_revalidation_ucb_stdev_multiplier_cfg=axes.get(
            "shap_proxied_revalidation_ucb_stdev_multiplier_cfg", None
        ),
        shap_proxied_inner_n_jobs_cap_cfg=axes.get(
            "shap_proxied_inner_n_jobs_cap_cfg", False
        ),
        early_stop_on_worsening_coeff_cfg=axes.get(
            "early_stop_on_worsening_coeff_cfg", 5
        ),
        early_stop_on_worsening_min_iters_cfg=axes.get(
            "early_stop_on_worsening_min_iters_cfg", 5
        ),
        mrmr_relaxmrmr_alpha_cfg=axes.get("mrmr_relaxmrmr_alpha_cfg", 0.0),
        mrmr_uaed_auto_size_cfg=axes.get("mrmr_uaed_auto_size_cfg", False),
        mrmr_cpt_test_cfg=axes.get("mrmr_cpt_test_cfg", False),
        mrmr_pid_synergy_bonus_cfg=axes.get("mrmr_pid_synergy_bonus_cfg", 0.0),
        cv_selector_alpha_cfg=axes.get("cv_selector_alpha_cfg", 1.0),
        cv_selector_confidence_cfg=axes.get("cv_selector_confidence_cfg", 0.9),
        cv_selector_quantile_level_cfg=axes.get("cv_selector_quantile_level_cfg", 0.9),
        cv_persist_fold_scores_cfg=axes.get("cv_persist_fold_scores_cfg", False),
        # 2026-05-31 audit-pass-8 HIGH (#1-#4). Defaults source-verified at
        # filters/mrmr.py:334 / :326 and training/neural/base.py:217 / :218.
        mrmr_cardinality_bias_correction_cfg=axes.get(
            "mrmr_cardinality_bias_correction_cfg", True
        ),
        mrmr_min_relevance_gain_relative_to_first_cfg=axes.get(
            "mrmr_min_relevance_gain_relative_to_first_cfg", 0.05
        ),
        mlp_random_state_cfg=axes.get("mlp_random_state_cfg", None),
        mlp_class_weight_cfg=axes.get("mlp_class_weight_cfg", None),
        # 2026-05-31 audit-pass-8 MED + LOW->MED (#5/#7/#8/#9/#10). Defaults
        # source-verified at shap_proxied_fs.py:208 / flat.py:205 + library
        # defaults for the remaining MLP / frame-builder injection axes.
        shap_proxied_adaptive_prescreen_by_stability_cfg=axes.get(
            "shap_proxied_adaptive_prescreen_by_stability_cfg", False
        ),
        mlp_use_layernorm_cfg=axes.get("mlp_use_layernorm_cfg", False),
        mlp_l1_alpha_cfg=axes.get("mlp_l1_alpha_cfg", 0.0),
        mlp_inject_zero_sample_weight_batch_cfg=axes.get(
            "mlp_inject_zero_sample_weight_batch_cfg", False
        ),
        inject_xor_synergy_pair_cfg=axes.get(
            "inject_xor_synergy_pair_cfg", False
        ),
        # 2026-05-31 audit-pass-9 (W9). Defaults mirror source verbatim.
        mlp_adamw_betas_cfg=axes.get("mlp_adamw_betas_cfg", (0.9, 0.95)),
        mlp_use_ema_cfg=axes.get("mlp_use_ema_cfg", False),
        mlp_label_smoothing_cfg=axes.get("mlp_label_smoothing_cfg", 0.0),
        mlp_focal_loss_gamma_cfg=axes.get("mlp_focal_loss_gamma_cfg", None),
        mlp_use_residual_cfg=axes.get("mlp_use_residual_cfg", False),
        mlp_use_learnable_cat_embeddings_cfg=axes.get("mlp_use_learnable_cat_embeddings_cfg", True),
        mlp_categorical_embed_dim_cfg=axes.get("mlp_categorical_embed_dim_cfg", None),
        mlp_numerical_embedding_cfg=axes.get("mlp_numerical_embedding_cfg", None),
        mlp_numerical_embedding_kwargs_cfg=axes.get(
            "mlp_numerical_embedding_kwargs_cfg", "paper_default"
        ),
        mrmr_fe_hybrid_orth_enable_cfg=axes.get(
            "mrmr_fe_hybrid_orth_enable_cfg", False
        ),
        mrmr_fe_hybrid_orth_pair_enable_cfg=axes.get(
            "mrmr_fe_hybrid_orth_pair_enable_cfg", True
        ),
        # 2026-05-31 audit-pass-10 (W10). Defaults mirror source verbatim.
        mlp_optimizer_cfg=axes.get("mlp_optimizer_cfg", "adamw"),
        mrmr_fe_hybrid_orth_degrees_cfg=axes.get(
            "mrmr_fe_hybrid_orth_degrees_cfg", (2, 3)
        ),
        mrmr_fe_hybrid_orth_basis_cfg=axes.get(
            "mrmr_fe_hybrid_orth_basis_cfg", "auto"
        ),
        mrmr_fe_hybrid_orth_top_k_cfg=axes.get(
            "mrmr_fe_hybrid_orth_top_k_cfg", 5
        ),
        mrmr_fe_hybrid_orth_pair_max_degree_cfg=axes.get(
            "mrmr_fe_hybrid_orth_pair_max_degree_cfg", 2
        ),
        # 2026-05-31 audit-pass-12 (W12). Defaults mirror source verbatim
        # (Group A canon-only markers, Group B MRMR FE master switches,
        # Group C MRMR+ShapProxiedFS coupling).
        composite_target_multilabel_strategy_cfg=axes.get(
            "composite_target_multilabel_strategy_cfg", "per_target"
        ),
        enable_ct_ensemble_cfg=axes.get("enable_ct_ensemble_cfg", True),
        mtr_eval_metric_cfg=axes.get("mtr_eval_metric_cfg", None),
        mrmr_fe_kfold_te_enable_cfg=axes.get(
            "mrmr_fe_kfold_te_enable_cfg", False
        ),
        mrmr_fe_missingness_indicator_enable_cfg=axes.get(
            "mrmr_fe_missingness_indicator_enable_cfg", False
        ),
        mrmr_fe_missingness_count_enable_cfg=axes.get(
            "mrmr_fe_missingness_count_enable_cfg", False
        ),
        mrmr_fe_missingness_pattern_enable_cfg=axes.get(
            "mrmr_fe_missingness_pattern_enable_cfg", False
        ),
        mrmr_fe_cat_aux_enable_cfg=axes.get(
            "mrmr_fe_cat_aux_enable_cfg", "off"
        ),
        mrmr_fe_hybrid_orth_extra_bases_cfg=axes.get(
            "mrmr_fe_hybrid_orth_extra_bases_cfg", ()
        ),
        mrmr_fe_ratio_delta_diff_cfg=axes.get(
            "mrmr_fe_ratio_delta_diff_cfg", "off"
        ),
        mrmr_fe_mi_greedy_enable_cfg=axes.get(
            "mrmr_fe_mi_greedy_enable_cfg", False
        ),
        mrmr_shap_proxy_artifact_reuse_cfg=axes.get(
            "mrmr_shap_proxy_artifact_reuse_cfg", "off"
        ),
        mrmr_shap_proxy_align_mode_cfg=axes.get(
            "mrmr_shap_proxy_align_mode_cfg", "exact"
        ),
        # 2026-05-31 audit-pass-14 (W14). Defaults source-verified at HEAD:
        #   F14-1 shap_proxied_cluster_backend_cfg = "auto"
        #         (shap_proxied_fs.py:258)
        #   F14-3 mrmr_partial_fit_decay_cfg = 0.0
        #         mrmr_partial_fit_min_recompute_cfg = 100
        #         mrmr_partial_fit_window_cfg = None
        #         (filters/mrmr.py:845-847)
        #   F14-4 mrmr_dcd_tau_cluster_cfg = 0.7 (filters/mrmr.py:621)
        #   F14-5 mrmr_dcd_distance_cfg = "su" (filters/mrmr.py:622)
        #         mrmr_dcd_swap_method_cfg = "auto" (filters/mrmr.py:655)
        shap_proxied_cluster_backend_cfg=axes.get(
            "shap_proxied_cluster_backend_cfg", "auto"
        ),
        mrmr_partial_fit_decay_cfg=axes.get(
            "mrmr_partial_fit_decay_cfg", 0.0
        ),
        mrmr_partial_fit_min_recompute_cfg=axes.get(
            "mrmr_partial_fit_min_recompute_cfg", 100
        ),
        mrmr_partial_fit_window_cfg=axes.get(
            "mrmr_partial_fit_window_cfg", None
        ),
        mrmr_dcd_tau_cluster_cfg=axes.get(
            "mrmr_dcd_tau_cluster_cfg", 0.7
        ),
        mrmr_dcd_distance_cfg=axes.get(
            "mrmr_dcd_distance_cfg", "su"
        ),
        mrmr_dcd_swap_method_cfg=axes.get(
            "mrmr_dcd_swap_method_cfg", "auto"
        ),
        # iter639/640 audit-pass-15. Defaults source-verified at HEAD
        # against feature_selection/filters/mrmr.py and
        # training/neural/_flat_torch_module.py / flat.py.
        mrmr_fe_hybrid_orth_default_scorer_cfg=axes.get(
            "mrmr_fe_hybrid_orth_default_scorer_cfg", "plug_in"
        ),
        mrmr_fe_hybrid_orth_meta_enable_cfg=axes.get(
            "mrmr_fe_hybrid_orth_meta_enable_cfg", False
        ),
        mrmr_fe_hybrid_orth_bootstrap_enable_cfg=axes.get(
            "mrmr_fe_hybrid_orth_bootstrap_enable_cfg", False
        ),
        mrmr_fe_hybrid_orth_three_gate_enable_cfg=axes.get(
            "mrmr_fe_hybrid_orth_three_gate_enable_cfg", False
        ),
        mlp_use_sam_cfg=axes.get("mlp_use_sam_cfg", False),
        mlp_use_lookahead_cfg=axes.get("mlp_use_lookahead_cfg", False),
        mlp_use_mixup_cfg=axes.get("mlp_use_mixup_cfg", False),
        mlp_spectral_norm_output_only_cfg=axes.get(
            "mlp_spectral_norm_output_only_cfg", False
        ),
        # iter642 audit-pass-15 batch 2.
        mrmr_fe_hybrid_orth_ensemble_enable_cfg=axes.get(
            "mrmr_fe_hybrid_orth_ensemble_enable_cfg", False
        ),
        mrmr_fe_hybrid_orth_lasso_enable_cfg=axes.get(
            "mrmr_fe_hybrid_orth_lasso_enable_cfg", False
        ),
        mrmr_fe_hybrid_orth_elasticnet_enable_cfg=axes.get(
            "mrmr_fe_hybrid_orth_elasticnet_enable_cfg", False
        ),
        mrmr_fe_hybrid_orth_adaptive_arity_enable_cfg=axes.get(
            "mrmr_fe_hybrid_orth_adaptive_arity_enable_cfg", False
        ),
        mrmr_fe_hybrid_orth_diff_basis_enable_cfg=axes.get(
            "mrmr_fe_hybrid_orth_diff_basis_enable_cfg", False
        ),
        mrmr_fe_semi_supervised_enable_cfg=axes.get(
            "mrmr_fe_semi_supervised_enable_cfg", False
        ),
        # audit-pass-16 — MRMR Layers 87-91.
        mrmr_fe_grouped_agg_enable_cfg=axes.get("mrmr_fe_grouped_agg_enable_cfg", False),
        mrmr_fe_grouped_quantile_enable_cfg=axes.get("mrmr_fe_grouped_quantile_enable_cfg", False),
        mrmr_fe_grouped_quantile_target_aware_cfg=axes.get("mrmr_fe_grouped_quantile_target_aware_cfg", False),
        mrmr_fe_cat_pair_enable_cfg=axes.get("mrmr_fe_cat_pair_enable_cfg", False),
        mrmr_fe_numeric_decompose_enable_cfg=axes.get("mrmr_fe_numeric_decompose_enable_cfg", False),
        mrmr_fe_numeric_decompose_digits_cfg=axes.get("mrmr_fe_numeric_decompose_digits_cfg", (0, 1, 2)),
        mrmr_fe_local_mi_gate_cfg=axes.get("mrmr_fe_local_mi_gate_cfg", True),  # audit-pass-17: source default True
        mrmr_fe_unified_second_pass_gate_cfg=axes.get("mrmr_fe_unified_second_pass_gate_cfg", False),
        # audit-pass-17 — Param-Oracle / fe_auto + FE families L92-104.
        mrmr_fe_auto_cfg=axes.get("mrmr_fe_auto_cfg", False),
        mrmr_fe_temporal_agg_enable_cfg=axes.get("mrmr_fe_temporal_agg_enable_cfg", False),
        mrmr_fe_composite_group_agg_enable_cfg=axes.get("mrmr_fe_composite_group_agg_enable_cfg", False),
        mrmr_fe_modular_enable_cfg=axes.get("mrmr_fe_modular_enable_cfg", False),
        mrmr_fe_group_distance_enable_cfg=axes.get("mrmr_fe_group_distance_enable_cfg", False),
        mrmr_fe_rare_category_enable_cfg=axes.get("mrmr_fe_rare_category_enable_cfg", False),
        mrmr_fe_conditional_residual_enable_cfg=axes.get("mrmr_fe_conditional_residual_enable_cfg", False),
        # 2026-06-13 coverage refresh -- embedding passthrough + 5 default-ON /
        # 1 default-OFF MRMR FE families. Defaults mirror MRMR.__init__ source.
        mrmr_embedding_passthrough_cfg=axes.get("mrmr_embedding_passthrough_cfg", True),
        mrmr_embedding_passthrough_detect_embeddings_cfg=axes.get("mrmr_embedding_passthrough_detect_embeddings_cfg", True),
        mrmr_embedding_passthrough_detect_text_cfg=axes.get("mrmr_embedding_passthrough_detect_text_cfg", True),
        mrmr_fe_hinge_enable_cfg=axes.get("mrmr_fe_hinge_enable_cfg", True),
        mrmr_fe_conditional_dispersion_enable_cfg=axes.get("mrmr_fe_conditional_dispersion_enable_cfg", True),
        mrmr_fe_wavelet_enable_cfg=axes.get("mrmr_fe_wavelet_enable_cfg", True),
        mrmr_fe_stability_vote_enable_cfg=axes.get("mrmr_fe_stability_vote_enable_cfg", True),
        mrmr_fe_sufficient_summary_early_stop_cfg=axes.get("mrmr_fe_sufficient_summary_early_stop_cfg", True),
        mrmr_fe_gradient_interaction_enable_cfg=axes.get("mrmr_fe_gradient_interaction_enable_cfg", False),
    )



def _all_axis_pairs() -> set[tuple[str, Any, str, Any]]:
    pairs: set[tuple[str, Any, str, Any]] = set()
    # Also include model-count ("n_models" pseudo-axis) to balance single vs
    # multi-model combos across other axes.
    axes_ext: dict[str, tuple[Any, ...]] = {**AXES, "n_models": (1, 2, 3, 4, 5)}
    axis_names = list(axes_ext.keys())
    for i in range(len(axis_names)):
        for j in range(i + 1, len(axis_names)):
            ai, aj = axis_names[i], axis_names[j]
            for vi in axes_ext[ai]:
                for vj in axes_ext[aj]:
                    pairs.add((ai, vi, aj, vj))
    return pairs


def _combo_pairs(combo: FuzzCombo) -> set[tuple[str, Any, str, Any]]:
    values = {
        "input_type": combo.input_type,
        "n_rows": combo.n_rows,
        "cat_feature_count": combo.cat_feature_count,
        "null_fraction_cats": combo.null_fraction_cats,
        "use_mrmr_fs": combo.use_mrmr_fs,
        "weight_schemas": combo.weight_schemas,
        "target_type": combo.target_type,
        "auto_detect_cats": combo.auto_detect_cats,
        "align_polars_categorical_dicts": combo.align_polars_categorical_dicts,
        "prefer_polarsds": combo.prefer_polarsds,
        "use_text_features": combo.use_text_features,
        "honor_user_dtype": combo.honor_user_dtype,
        "text_col_count": combo.text_col_count,
        "embedding_col_count": combo.embedding_col_count,
        # 2026-04-24 combo-extension axes
        "outlier_detection": combo.outlier_detection,
        "use_ensembles": combo.use_ensembles,
        "continue_on_model_failure": combo.continue_on_model_failure,
        "iterations": combo.iterations,
        "prefer_calibrated_classifiers": combo.prefer_calibrated_classifiers,
        "inject_degenerate_cols": combo.inject_degenerate_cols,
        "inject_inf_nan": combo.inject_inf_nan,
        "with_datetime_col": combo.with_datetime_col,
        "inject_zero_col": combo.inject_zero_col,
        "fairness_col": combo.fairness_col,
        "custom_prep": combo.custom_prep,
        "input_storage": combo.input_storage,
        # 2026-04-24 round 2
        "fillna_value_cfg": combo.fillna_value_cfg,
        "scaler_name_cfg": combo.scaler_name_cfg,
        "categorical_encoding_cfg": combo.categorical_encoding_cfg,
        "skip_categorical_encoding_cfg": combo.skip_categorical_encoding_cfg,
        "val_placement_cfg": combo.val_placement_cfg,
        "test_size_cfg": combo.test_size_cfg,
        "trainset_aging_limit_cfg": combo.trainset_aging_limit_cfg,
        "cat_text_card_threshold_cfg": combo.cat_text_card_threshold_cfg,
        "early_stopping_rounds_cfg": combo.early_stopping_rounds_cfg,
        "use_robust_eval_metric_cfg": combo.use_robust_eval_metric_cfg,
        # Fix G
        "inject_label_leak": combo.inject_label_leak,
        "inject_rank_deficient": combo.inject_rank_deficient,
        "inject_all_nan_col": combo.inject_all_nan_col,
        # R3
        "inject_test_drift": combo.inject_test_drift,
        "imbalance_ratio": combo.imbalance_ratio,
        "weird_cat_content": combo.weird_cat_content,
        # Phase H
        "multilabel_strategy_cfg": combo.multilabel_strategy_cfg,
        # 2026-04-26 batch 1
        "fix_infinities_cfg": combo.fix_infinities_cfg,
        "ensure_float32_cfg": combo.ensure_float32_cfg,
        "remove_constant_columns_cfg": combo.remove_constant_columns_cfg,
        "imputer_strategy_cfg": combo.imputer_strategy_cfg,
        "shuffle_val_cfg": combo.shuffle_val_cfg,
        "shuffle_test_cfg": combo.shuffle_test_cfg,
        "wholeday_splitting_cfg": combo.wholeday_splitting_cfg,
        "val_sequential_fraction_cfg": combo.val_sequential_fraction_cfg,
        # batch 3 — multilabel dispatch
        "multilabel_n_chains_cfg": combo.multilabel_n_chains_cfg,
        "multilabel_chain_order_cfg": combo.multilabel_chain_order_cfg,
        "multilabel_cv_cfg": combo.multilabel_cv_cfg,
        # batch 4 — PreprocessingExtensionsConfig
        "prep_ext_scaler_cfg": combo.prep_ext_scaler_cfg,
        "prep_ext_kbins_cfg": combo.prep_ext_kbins_cfg,
        "prep_ext_polynomial_degree_cfg": combo.prep_ext_polynomial_degree_cfg,
        "prep_ext_dim_reducer_cfg": combo.prep_ext_dim_reducer_cfg,
        "prep_ext_nonlinear_cfg": combo.prep_ext_nonlinear_cfg,
        "prep_ext_pysr_enabled_cfg": combo.prep_ext_pysr_enabled_cfg,
        "mrmr_nan_strategy_cfg": combo.mrmr_nan_strategy_cfg,
        # batch 5
        "rfecv_estimator_cfg": combo.rfecv_estimator_cfg,
        # batch 6
        "recurrent_model_cfg": combo.recurrent_model_cfg,
        # 2026-04-28 batch 4 followup
        "include_confidence_analysis_cfg": combo.include_confidence_analysis_cfg,
        "n_models": len(combo.models),
    }
    names = list(values.keys())
    out: set[tuple[str, Any, str, Any]] = set()
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            ai, aj = names[i], names[j]
            out.add((ai, values[ai], aj, values[aj]))
    return out


def enumerate_combos(
    target: int = 150,
    master_seed: int = 2026_04_22,
    model_universe: tuple[str, ...] = MODELS,
) -> list[FuzzCombo]:
    """Return `target` unique FuzzCombo instances covering the axis space.

    Phase A seeds with one combo per non-empty model-subset.
    Phase B greedy-fills until pairwise coverage is achieved.
    Phase C random-fills until len == target.
    """
    rng = random.Random(master_seed)
    seen: set[tuple] = set()
    combos: list[FuzzCombo] = []

    # Phase A — model subsets
    for subset in _powerset_nonempty(model_universe):
        axes = _sample_axes(rng)
        # iter373: skip unrunnable LTR-without-ranker combos before they hit
        # training and crash with NotImplementedError. Pull a fresh axes
        # sample only if the FIRST pick happened to pair this subset with
        # LTR; that way the enumerator doesn't deterministically lose its
        # LTR slot for hgb-only / linear-only model subsets.
        for _retry in range(20):
            if _combo_is_runnable(subset, axes["target_type"]):
                break
            axes = _sample_axes(rng)
        else:
            # 20 resamples couldn't land on a runnable target_type; skip.
            continue
        combo = _build_combo(subset, axes, len(combos))
        key = combo.canonical_key()
        if key in seen:
            continue
        seen.add(key)
        combos.append(combo)

    # Phase B — pairwise coverage
    required = _all_axis_pairs()
    covered: set[tuple[str, Any, str, Any]] = set()
    for c in combos:
        covered.update(_combo_pairs(c))

    tries = 0
    max_tries = 10_000
    while covered < required and tries < max_tries and len(combos) < target:
        uncovered = required - covered
        best_combo = None
        best_new = 0
        for _ in range(50):
            subset = rng.choice(_powerset_nonempty(model_universe))
            axes = _sample_axes(rng)
            if not _combo_is_runnable(subset, axes["target_type"]):
                continue
            candidate = _build_combo(subset, axes, len(combos))
            if candidate.canonical_key() in seen:
                continue
            cand_pairs = _combo_pairs(candidate)
            new = len(cand_pairs & uncovered)
            if new > best_new:
                best_new = new
                best_combo = candidate
        if best_combo is None:
            break
        seen.add(best_combo.canonical_key())
        combos.append(best_combo)
        covered.update(_combo_pairs(best_combo))
        tries += 1

    # Phase C — random fill until target
    while len(combos) < target:
        subset = rng.choice(_powerset_nonempty(model_universe))
        axes = _sample_axes(rng)
        if not _combo_is_runnable(subset, axes["target_type"]):
            continue
        candidate = _build_combo(subset, axes, len(combos))
        key = candidate.canonical_key()
        if key in seen:
            continue
        seen.add(key)
        combos.append(candidate)

    return combos[:target]


# ---------------------------------------------------------------------------
# Fix A — 3-wise covering over a curated subset of load-bearing axes
# ---------------------------------------------------------------------------

# Only the axes where 3-way interaction bugs have historically lived or
# are most plausible. Restricting the triple-space from the full 36
# axes to these 13 keeps the covering algorithm tractable (~286 axis-
# triples × ~12 value-triples = ~3.5k triples to cover) while still
# probing the interactions that matter. Expand cautiously — adding one
# axis bumps the triple count by ~C(N-1, 2) new axis-triples.
_3WAY_AXES: tuple[str, ...] = (
    "input_type",
    "n_rows",
    "cat_feature_count",
    "use_mrmr_fs",
    "target_type",
    "outlier_detection",
    "use_ensembles",
    "inject_inf_nan",
    "inject_degenerate_cols",
    "custom_prep",
    "categorical_encoding_cfg",
    "scaler_name_cfg",
    "inject_label_leak",
    "inject_rank_deficient",
    "inject_all_nan_col",
    # 2026-04-28 batch 4 followup - the confidence-analysis path interacts
    # with the test-set metrics, the calibration loop, and SHAP/permutation
    # paths; keep it triple-covered against the load-bearing axes that
    # historically interact with that side of the pipeline.
    "include_confidence_analysis_cfg",
)


def _all_axis_triples() -> set[tuple[str, Any, str, Any, str, Any]]:
    axes_ext: dict[str, tuple[Any, ...]] = {
        name: AXES[name] for name in _3WAY_AXES if name in AXES
    }
    axes_ext["n_models"] = (1, 2, 3, 4, 5)
    names = list(axes_ext.keys())
    out: set[tuple] = set()
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            for k in range(j + 1, len(names)):
                ai, aj, ak = names[i], names[j], names[k]
                for vi in axes_ext[ai]:
                    for vj in axes_ext[aj]:
                        for vk in axes_ext[ak]:
                            out.add((ai, vi, aj, vj, ak, vk))
    return out


def _combo_triples(combo: FuzzCombo) -> set[tuple[str, Any, str, Any, str, Any]]:
    values = {
        name: getattr(combo, name) for name in _3WAY_AXES if hasattr(combo, name)
    }
    values["n_models"] = len(combo.models)
    names = list(values.keys())
    out: set[tuple] = set()
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            for k in range(j + 1, len(names)):
                out.add(
                    (names[i], values[names[i]],
                     names[j], values[names[j]],
                     names[k], values[names[k]])
                )
    return out


def enumerate_combos_3way(
    target: int = 600,
    master_seed: int = 2026_04_24,
    model_universe: tuple[str, ...] = MODELS,
) -> list[FuzzCombo]:
    """Greedy 3-wise (triple) covering over ``_3WAY_AXES``.

    Same shape as ``enumerate_combos`` but optimises for triple-coverage
    instead of pair-coverage. Seeded separately (default ``2026_04_24``
    so the 3-wise suite doesn't stomp the pairwise seed's sample).
    """
    rng = random.Random(master_seed)
    seen: set[tuple] = set()
    combos: list[FuzzCombo] = []

    # Phase A — model subsets (iter373: skip LTR-without-ranker)
    for subset in _powerset_nonempty(model_universe):
        axes = _sample_axes(rng)
        for _retry in range(20):
            if _combo_is_runnable(subset, axes["target_type"]):
                break
            axes = _sample_axes(rng)
        else:
            continue
        combo = _build_combo(subset, axes, len(combos))
        key = combo.canonical_key()
        if key in seen:
            continue
        seen.add(key)
        combos.append(combo)

    # Phase B — greedy triple coverage
    required = _all_axis_triples()
    covered: set[tuple] = set()
    for c in combos:
        covered.update(_combo_triples(c))

    tries = 0
    max_tries = 40_000
    while covered < required and tries < max_tries and len(combos) < target:
        uncovered = required - covered
        best_combo = None
        best_new = 0
        for _ in range(80):
            subset = rng.choice(_powerset_nonempty(model_universe))
            axes = _sample_axes(rng)
            if not _combo_is_runnable(subset, axes["target_type"]):
                continue
            candidate = _build_combo(subset, axes, len(combos))
            if candidate.canonical_key() in seen:
                continue
            cand = _combo_triples(candidate)
            new = len(cand & uncovered)
            if new > best_new:
                best_new = new
                best_combo = candidate
        if best_combo is None:
            break
        seen.add(best_combo.canonical_key())
        combos.append(best_combo)
        covered.update(_combo_triples(best_combo))
        tries += 1

    # Phase C — random fill
    while len(combos) < target:
        subset = rng.choice(_powerset_nonempty(model_universe))
        axes = _sample_axes(rng)
        if not _combo_is_runnable(subset, axes["target_type"]):
            continue
        candidate = _build_combo(subset, axes, len(combos))
        key = candidate.canonical_key()
        if key in seen:
            continue
        seen.add(key)
        combos.append(candidate)

    return combos[:target]

