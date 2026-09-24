# Complexity refactor (C901, threshold 25)

Scope: every production function (``_benchmarks`` excluded) whose McCabe cyclomatic complexity exceeds **25**, measured
with ``ruff check src/mlframe --select C901`` on 2026-09-24. Queued to start after every other finding of this audit
wave is implemented.

## Why

``[tool.ruff.lint.mccabe] max-complexity = 40`` was configured but never enforced: both blocking CI ruff jobs and both
pre-commit ruff hooks pass ``--ignore C901``, and no advisory job runs it. The comment justifying 40 states that "median
complexity in this codebase is 30"; measured over all 11,028 production functions the median is 2 (p90 9, p95 13, p99
30), so that triage most likely looked only at already-flagged functions.

## Threshold

25, confirmed by the project owner. Below ~20 the check flags config-driven dispatch functions whose branching IS their
job (466 functions above 15); 25 sits just past the natural tail (p99 = 30), so everything above it is an outlier
(1.7% of functions); 40 lets through functions that cannot be read today (84 in an ~800-line function).

## Gate

Blocking from day one without breaking the current tree, the same ratchet as the function-length gate:
- a new function may not exceed 25;
- every function listed below is frozen at its current value in a baseline and may only go DOWN; it drops off the
  baseline once at or under 25;
- CI and pre-commit stop passing ``--ignore C901`` for this check; the misleading pyproject comment is corrected.

## Method

Top-down by complexity. Each function is split into phase helpers moved into a sibling module or a subdirectory package
(re-exported from the original location so importers are unaffected), with behaviour pinned before the split and
verified identical after it. Functions whose branching is inherent dispatch are split by dispatch target, not by
arbitrary line ranges.

## Summary

| Complexity | Functions |
|---|---|
| >100 | 6 |
| 76-100 | 20 |
| 51-75 | 33 |
| 41-50 | 27 |
| 26-40 | 107 |
| **Total** | **193** |

- **Disposition**: TODO - 193 functions queued; the gate is the first item.

## Findings

| ID | Complexity | Function | Location | Status |
|---|---|---|---|---|
| CX-GATE | - | blocking C901 ratchet at 25 + baseline | `tests/test_meta/test_function_complexity.py`, `_function_complexity_baseline.json` (193 entries), `py_ci_shared.function_complexity`; `pyproject.toml` max-complexity 25 with the measured distribution | DONE - the ratchet (ruff C901 numbers, path::qualname keys) is the blocking gate; the plain ruff steps keep `--ignore C901` since a baseline-less ruff run would fail on the 193 tracked functions |
| CX-001 | 317 | `_fit_impl` | `src/mlframe/feature_selection/filters/_mrmr_fit_impl/_fit_impl_core.py:123` | TODO |
| CX-002 | 147 | `materialise_and_finalise_fe_candidates` | `src/mlframe/feature_selection/filters/_mrmr_fe_step/_step_score.py:50` | TODO |
| CX-003 | 115 | `_score_one_pair` | `src/mlframe/feature_selection/filters/_feature_engineering_pairs/_pairs_score.py:79` | TODO |
| CX-004 | 107 | `train_mlframe_ranker_suite` | `src/mlframe/training/ranking/_ranker_suite_train.py:23` | TODO |
| CX-005 | 104 | `_fe_stage_cascade_mid_b` | `src/mlframe/feature_selection/filters/_mrmr_fit_impl/_fe_stage_cascade_mid_b.py:26` | TODO |
| CX-006 | 101 | `run_cat_interaction_step` | `src/mlframe/feature_selection/filters/_cat_interactions_step.py:107` | TODO |
| CX-007 | 98 | `_fe_stage_cascade_early_b` | `src/mlframe/feature_selection/filters/_mrmr_fit_impl/_fe_stage_cascade_early_b.py:28` | TODO |
| CX-008 | 97 | `_build_cross_target_ensemble_for_target` | `src/mlframe/training/core/_phase_composite_post_xt_ensemble/__init__.py:121` | TODO |
| CX-009 | 96 | `_fe_stage_cascade_mid_a` | `src/mlframe/feature_selection/filters/_mrmr_fit_impl/_fe_stage_cascade_mid_a.py:25` | TODO |
| CX-010 | 96 | `_auto_base` | `src/mlframe/training/composite/discovery/_auto_base.py:50` | TODO |
| CX-011 | 94 | `_assign_support_tail` | `src/mlframe/feature_selection/filters/_mrmr_fit_impl/_assign_support_tail.py:28` | TODO |
| CX-012 | 93 | `check_prospective_fe_pairs` | `src/mlframe/feature_selection/filters/_feature_engineering_pairs/_pairs_core.py:328` | TODO |
| CX-013 | 92 | `_fit_common` | `src/mlframe/training/neural/base/_base_fit.py:74` | TODO |
| CX-014 | 91 | `_phase_fit_pipeline` | `src/mlframe/training/core/_phase_helpers_fit_pipeline.py:109` | TODO |
| CX-015 | 89 | `optimise_hermite_pair` | `src/mlframe/feature_selection/filters/_hermite_fe_optimise_pair.py:68` | TODO |
| CX-016 | 88 | `_fit_body` | `src/mlframe/feature_selection/filters/mrmr/_mrmr_class.py:3379` | TODO |
| CX-017 | 88 | `configure_training_params` | `src/mlframe/training/_trainer_configure.py:215` | TODO |
| CX-018 | 85 | `train_and_evaluate_model` | `src/mlframe/training/_trainer_train_and_evaluate.py:104` | TODO |
| CX-019 | 85 | `make_train_test_split` | `src/mlframe/training/splitting.py:36` | TODO |
| CX-020 | 84 | `report_probabilistic_model_perf` | `src/mlframe/training/reporting/_reporting_probabilistic.py:127` | TODO |
| CX-021 | 83 | `_train_model_with_fallback_unguarded` | `src/mlframe/training/_training_loop.py:283` | TODO |
| CX-022 | 83 | `generate_mlp` | `src/mlframe/training/neural/flat.py:215` | TODO |
| CX-023 | 82 | `predict_from_models` | `src/mlframe/training/core/_predict_main_from_models.py:33` | TODO |
| CX-024 | 80 | `_train_one_target` | `src/mlframe/training/core/_phase_train_one_target_body.py:72` | TODO |
| CX-025 | 79 | `_sanitize_X_inputs` | `src/mlframe/feature_selection/wrappers/rfecv/_validate.py:38` | TODO |
| CX-026 | 76 | `_assign_support` | `src/mlframe/feature_selection/filters/_mrmr_fit_impl/_assign_support.py:85` | TODO |
| CX-027 | 75 | `fit` | `src/mlframe/feature_selection/shap_proxied_fs/_shap_proxied_fit.py:146` | TODO |
| CX-028 | 73 | `_tiny_model_rerank` | `src/mlframe/training/composite/discovery/_tiny_rerank.py:76` | TODO |
| CX-029 | 71 | `retain_usable_pure_forms` | `src/mlframe/feature_selection/filters/_fe_pure_form_retention.py:104` | TODO |
| CX-030 | 70 | `run_composite_target_discovery` | `src/mlframe/training/core/_phase_composite_discovery.py:63` | TODO |
| CX-031 | 67 | `_emit_pair_features` | `src/mlframe/feature_selection/filters/_feature_engineering_pairs/_pairs_emit.py:49` | TODO |
| CX-032 | 67 | `screen_predictors` | `src/mlframe/feature_selection/filters/_screen_predictors.py:75` | TODO |
| CX-033 | 66 | `_oof_holdout_predictions_with_rows` | `src/mlframe/training/composite/ensemble/__init__.py:397` | TODO |
| CX-034 | 64 | `predict_mlframe_models_suite` | `src/mlframe/training/core/_predict_main_suite.py:116` | TODO |
| CX-035 | 64 | `apply_preprocessing_extensions` | `src/mlframe/training/pipeline/_pipeline_extensions.py:298` | TODO |
| CX-036 | 63 | `compute_numerical_aggregates_numba` | `src/mlframe/feature_engineering/_numerical_numba.py:48` | TODO |
| CX-037 | 62 | `build_raw_redundancy_anchors` | `src/mlframe/feature_selection/filters/_fe_raw_redundancy_anchors.py:38` | TODO |
| CX-038 | 62 | `_friend_graph_and_redundancy_passes_group3` | `src/mlframe/feature_selection/filters/_mrmr_fit_impl/_friend_graph_and_redundancy/_group3.py:22` | TODO |
| CX-039 | 60 | `get_feature_importances` | `src/mlframe/feature_selection/wrappers/_helpers_importance.py:359` | TODO |
| CX-040 | 60 | `report_regression_model_perf` | `src/mlframe/training/reporting/_reporting_regression/__init__.py:54` | TODO |
| CX-041 | 59 | `drop_redundant_raw_operands` | `src/mlframe/feature_selection/filters/_fe_raw_redundancy_drop.py:181` | TODO |
| CX-042 | 59 | `_apply_pre_pipeline_transforms` | `src/mlframe/training/pipeline/_pipeline_helpers_apply.py:61` | TODO |
| CX-043 | 58 | `_make_compute_moments_slope_mi` | `src/mlframe/feature_engineering/_numerical_numba.py:458` | TODO |
| CX-044 | 58 | `usability_greedy_clf_gpu_resident` | `src/mlframe/feature_selection/filters/_usability_greedy_clf_gpu_resident.py:67` | TODO |
| CX-045 | 58 | `fit` | `src/mlframe/training/composite/discovery/_fit.py:273` | TODO |
| CX-046 | 57 | `kernel` | `src/mlframe/feature_engineering/_numerical_numba.py:471` | TODO |
| CX-047 | 57 | `_prewarm_numba_cache_body` | `src/mlframe/metrics/_core_numba_warmup.py:147` | TODO |
| CX-048 | 55 | `_friend_graph_and_redundancy_passes_group1` | `src/mlframe/feature_selection/filters/_mrmr_fit_impl/_friend_graph_and_redundancy/_group1.py:24` | TODO |
| CX-049 | 55 | `apply_recipe` | `src/mlframe/feature_selection/filters/engineered_recipes/_recipe_dispatch.py:26` | TODO |
| CX-050 | 54 | `run_polynom_pair_fe` | `src/mlframe/feature_selection/filters/polynom_pair_fe.py:58` | TODO |
| CX-051 | 54 | `_init_fit_state` | `src/mlframe/feature_selection/wrappers/rfecv/_fit_init.py:76` | TODO |
| CX-052 | 54 | `run_confidence_analysis` | `src/mlframe/training/_confidence_analysis.py:33` | TODO |
| CX-053 | 54 | `_phase_train_val_test_split` | `src/mlframe/training/core/_phase_helpers_fit_split.py:187` | TODO |
| CX-054 | 53 | `_friend_graph_and_redundancy_passes_group2` | `src/mlframe/feature_selection/filters/_mrmr_fit_impl/_friend_graph_and_redundancy/_group2.py:25` | TODO |
| CX-055 | 53 | `usability_greedy` | `src/mlframe/feature_selection/filters/_usability_aware_selection.py:491` | TODO |
| CX-056 | 52 | `apply_cmi_redundancy_gate` | `src/mlframe/feature_selection/filters/_fe_cmi_redundancy_gate.py:196` | TODO |
| CX-057 | 52 | `_auto_detect_feature_types` | `src/mlframe/training/core/_misc_helpers_feature_types.py:51` | TODO |
| CX-058 | 51 | `prepare_df_for_catboost` | `src/mlframe/preprocessing/transforms.py:32` | TODO |
| CX-059 | 51 | `_run_composite_target_wrapping` | `src/mlframe/training/core/_phase_composite_wrapping.py:359` | TODO |
| CX-060 | 50 | `greedy_cmi_fe_construct` | `src/mlframe/feature_selection/filters/_mi_greedy_cmi_fe.py:1418` | TODO |
| CX-061 | 49 | `usability_greedy_gpu_resident` | `src/mlframe/feature_selection/filters/_usability_greedy_gpu_resident.py:70` | TODO |
| CX-062 | 49 | `run_outer_loop_iteration` | `src/mlframe/feature_selection/wrappers/rfecv/_fit_outer_loop.py:73` | TODO |
| CX-063 | 49 | `_run_target_distribution_analyzer` | `src/mlframe/training/core/_main_train_suite_target_distribution.py:298` | TODO |
| CX-064 | 48 | `_run_fe_step_impl` | `src/mlframe/feature_selection/filters/_mrmr_fe_step/_step_core.py:144` | TODO |
| CX-065 | 48 | `_prewarm_fs_numba_cache_impl` | `src/mlframe/feature_selection/filters/_prewarm.py:41` | TODO |
| CX-066 | 48 | `_eval_fold_body` | `src/mlframe/feature_selection/wrappers/rfecv/_fit_fold.py:62` | TODO |
| CX-067 | 48 | `apply_polars_categorical_fixes` | `src/mlframe/training/core/_phase_polars_fixes.py:84` | TODO |
| CX-068 | 47 | `evaluate_swap_candidate` | `src/mlframe/feature_selection/filters/_dynamic_cluster_discovery/_dcd_swap.py:180` | TODO |
| CX-069 | 46 | `within_cluster_refine` | `src/mlframe/feature_selection/shap_proxied_fs/_shap_proxy_revalidate/_shap_proxy_refine.py:548` | TODO |
| CX-070 | 46 | `fit` | `src/mlframe/feature_selection/wrappers/rfecv/_fit.py:178` | TODO |
| CX-071 | 46 | `analyse_and_clean_features` | `src/mlframe/preprocessing/cleaning.py:505` | TODO |
| CX-072 | 45 | `per_feature_edges` | `src/mlframe/feature_selection/filters/_adaptive_nbins.py:604` | TODO |
| CX-073 | 45 | `run_fe_auto_escalation` | `src/mlframe/feature_selection/filters/_fe_auto_escalation.py:588` | TODO |
| CX-074 | 45 | `build_usability_candidate_pool` | `src/mlframe/feature_selection/filters/_usability_aware_selection.py:127` | TODO |
| CX-075 | 45 | `_scatter` | `src/mlframe/reporting/renderers/_plotly_scatter.py:36` | TODO |
| CX-076 | 45 | `get_training_configs` | `src/mlframe/training/_helpers_training_configs.py:57` | TODO |
| CX-077 | 44 | `compute_probabilistic_multiclass_error` | `src/mlframe/metrics/_ice_metric.py:70` | TODO |
| CX-078 | 44 | `fit` | `src/mlframe/training/lgb_shim.py:327` | TODO |
| CX-079 | 43 | `generate_adaptive_arity_cross_basis` | `src/mlframe/feature_selection/filters/_orthogonal_adaptive_arity_fe.py:129` | TODO |
| CX-080 | 43 | `evaluate_candidate` | `src/mlframe/feature_selection/filters/evaluation.py:434` | TODO |
| CX-081 | 43 | `_bootstrap_ci_for_strongest` | `src/mlframe/training/baselines/_dummy_bootstrap.py:354` | TODO |
| CX-082 | 43 | `_build_feature_selection_report` | `src/mlframe/training/core/_phase_train_one_target_helpers.py:21` | TODO |
| CX-083 | 42 | `__init__` | `src/mlframe/feature_selection/wrappers/rfecv/__init__.py:165` | TODO |
| CX-084 | 42 | `_render_post_fit_diagnostics` | `src/mlframe/training/reporting/_reporting_diagnostics.py:252` | TODO |
| CX-085 | 41 | `bootstrap_metrics` | `src/mlframe/evaluation/bootstrap.py:318` | TODO |
| CX-086 | 41 | `_phase_auto_detect_feature_types` | `src/mlframe/training/core/_phase_helpers_fit_split.py:586` | TODO |
| CX-087 | 40 | `_friend_graph_and_redundancy_passes_group4` | `src/mlframe/feature_selection/filters/_mrmr_fit_impl/_friend_graph_and_redundancy/_group4.py:19` | TODO |
| CX-088 | 40 | `is_variable_truly_continuous` | `src/mlframe/preprocessing/cleaning.py:190` | TODO |
| CX-089 | 40 | `forward_stepwise_multi_base` | `src/mlframe/training/composite/discovery/forward_stepwise.py:40` | TODO |
| CX-090 | 40 | `setup_configuration` | `src/mlframe/training/core/_phase_config_setup.py:120` | TODO |
| CX-091 | 39 | `_conditional_perm_null` | `src/mlframe/feature_selection/filters/_fe_cmi_redundancy_null.py:38` | TODO |
| CX-092 | 39 | `_fe_stage_cascade_early_a` | `src/mlframe/feature_selection/filters/_mrmr_fit_impl/_fe_stage_cascade_early_a.py:24` | TODO |
| CX-093 | 39 | `generate_extra_basis_features` | `src/mlframe/feature_selection/filters/_orthogonal_univariate_fe/_orth_extra_basis_fe_generate.py:52` | TODO |
| CX-094 | 39 | `compute_shap_matrix` | `src/mlframe/feature_selection/shap_proxied_fs/_shap_proxy_explain.py:603` | TODO |
| CX-095 | 39 | `show_calibration_plot` | `src/mlframe/metrics/calibration/_calibration_plot.py:551` | TODO |
| CX-096 | 39 | `fit_and_transform_pipeline` | `src/mlframe/training/pipeline/_pipeline_fit_transform.py:36` | TODO |
| CX-097 | 38 | `propose_additive_fusions` | `src/mlframe/feature_selection/filters/_fe_additive_fusion.py:105` | TODO |
| CX-098 | 38 | `_resample_metric` | `src/mlframe/training/baselines/_dummy_bootstrap.py:376` | TODO |
| CX-099 | 37 | `finalize_suite` | `src/mlframe/training/core/_phase_finalize.py:545` | TODO |
| CX-100 | 37 | `analyze_feature_distribution` | `src/mlframe/training/targets/_target_distribution_analyzer_features.py:267` | TODO |
| CX-101 | 36 | `create_aggregated_features` | `src/mlframe/feature_engineering/timeseries.py:189` | TODO |
| CX-102 | 36 | `_hybrid_orth_family_variants_group1` | `src/mlframe/feature_selection/filters/_mrmr_fit_impl/_hybrid_orth_family_variants/_group1.py:20` | TODO |
| CX-103 | 36 | `__init__` | `src/mlframe/models/_optimization_search.py:84` | TODO |
| CX-104 | 36 | `render_multi_target_panels` | `src/mlframe/reporting/auto_dispatch.py:142` | TODO |
| CX-105 | 36 | `pinned_train_val_test_split` | `src/mlframe/training/_fixed_splits.py:299` | TODO |
| CX-106 | 36 | `_tiny_cv_rmse_y_scale` | `src/mlframe/training/composite/discovery/_screening_tiny_perbin.py:195` | TODO |
| CX-107 | 36 | `run_dummy_baselines` | `src/mlframe/training/core/_phase_dummy_baselines.py:36` | TODO |
| CX-108 | 36 | `_phase_pandas_conversion_and_cat_prep` | `src/mlframe/training/core/_phase_helpers.py:434` | TODO |
| CX-109 | 36 | `_run_one_weight_iteration` | `src/mlframe/training/core/_phase_train_one_target_weight_iteration.py:51` | TODO |
| CX-110 | 35 | `evaluate_estimators` | `src/mlframe/evaluation/reports.py:135` | TODO |
| CX-111 | 35 | `_resolve_cv_and_val_cv` | `src/mlframe/feature_selection/wrappers/rfecv/_cv_setup.py:72` | TODO |
| CX-112 | 35 | `_process_single_ensemble_method` | `src/mlframe/models/ensembling/process_method.py:82` | TODO |
| CX-113 | 35 | `check_rules` | `src/mlframe/models/tuning_rules.py:83` | TODO |
| CX-114 | 35 | `_scatter` | `src/mlframe/reporting/renderers/_matplotlib_scatter.py:33` | TODO |
| CX-115 | 35 | `_iterative_stratification_njit` | `src/mlframe/training/_iterative_stratification_njit.py:29` | TODO |
| CX-116 | 35 | `_paired_bootstrap_vs_runner_up` | `src/mlframe/training/baselines/_dummy_bootstrap.py:44` | TODO |
| CX-117 | 35 | `get_pandas_view_of_polars_df` | `src/mlframe/training/utils.py:461` | TODO |
| CX-118 | 34 | `train_postcalibrators` | `src/mlframe/calibration/_post_train_calibrators.py:22` | TODO |
| CX-119 | 34 | `_finalise_empty_support_fallback` | `src/mlframe/feature_selection/filters/_mrmr_fit_impl/_finalise.py:38` | TODO |
| CX-120 | 34 | `_validate_inputs` | `src/mlframe/feature_selection/filters/_mrmr_validate_transform.py:136` | TODO |
| CX-121 | 34 | `categorize_dataset` | `src/mlframe/feature_selection/filters/discretization/_discretization_dataset.py:230` | TODO |
| CX-122 | 34 | `save_mlframe_model` | `src/mlframe/training/_io_save.py:107` | TODO |
| CX-123 | 34 | `_compute_metrics_table` | `src/mlframe/training/baselines/_dummy_metrics_pick_plot.py:54` | TODO |
| CX-124 | 34 | `format_suite_end_summary` | `src/mlframe/training/baselines/_dummy_summary_format.py:23` | TODO |
| CX-125 | 34 | `fit` | `src/mlframe/training/composite/estimator/_estimator.py:464` | TODO |
| CX-126 | 34 | `_apply_loss_recommendation_in_place` | `src/mlframe/training/core/_phase_train_one_target.py:80` | TODO |
| CX-127 | 34 | `select_target` | `src/mlframe/training/targets/_train_eval_select_target.py:46` | TODO |
| CX-128 | 33 | `binned_numeric_agg_with_recipes` | `src/mlframe/feature_selection/filters/_binned_numeric_agg_fe.py:708` | TODO |
| CX-129 | 33 | `_hybrid_orth_family_variants_group2` | `src/mlframe/feature_selection/filters/_mrmr_fit_impl/_hybrid_orth_family_variants/_group2.py:20` | TODO |
| CX-130 | 33 | `train_recurrent_models` | `src/mlframe/training/core/_phase_recurrent.py:433` | TODO |
| CX-131 | 33 | `_build_pre_pipelines` | `src/mlframe/training/core/_setup_helpers_pre_pipelines.py:57` | TODO |
| CX-132 | 32 | `generate_modelling_data` | `src/mlframe/data/synthetic.py:141` | TODO |
| CX-133 | 32 | `_batch_per_class_ice_kernel` | `src/mlframe/metrics/classification/_ice_kernel.py:34` | TODO |
| CX-134 | 32 | `_maybe_get_or_build_cb_pool` | `src/mlframe/training/cb/_cb_pool_build.py:31` | TODO |
| CX-135 | 32 | `_run_suite_end_dummy_baselines_summary` | `src/mlframe/training/core/_phase_composite_post_summary.py:20` | TODO |
| CX-136 | 32 | `build_targets` | `src/mlframe/training/extractors/_extractors_simple.py:161` | TODO |
| CX-137 | 32 | `create_polarsds_pipeline` | `src/mlframe/training/pipeline/__init__.py:585` | TODO |
| CX-138 | 31 | `confirm_one_predictor` | `src/mlframe/feature_selection/filters/_confirm_predictor.py:658` | TODO |
| CX-139 | 31 | `discover_cluster_members` | `src/mlframe/feature_selection/filters/_dynamic_cluster_discovery/__init__.py:535` | TODO |
| CX-140 | 31 | `_hybrid_orth_family_variants_group3` | `src/mlframe/feature_selection/filters/_mrmr_fit_impl/_hybrid_orth_family_variants/_group3.py:19` | TODO |
| CX-141 | 31 | `_detect_fourier_freqs_for_col` | `src/mlframe/feature_selection/filters/_orthogonal_univariate_fe/_orth_extra_basis_fe.py:536` | TODO |
| CX-142 | 31 | `dispatch_batch_pair_mi` | `src/mlframe/feature_selection/filters/batch_pair_mi_gpu.py:378` | TODO |
| CX-143 | 31 | `revalidate_top_n` | `src/mlframe/feature_selection/shap_proxied_fs/_shap_proxy_revalidate/_shap_proxy_refine.py:101` | TODO |
| CX-144 | 31 | `select_optimal_nfeatures_` | `src/mlframe/feature_selection/wrappers/rfecv/_stability_select.py:339` | TODO |
| CX-145 | 31 | `_batch_per_class_ice_kernel_serial` | `src/mlframe/metrics/classification/_ice_kernel.py:286` | TODO |
| CX-146 | 31 | `combine_probs` | `src/mlframe/models/ensembling/base.py:585` | TODO |
| CX-147 | 30 | `propose_additive_fusions_gpu` | `src/mlframe/feature_selection/filters/_fe_additive_fusion_gpu_resident.py:104` | TODO |
| CX-148 | 30 | `sufficient_summary_reached` | `src/mlframe/feature_selection/filters/_fe_sufficient_summary.py:189` | TODO |
| CX-149 | 30 | `generate_conditional_basis_routing_features` | `src/mlframe/feature_selection/filters/_orthogonal_routing_fe.py:160` | TODO |
| CX-150 | 30 | `mi_direct` | `src/mlframe/feature_selection/filters/permutation.py:570` | TODO |
| CX-151 | 30 | `compute_unsupervised_drops` | `src/mlframe/feature_selection/pre_screen.py:44` | TODO |
| CX-152 | 30 | `_finalize_fit_results` | `src/mlframe/feature_selection/wrappers/rfecv/_finalize.py:28` | TODO |
| CX-153 | 30 | `_tiny_cv_rmse_raw_y` | `src/mlframe/training/composite/discovery/_screening_tiny.py:261` | TODO |
| CX-154 | 30 | `_validate_input_columns_against_metadata` | `src/mlframe/training/core/_misc_helpers.py:324` | TODO |
| CX-155 | 30 | `_select_scalable_numeric_columns` | `src/mlframe/training/pipeline/__init__.py:403` | TODO |
| CX-156 | 29 | `score_prospective_pairs` | `src/mlframe/feature_selection/filters/_mrmr_fe_step/_step_pairs_rank.py:475` | TODO |
| CX-157 | 29 | `_finalise_fs_results` | `src/mlframe/feature_selection/filters/_mrmr_fit_impl/_finalise.py:349` | TODO |
| CX-158 | 29 | `batch_mi_with_noise_gate_cuda_resident` | `src/mlframe/feature_selection/filters/batch_mi_noise_gate_gpu.py:477` | TODO |
| CX-159 | 29 | `estimate_features_relevancy` | `src/mlframe/feature_selection/general.py:106` | TODO |
| CX-160 | 29 | `suggest_candidate` | `src/mlframe/models/_optimization_search.py:346` | TODO |
| CX-161 | 29 | `get_model_feature_importances` | `src/mlframe/training/_feature_importances.py:515` | TODO |
| CX-162 | 29 | `_apply_pre_pipeline_with_passthrough` | `src/mlframe/training/core/_predict_pre_pipeline.py:454` | TODO |
| CX-163 | 29 | `_predict_raw` | `src/mlframe/training/neural/base/_base_predict.py:36` | TODO |
| CX-164 | 29 | `_passthrough_cols_fit_transform` | `src/mlframe/training/pipeline/_pipeline_helpers.py:484` | TODO |
| CX-165 | 28 | `pick_best_calibrator` | `src/mlframe/calibration/policy.py:676` | TODO |
| CX-166 | 28 | `cheap_conditional_gate_scan` | `src/mlframe/feature_selection/filters/_conditional_gate_fe.py:604` | TODO |
| CX-167 | 28 | `_build_operand_table` | `src/mlframe/feature_selection/filters/_feature_engineering_pairs/_pairs_setup.py:313` | TODO |
| CX-168 | 28 | `compute_pair_mis_and_floor` | `src/mlframe/feature_selection/filters/_mrmr_fe_step/_step_pairmi.py:106` | TODO |
| CX-169 | 28 | `scan_engineered_duplicates` | `src/mlframe/feature_selection/filters/_mrmr_fit_impl/_eng_dedup_scan.py:106` | TODO |
| CX-170 | 28 | `_align_xgb_cat_categories` | `src/mlframe/training/_eval_helpers.py:39` | TODO |
| CX-171 | 28 | `_maybe_refit_on_collapsed_predictions` | `src/mlframe/training/_training_loop_refit.py:339` | TODO |
| CX-172 | 28 | `_maybe_auto_drop_after_feature_analyzer` | `src/mlframe/training/core/_main_train_suite_target_distribution.py:81` | TODO |
| CX-173 | 28 | `_apply_outlier_detection_global` | `src/mlframe/training/core/_setup_helpers_outliers.py:82` | TODO |
| CX-174 | 28 | `compute_label_distribution_drift` | `src/mlframe/training/drift_report.py:207` | TODO |
| CX-175 | 28 | `process_model` | `src/mlframe/training/train_eval.py:337` | TODO |
| CX-176 | 27 | `generate_diff_basis_features` | `src/mlframe/feature_selection/filters/_orthogonal_diff_basis_fe.py:194` | TODO |
| CX-177 | 27 | `build_friend_graph` | `src/mlframe/feature_selection/filters/friend_graph.py:329` | TODO |
| CX-178 | 27 | `find_weak_slices` | `src/mlframe/reporting/charts/slice_finder.py:271` | TODO |
| CX-179 | 27 | `_patch_dataset_constructors_with_logging` | `src/mlframe/training/_model_factories.py:155` | TODO |
| CX-180 | 27 | `showcase_features_and_targets` | `src/mlframe/training/extractors/_extractors_showcase.py:26` | TODO |
| CX-181 | 26 | `_shannon_entropy_binned_kernel` | `src/mlframe/feature_engineering/windowed_shape.py:183` | TODO |
| CX-182 | 26 | `confirm_recipes_cross_fold` | `src/mlframe/feature_selection/filters/_fe_stability_vote.py:148` | TODO |
| CX-183 | 26 | `_eval_coef_pair_batch` | `src/mlframe/feature_selection/filters/_hermite_fe_optimise.py:189` | TODO |
| CX-184 | 26 | `evaluate_gain` | `src/mlframe/feature_selection/filters/evaluation.py:142` | TODO |
| CX-185 | 26 | `apply_gpu_unary_batched` | `src/mlframe/feature_selection/filters/feature_engineering.py:294` | TODO |
| CX-186 | 26 | `_line` | `src/mlframe/reporting/renderers/_plotly_line.py:46` | TODO |
| CX-187 | 26 | `render` | `src/mlframe/reporting/renderers/plotly.py:267` | TODO |
| CX-188 | 26 | `_process_special_values` | `src/mlframe/training/_nan_processing.py:24` | TODO |
| CX-189 | 26 | `compute_dummy_baselines` | `src/mlframe/training/baselines/dummy.py:324` | TODO |
| CX-190 | 26 | `_eval_one_transform_impl` | `src/mlframe/training/composite/discovery/_eval.py:294` | TODO |
| CX-191 | 26 | `run_composite_moe_and_value_report` | `src/mlframe/training/core/_phase_composite_post_moe.py:146` | TODO |
| CX-192 | 26 | `feature_handling_apply` | `src/mlframe/training/feature_handling/apply.py:106` | TODO |
| CX-193 | 26 | `analyze_target_distribution` | `src/mlframe/training/targets/_target_distribution_analyzer_target_fn.py:77` | TODO |

## Dispositions

| ID | Status | Note |
|---|---|---|
