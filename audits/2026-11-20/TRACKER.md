# Audit 2026-11-20 - tracker

Scope: all src/mlframe except feature_selection/filters/** (MRMR engine). 42 clusters, ~505k LOC.

Status legend: AUDITED = report written, findings not yet verified or dispositioned; VERIFIED = cited lines read by parent; RESOLVED / WON'T FIX / DEFERRED = disposition written in the report.

| # | Report | P0 | P1 | P2 | P3 | Total | Coverage | Status |
|---|---|---|---|---|---|---|---|---|
| 01 | 01_calibration.md | 0 | 1 | 5 | 7 | 13 | full | AUDITED |
| 02 | 02_competition.md | 0 | 0 | 3 | 3 | 6 | full | AUDITED |
| 03 | 03_core.md | 0 | 0 | 3 | 3 | 6 | full | AUDITED |
| 04 | 04_data.md | 0 | 13 | 13 | 8 | 34 | full | AUDITED |
| 05 | 05_data_valuation.md | 0 | 0 | 2 | 4 | 6 | full | AUDITED |
| 06 | 06_estimators.md | 0 | 0 | 2 | 3 | 5 | full | AUDITED |
| 07 | 07_evaluation.md | 0 | 0 | 6 | 8 | 14 | full | AUDITED |
| 08 | 08_feature_engineering-1.md | 0 | 2 | 15 | 4 | 21 | partial | AUDITED |
| 09 | 09_feature_engineering-2.md | 3 | 10 | 18 | 14 | 45 | full | AUDITED |
| 10 | 10_feature_selection-1.md | 0 | 3 | 3 | 3 | 9 | full | AUDITED |
| 11 | 11_feature_selection-2.md | 0 | 3 | 10 | 7 | 20 | partial | AUDITED |
| 12 | 12_feature_selection-3.md | 0 | 22 | 57 | 22 | 101 | partial | AUDITED |
| 13 | 13_feature_selection-4.md | 0 | 2 | 12 | 22 | 36 | partial | AUDITED |
| 14 | 14_feature_selection-5.md | 0 | 2 | 5 | 6 | 13 | partial | AUDITED |
| 15 | 15_feature_selection-6.md | 0 | 2 | 9 | 8 | 19 | partial | AUDITED |
| 16 | 16_inference.md | 0 | 0 | 3 | 8 | 11 | full | AUDITED |
| 17 | 17_metrics.md | 0 | 9 | 11 | 11 | 31 | full | AUDITED |
| 18 | 18_models.md | 0 | 5 | 9 | 6 | 20 | full | AUDITED |
| 19 | 19_preprocessing.md | 0 | 0 | 3 | 6 | 9 | full | AUDITED |
| 20 | 20_reporting-1.md | 0 | 6 | 21 | 18 | 45 | full | AUDITED |
| 21 | 21_reporting-2.md | 0 | 4 | 12 | 12 | 28 | full | AUDITED |
| 22 | 22_root_modules-1.md | 0 | 8 | 18 | 19 | 45 | full | AUDITED |
| 23 | 23_root_modules-2.md | 0 | 4 | 4 | 6 | 14 | partial | AUDITED |
| 24 | 24_root_modules-3.md | 1 | 21 | 28 | 16 | 66 | full | AUDITED |
| 25 | 25_root_modules-4.md | 0 | 19 | 35 | 17 | 71 | full | AUDITED |
| 26 | 26_root_modules-5.md | 0 | 5 | 16 | 17 | 38 | full | AUDITED |
| 27 | 27_root_modules-6.md | 0 | 6 | 10 | 10 | 26 | full | AUDITED |
| 28 | 28_root_modules-7.md | 0 | 18 | 32 | 26 | 76 | full | AUDITED |
| 29 | 29_root_modules-8.md | 0 | 6 | 20 | 15 | 41 | full | AUDITED |
| 30 | 30_signal.md | 0 | 0 | 3 | 3 | 6 | full | AUDITED |
| 31 | 31_system.md | 0 | 3 | 6 | 4 | 13 | full | AUDITED |
| 32 | 32_training-1.md | 0 | 6 | 19 | 3 | 28 | full | AUDITED |
| 33 | 33_training-2.md | 0 | 2 | 1 | 1 | 4 | partial | AUDITED |
| 34 | 34_training-3.md | 0 | 4 | 14 | 17 | 35 | full | AUDITED |
| 35 | 35_training-4.md | 2 | 12 | 29 | 14 | 57 | full | AUDITED |
| 36 | 36_training-5.md | 0 | 4 | 4 | 1 | 9 | partial | AUDITED |
| 37 | 37_training-6.md | 0 | 8 | 21 | 23 | 52 | full | AUDITED |
| 38 | 38_training-7.md | 0 | 3 | 12 | 12 | 27 | partial | AUDITED |
| 39 | 39_training-8.md | 0 | 20 | 26 | 26 | 72 | full | AUDITED |
| 40 | 40_training-9.md | 2 | 17 | 12 | 3 | 34 | full | AUDITED |
| 41 | 41_training-10.md | 0 | 6 | 21 | 17 | 44 | full | AUDITED |
| 42 | 42_votenrank.md | 0 | 2 | 10 | 3 | 15 | full | AUDITED |
| | **Total** | **8** | **258** | **563** | **436** | **1265** | | |

## P0 findings: parent verification

Each P0 was checked by reading the cited code. Disposition is the closing state.

| ID | Report | Disposition | Note |
|---|---|---|---|
| 09-F1 | 09_feature_engineering-2.md | RESOLVED | Inputs scaled by d^{-1/4} with unscaled W, so the kernel is exp(q.k/sqrt(d)) as documented. Tests test_performer_kernel_identity.py. |
| 09-F2 | 09_feature_engineering-2.md | RESOLVED | Double d^-1/2 removed at both call sites; test test_nystrom_scales_each_neighbour_once.py. Commit 4863686a5. |
| 09-F3 | 09_feature_engineering-2.md | RESOLVED | Documented as the label-free proxy |p - median(y)| on both sides (no query label exists). Behaviour unchanged. Commit 8dc4360c0. |
| 24-F1 | 24_root_modules-3.md | RESOLVED | Per-column finite-row AUC when a cell is non-finite; single-class columns are kept. Test test_drop_near_noise_handles_nan.py. Commit 2171ff34e. Siblings drop_raw_after_embedding.py:53,142 still open. |
| 35-F1 | 35_training-4.md | RESOLVED | from_fitted_inner stamps smearing_quantiles from train features (X_train) and reconstructed T, as fit() does. Parity tests test_smearing_fit_from_fitted_parity.py. |
| 35-F2 | 35_training-4.md | RESOLVED | CRPS weights are cell widths (helper _crps_cell_weights); reduces to 2/K on midpoint grids. Also fixed sibling in qrf.py (new). Tests test_crps_cell_weights.py. Commit bd2709ef9. |
| 40-F1 | 40_training-9.md | RESOLVED | save/load store classes and the multilabel flag; model rebuilt with task_type. Tests test_recurrent_classifier_save_keeps_classes_and_multilabel.py. Commit 7e0cd9b40. |
| 40-F2 | 40_training-9.md | RESOLVED | Unweighted score() returns accuracy in both branches. Tests test_classifier_score_is_accuracy.py. Commit 2798b5a6c. |

## Dispositions after the P0 pass

| ID | Report | Disposition | Note |
|---|---|---|---|
| 01-F3 | 01_calibration.md | WON'T FIX | Rejected by the in-code bench (probabilities.py:292-298): the argsort variant is 0.43x at n=1e6. Not ported. |
| 35-F10 | 35_training-4.md | FIXED | Trim branch carries kept _component_col_means and logs the meta-model drop. Test test_cluster35_cap_keeps_component_means.py. |
| 35-F12 | 35_training-4.md | FIXED | Point-mass match uses rtol=0. Test test_cluster35_point_mass_tolerance.py. |
| 35-F13 | 35_training-4.md | FIXED | Per-fold catch removed; a failing fold raises. Test test_cluster35_group_oof_no_silent_fold_nan.py. |
| 35-F14 | 35_training-4.md | FIXED | predict_cdf sorts rows before the search. Test test_cluster35_cdf_unsorted_quantile_rows.py. |
| 35-F3 | 35_training-4.md | FIXED | Constant or all-zero base set the early time-aware flag. Base must vary. Test test_cluster35_digest_and_flags.py. |
| 35-F4 | 35_training-4.md | FIXED | Non-finite raw baseline silently disabled the gate. Fails closed with WARNING. Test test_cluster35_raw_baseline_fail_closed.py. |
| 35-F6 | 35_training-4.md | FIXED | Target float32 before the residual. float64 at both entry points. Test test_cluster35_stacked_target_float64_guard.py. |
| 35-F8 | 35_training-4.md | FIXED | Digest of U-dtype strings joined without boundaries. Length-prefixed. Test test_cluster35_digest_and_flags.py. |
| 35-F15 | 35_training-4.md | FIXED | _kwargs unbound on signature failure; weighted-fit fallback logged at DEBUG. Test test_cluster35_sample_weight_fallback.py. |
| 35-F16 | 35_training-4.md | FIXED | Group-holdout counts were O(groups x rows). One np.unique pass. Test test_cluster35_group_counts_one_pass.py. |
| 35-F17 | 35_training-4.md | FIXED | predict blended non-finite component outputs with no warning. Now a WARNING names the component. Test test_cluster35_predict_nonfinite_component_warns.py. |
| 35-F18 | 35_training-4.md | FIXED | GPU probe failures and the refit-groups fallback logged at DEBUG. Now WARNING. Test test_cluster35_gpu_probe_failure_warns.py (refit path untested). |
| 35-F19 | 35_training-4.md | DEFERRED | Composite refit implemented three times with different group handling. Consolidation changes behaviour. Owner decision on group semantics first. |
| 35-F20 | 35_training-4.md | FIXED | GBM stacker ignored group_ids and time_ordered; row-random early stopping leaked across groups. Early stopping off when either is set; dispatcher forwards both. Test test_cluster35_gbm_stacker_group_aware.py. |
| 35-F21 | 35_training-4.md | FIXED | OOF cache key omitted sample_weight and component specs. Both keyed with full digests. Test test_cluster35_oof_cache_key_weights_and_specs.py. |
| 35-F22 | 35_training-4.md | FIXED | Auto gate tested prediction correlation; protocol defines residual correlation. Now residuals. Test test_cluster35_auto_gate_measures_residuals.py. |
| 35-F23 | 35_training-4.md | FIXED | Degenerate fit permanently switched the calibrator to linear. Method restored per fit. Test test_cluster35_calibrator_method_restored_per_fit.py. |
| 35-F24 | 35_training-4.md | FIXED | Lag extraction failure logged at DEBUG; deep-OOD rows silently took the median. Now WARNING. Test test_cluster35_lag_extraction_failure_warns.py. |
| 35-F25 | 35_training-4.md | FIXED | Stamped base-fit range overwritten by from_fitted, moving soft-shrink thresholds. Stamped range kept. Test test_cluster35_from_fitted_keeps_stamped_range.py. |
| 35-F26 | 35_training-4.md | PARTIAL | Discovery smearing used unweighted quantiles over all rows. Now shares residual_quantiles with fit. Discovery callers do not pass sample weights yet. Test test_cluster35_smeared_prediction_matches_fit.py. |
| 35-F27 | 35_training-4.md | FIXED | predict_quantile re-resolved the causal lag once per quantile column. Resolved once per batch. Test test_cluster35_smart_fallback_resolves_lag_once.py. |
| 35-F28 | 35_training-4.md | DEFERRED | Distributional grid 0.05..0.95 truncates sample() tails and omits tail mass in crps(). Adding tail levels or a GPD tail is a modelling decision for the owner. |
| 35-F29 | 35_training-4.md | FIXED | TypeError fallback dropped sample_weight for any TypeError. Support read from signature; other TypeErrors propagate. Test test_cluster35_mixture_sample_weight_support.py. |
| 35-F30 | 35_training-4.md | FIXED | Dict config seed ignored; report recorded 42. Coerced before the seed is read. Test test_cluster35_highlevel_dict_config_seed.py. |
| 35-F31 | 35_training-4.md | FIXED | Group broadcast was a per-row dict loop; NaN group ids fell back silently. Vectorised reindex; NaN rows counted and logged. Test test_cluster35_group_broadcast_vectorised.py. |
| 35-F32 | 35_training-4.md | FIXED | Single numeric column raised ValueError in linkage. Returns that column as one subset. Test test_cluster35_subset_bagging_single_column.py. |
| 35-F33 | 35_training-4.md | PARTIAL | Tail fitted on in-sample residuals by default, under-dispersing held-out tails. WARNING added when residual_X is absent. Default unchanged (owner decision). Test test_cluster35_tail_in_sample_residuals_warn.py. |
| 35-F34 | 35_training-4.md | FIXED | Worst-candidate selection filled NaN from all fit rows (judge fills from the used rows) and max() depended on NaN order. Judge's fill, NaN as worst. Test test_cluster35_worst_candidate_nan_order.py. |
| 35-F35 | 35_training-4.md | PARTIAL | Spec with under 50 in-domain fit rows passed the y-scale gate unchecked. Now rejected with a ledger entry. No direct test (needs a full gate fixture). |
| 35-F36 | 35_training-4.md | PARTIAL | Multi-base specs bypass the structural fragility gate silently. Bypass now logged at WARNING; gating them is an owner decision. Test test_cluster35_structural_gate_multibase_logged.py. |
| 35-F37 | 35_training-4.md | FIXED | WAIC scoring failures swallowed at DEBUG; spec silently skipped in tie-break. Now WARNING naming spec and exception. Test test_cluster35_waic_failure_logged.py. |
| 35-F38 | 35_training-4.md | FIXED | kNN MI loop duplicated in _mi_to_target and _mi_per_feature_knn. One copy now. Test test_cluster35_knn_mi_single_loop.py. |
| 35-F39 | 35_training-4.md | FIXED | fit_stacked and fit_stacked_on_residual repeated ranking and group derivation. Shared helpers _rank_pass1_specs and _train_group_ids. Test test_cluster35_stacked_shared_helpers.py. |
| 35-F40 | 35_training-4.md | FIXED | Polars int-as-cat candidate logic duplicated; the two copies differed on temporal columns. One helper. Test test_cluster35_polars_candidates_one_helper.py. |
| 35-F41 | 35_training-4.md | FIXED | bayesian_alpha_fit let a NaN pair reach the regression and returned a NaN posterior silently. Pairs dropped, counted and logged. Test test_cluster35_bayesian_nonfinite_dropped.py. |
| 35-F42 | 35_training-4.md | DEFERRED | A broken process pool reruns every task serially. Documented as deliberate; correctness unaffected. Partial retrieval is a design change, owner decision. |
| 35-F43 | 35_training-4.md | FIXED | Three paths had no test: CDF with enforce_non_crossing=False, mixture soft_routing, calibrated gated outlier. Smoke tests added. Test test_cluster35_untested_paths_smoke.py. |
| 35-F44 | 35_training-4.md | PARTIAL | Repeated integer check at the grouping call site removed. Dead helpers kept as allowlisted test references. Test test_cluster35_low_card_check_once.py. |
| 35-F45 | 35_training-4.md | FIXED | Bootstrap posterior result lacked posterior_kind, n_samples, degrees_of_freedom and sigma2_posterior_mean. Added on both variants. Test test_cluster35_bootstrap_result_keys.py. |
| 35-F46 | 35_training-4.md | PARTIAL | _residualise used an absolute 1e-12 constant test. Now relative. Mid-module imports left in place. Test test_cluster35_residualise_relative_constant.py. |
| 35-F47 | 35_training-4.md | FIXED | Dead _groups_screen assignment and a missing blank line in _tiny_rerank.py. The None now goes straight into the call. Test: 231 existing rerank tests pass. |
| 35-F48 | 35_training-4.md | FIXED | Ridge import unused; __all__ listed typing names and omitted public functions. Import removed, __all__ rebuilt, cache-key block moved to _oof_cache_keys.py (facade 955 lines, under the 1k gate). Test test_cluster35_ensemble_public_surface.py. |
| 35-F49 | 35_training-4.md | FIXED | Uncached transform-pair path dropped refit_supervised, so a supervised selector was not refit on the fold. Forwarded. Test test_cluster35_uncached_pair_forwards_refit.py. |
| 35-F50 | 35_training-4.md | FIXED | Write-only _sig_degenerate attribute. Deleted. Existing 27 calibration tests pass; no separate test (removal). |
| 35-F51 | 35_training-4.md | FIXED | n_oof_rows counted rows finite in y only; the fit uses rows finite in y and every OOF column. Count now matches the fit. Test test_cluster35_meta_stack_row_count.py. |
| 35-F11 | 35_training-4.md | FIXED | Early logratio raise removed; base<=0 rows go to the fallback. Test test_cluster35_quantile_logratio_nonpositive_base.py. |
| 35-F7 | 35_training-4.md | FIXED | time_aware with groups raises. Test test_cluster35_stacking_time_and_groups_refused.py. |
| 35-F9 | 35_training-4.md | FIXED | External holdout digest covers frame content. Test test_cluster35_holdout_digest_distinguishes_content.py. |
| 12-F38 | 12_feature_selection-3.md | FIXED | downstream returns NaN for an empty selection. Test test_fs_hybrid_empty_selection_downstream.py. |
| 12-F31 | 12_feature_selection-3.md | PARTIAL | Two named benches narrowed; 61 other fs_hybrid benches still use a catch-all ignore (open). Test test_fs_hybrid_warning_filter_keeps_degraded_warning.py. |
| 12-F20 | 12_feature_selection-3.md | FIXED | Empty RFECV selection recorded as empty via score_selection, not scored. Test test_fs_hybrid_empty_rfecv_selection_not_scored.py. |
| 35-F5 | 35_training-4.md | FIXED | Every fold keeps a slot (NaN when unusable). Test test_cluster35_stepwise_fold_slots.py. |
| 12-F26 | 12_feature_selection-3.md | FIXED | Flip needs a sign majority and a positive mean delta (decide_flip). Test test_fs_hybrid_flip_needs_positive_mean.py. |
| 12-F27 | 12_feature_selection-3.md | FIXED | Proxy Brier computed out-of-fold. Test test_fs_hybrid_s2_proxy_is_out_of_fold.py. |
| 12-F28 | 12_feature_selection-3.md | FIXED | Sizes with a failed cell get no family mean. Test test_fs_hybrid_family_curve_failed_cells.py. |
| 12-F29 | 12_feature_selection-3.md | FIXED | Designed verdict uses the pre-specified operator, not the max over test AUC. Test test_fs_hybrid_designed_verdict_uses_prespecified_op.py. |
| 12-F30 | 12_feature_selection-3.md | FIXED | run_bed uses a fixed-seed split shared by all presets. Test test_fs_hybrid_cost_split_is_fixed.py. |
| 12-F32 | 12_feature_selection-3.md | FIXED | Failures at WARNING; per-config completed/failed counts printed with the means. Test test_fs_hybrid_shap_run_counts.py. |
| 12-F33 | 12_feature_selection-3.md | FIXED | Flip verdict on paired deltas over 5 splits with a 95% bound (flip_verdict). Test test_fs_hybrid_flip_verdict_paired.py. |
| 12-F34 | 12_feature_selection-3.md | DEFERRED | H35/H32 pick configs on rows their members were fit on. Fix needs an inner-split refit and held-out feature transform the bench's HybridSelector lacks. Benchmark-only. |
| 12-F35 | 12_feature_selection-3.md | FIXED | Five RFECV benches budget by refits only. Test test_fs_hybrid_rfecv_benches_have_no_wall_clock_cap.py. |
| 12-F36 | 12_feature_selection-3.md | FIXED | Configs with failed runs INCOMPLETE and excluded from ranking. Test test_fs_hybrid_ablation_failed_runs_excluded.py. |
| 12-F37 | 12_feature_selection-3.md | FIXED | Failed runs excluded from kept-share sizes. Test test_fs_hybrid_ablation_failed_runs_excluded.py. |
| 12-F39 | 12_feature_selection-3.md | FIXED | Failures at WARNING; completed/failed counts printed with the means. Test test_fs_hybrid_run_counts_by_key.py. |
| 12-F23 | 12_feature_selection-3.md | FIXED (test pending) | Baseline and relay use h.transform output, keeping engineered columns. No dedicated test yet (needs a full HybridSelector fit). |
| 12-F25 | 12_feature_selection-3.md | FIXED | Refit-only budget; test extended in test_fs_hybrid_rfecv_benches_have_no_wall_clock_cap.py. |
| 39-F2 | 39_training-8.md | FIXED | Batch alignment refused. Test test_cluster39_run_batched_alignment.py. |
| 39-F3 | 39_training-8.md | FIXED | Failed concatenate raises. Test test_cluster39_run_batched_alignment.py. |
| 39-F4 | 39_training-8.md | FIXED | Named target never borrows another target's params (suite-wide fallback also gated). Test test_cluster39_chosen_params_target_isolation.py. |
| 39-F5 | 39_training-8.md | FIXED | Missing group column logged at WARNING. Test test_cluster39_missing_group_column_logged.py. |
| 39-F17 | 39_training-8.md | FIXED | fit_transform applies the woe binary check. Test test_cluster39_woe_fit_transform_binary_check.py. |
| 39-F18 | 39_training-8.md | FIXED | NaN text embedded as empty string. Test test_cluster39_hf_nan_text_is_empty.py. |
| 39-F35 | 39_training-8.md | FIXED | 'no terms remain' handled as empty vocabulary. Test test_cluster39_tfidf_no_terms_remain.py. |
| 39-F54 | 39_training-8.md | FIXED | orthogonal_/sparse_ fall back on 1-D tensors. Test test_cluster39_orthogonal_init_on_batchnorm.py. |
| 39-F67 | 39_training-8.md | FIXED | L1 exclusion ids are a local, never stored on the module (the earlier fix still stored them). Test test_cluster39_l1_exclusion_not_pickled.py. |
| 39-F36 | 39_training-8.md | FIXED | Prewarm pins the entry in the strong tier while loading. Test test_cluster39_prewarm_pins_entry_during_load.py. |
| 39-F37 | 39_training-8.md | FIXED | NaN/inf refused with a clear message (default choice). Test test_cluster39_polynomial_refuses_nonfinite.py. |
| 39-F6 | 39_training-8.md | FIXED | Missing timestamp gets the floor weight (default choice). Test test_cluster39_recency_missing_timestamp.py. |
| 39-F38 | 39_training-8.md | DEFERRED | Cap has no pipeline caller; project policy is wire-in, not delete; insertion point is a design decision. |
| 39-F32 | 39_training-8.md | FIXED | Only float NaN rebranded; string 'NaN' kept. Test test_cluster39_polars_string_nan_is_data.py. |
| 39-F19 | 39_training-8.md | FIXED | Low RAM evicts one entry per check (default choice). Test test_cluster39_low_ram_evicts_one_entry.py. |
| 04-F11 | 04_data.md | FIXED | Provenance now states the beds are approximations, not published functions. Test test_cluster04_friedman_provenance_is_honest.py. |
| 04-F13 | 04_data.md | FIXED | Weights 1/(levels-1). Test test_cluster04_graded_cardinality_weights.py. |
| 23-F2 | 23_root_modules-2.md | FIXED | Explicit leak_tolerance decides the verdict. Test test_cluster23_leak_verdict_uses_caller_tolerance.py. |
| 40-F13 | 40_training-9.md | FIXED | Cache hit stamps the pre-pipeline on the model. Test test_cluster40_cache_hit_stamps_pre_pipeline.py. |
| 40-F18 | 40_training-9.md | FIXED | Unaligned split with group ids refused. Test test_cluster40_ma_crossover_refuses_unaligned_groups.py. |
| 40-F3 | 40_training-9.md | FIXED | Multi-target regression monitors RMSE, not ICE. Test test_cluster40_multitarget_regression_uses_rmse.py. |
| 40-F4 | 40_training-9.md | FIXED | Multilabel predict returns indicators. Test test_cluster40_multilabel_predict_returns_indicators.py. |
| 40-F5 | 40_training-9.md | FIXED | Class weights follow original labels via classes_. Test test_cluster40_class_weight_follows_original_labels.py. |
| 40-F8 | 40_training-9.md | FIXED | class_weight non-persistent; load drops stale key. Test test_cluster40_class_weight_not_persisted.py. |
| 40-F7 | 40_training-9.md | FIXED | Scaling copies before the block write. Test test_cluster40_scaling_does_not_mutate_input.py. |
| 34-F7 | 34_training-3.md | FIXED | Spec under 50 valid rows passes the RMSE gate unscored; now logs WARNING naming spec and row count. Test test_cluster34_discovery_silent_skips_are_logged.py. |
| 34-F8 | 34_training-3.md | FIXED | Residual-std probe failure logged at DEBUG; now WARNING with base and transform. Test test_cluster34_discovery_silent_skips_are_logged.py. |
| 34-F5 | 34_training-3.md | FIXED | Negative train/test positions were not bounds-checked or normalised before the overlap test; now wrapped by len(df) and checked. Test test_cluster34_negative_indices_are_positions.py. |
| 34-F6 | 34_training-3.md | FIXED | Holdout docstring claimed the same quantity as in-screen MI; now states the feature sets differ. Feature-set alignment not implemented. |
| 34-F1 | 34_training-3.md | FIXED | Trailing mean NaN poisons later rows; kernel skips NaN (commit 9c32cf0b1). Test test_cluster34_trailing_mean_nan.py. |
| 34-F2 | 34_training-3.md | FIXED | Hints dropped by the finite filter still take top-k slots and are returned (commit 9c32cf0b1). Test test_cluster34_auto_base_mrmr_and_collinear.py. |
| 34-F3 | 34_training-3.md | FIXED | Demoted candidates set the MRMR relevance scale (commit 9c32cf0b1). Test test_cluster34_auto_base_mrmr_and_collinear.py. |
| 34-F4 | 34_training-3.md | FIXED | Collinear replay skipped re-checking non-borderline columns after a flip (commit 9c32cf0b1). Test test_cluster34_auto_base_mrmr_and_collinear.py. |
| 34-F9 | 34_training-3.md | PARTIAL | Per-fold refit failure silently fell back to leaked global params; now WARNING. Spec not marked (unreachable from default discovery). Test test_cluster34_fold_refit_failure_is_visible.py. |
| 34-F10 | 34_training-3.md | FIXED | Upgraded multi-base spec carried the seed's tiny-rerank score; now uses the final kept set's CV RMSE. Test test_cluster34_multibase_upgrade_score.py. |
| 34-F11 | 34_training-3.md | FIXED | Block-size search ignored sample_weight used by the final fit. Test test_cluster34_direct_multi_horizon_weighted_search.py. |
| 34-F12 | 34_training-3.md | FIXED | Column set rebuilt per candidate column in warm-start rescore; built once per spec. Test test_cluster34_incremental_column_set_built_once.py. |
| 34-F18 | 34_training-3.md | FIXED | Degenerate fits skipped bookkeeping; elapsed_seconds_ etc raised AttributeError. Test test_cluster34_degenerate_fit_publishes_attributes.py. |
| 34-F24 | 34_training-3.md | FIXED | domain_check numeric mask indexed rows wrongly; cast to bool at both sites. Test test_cluster34_domain_mask_is_boolean.py. |
| 34-F22 | 34_training-3.md | FIXED | logger and _UNARY_BASE_SENTINEL defined twice. Test test_cluster34_fit_steps_single_definitions.py. |
| 34-F13 | 34_training-3.md | FIXED | Per-group loops built full-length masks per group and re-extracted base columns. Sorted slices plus one extraction per base. Test test_cluster34_route_extracts_base_once.py. |
| 34-F31 | 34_training-3.md | FIXED | Auto-base log counted an MI candidate not in the output when hints fill top_k. Test test_cluster34_auto_base_log_and_wording.py. |
| 34-F25 | 34_training-3.md | FIXED | Skipped-spec messages said fell back to its default. Wording only. |
| 34-F23 | 34_training-3.md | FIXED | Comment said non-finite keys sort first; they sort last. Comment only. |
| 34-F19 | 34_training-3.md | FIXED | Carry-null permuted the whole base, so NaN base entries could fall inside the scored rows. Permutes the scored rows only. Test test_cluster34_base_carry_null_rows.py. |
| 34-F27 | 34_training-3.md | FIXED | Shared-fold cache keyed on a 64-bit hash() of row bytes. Now a blake2b digest. Test test_cluster34_fold_cache_row_digest.py. |
| 34-F29 | 34_training-3.md | FIXED | clear borrowed from ContentMemo via shared attribute names. Own clear() now. Test test_cluster34_matrix_cache_clear.py. |
| 34-F30 | 34_training-3.md | FIXED | Val-side target cast to float32 when attached, so bases differed from the train side. Target kept float64. Test test_cluster34_attach_keeps_target_float64.py. |
| 34-F15 | 34_training-3.md | FIXED | Per-element Python tie ranking and PAVA loops. Vectorised ranking for finite input; PAVA compiled. Test test_cluster34_leakage_rank_and_pava.py. |
| 34-F14 | 34_training-3.md | DEFERRED | Column gathers per 8192-row block. One gather per column needs n x F memory (about 6.4 GB at 3.2M x 500); a column-chunked Gram is a redesign. Owner decision on memory budget. |
| 34-F17 | 34_training-3.md | DEFERRED | Carry test: 20 MI calls per column. The batched kernel has a different null (block-shuffled codes). Numerical change; needs benchmark and owner sign-off. |
| 34-F20 | 34_training-3.md | DEFERRED | Warm-start data signature on the full frame. A sampled signature can reuse stale specs. Owner decision. |
| 34-F21 | 34_training-3.md | PARTIAL | Finite mask reused after impute (4 passes to 2). No new test; 40 existing auto-base tests pass. |
| 34-F26 | 34_training-3.md | FIXED | Helper misnamed; renamed to _run_folds_serial_early_stop. Rename only. |
| 34-F28 | 34_training-3.md | FIXED | BoundedMemo ran builds under its lock; re-entrant build deadlocked. Build now outside the lock. Test test_cluster34_bounded_memo_build_outside_lock.py. |
| 34-F34 | 34_training-3.md | FIXED | No test for BoundedMemo eviction. Test test_cluster34_bounded_memo_eviction.py. |
| 34-F35 | 34_training-3.md | FIXED | No test for abs_corr_all_no_copy. Reference test on float32 strided input. Test test_cluster34_abs_corr_no_copy_matches_reference.py. |
| 34-F32 | 34_training-3.md | FIXED | Bench NEW arm used the config default, which excludes the widened transforms. Built explicitly. Test test_cluster34_bench_new_pool_is_explicit.py. |
| 34-F33 | 34_training-3.md | FIXED | Strict-win counts: the default counted per challenger, not per scenario. Per-scenario helper. Test test_cluster34_bench_strict_wins_per_scenario.py. |
| 34-F16 | 34_training-3.md | DEFERRED | Calibration gate has no caller; wire in or remove the public export. Owner decision. |
| 40-F9 | 40_training-9.md | FIXED | All-null categorical maps to unknown code at predict. Test test_cluster40_all_null_categorical_predicts.py. |
| 40-F10 | 40_training-9.md | FIXED | Unequal hidden_layers refused at construction (default choice; per-layer sizes not supported by the builder). Test test_cluster40_ranker_refuses_unequal_hidden_layers.py. |
| 40-F15 | 40_training-9.md | FIXED | Entity-time module refuses an unaligned split (test test_cluster40_entity_time_refuses_unaligned_split.py). Latent-SVD, MA no-group path and nearest-past-join still open. |
| 40-F16 | 40_training-9.md | FIXED | Powerset composites attached to every split or none. Test test_cluster40_powerset_all_or_nothing.py. |
| 40-F17 | 40_training-9.md | FIXED | Replay uses the fitted effective_k. Test test_cluster40_replay_uses_fitted_effective_k.py. |
| 40-F11 | 40_training-9.md | FIXED | Precomputed ranking pairs capped per query like the loss. Test test_cluster40_ranker_pairs_are_capped.py. |
| 25-F2 | 25_root_modules-4.md | FIXED | Monotonicity by comparison for unsigned ids. Test test_cluster25_unsigned_group_ids_unsorted.py. |
| 26-F4 | 26_root_modules-5.md | FIXED | Expanding stats use observed count; the earlier test was weak (finiteness only). Strict test test_cluster26_expanding_uses_observed_count.py. |
| 28-F12 | 28_root_modules-7.md | FIXED | Absent class_weight key skipped. Test test_cluster28_class_weight_absent_key_skipped.py. |
| 28-F2 | 28_root_modules-7.md | FIXED (test pending) | Multilabel decided from shape only; no targeted test yet (needs full calibrator run). |
| 28-F6 | 28_root_modules-7.md | FIXED | Inf guard runs on the batched path. Test test_cluster28_polars_inf_guard_runs_when_not_fixing.py. |
| 28-F17 | 28_root_modules-7.md | FIXED | set_label applies the fitted label encoder. Test test_cluster28_set_label_encodes_like_fit.py. |
| 28-F7 | 28_root_modules-7.md | FIXED | Split artifacts always rewritten. Test test_cluster28_split_artifacts_rewritten.py. |
| 28-F11 | 28_root_modules-7.md | FIXED | Auto-wrap switch carried on the control config. Test test_cluster28_wrap_flag_reaches_control.py. |
| 28-F14 | 28_root_modules-7.md | FIXED | Unreadable existing bundle raises. Test test_cluster28_unreadable_bundle_raises.py. |
| 28-F16 | 28_root_modules-7.md | FIXED (test pending) | Validation cache hit re-applies init_score. Needs a full cached fit to test. |
| 01-F1 | 01_calibration.md | FIXED | Bench groups from the bias-index codes. Bench-only, no test. |
| 01-F2 | 01_calibration.md | FIXED | Bench seeds via zlib.crc32. Bench-only, no test. |
| 01-F4 | 01_calibration.md | DEFERRED | Bench re-implements shipped ECI code. Bench-only rewrite. |
| 01-F6 | 01_calibration.md | DEFERRED | Bench per-row label loop (perf). Bench-only. |
| 01-F8 | 01_calibration.md | DEFERRED | Bench verdict needs re-measuring. Bench-only. |
| 02-F1 | 02_competition.md | FIXED | Wall time from an un-profiled call. Bench-only. |
| 02-F2 | 02_competition.md | DEFERRED | Bench-only group, not read. |
| 02-F3 | 02_competition.md | DEFERRED | Bench-only group, not read. |
| 03-F1 | 03_core.md | DEFERRED | Bench-only perf. |
| 03-F2 | 03_core.md | DEFERRED | Bench-only; production covered by test. |
| 03-F3 | 03_core.md | FIXED | Production parallel-vs-serial test passing (test_robust_mean_parallel_matches_serial.py). |
| 04-F25 | 04_data.md | FIXED | SAPP shape-then-loc/scale. Test test_cluster04_sapp_shape_then_loc_scale.py. |
| 04-F20 | 04_data.md | FIXED | id_trap row_id is a permutation; test asserts exact uniqueness. |
| 04-F23 | 04_data.md | FIXED | Coerced non-numeric counted in notes. Test test_cluster04_sapp_args_and_coerce.py (passing). |
| 04-F17 | 04_data.md | FIXED | No-op threshold line removed (behaviour identical). Existing link tests pass. |
| 04-F16 | 04_data.md | FIXED | Caveat threshold from the spec's ceiling tolerance. Test test_cluster04_ceiling_tolerance_drives_caveat.py. |
| 04-F18 | 04_data.md | FIXED | Renamed to coverage_adjusted_plugin. Test test_cluster04_oracle_estimator_names_are_honest.py. |
| 04-F15 | 04_data.md | DEFERRED | Allowlisted exploration helper with a wrong body; intent needs confirming. |
| 04-F21 | 04_data.md | FIXED | Probes keep donor missingness. Test test_cluster04_probe_keeps_donor_missingness.py. |
