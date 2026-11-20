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
