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
| 09-F1 | 09_feature_engineering-2.md | RESOLVED | Random directions are unscaled; the d^{-1/4} scale moved to the inputs so the kernel is exp(q.k/sqrt(d)) as documented. Tests tests/feature_engineering/transformer/test_performer_kernel_identity.py; reference in ... |
| 09-F2 | 09_feature_engineering-2.md | RESOLVED | Double d^-1/2 removed at both call sites; test test_nystrom_scales_each_neighbour_once.py. Commit 4863686a5. |
| 09-F3 | 09_feature_engineering-2.md | RESOLVED | Documented as the label-free proxy |p - median(y)| on both sides (no query label exists). Behaviour unchanged. Commit 8dc4360c0. |
| 24-F1 | 24_root_modules-3.md | RESOLVED | Per-column finite-row AUC when a cell is non-finite; single-class columns are kept. Test test_drop_near_noise_handles_nan.py. Commit 2171ff34e. Siblings drop_raw_after_embedding.py:53,142 still open. |
| 35-F1 | 35_training-4.md | RESOLVED | from_fitted_inner now stamps smearing_quantiles from the train features (X_train) and the reconstructed T, as fit() does; call sites pass the raw train rows. Tests ... |
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
| 35-F21 | 35_training-4.md | FIXED | OOF cache key now includes a full digest of sample_weight and of the component specs (arrays digested in full, nested dicts walked); tests test_cluster35_oof_cache_key_weights_and_specs.py; 48 existing cache-key tests ... |
| 35-F22 | 35_training-4.md | FIXED | Auto gate tested prediction correlation; protocol defines residual correlation. Now residuals. Test test_cluster35_auto_gate_measures_residuals.py. |
| 35-F23 | 35_training-4.md | FIXED | Requested calibration method stored at construction and restored at the start of each fit; a degenerate fit no longer changes later fits (test_cluster35_calibrator_method_restored_per_fit; 27 existing calibration tests ... |
| 35-F24 | 35_training-4.md | FIXED | Lag extraction failure logged at DEBUG; deep-OOD rows silently took the median. Now WARNING. Test test_cluster35_lag_extraction_failure_warns.py. |
| 35-F25 | 35_training-4.md | FIXED | Stamped base-fit range overwritten by from_fitted, moving soft-shrink thresholds. Stamped range kept. Test test_cluster35_from_fitted_keeps_stamped_range.py. |
| 35-F26 | 35_training-4.md | PARTIAL | smeared_prediction now uses residual_quantiles (same subsample, seed and weighting path as fit) and takes an optional sample_weight. Callers in discovery do not pass weights yet. Test ... |
| 35-F27 | 35_training-4.md | FIXED | predict_quantile re-resolved the causal lag once per quantile column. Resolved once per batch. Test test_cluster35_smart_fallback_resolves_lag_once.py. |
| 35-F28 | 35_training-4.md | DEFERRED | The grid spans 0.05 to 0.95, so sample() flattens tails and crps() omits the 0-0.05 and 0.95-1 mass. Fixing it means adding tail levels (0.001/0.999) or a GPD tail, a modelling choice the owner should make; the ... |
| 35-F29 | 35_training-4.md | FIXED | TypeError fallback dropped sample_weight for any TypeError. Support read from signature; other TypeErrors propagate. Test test_cluster35_mixture_sample_weight_support.py. |
| 35-F30 | 35_training-4.md | FIXED | Dict config seed ignored; report recorded 42. Coerced before the seed is read. Test test_cluster35_highlevel_dict_config_seed.py. |
| 35-F31 | 35_training-4.md | FIXED | Group broadcast was a per-row dict loop; NaN group ids fell back silently. Vectorised reindex; NaN rows counted and logged. Test test_cluster35_group_broadcast_vectorised.py. |
| 35-F32 | 35_training-4.md | FIXED | Single numeric column raised ValueError in linkage. Returns that column as one subset. Test test_cluster35_subset_bagging_single_column.py. |
| 35-F33 | 35_training-4.md | PARTIAL | TailCompositeEstimator.fit logs a WARNING when residual_X is absent (in-sample residuals under-disperse the tail). Default still in-sample: out-of-fold residuals by default would change every caller's fit cost, so it is ... |
| 35-F34 | 35_training-4.md | FIXED | Worst-candidate choice uses the judge's fill (fit rows that the judge uses) and treats a NaN score as worst, so it no longer depends on order (test_cluster35_worst_candidate_nan_order; 142 existing y-scale gate tests ... |
| 35-F35 | 35_training-4.md | PARTIAL | Spec with fewer than 50 in-domain fit rows is now rejected with a ledger entry instead of passing unchecked. No direct test: that branch needs a full gate fixture (a transform whose domain keeps under 50 rows). 142 ... |
| 35-F36 | 35_training-4.md | PARTIAL | Multi-base specs still pass the structural fragility gate unjudged; the bypass is now logged at WARNING. Gating on the primary base changes which specs ship, so it is left as an owner decision. Test ... |
| 35-F37 | 35_training-4.md | FIXED | WAIC scoring failures swallowed at DEBUG; spec silently skipped in tie-break. Now WARNING naming spec and exception. Test test_cluster35_waic_failure_logged.py. |
| 35-F38 | 35_training-4.md | PARTIAL | Pair-mask fix present in origin's kNN loop; de-duplication into one shared loop not carried (conflicts with origin's version). Follow-up. |
| 35-F39 | 35_training-4.md | FIXED | Both stacking passes share one ranking helper (_rank_pass1_specs) and one group-id derivation (_train_group_ids); the pass-specific cap, warnings and aggregation stay in each caller ... |
| 35-F40 | 35_training-4.md | FIXED | Group path and default path use one helper (_polars_default_cat_candidates), so the two candidate sets cannot drift; temporal columns are excluded on both (test_cluster35_polars_candidates_one_helper; 67 existing ... |
| 35-F41 | 35_training-4.md | FIXED | bayesian_alpha_fit let a NaN pair reach the regression and returned a NaN posterior silently. Pairs dropped, counted and logged. Test test_cluster35_bayesian_nonfinite_dropped.py. |
| 35-F42 | 35_training-4.md | DEFERRED | A broken process pool serially rescores every task, not only the unfinished ones. Documented as deliberate, and correctness is unaffected. Rescoring only the unfinished tasks needs partial results from the broken pool ... |
| 35-F43 | 35_training-4.md | FIXED | Three paths had no test: CDF with enforce_non_crossing=False, mixture soft_routing, calibrated gated outlier. Smoke tests added. Test test_cluster35_untested_paths_smoke.py. |
| 35-F44 | 35_training-4.md | PARTIAL | Repeated integer-valuedness check removed from the grouping call site (level-count helper). The two dead helpers stay as test-facing references, already allowlisted in the uncalled-functions gate. Test ... |
| 35-F45 | 35_training-4.md | FIXED | Bootstrap posterior result carries posterior_kind, n_samples, degrees_of_freedom, sigma2_posterior_mean and n_dropped_nonfinite, like the conjugate one (test_cluster35_bootstrap_result_keys; 27 existing bayesian tests ... |
| 35-F46 | 35_training-4.md | PARTIAL | _residualise treats a column as constant only relative to its own magnitude (test_cluster35_residualise_relative_constant; existing _residualise tests pass). Mid-module imports left in place (not circular-import ... |
| 35-F47 | 35_training-4.md | FIXED | Dead _groups_screen assignment and a missing blank line in _tiny_rerank.py. The None now goes straight into the call. Test: 231 existing rerank tests pass. |
| 35-F48 | 35_training-4.md | FIXED | Unused Ridge import removed; __all__ rebuilt as the public surface; cache-key block moved to ensemble/_oof_cache_keys.py so the facade is under the 1k module gate (955 lines), with re-exports ... |
| 35-F49 | 35_training-4.md | FIXED | Uncached transform-pair path dropped refit_supervised, so a supervised selector was not refit on the fold. Forwarded. Test test_cluster35_uncached_pair_forwards_refit.py. |
| 35-F50 | 35_training-4.md | FIXED | Write-only _sig_degenerate attribute. Deleted. Existing 27 calibration tests pass; no separate test (removal). |
| 35-F51 | 35_training-4.md | FIXED | n_oof_rows counted rows finite in y only; the fit uses rows finite in y and every OOF column. Count now matches the fit. Test test_cluster35_meta_stack_row_count.py. |
| 35-F52 | 35_training-4.md | PARTIAL | Subsample bench fits weights against the returned holdout target, row for row (test_cluster35_bench_subsample_targets_align). Calibration bench already uses crc32 (no salted hash() left in the ensemble benches). ... |
| 35-F53 | 35_training-4.md | FIXED | getattr(self, 'smearing', True) read an attribute no constructor sets; always True. Gate removed. Test test_cluster35_smearing_always_on.py. |
| 35-F54 | 35_training-4.md | FIXED | predict before fit raises sklearn NotFittedError (a ValueError subclass, so existing catches still work) (test_cluster35_dual_direction_not_fitted). 31 existing dual-direction tests pass; 2 fail inside the numba metrics ... |
| 35-F55 | 35_training-4.md | PARTIAL | One shared subset_rows in _frame_ops for both gated models (test_cluster35_shared_subset_rows; 61 existing gated tests pass, one pre-existing numba failure confirmed on master). The glm _is_polars_df re-export is kept ... |
| 35-F56 | 35_training-4.md | WON'T FIX | Docstring cites the defining module (varying_size_top_k_subsets._cluster_anchors), which is correct; the shared import is a re-export. No drift. |
| 35-F57 | 35_training-4.md | FIXED | Pointless try/except around list(names). Removed. 13 existing extremes tests pass; no separate test (removal). |
| 36-F1 | 36_training-5.md | FIXED | time_ordering restricted to covered rows, length-checked before nested selection. See DISPOSITIONS. |
| 36-F2 | 36_training-5.md | FIXED | sample_weight to raw estimator gated by _model_fit_accepts_sample_weight on both fits. See DISPOSITIONS. |
| 36-F3 | 36_training-5.md | FIXED | Stage-2 sample_weight gated like stage 1. No separate test. |
| 36-F4 | 36_training-5.md | FIXED | Proba wrapper refuses non-binary classifiers at fit. Test tests/training/composite/test_cluster36_s*. |
| 36-F5 | 36_training-5.md | FIXED | Failed candidate fold logs at WARNING with the exception class. Read and verified; no separate test. |
| 36-F6 | 36_training-5.md | FIXED | More CV folds than rows raises ValueError naming both numbers. Test test_cluster36_hpo_refuses_more_folds_than_rows.py. |
| 36-F7 | 36_training-5.md | FIXED | Imputed NaN meta-feature rows logged at WARNING. Read and verified; no separate test. |
| 36-F8 | 36_training-5.md | DEFERRED | Stage-2 trained on OOF meta-features but predicts from full-refit ones (design train/serve skew). Owner decision. |
| 36-F9 | 36_training-5.md | FIXED | Class factories are inspected for sample_weight from the fit signature without constructing an estimator; callables are still called (test_cluster36_factory_signature_no_construction; 44 existing multi-stage tests pass ... |
| 37-F1 | 37_training-6.md | FIXED | Median fill of out-of-domain T reached training rows, a fabricated label. Fill now only outside the train split. Test test_cluster37_composite_t_no_train_label_fill.py; pinned test updated. |
| 37-F2 | 37_training-6.md | FIXED | Supplied lag with no lag column was never offered as a base, understating the ceiling. Attached as a named column. Test test_cluster37_supplied_lag_is_a_base.py. |
| 37-F3 | 37_training-6.md | FIXED | Venn-Abers calibration let NaN scores become a grid point; lookup silently returned the top bin for NaN. Calibration now refuses non-finite scores; lookup yields NaN. Test test_cluster37_venn_abers_nan_refused.py. |
| 37-F4 | 37_training-6.md | FIXED | Member individual score mixed validation metric with negated OOF log-loss; now one source is used for all members. Test test_cluster37_diversity_single_score_scale.py. |
| 37-F5 | 37_training-6.md | FIXED | Stacking-aware gate pruned columns and the stacker was fit unweighted; per-row sample weights are now passed regardless of pruning (commit 4ef46cf95). |
| 37-F6 | 37_training-6.md | FIXED | refit_capped_stack ignored sample_weight; now takes it and the call site passes OOF weights. Test test_cluster37_venn_refit.py. |
| 37-F7 | 37_training-6.md | FIXED | Group subsample recounted group sizes over the full array per group; sizes now come from one np.unique pass (commit 4ef46cf95). |
| 37-F8 | 37_training-6.md | FIXED | Fingerprint cross-check exception left the precomputed bundle trusted at DEBUG; now distrusted and logged at WARNING. Test test_cluster37_fingerprint_exception_distrusts.py. |
| 37-F9 | 37_training-6.md | FIXED | Render scope entered inside the try; cleanup guarded. |
| 37-F10 | 37_training-6.md | FIXED | Pandas leaderboard copied before insert; metadata frame no longer mutated. |
| 37-F11 | 37_training-6.md | FIXED | Knob overrides use model_copy only; caller config not mutated. Test test_cluster37_knob_overrides_no_caller_mutation.py. |
| 37-F12 | 37_training-6.md | FIXED | Misleading view-select comment corrected (pandas copies). |
| 37-F13 | 37_training-6.md | FIXED | Group-aware subsample no longer abandoned when kept groups cover all rows. Test test_cluster37_subsample_groups_and_nan_gate.py. |
| 37-F14 | 37_training-6.md | FIXED | Non-finite ensemble RMSE is a gate failure, not a pass. |
| 37-F15 | 37_training-6.md | FIXED | OOF gate skip logged at WARNING and marked UNVALIDATED. |
| 37-F16 | 37_training-6.md | FIXED | Leaky-RMSE predict failure logged at WARNING per component. |
| 37-F17 | 37_training-6.md | FIXED | NNLS non-convergence isolated to its column. |
| 37-F18 | 37_training-6.md | PARTIAL | Base columns built once per name across specs; full-length allocation remains (needs benchmark for index-subset build). |
| 37-F19 | 37_training-6.md | FIXED | Pandas unseen-category drift check added. Test test_cluster37_pandas_unseen_category_drift.py. |
| 37-F20 | 37_training-6.md | FIXED | Ignored plot_format_subfolders / calibration_colormap logged at WARNING. |
| 37-F21 | 37_training-6.md | FIXED | Numba prewarm also runs when feature selection will run. |
| 37-F22 | 37_training-6.md | FIXED | Label distribution drift is best-effort. |
| 37-F23 | 37_training-6.md | WON'T FIX | MTRPerColumnEqualMeanEnsemble.fit is the public E3 API, exercised by tests/training/test_mtr_nnls_ensemble.py; production path uses compute_mtr_oof_nnls_weights. Premise 'no caller' holds only for production; semantics ... |
| 37-F24 | 37_training-6.md | FIXED | Already classified as intentionally unused in uncalled-functions allowlist. |
| 37-F50 | 37_training-6.md | FIXED | Redundant tier alias removed. |
| 37-F51 | 37_training-6.md | PARTIAL | Redundant placeholder removed; helper rename not done. |
| 37-F52 | 37_training-6.md | FIXED | Binary-only Venn-Abers message tidied. |
| 38-F1 | 38_training-7.md | FIXED | Suite-wide ensemble no longer spans targets (9d0472dca). |
| 38-F2 | 38_training-7.md | FIXED | Conformal residuals use pre-recalibration predictions (9d0472dca). |
| 38-F3 | 38_training-7.md | FIXED | Fit-predict-only detector skips val OD filter explicitly (9d0472dca). |
| 38-F4 | 38_training-7.md | FIXED | Warning matches caller behaviour (prior ensemble kept). |
| 38-F5 | 38_training-7.md | FIXED | Bool cast on a copy. Test test_cluster38_predict_and_fit_side_effects.py. |
| 38-F6 | 38_training-7.md | FIXED | Empty per-type dict skipped. Test test_cluster38_predict_and_fit_side_effects.py. |
| 38-F7 | 38_training-7.md | FIXED | PySR skip logged at WARNING regardless of verbose. |
| 38-F8 | 38_training-7.md | FIXED | Non-neural time reset once per target. |
| 38-F9 | 38_training-7.md | DEFERRED | Needs explicit group-column name from extractor (design). |
| 38-F10 | 38_training-7.md | DEFERRED | Owner decision: delete or finish the weight-iteration refactor. |
| 38-F11 | 38_training-7.md | FIXED | Stale other-format metadata file removed on save. |
| 38-F12 | 38_training-7.md | FIXED | Embedding detection requires non-scalar shape. |
| 38-F13 | 38_training-7.md | FIXED | Failed split extraction logged at WARNING. |
| 38-F14 | 38_training-7.md | FIXED | Composition states described entry and count. |
| 38-F15 | 38_training-7.md | FIXED | No bare clone appended on prediction failure. |
| 38-F16 | 38_training-7.md | FIXED | Temporal audit receives real target type for regression/binary. |
| 38-F17 | 38_training-7.md | FIXED | set() hoisted out of comprehensions. |
| 38-F18 | 38_training-7.md | FIXED | Dead expression removed. |
| 38-F19 | 38_training-7.md | FIXED | Duplicate logger removed. |
| 38-F20 | 38_training-7.md | FIXED | Docstring corrected (no strong-ref window). |
| 38-F21 | 38_training-7.md | PARTIAL | Comment corrected; repr-keyed fallback documented, not re-keyed. |
| 38-F22 | 38_training-7.md | DEFERRED | Owner decision: remove unused parameter or guard cast. |
| 38-F23 | 38_training-7.md | FIXED | Documented: first regression target drives stratification. |
| 38-F24 | 38_training-7.md | DEFERRED | Needs explicit target_type contract decision. |
| 38-F25 | 38_training-7.md | FIXED | Narrowing suffix widened to 8 hex characters. |
| 38-F26 | 38_training-7.md | FIXED | AssertionError message and warning text corrected. |
| 38-F27 | 38_training-7.md | FIXED | Filtered-subset threshold skip logged and default stamped. |
| 39-F1 | 39_training-8.md | FIXED | Object-dtype targets range-checked before int8 cast (earlier commit). |
| 39-F2 | 39_training-8.md | FIXED | Batch missing a key raises instead of misaligning rows. |
| 39-F3 | 39_training-8.md | FIXED | Failed concatenate raises with shapes. |
| 39-F4 | 39_training-8.md | FIXED | Named-target lookup no longer borrows another target's params. |
| 39-F5 | 39_training-8.md | FIXED | Configured group column missing logs WARNING. |
| 39-F6 | 39_training-8.md | FIXED | Missing timestamp gets floor weight (d38c3c2f5). |
| 39-F17 | 39_training-8.md | FIXED | fit_transform applies binary check for woe. |
| 39-F18 | 39_training-8.md | FIXED | Float NaN text embedded as empty string. |
| 39-F19 | 39_training-8.md | FIXED | Low-RAM eviction bounded to one entry per check (9f2781207). |
| 39-F20 | 39_training-8.md | FIXED | Persistence on without DiskKey logs WARNING. Test test_cluster39_disk_tier_key_warning.py. |
| 39-F32 | 39_training-8.md | FIXED | Only float NaN becomes null sentinel (0697b9a6b). |
| 39-F33 | 39_training-8.md | FIXED | Null categories grouped with dropna=False. |
| 39-F34 | 39_training-8.md | FIXED | Zero smoothing falls back to prior. |
| 39-F35 | 39_training-8.md | FIXED | 'No terms remain' handled like empty vocabulary. |
| 39-F36 | 39_training-8.md | FIXED | Prewarm pins entry in strong tier while loading. |
| 39-F37 | 39_training-8.md | FIXED | NaN/inf refused in polynomial fit and transform (04ea4f9c7). |
| 39-F38 | 39_training-8.md | DEFERRED | hgb_max_features_cap has no caller; wire in or remove is an owner decision. |
| 39-F53 | 39_training-8.md | FIXED | Multilabel heads use per-label sigmoid. |
| 39-F54 | 39_training-8.md | FIXED | orthogonal_ and sparse_ fall back on 1-D tensors. |
| 39-F67 | 39_training-8.md | FIXED | L1 exclusion ids no longer stored or pickled. |
| 35-F11 | 35_training-4.md | FIXED | Early logratio raise removed; base<=0 rows go to the fallback. Test test_cluster35_quantile_logratio_nonpositive_base.py. |
| 35-F7 | 35_training-4.md | FIXED | time_aware with groups raises. Test test_cluster35_stacking_time_and_groups_refused.py. |
| 35-F9 | 35_training-4.md | FIXED | External holdout digest covers frame content. Test test_cluster35_holdout_digest_distinguishes_content.py. |
| 12-F38 | 12_feature_selection-3.md | FIXED | downstream returns NaN for an empty selection. Test test_fs_hybrid_empty_selection_downstream.py. |
| 12-F31 | 12_feature_selection-3.md | PARTIAL | The two named benches (round3_hard_bed_bench, round3_hybrid_refine_bench) now silence only FutureWarning/DeprecationWarning. 61 other fs_hybrid bench files still install a catch-all ignore; not changed in this pass. ... |
| 12-F20 | 12_feature_selection-3.md | FIXED | An RFECV that selects nothing is replaced by the first column and recorded as its selection (importance_shootout.py cell loop). The fix (empty selection scores chance) is a one-branch change, but the cell loop runs a ... |
| 35-F5 | 35_training-4.md | FIXED | Every fold keeps a slot (NaN when unusable). Test test_cluster35_stepwise_fold_slots.py. |
| 12-F26 | 12_feature_selection-3.md | FIXED | Flip needs a sign majority and a positive mean delta (decide_flip). Test test_fs_hybrid_flip_needs_positive_mean.py. |
| 12-F27 | 12_feature_selection-3.md | FIXED | Proxy Brier computed out-of-fold. Test test_fs_hybrid_s2_proxy_is_out_of_fold.py. |
| 12-F28 | 12_feature_selection-3.md | FIXED | Sizes with a failed cell get no family mean. Test test_fs_hybrid_family_curve_failed_cells.py. |
| 12-F29 | 12_feature_selection-3.md | FIXED | The designed-bed verdict reads the pre-specified right operator (designed_recovery, DESIGNED_RIGHT_OP) instead of the max over test AUC. Tests ... |
| 12-F30 | 12_feature_selection-3.md | FIXED | run_bed uses a fixed-seed split shared by all presets. Test test_fs_hybrid_cost_split_is_fixed.py. |
| 12-F32 | 12_feature_selection-3.md | FIXED | Failures at WARNING; per-config completed/failed counts printed with the means. Test test_fs_hybrid_shap_run_counts.py. |
| 12-F33 | 12_feature_selection-3.md | FIXED | Flip verdict on paired deltas over 5 splits with a 95% bound (flip_verdict). Test test_fs_hybrid_flip_verdict_paired.py. |
| 12-F34 | 12_feature_selection-3.md | DEFERRED | H35 (and H32) pick the config by OOF AUC on rows the members were fit on. A real fix refits members on an inner split and scores engineered features on held-out rows; this bench's HybridSelector has no transform path ... |
| 12-F35 | 12_feature_selection-3.md | FIXED | Five RFECV benches budget by refits only. Test test_fs_hybrid_rfecv_benches_have_no_wall_clock_cap.py. |
| 12-F36 | 12_feature_selection-3.md | FIXED | Configs with failed runs INCOMPLETE and excluded from ranking. Test test_fs_hybrid_ablation_failed_runs_excluded.py. |
| 12-F37 | 12_feature_selection-3.md | FIXED | Failed runs excluded from kept-share sizes. Test test_fs_hybrid_ablation_failed_runs_excluded.py. |
| 12-F39 | 12_feature_selection-3.md | FIXED | Failures at WARNING; completed/failed counts printed with the means. Test test_fs_hybrid_run_counts_by_key.py. |
| 12-F23 | 12_feature_selection-3.md | FIXED (test pending) | The baseline filters raw_selected_ to raw columns, dropping engineered names. The fix must use the hybrid's transform output (Ztr = h.transform(Xtr)) and filter against it. Confirming which names survive takes a full ... |
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
| 04-F11 | 04_data.md | FIXED | Published Friedman-2 is sqrt(x0^2 + (x1*x2 - 1/(x1*x3))^2) and Friedman-3 is atan((x1*x2 - 1/(x1*x3))/x0). The link vocabulary (BasisTerm kinds) has no 1/(a*b) term and no compound sqrt/atan form, so neither can be ... |
| 04-F13 | 04_data.md | FIXED | Weights 1/(levels-1). Test test_cluster04_graded_cardinality_weights.py. |
| 23-F2 | 23_root_modules-2.md | FIXED | Leaking folds are decided with the same tolerance as the verdict (the caller's leak_tolerance when given), not a fixed 0.02, so leaky_row_positions agree with leak_detected. Regression coverage: the existing 50 ... |
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
| 34-F9 | 34_training-3.md | PARTIAL | Per-fold refit failure logged at WARNING with exception class (test_cluster34_fold_refit_failure_is_visible). Not done: marking the spec as leaked-param-scored; unreachable from default discovery since grouped ... |
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
| 34-F14 | 34_training-3.md | DEFERRED | Gathering every column once for all rows needs n x F floats (about 6.4 GB at 3.2M x 500 float32), which defeats the block design. A column-chunked Gram would need a redesign of the pairwise pass. Owner decision on the ... |
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
| 28-F2 | 28_root_modules-7.md | FIXED | Multilabel decided from array shape only; string class labels stay MULTICLASS. Test tests/training/test_cluster28_multiclass_string_labels_not_multilabel.py. |
| 28-F6 | 28_root_modules-7.md | FIXED | Inf guard runs on the batched path. Test test_cluster28_polars_inf_guard_runs_when_not_fixing.py. |
| 28-F17 | 28_root_modules-7.md | FIXED | set_label applies the fitted label encoder. Test test_cluster28_set_label_encodes_like_fit.py. |
| 28-F7 | 28_root_modules-7.md | FIXED | Split artifacts always rewritten. Test test_cluster28_split_artifacts_rewritten.py. |
| 28-F11 | 28_root_modules-7.md | FIXED | Auto-wrap switch carried on the control config. Test test_cluster28_wrap_flag_reaches_control.py. |
| 28-F14 | 28_root_modules-7.md | FIXED | Unreadable existing bundle raises. Test test_cluster28_unreadable_bundle_raises.py. |
| 28-F16 | 28_root_modules-7.md | FIXED | Val cache hit re-applies init_score; LightGBM cannot clear an init_score once set, so a val Dataset carrying one is rebuilt for a call without offsets (lgb_shim.py). Test tests/training/test_cluster28_val_cache_hit_reapplies_init_score.py. |
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
| 04-F1 | 04_data.md | FIXED | count branch uses count_mean(scale*unit_score), intercept no longer added twice. Test tests/data/datasets/test_count_target_intercept_once.py. Commit below. |
| 04-F2 | 04_data.md | FIXED | TargetSpec rejects a logloss or accuracy ceiling on a binary target (was silently ignored). Tests tests/data/datasets/test_binary_ceiling_metric_rejected.py. Commit below. |
| 04-F3 | 04_data.md | FIXED | Multiclass class scores come from multiclass_class_scores: rotated additive weights per class plus the link's interactions and basis terms for every class. Tests tests/data/datasets/test_multiclass_scores_keep_interactio |
| 04-F4 | 04_data.md | FIXED | link_regions passes the link's regional gate to every column it reads, so FeatureTruth.region is set at both generator call sites. Test tests/data/datasets/test_link_region_reaches_feature_truth.py. Commit below. |
| 04-F5 | 04_data.md | FIXED | Latent reflections are now graph edges (latent -> reflected column), so the M-bias collider sees its second arm and gets role M_COLLIDER. Also: instrument-convention test aligned with 04-F6; explicit t latent family kept |
| 04-F6 | 04_data.md | FIXED | Convention fixed in _declared_kind_role: copy/proxy kinds mark the target, the instrument kind marks its source (instrument -> treatment is causal). Test tests/data/datasets/test_declared_roles_follow_the_edge_convention |
| 04-F7 | 04_data.md | FIXED | realize_latent refuses an unknown family instead of drawing Student-t(4). Tests tests/data/datasets/test_latent_unknown_family_rejected.py. Commit below. |
| 04-F8 | 04_data.md | FIXED | _exact_mi_independent averages the row entropies within each bin (H(Y/X in bin) = mean H(p_i)) instead of the entropy at the bin-mean probability. Test tests/data/datasets/test_exact_mi_averages_row_entropy.py. Commit be |
| 04-F9 | 04_data.md | FIXED | mi_reference memo key includes a blake2b digest of the column data and labels. Test tests/data/datasets/test_truth_mi_memo_keys_on_data.py. Commit below. |
| 04-F10 | 04_data.md | FIXED | count_ceiling sums every row exactly in blocks (the 2001-row prefix extrapolation is gone). Test tests/data/datasets/test_count_ceiling_sums_every_row.py; 5 existing count tests pass. Commit below. |
| 04-F12 | 04_data.md | FIXED | Simpson group is a fair sign (new rademacher family, -1/+1), so the interaction reverses sign between subgroups as documented. Tests tests/data/datasets/test_simpson_group_is_a_sign.py; existing simpson bed test passes. ... |
| 08-F1 | 08_feature_engineering-1.md | FIXED | build_nca_projection fits NCA on the rows it is applied to: the internal StandardScaler (fit then discarded) is removed. Test tests/feature_engineering/transformer/test_nca_projection_lives_in_the_applied_space.py; exist |
| 08-F2 | 08_feature_engineering-1.md | FIXED | All 15 Mode-A out-of-fold output arrays start as NaN (baseline_surprise, adversarial_flip, anomaly_score_features, class_distance, and the 11 grep-found files), so an uncovered row is NaN, not 0.0. Test tests/feature_eng |
| 09-F4 | 09_feature_engineering-2.md | FIXED | Jackknife endpoint weights are no longer proportional: the upper endpoint favours positives and the lower negatives. Test tests/feature_engineering/transformer/test_jackknife_endpoints_weight_classes_differently.py. Comm |
| 09-F5 | 09_feature_engineering-2.md | FIXED | Regression quantile heads of the spread fan are sorted per row, so q10 <= q50 <= q90 holds. Test tests/feature_engineering/transformer/test_quantile_fan_is_ordered.py. Commit below. |
| 09-F6 | 09_feature_engineering-2.md | FIXED | Degenerate folds return NaN (not 0.0) in density_weighted_smote, pure_pos_smote and pseudo_smote. Test tests/feature_engineering/transformer/test_smote_degenerate_fold_is_nan.py covers the density-weighted path. Commit b |
| 09-F7 | 09_feature_engineering-2.md | DEFERRED | Layer 2 in Mode A consumes layer-1 OOF features of training rows that were produced by folds sharing those rows, so the stacking is optimistic. A fix re-folds layer 1 inside each layer-2 fold (nested OOF), which changes ... |
| 09-F8 | 09_feature_engineering-2.md | FIXED | Positional encoding no longer applies a 1e6 modulus, so distinct positions are not aliased; precision in float32 is documented and controlled by dtype. Test tests/feature_engineering/transformer/test_positional_encoding_ |
| 09-F9 | 09_feature_engineering-2.md | FIXED | Geodesic minimum over targets uses one virtual-source Dijkstra (O(rows) memory) instead of a dense (targets x rows) matrix. Test tests/feature_engineering/transformer/test_geodesic_min_over_sources.py equals the per-sour |
| 09-F10 | 09_feature_engineering-2.md | FIXED | Top-k proximity selects the largest k directly instead of partitioning a negated copy of the dense block. The dense block itself stays: the measured note in the code shows every pair shares a leaf. Test tests/feature_eng |
| 09-F11 | 09_feature_engineering-2.md | FIXED | Binary MDL binning refuses a target with more than two labels (was mis-indexing the two-class count arrays). Test tests/feature_engineering/transformer/test_mdl_binary_refuses_three_labels.py. Commit below. |
| 09-F12 | 09_feature_engineering-2.md | FIXED | Empty correct-subset distances are the package far sentinel (1e6) for both mean and nearest, in a module-level _nn_dist_to_subset. Test tests/feature_engineering/transformer/test_trust_empty_subset_is_far.py. Commit belo |
| 09-F13 | 09_feature_engineering-2.md | FIXED | Local PCA centres each neighbour matrix on the neighbour mean instead of the query row. Test tests/feature_engineering/transformer/test_lid_centres_on_the_neighbour_mean.py. Commit below. |
| 10-F1 | 10_feature_selection-1.md | FIXED | Adaptive-guards bench scores the refit on the untouched holdout split; the concatenated re-split is gone. Test tests/feature_selection/test_bench_holdout_auc_is_unseen.py. Commit below. |
| 10-F2 | 10_feature_selection-1.md | FIXED | honest_auc fits on the search rows and scores the untouched holdout, at all three call sites. Test tests/feature_selection/test_bench_interaction_holdout_is_unseen.py. Commit below. |
| 10-F3 | 10_feature_selection-1.md | FIXED | logreg_holdout_auc returns chance (0.5) for an empty selection instead of scoring all columns. Test tests/feature_selection/test_bench_logreg_empty_selection_is_chance.py. Commit below. |
| 11-F1 | 11_feature_selection-2.md | FIXED | _integer_bins ranks finite values only; missing rows take bin 0 instead of the top bin. Test tests/feature_selection/test_bench_integer_bins_missing_not_top.py. Commit below. |
| 11-F2 | 11_feature_selection-2.md | FIXED | RFECV empty selection scores chance (0.5) in the importance-aggregation bench instead of refitting on every column. Test tests/feature_selection/test_bench_rfecv_empty_selection_is_chance.py. Commit below. |
| 11-F3 | 11_feature_selection-2.md | DEFERRED | Searched arms report the max over many in-sample candidates while truth, joint and als are single candidates, so the comparison is not like for like. The fix is a study-design change (same candidate sets, or held-out sco |
| 12-F1 | 12_feature_selection-3.md | FIXED | Staleness default root walks up to the .git folder instead of the module directory, so the git paths match. Test tests/feature_selection/test_staleness_repo_root_is_the_git_root.py. Commit below. |
| 12-F19 | 12_feature_selection-3.md | FIXED | Bench arms are cap60 (the old MRMR width) and cap250; the earlier None arm equalled MRMR's default 250 and so duplicated cap250. Test tests/feature_selection/test_mrmrcap_bench_arms_differ.py. Commit below. |
| 12-F21 | 12_feature_selection-3.md | FIXED | cv_curve takes a ranker and ranks each fold on its training rows; the noise-floor bench passes lgbm_gain_ranking. Test tests/feature_selection/test_noise_floor_ranks_inside_folds.py. Commit below. |
| 12-F22 | 12_feature_selection-3.md | FIXED | Robust noise-floor bench passes a per-fold gain ranker (_gain_order) to cv_curve, so folds rank their own training rows. Test tests/feature_selection/test_noise_floor_robust_gain_order.py. Commit below. |
| 12-F24 | 12_feature_selection-3.md | FIXED | Protected core is the intersection of Boruta and MRMR accepts (agreed_core), matching the docstring. Test tests/feature_selection/test_protected_core_is_the_intersection.py. Commit below. |
| 13-F1 | 13_feature_selection-4.md | FIXED | Under premerge, Subset() expands accepted representatives back to their cluster members and indexes the pre-collapse frame. Test tests/feature_selection/test_boruta_subset_keeps_premerge_members.py; existing premerge tes |
| 13-F2 | 13_feature_selection-4.md | FIXED | The stability path builds the shared fit state before returning, so Subset, results_to_csv and plot read set attributes. Test tests/feature_selection/test_boruta_stability_path_has_fit_state.py; existing stability tests ... |
| 14-F1 | 14_feature_selection-5.md | FIXED | su_seeded_snr_z defaults to None; explicit values always win, so explicit 3.0 under auto is kept. Test test_cluster14_explicit_snr_z_pin_wins.py. |
| 14-F2 | 14_feature_selection-5.md | FIXED | Target imbalance of a float target counts labelled rows only; NaN is no longer a class. Test tests/feature_selection/test_auto_tune_imbalance_ignores_missing_labels.py; 22 existing auto-tune tests pass. Commit below. |
| 15-F1 | 15_feature_selection-6.md | FIXED | A NaN estimator score makes the fold score NaN (worst case cannot be read from the other estimators); a fully scored fold still takes the minimum. Test tests/feature_selection/test_rfecv_fold_min_counts_failed_estimators |
| 15-F2 | 15_feature_selection-6.md | FIXED | The dispatcher completes ragged per-run tables with fi_missing_policy before the tree and linear aggregators run, as the legacy path does. Test tests/feature_selection/test_dispatched_agg_imputes_ragged_runs.py; 3 existi |
| 17-F1 | 17_metrics.md | FIXED | MaxError kernels propagate NaN residuals (x != x check in the three comparison sites of both regression modules), matching MAE/MSE. Test tests/metrics/test_max_error_propagates_nan.py; 142 existing regression metric test |
| 17-F2 | 17_metrics.md | FIXED | variance_weighted R2 with all-constant outputs averages the per-output sklearn scores (1.0 for a perfect fit) instead of returning 0.0. Tests tests/metrics/test_r2_variance_weighted_constant_targets.py; 74 existing R2 te |
| 17-F3 | 17_metrics.md | FIXED | fast_tweedie_deviance (power 0) checks lengths and shapes before the difference, so a column prediction no longer broadcasts to N*N pairs. Test tests/metrics/test_tweedie_power_zero_length_check.py; 18 existing deviance ... |
| 17-F4 | 17_metrics.md | FIXED | Weighted regression kernels check sample_weight against the row count before indexing (the three weight-taking entry points); the deviance wrappers were already covered by the shape check in 17-F3. Test tests/metrics/tes |
| 17-F5 | 17_metrics.md | FIXED | accuracy_ratio returns NaN when any score is non-finite, instead of a ratio built from NaN rows as tie blocks. Test tests/metrics/test_accuracy_ratio_nan_scores.py; 28 existing CAP/AR tests pass. Commit below. |
| 17-F6 | 17_metrics.md | FIXED | apply_cutpoints refuses non-finite predictions (integer grades cannot carry NaN) instead of grading NaN as the top class. Tests tests/metrics/classification/test_apply_cutpoints_refuses_nan.py; the existing ordinal-cutpo |
| 17-F7 | 17_metrics.md | FIXED | Multiclass confusion wrappers (MCC and the confusion block) refuse labels outside [0, K) before the kernel, which skips them silently. Tests tests/metrics/test_multiclass_confusion_refuses_out_of_range_labels.py; 36 existing multiclass/MCC tests pass. Commit below. |
| 17-F8 | 17_metrics.md | FIXED | fast_calibration_metrics checks that y_true and y_pred have the same length before its serial kernel (inline, since the body is compiled). Test tests/metrics/test_fast_calibration_metrics_length_checked.py; 99 existing c |
| 17-F9 | 17_metrics.md | DEFERRED | Degenerate reports use ICE = 1.0 as a worst-case placeholder, but a real model's ICE can exceed 1.0, so the placeholder can beat it. The report contract (module docstring) promises finite floats to consumers (plots, threshold search, string builders); choosing a larger finite placeholder or NaN changes that contract for every caller, so it waits for a decision. |
| 18-F1 | 18_models.md | FIXED | Confidence threshold uses the finite uncertainties only; a NaN row is never confident, so one missing member no longer empties the confident set (on by default). Test tests/models/test_confidence_estimates_ignore_nan_row |
| 18-F2 | 18_models.md | PARTIAL | Multiclass member probabilities now get an explicit 'not supported' warning; per-class weighting not built. Test test_cluster18_multiclass_gate_skip_is_explicit.py. |
| 18-F3 | 18_models.md | FIXED | Target-MAE gate reads a binary (n, 2) predict_proba as its positive column (_as_gate_vector), not as 2n flattened values, so the MAE is no longer NaN and the block is no longer skipped. Test tests/models/test_gate_vector |
| 18-F4 | 18_models.md | FIXED | Level-1 member tuples are unwrapped to the entry namespace in the probability accessors. Test test_cluster18_level_member_tuple_unwrapped.py. |
| 18-F5 | 18_models.md | FIXED | batch_numaggs passes whiten_means=False as the serial path does. Test tests/models/test_batch_numaggs_matches_serial_whitening.py is a source-level guard (weak: it checks the call text, not the output); 74 existing numag |
| 20-F1 | 20_reporting-1.md | WON'T FIX | Not reproduced: the WORM verdict already labels a long positive-tail residual RIGHT-SKEW (test_worm_plot_verdict_is_operative.py pins it). |
| 20-F2 | 20_reporting-1.md | FIXED | Implied AUC of a Fisher J is Phi(sqrt(J)/sqrt(2)) for J as the squared Mahalanobis distance; the earlier Phi(sqrt(J)/2) understated every separability figure. Test tests/reporting/test_implied_auc_formula.py; 25 existing |
| 20-F3 | 20_reporting-1.md | FIXED | NDCG bootstrap builds its resample gather in row blocks (bounded memory) with the same seeded draws. Test tests/reporting/test_bootstrap_ndcg_ci_blocks.py matches a row-by-row reference; 115 existing LTR/bootstrap tests ... |
| 20-F4 | 20_reporting-1.md | FIXED | Calendar encodings are used as computed on the non-missing rows; the second full-length mask index (IndexError on any NaT, swallowed by the caller) is gone. Test tests/reporting/test_calendar_features_with_nat.py; 4 existing calendar tests pass. Commit below. |
| 20-F5 | 20_reporting-1.md | FIXED | Target drift verdict does not run the binary class-1 proportion test for multiclass targets; it reports the mean shift and states that no drift flag applies. Test tests/reporting/test_target_drift_multiclass_no_false_fla |
| 20-F6 | 20_reporting-1.md | FIXED | Adversarial validation on a frame with no scalar column returns an annotated skip instead of fitting a 0-feature model. Test tests/reporting/test_adversarial_all_embedding_frame.py; 150 existing adversarial/drift tests p |
| 21-F1 | 21_reporting-2.md | FIXED | Empty label arrays map to an empty index instead of raising at classes.min(), so the designed no-data path is reached. Test tests/reporting/test_risk_coverage_empty_labels.py; 28 existing risk-coverage tests pass. Commit |
| 21-F2 | 21_reporting-2.md | FIXED | Slice finder feeds the 3-way combo builder the feature rows that match the finite errors (mat filtered with row_mask on the NaN path). Test tests/reporting/test_slice_finder_nan_errors_arity3.py; 50 existing slice tests ... |
| 21-F3 | 21_reporting-2.md | FIXED | Per-instance and interaction SHAP builders skip a multi-output CatBoost model before shap's TreeExplainer, as shap_panels already does. Test tests/reporting/test_shap_multioutput_catboost_is_skipped.py (monkeypatched; it |
| 21-F4 | 21_reporting-2.md | FIXED | Scatter subsample past the cap slices symmetric and asymmetric per-point error arrays with the same indices (on a copy of the spec). Test tests/reporting/test_scatter_subsample_keeps_error_bars_aligned.py; 61 existing sc |
| 22-F1 | 22_root_modules-1.md | FIXED | Independence summary reads 1.0 (fully redundant) when every member is constant, instead of 0.0 (independent), so odds_ratio_combine warns. Test tests/calibration/test_independence_all_constant_members_is_redundant.py; 30 |
| 22-F2 | 22_root_modules-1.md | FIXED | A calibrator whose OOF outputs are non-finite is excluded from candidate scoring (logged), instead of being scored over its finite rows. Test tests/calibration/test_candidate_with_nan_outputs_is_excluded.py; 340 existing |
| 22-F3 | 22_root_modules-1.md | FIXED | Each optional calibration library is imported separately; an absent one skips only its own candidates. Test test_cluster22_missing_calibrator_library_skipped.py. |
| 22-F4 | 22_root_modules-1.md | FIXED | Confidence shrinkage fully shrinks an output with no confidence (weight 0), including when the confidence map is empty; the early return that gave raw predictions is removed, and a segment without a confidence is also fu |
| 22-F5 | 22_root_modules-1.md | FIXED | Band correction clips only the corrected in-band values; out-of-band predictions are unchanged, as the docstring states. Test tests/calibration/test_band_correction_leaves_out_of_band_unchanged.py; 7 existing band tests ... |
| 22-F6 | 22_root_modules-1.md | FIXED | replay_cv_results returns the loaded mapping; the loop variable no longer shadows it. The existing replay assertions had encoded the bug (they checked for the inner run's key) and now assert the loaded structure. Tests ... |
| 22-F7 | 22_root_modules-1.md | FIXED | Probability generator names only the n_informative columns that reach X, so n_informative < n_classes no longer crashes. Test tests/data/test_synthetic_probability_fewer_informative_than_classes.py; 16 existing synthetic |
| 22-F8 | 22_root_modules-1.md | FIXED | Stratified bootstrap groups are keyed by the label value itself (a list of per-label index arrays), so fractional labels no longer merge. Test tests/evaluation/test_stratified_bootstrap_fractional_labels.py; 72 existing ... |
| 23-F1 | 23_root_modules-2.md | FIXED | When more than a quarter of resamples fail, the bootstrap CI spans the surviving range (as the docstring documents) and the result carries degraded=True. Test tests/evaluation/test_bootstrap_degraded_ci_spans_survivors.p |
| 23-F3 | 23_root_modules-2.md | FIXED | Missing group ids (NaN/None) are one group in the leakage guard, so a shared missing id across a split raises. Test tests/evaluation/test_group_leakage_nan_ids.py; 43 existing leakage tests pass. Commit below. |
| 23-F4 | 23_root_modules-2.md | FIXED | rank_agreement counts only concordant pairs (both orderings equal); a tie in either ranking is not agreement. Test tests/evaluation/test_rank_agreement_ignores_ties.py; 23 existing blend-selection tests pass. Commit belo |
| 24-F2 | 24_root_modules-3.md | FIXED | timeseries walk kernels now return exclusive end including the crossing element (forward) and inclusive start for the backward walk; regression tests in tests/feature_engineering/test_cumsum_walk_window_bounds.py; test_c |
| 24-F3 | 24_root_modules-3.md | FIXED | backward walk now visits index 0 (range covers right_index-1..0); covered by test_backward_walk_reaches_index_zero |
| 24-F4 | 24_root_modules-3.md | FIXED | integer input promoted to float64 in rolling_moving_average; test_rolling_moving_average_integer_input.py |
| 24-F5 | 24_root_modules-3.md | FIXED | adjust=True now equals pandas ewm(adjust=True) via zero-seed numerator over 1-(1-alpha)^(t+1); test_ewma_adjust_is_causal.py; test_ewma_residual_recurrence_njit reference updated |
| 24-F6 | 24_root_modules-3.md | FIXED | CUSUM reference mean is the expanding mean of prior rows; auto threshold is MAD of the first 30 rows; test_cusum_is_causal.py; test_cusum_walk_njit reference updated to causal form |
| 24-F7 | 24_root_modules-3.md | FIXED | NaN/None states no longer count as possession; test_state_duration_and_sentinel_guards.py |
| 24-F8 | 24_root_modules-3.md | FIXED | sentinel detection is numeric-only, non-bool, and needs at least 3 other levels; same test file |
| 24-F9 | 24_root_modules-3.md | FIXED | IDW, label-dispersion and gradient drop the query point itself when query is the reference pool (shared _drop_self_match); tests/feature_engineering/test_spatial_query_is_ref_and_nan_queries.py |
| 24-F10 | 24_root_modules-3.md | FIXED | knn_within_bucket_aggregate drops the self-match when query is reference; same test file |
| 24-F11 | 24_root_modules-3.md | FIXED | knn_gradient uses zero-filled coords for the design matrix and marks non-finite query rows singular, so no NaN reaches the batched SVD/solve; same test file |
| 24-F12 | 24_root_modules-3.md | FIXED | rolling_longest_monotone_run returns NaN for windows containing a non-finite value; tests/feature_engineering/test_feature_edge_cases_24_f12_to_f15.py |
| 24-F13 | 24_root_modules-3.md | FIXED | nearest_past_join indexes unresolved rows through the positional copy; same test file |
| 24-F14 | 24_root_modules-3.md | FIXED | create_ts_features_parallel returns (None, None) for an empty span before dividing; same test file |
| 24-F15 | 24_root_modules-3.md | FIXED | mps shift fills the vacated tail from the last shifted position; same test file |
| 24-F16 | 24_root_modules-3.md | DEFERRED | Poly weights depend on window length; exact O(1) update not possible; cap changes results. Owner decision. |
| 24-F17 | 24_root_modules-3.md | DEFERRED | Causal KDE recomputed per prefix; bounded lookback changes the estimate. Owner decision. |
| 24-F18 | 24_root_modules-3.md | FIXED | NaN entity kept as its own group (dropna=False); test_tfidf_entity_nan_and_case.py |
| 24-F19 | 24_root_modules-3.md | FIXED | OOV check compares lowercased tokens when the vectorizer lowercases; same test file |
| 24-F20 | 24_root_modules-3.md | FIXED | with fit_on_train_only the member stage receives the train partition, matching MRMR |
| 24-F21 | 24_root_modules-3.md | FIXED | corr_clusters divides Z.T@Z by n (population sd); test_hybrid_ddof_and_raw_labels.py |
| 24-F22 | 24_root_modules-3.md | FIXED | cohort helper returns raw column labels, so non-string labels match; same test file |
| 25-F1 | 25_root_modules-4.md | FIXED | weighted quantile above the last midpoint returns the group max; tests/inference/test_entity_collapse_quantile_above_last_midpoint.py |
| 25-F3 | 25_root_modules-4.md | FIXED | label_ranking_average_precision scores degenerate rows 1.0 and keeps them in the mean (sklearn); tests/metrics/test_multilabel_extras.py compares full arrays |
| 25-F4 | 25_root_modules-4.md | FIXED | label_ranking_loss keeps degenerate rows with loss 0 in the mean (sklearn) |
| 25-F5 | 25_root_modules-4.md | FIXED | multilabel inference remap also for two columns (len(probs) >= 2); tests/metrics/test_multiclass_labels_fairness_bins.py |
| 25-F6 | 25_root_modules-4.md | FIXED | isinstance(str) before the special-name membership test; same test file |
| 25-F7 | 25_root_modules-4.md | FIXED | excluded rare bins become NaN in bins; same test file |
| 25-F8 | 25_root_modules-4.md | FIXED | bootstrap distribution and resampler factory check length and return NaN for NaN scores (sibling convention); tests/metrics/test_auc_entry_checks_and_thread_safe_resampler.py |
| 25-F9 | 25_root_modules-4.md | FIXED | resampler scratch buffers are thread-local; concurrency regression in the same file |
| 25-F10 | 25_root_modules-4.md | FIXED | fast_aucs_per_group_optimized checks length, refuses NaN with ValueError and normalises labels; same file |
| 25-F11 | 25_root_modules-4.md | FIXED | binary label normaliser raises for more than two distinct labels; same file |
| 25-F12 | 25_root_modules-4.md | FIXED | non-finite values are still dropped, now logged with counts (warn option of the finding); tests/metrics/test_cluster25_entry_checks_and_fold_fits.py |
| 25-F13 | 25_root_modules-4.md | FIXED | MAPE checks length before the kernels; same test file |
| 25-F14 | 25_root_modules-4.md | FIXED | XGBoost interaction path uses native_contrib_kwargs for DMatrix and predict; same test file |
| 25-F15 | 25_root_modules-4.md | FIXED | eval_fraction splits within the training fold; the test fold never reaches fit; same test file |
| 25-F16 | 25_root_modules-4.md | FIXED | ETR no longer receives the CatBoost iterations key; same test file |
| 25-F17 | 25_root_modules-4.md | FIXED | isotonic refit slices sorted contiguous group segments (one sort); same test file checks per-group monotonicity |
| 25-F18 | 25_root_modules-4.md | FIXED | a column is a drop candidate only if unimportant in every training fold, importances on training rows only. Residual: the fold's held-out rows still shape the other folds' masks; a fully nested variant is a design option |
| 25-F19 | 25_root_modules-4.md | FIXED | NaN epochs are skipped before the baseline and weights; a RuntimeWarning fires when no epoch is finite; same test file |
| 26-F1 | 26_root_modules-5.md | FIXED | non-mutating cleaning path assigns per column (no assign(**kwargs)); tests/preprocessing/test_cluster26_preprocessing_fixes.py |
| 26-F2 | 26_root_modules-5.md | FIXED | dedup gate tests only float NaN, so pd.NA no longer hits bool(); same file |
| 26-F3 | 26_root_modules-5.md | FIXED | tree branch scores against supplied bounds as the non-tree branch does; same file |
| 26-F5 | 26_root_modules-5.md | FIXED | auto-discovered non-numeric columns take mode under median/mean; same file |
| 27-F1 | 27_root_modules-6.md | FIXED | edge blend reads a snapshot of the hard predictions; tests/training/test_direct_horizon_edge_blend_from_hard.py |
| 27-F2 | 27_root_modules-6.md | FIXED | polars persist path fits a real SimpleImputer, so sklearn's dtype state is set; covered by tests/inference/test_predict_nan_guard_polars_native.py and the schema drift matrix |
| 27-F3 | 27_root_modules-6.md | FIXED | fingerprint digests __slots__ and __dict__ state for objects without get_params; tests/training/test_cluster27_fingerprint_threshold_backends.py |
| 27-F4 | 27_root_modules-6.md | FIXED | positive class is the larger label (y == max) in _counts_at and the minority count; same file |
| 27-F5 | 27_root_modules-6.md | FIXED | duplicated buckets read member slices from one stable argsort; covered by test_biz_val_training_pseudo_group_reconstruction.py |
| 27-F6 | 27_root_modules-6.md | FIXED | cb/lgb RFECV built only when the class is installed and the backend is requested, else None; same file tests the helper |
| 28-F1 | 28_root_modules-7.md | FIXED | NaT rows dropped before binning, order maps back to original rows; tests/training/test_cluster28_suite_fixes.py |
| 28-F3 | 28_root_modules-7.md | FIXED | single-class y_true uses label 1 as the positive class, matching probs column 1; no separate test (internal branch of the exploss metric) |
| 28-F4 | 28_root_modules-7.md | FIXED | sort mode returns NaN rows verbatim with a RuntimeWarning; same test file |
| 28-F5 | 28_root_modules-7.md | FIXED | TTA jitter uses nanstd (NaN-safe) at all three sites; same test file |
| 28-F8 | 28_root_modules-7.md | FIXED | dotted components checked against denylist prefixes in the older loader; same test file |
| 28-F9 | 28_root_modules-7.md | FIXED | write_ added to denied prefixes; same test file |
| 28-F10 | 28_root_modules-7.md | FIXED | automl_target_label defaults to None and the run target name is used; same test file |
| 28-F13 | 28_root_modules-7.md | FIXED | fallback retry goes through _train_model_with_f_phase; no separate test |
| 28-F15 | 28_root_modules-7.md | FIXED | drift cache key digests the full content (blake2b over row hashes); same test file |
| 28-F18 | 28_root_modules-7.md | FIXED | the unimplemented carrier stamp is replaced by a logged warning; no separate test |
| 29-F1 | 29_root_modules-8.md | FIXED | -P only on Python 3.11+ (empty flag tuple on older interpreters); tests/utils/test_native_import_probe.py and tests/utils/test_cluster29_utils_fixes.py |
| 29-F2 | 29_root_modules-8.md | FIXED | strided sample taken before the dense copy; n keeps the true row count; same test file |
| 29-F3 | 29_root_modules-8.md | FIXED | unmeasurable dimensions bucket to the string marker nan, not 0.0; same test file |
| 29-F4 | 29_root_modules-8.md | FIXED | structured/void dtype hashes the whole buffer; same test file |
| 29-F5 | 29_root_modules-8.md | FIXED | preprocess_glue NaN guard matches preprocess_sglue; no separate test (votenrank suite passed) |
| 29-F6 | 29_root_modules-8.md | FIXED | array-likes are written as a frame, other types raise TypeError; same test file |
| 31-F1 | 31_system.md | FIXED | -P gated on Python 3.11+; PYTHONPATH still carries the import root; no separate test (subprocess command construction) |
| 31-F2 | 31_system.md | FIXED | an ok sweep that persisted 0 regions counts as failed; cmd_refresh returns non-zero for 0; tests/system/test_cluster31_kernel_tuning_fixes.py |
| 31-F3 | 31_system.md | FIXED | lock records pid:start-token; a live pid with a mismatching token is stale; same test file; test_kernel_tuning_lock_race.py assertion updated to the pid field |
| 32-F1 | 32_training-1.md | FIXED | an empty rows list returns an empty frame before set_index; no separate test (LTR skip path) |
| 32-F2 | 32_training-1.md | FIXED | a probability matrix takes the true-class probability and its bootstrap CI; tests/training/test_cluster32_baseline_fixes.py |
| 32-F3 | 32_training-1.md | FIXED | one validation shard falls back to single-validation early stopping with one warning; no separate test |
| 32-F4 | 32_training-1.md | FIXED | a non-finite global mean returns the minimum pseudocount (finite); tests/training/test_cluster32_baseline_fixes.py. Residual: the finding's preferred option, dropping non-finite y before the estimate, is not implemented; |
| 32-F5 | 32_training-1.md | FIXED | nanmedian and nanquantile for the quantile baselines; no separate test |
| 32-F6 | 32_training-1.md | FIXED | nanstd for the cross-output scale, non-finite scales skipped, non-finite normalized means excluded before the pick; no separate test |
| 33-F1 | 33_training-2.md | FIXED | the CatBoost categorical NaN fill failure is logged at WARNING with the error type; the except is kept broad (not narrowed) so the fill never raises into predict; no separate test |
| 33-F2 | 33_training-2.md | FIXED | label-stacking failure logged at WARNING; no separate test |
| 40-F6 | 40_training-9.md | FIXED | unseen validation labels raise instead of mapping to a neighbouring class; no separate test beyond the suite run |
| 40-F12 | 40_training-9.md | FIXED | kwargs are filtered by signature and the fit is called once, so a TypeError inside a fit propagates; tests/training/test_cluster40_pipeline_fixes.py |
| 40-F14 | 40_training-9.md | FIXED | duplicated index labels no longer multiply rows: both FE joins are positional; tests/training/test_cluster40_pipeline_fixes.py |
| 40-F19 | 40_training-9.md | FIXED | array-valued parameters are keyed by shape, dtype and content digest; tests/training/test_cluster40_pipeline_fixes.py |
| 40-F27 | 40_training-9.md | FIXED | Per-column float32 fallback rebuilds the frame from its labels; integer column labels no longer raise. Test test_cluster40_int_column_labels_convert.py. |
| 40-F22 | 40_training-9.md | PARTIAL | fit refuses NaN/inf relevance labels; predict checks feature count. Column names and order not checked. Test test_cluster40_ranker_refuses_nan_labels_and_bad_width.py. |
| 40-F29 | 40_training-9.md | PARTIAL | Config refuses bad batch_size, sequence_preprocessing, ema_decay, num_classes at construction. mixup_alpha not checked (0 disables mixup). Test test_cluster40_recurrent_config_validates_fields.py. |
| 40-F28 | 40_training-9.md | FIXED | transform adds PySR columns to a shallow copy; the caller's frame is not mutated. Test test_cluster40_pysr_transform_does_not_mutate_input.py. |
| 40-F23 | 40_training-9.md | FIXED | Join looks up each unique id once with normalised keys; 3 matches 3.0; unknown and NaN ids zero-filled. Test test_cluster40_embedding_join_normalised_keys.py. |
| 40-F30 | 40_training-9.md | FIXED | Weighted classification loss divides by class-times-sample weights when class weights are set, matching the unweighted CrossEntropy mean. Test test_cluster40_class_weighted_loss_scale.py. |
| 40-F25 | 40_training-9.md | DEFERRED | Fit-time lookup raw-key vs replay normalised-key mismatch. Not changed: no discriminating case found (encoder returns the prior for the tested entities). Needs the terminal-encoding semantics. |
| 40-F20 | 40_training-9.md | FIXED | Frame content hash hashes per-row hash bytes in row order, not their sum; row-permuted frames no longer share a key. Test test_cluster40_frame_hash_is_row_order_sensitive.py. |
| 40-F21 | 40_training-9.md | FIXED | Trunk-residual and field-grouped regressors refuse NaN/inf X at fit and predict, as Tabular1DCNN does. Test test_cluster40_residual_regressors_refuse_nonfinite.py. |
| 40-F24 | 40_training-9.md | FIXED | With keep_cols pinned, the nullable-bool scan covers only the pinned columns. Test test_cluster40_filter_numeric_keep_path.py pins the behaviour; the speed-up is not measured. |
| 40-F26 | 40_training-9.md | DEFERRED | Per-entity history restarts per split; changing it shifts feature distributions. Owner decision on history semantics. |
| 40-F31 | 40_training-9.md | DEFERRED | Stratified sampler duplicated; owner decision on the canonical copy before consolidating. |
| 41-F1 | 41_training-10.md | FIXED | Predict maps categorical-dtype NaN to the fit-time sentinel. Test test_cluster41_catboost_predict_nan_categorical.py. |
| 41-F2 | 41_training-10.md | FIXED | verbose NDCG logging uses .get so an eval_at without 10 cannot raise; no separate test |
| 41-F3 | 41_training-10.md | FIXED | NaN temporal labels count as missing in the polars path, matching the pandas twin; no separate test |
| 41-F4 | 41_training-10.md | FIXED | Skew check now runs before the kurtosis-band returns; a skewed residual is no longer labelled Near-Gaussian. Test test_cluster41_skew_not_masked_by_kurtosis_band.py. |
| 41-F5 | 41_training-10.md | FIXED | Group loops use one stable sort and contiguous slices instead of O(G*n) masks. Test test_cluster41_group_stats_match_reference.py. |
| 41-F6 | 41_training-10.md | FIXED | GroupKFold with fewer groups than k returns no shards, as the docstring promises; no separate test |
| 41-F13 | 41_training-10.md | FIXED | Failing normality test logged at WARNING; the fallback is visible. Test test_cluster41_normality_fallback_warns.py. |
| 41-F27 | 41_training-10.md | FIXED | A failing collapse sensor is logged at WARNING, not debug. Test test_cluster41_collapse_sensor_failure_warns.py. |
| 42-F1 | 42_votenrank.md | FIXED | tie-number keys use the tolerant PRETTY_NAMES lookup like the result rename; no separate test |
| 42-F2 | 42_votenrank.md | FIXED | sub-leaderboards inherit allow_partial and partial_fill; no separate test |
