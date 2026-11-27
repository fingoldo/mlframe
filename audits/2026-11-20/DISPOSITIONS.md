# Disposition log (every finding, one line each)

| ID | Status | Note |
|---|---|---|
| 01-F3 | WON'T FIX | Production loop already measured against the argsort variant in probabilities.py (argsort 0.43x at n=1e6, n_bins=10); the O(n_bins*N) scan stays. Bench finding is about the bench, not the shipped code. |
| 04-F1 | RESOLVED | count branch uses count_mean(scale*unit_score), intercept no longer added twice. Test tests/data/datasets/test_count_target_intercept_once.py. Commit below. |
| 04-F11 | DEFERRED | Published Friedman-2 is sqrt(x0^2 + (x1*x2 - 1/(x1*x3))^2) and Friedman-3 is atan((x1*x2 - 1/(x1*x3))/x0). The link vocabulary (BasisTerm kinds) has no 1/(a*b) term and no compound sqrt/atan form, so neither can be written exactly. Needs new basis kinds and a decision on the provenance text. |
| 04-F2 | RESOLVED | TargetSpec rejects a logloss or accuracy ceiling on a binary target (was silently ignored). Tests tests/data/datasets/test_binary_ceiling_metric_rejected.py. Commit below. |
| 04-F7 | RESOLVED | realize_latent refuses an unknown family instead of drawing Student-t(4). Tests tests/data/datasets/test_latent_unknown_family_rejected.py. Commit below. |
| 04-F12 | RESOLVED | Simpson group is a fair sign (new rademacher family, -1/+1), so the interaction reverses sign between subgroups as documented. Tests tests/data/datasets/test_simpson_group_is_a_sign.py; existing simpson bed test passes. Commit below. |
| 04-F9 | RESOLVED | mi_reference memo key includes a blake2b digest of the column data and labels. Test tests/data/datasets/test_truth_mi_memo_keys_on_data.py. Commit below. |
| 04-F10 | RESOLVED | count_ceiling sums every row exactly in blocks (the 2001-row prefix extrapolation is gone). Test tests/data/datasets/test_count_ceiling_sums_every_row.py; 5 existing count tests pass. Commit below. |
| 04-F6 | RESOLVED | Convention fixed in _declared_kind_role: copy/proxy kinds mark the target, the instrument kind marks its source (instrument -> treatment is causal). Test tests/data/datasets/test_declared_roles_follow_the_edge_convention.py. Commit below. |
| 04-F8 | RESOLVED | _exact_mi_independent averages the row entropies within each bin (H(Y/X in bin) = mean H(p_i)) instead of the entropy at the bin-mean probability. Test tests/data/datasets/test_exact_mi_averages_row_entropy.py. Commit below. |
| 04-F4 | RESOLVED | link_regions passes the link's regional gate to every column it reads, so FeatureTruth.region is set at both generator call sites. Test tests/data/datasets/test_link_region_reaches_feature_truth.py. Commit below. |
| 04-F3 | RESOLVED | Multiclass class scores come from multiclass_class_scores: rotated additive weights per class plus the link's interactions and basis terms for every class. Tests tests/data/datasets/test_multiclass_scores_keep_interactions.py; existing multiclass tests pass. Commit below. |
| 04-F5 | RESOLVED | Latent reflections are now graph edges (latent -> reflected column), so the M-bias collider sees its second arm and gets role M_COLLIDER. Also: instrument-convention test aligned with 04-F6; explicit t latent family kept as Student-t(4). Tests tests/data/datasets/test_mbias_collider_sees_the_reflected_arm.py; related tests pass. Commit below. |
| 08-F1 | RESOLVED | build_nca_projection fits NCA on the rows it is applied to: the internal StandardScaler (fit then discarded) is removed. Test tests/feature_engineering/transformer/test_nca_projection_lives_in_the_applied_space.py; existing nca selection passes. Note: the unit test uses production-standardised input, so it does not separate old from new behaviour on its own. Commit below. |
| 08-F2 | RESOLVED | All 15 Mode-A out-of-fold output arrays start as NaN (baseline_surprise, adversarial_flip, anomaly_score_features, class_distance, and the 11 grep-found files), so an uncovered row is NaN, not 0.0. Test tests/feature_engineering/transformer/test_oof_uncovered_rows_are_nan.py; 15 selected OOF tests pass. Commit below. |
| 09-F4 | RESOLVED | Jackknife endpoint weights are no longer proportional: the upper endpoint favours positives and the lower negatives. Test tests/feature_engineering/transformer/test_jackknife_endpoints_weight_classes_differently.py. Commit below. |
| 09-F5 | RESOLVED | Regression quantile heads of the spread fan are sorted per row, so q10 <= q50 <= q90 holds. Test tests/feature_engineering/transformer/test_quantile_fan_is_ordered.py. Commit below. |
| 09-F6 | RESOLVED | Degenerate folds return NaN (not 0.0) in density_weighted_smote, pure_pos_smote and pseudo_smote. Test tests/feature_engineering/transformer/test_smote_degenerate_fold_is_nan.py covers the density-weighted path. Commit below. |
| 09-F8 | RESOLVED | Positional encoding no longer applies a 1e6 modulus, so distinct positions are not aliased; precision in float32 is documented and controlled by dtype. Test tests/feature_engineering/transformer/test_positional_encoding_does_not_alias_positions.py; existing positional tests pass. Commit below. |
| 09-F9 | RESOLVED | Geodesic minimum over targets uses one virtual-source Dijkstra (O(rows) memory) instead of a dense (targets x rows) matrix. Test tests/feature_engineering/transformer/test_geodesic_min_over_sources.py equals the per-source minimum; existing geodesic tests pass. Commit below. |
| 09-F11 | RESOLVED | Binary MDL binning refuses a target with more than two labels (was mis-indexing the two-class count arrays). Test tests/feature_engineering/transformer/test_mdl_binary_refuses_three_labels.py. Commit below. |
| 09-F12 | RESOLVED | Empty correct-subset distances are the package far sentinel (1e6) for both mean and nearest, in a module-level _nn_dist_to_subset. Test tests/feature_engineering/transformer/test_trust_empty_subset_is_far.py. Commit below. |
| 09-F13 | RESOLVED | Local PCA centres each neighbour matrix on the neighbour mean instead of the query row. Test tests/feature_engineering/transformer/test_lid_centres_on_the_neighbour_mean.py. Commit below. |
| 09-F10 | RESOLVED | Top-k proximity selects the largest k directly instead of partitioning a negated copy of the dense block. The dense block itself stays: the measured note in the code shows every pair shares a leaf. Test tests/feature_engineering/transformer/test_rf_topk_matches_full_sort.py; existing proximity tests pass. Commit below. |
| 10-F3 | RESOLVED | logreg_holdout_auc returns chance (0.5) for an empty selection instead of scoring all columns. Test tests/feature_selection/test_bench_logreg_empty_selection_is_chance.py. Commit below. |
| 10-F1 | RESOLVED | Adaptive-guards bench scores the refit on the untouched holdout split; the concatenated re-split is gone. Test tests/feature_selection/test_bench_holdout_auc_is_unseen.py. Commit below. |
| 10-F2 | RESOLVED | honest_auc fits on the search rows and scores the untouched holdout, at all three call sites. Test tests/feature_selection/test_bench_interaction_holdout_is_unseen.py. Commit below. |
| 11-F2 | RESOLVED | RFECV empty selection scores chance (0.5) in the importance-aggregation bench instead of refitting on every column. Test tests/feature_selection/test_bench_rfecv_empty_selection_is_chance.py. Commit below. |
| 11-F1 | RESOLVED | _integer_bins ranks finite values only; missing rows take bin 0 instead of the top bin. Test tests/feature_selection/test_bench_integer_bins_missing_not_top.py. Commit below. |
| 12-F1 | RESOLVED | Staleness default root walks up to the .git folder instead of the module directory, so the git paths match. Test tests/feature_selection/test_staleness_repo_root_is_the_git_root.py. Commit below. |
| 12-F19 | RESOLVED | Bench arms are cap60 (the old MRMR width) and cap250; the earlier None arm equalled MRMR's default 250 and so duplicated cap250. Test tests/feature_selection/test_mrmrcap_bench_arms_differ.py. Commit below. |
| 12-F20 | DEFERRED | An RFECV that selects nothing is replaced by the first column and recorded as its selection (importance_shootout.py cell loop). The fix (empty selection scores chance) is a one-branch change, but the cell loop runs a full RFECV and cannot be tested in isolation without first extracting its scoring into a helper; deferred until that extraction is done. |
| 12-F21 | RESOLVED | cv_curve takes a ranker and ranks each fold on its training rows; the noise-floor bench passes lgbm_gain_ranking. Test tests/feature_selection/test_noise_floor_ranks_inside_folds.py. Commit below. |
| 12-F22 | RESOLVED | Robust noise-floor bench passes a per-fold gain ranker (_gain_order) to cv_curve, so folds rank their own training rows. Test tests/feature_selection/test_noise_floor_robust_gain_order.py. Commit below. |
| 12-F23 | DEFERRED | The baseline filters raw_selected_ to raw columns, dropping engineered names. The fix must use the hybrid's transform output (Ztr = h.transform(Xtr)) and filter against it. Confirming which names survive takes a full HybridSelector fit (minutes), so it cannot be covered by a cheap test here; deferred until a cheap fixture for the transform output exists. |
| 12-F24 | RESOLVED | Protected core is the intersection of Boruta and MRMR accepts (agreed_core), matching the docstring. Test tests/feature_selection/test_protected_core_is_the_intersection.py. Commit below. |
| 12-F25 | DEFERRED | Whether RFECV with max_runtime_mins=3 timed out on this data is only visible from a real run (cv_results_ length); the fix (fail or flag a timed-out run) needs that run to be defined. Deferred. |
| 13-F2 | RESOLVED | The stability path builds the shared fit state before returning, so Subset, results_to_csv and plot read set attributes. Test tests/feature_selection/test_boruta_stability_path_has_fit_state.py; existing stability tests (6) pass. Commit below. |
| 13-F1 | RESOLVED | Under premerge, Subset() expands accepted representatives back to their cluster members and indexes the pre-collapse frame. Test tests/feature_selection/test_boruta_subset_keeps_premerge_members.py; existing premerge tests (3) pass. Commit below. |
| 14-F2 | RESOLVED | Target imbalance of a float target counts labelled rows only; NaN is no longer a class. Test tests/feature_selection/test_auto_tune_imbalance_ignores_missing_labels.py; 22 existing auto-tune tests pass. Commit below. |
| 14-F1 | DEFERRED | The auto tightening compares the value to its default (3.0), so an explicit 3.0 is overridden. Telling explicit from default needs a None sentinel in the public signature, and the training config (training/fs_params/shap_proxied_fs.py:56) always passes 3.0, so the sentinel would never reach the tightening. That is a config-design change: deferred. |
| 15-F1 | RESOLVED | A NaN estimator score makes the fold score NaN (worst case cannot be read from the other estimators); a fully scored fold still takes the minimum. Test tests/feature_selection/test_rfecv_fold_min_counts_failed_estimators.py; 33 existing RFECV fold tests pass. Commit below. |
| 15-F2 | RESOLVED | The dispatcher completes ragged per-run tables with fi_missing_policy before the tree and linear aggregators run, as the legacy path does. Test tests/feature_selection/test_dispatched_agg_imputes_ragged_runs.py; 3 existing importance-agg tests pass. Commit below. |
| 17-F5 | RESOLVED | accuracy_ratio returns NaN when any score is non-finite, instead of a ratio built from NaN rows as tie blocks. Test tests/metrics/test_accuracy_ratio_nan_scores.py; 28 existing CAP/AR tests pass. Commit below. |
| 17-F1 | RESOLVED | MaxError kernels propagate NaN residuals (x != x check in the three comparison sites of both regression modules), matching MAE/MSE. Test tests/metrics/test_max_error_propagates_nan.py; 142 existing regression metric tests pass. Commit below. |
| 17-F2 | RESOLVED | variance_weighted R2 with all-constant outputs averages the per-output sklearn scores (1.0 for a perfect fit) instead of returning 0.0. Tests tests/metrics/test_r2_variance_weighted_constant_targets.py; 74 existing R2 tests pass. Commit below. |
| 17-F3 | RESOLVED | fast_tweedie_deviance (power 0) checks lengths and shapes before the difference, so a column prediction no longer broadcasts to N*N pairs. Test tests/metrics/test_tweedie_power_zero_length_check.py; 18 existing deviance tests pass. Commit below. |
| 17-F6 | RESOLVED | apply_cutpoints refuses non-finite predictions (integer grades cannot carry NaN) instead of grading NaN as the top class. Tests tests/metrics/classification/test_apply_cutpoints_refuses_nan.py; the existing ordinal-cutpoint tests pass. Commit below. |
| 17-F4 | RESOLVED | Weighted regression kernels check sample_weight against the row count before indexing (the three weight-taking entry points); the deviance wrappers were already covered by the shape check in 17-F3. Test tests/metrics/test_sample_weight_length_checked.py; 192 existing weighted/regression metric tests pass. Commit below. |
| 17-F8 | RESOLVED | fast_calibration_metrics checks that y_true and y_pred have the same length before its serial kernel (inline, since the body is compiled). Test tests/metrics/test_fast_calibration_metrics_length_checked.py; 99 existing calibration metric tests pass. Commit below. |
| 17-F7 | RESOLVED | Multiclass confusion wrappers (MCC and the confusion block) refuse labels outside [0, K) before the kernel, which skips them silently. Tests tests/metrics/test_multiclass_confusion_refuses_out_of_range_labels.py; 36 existing multiclass/MCC tests pass. Commit below. |
| 18-F1 | RESOLVED | Confidence threshold uses the finite uncertainties only; a NaN row is never confident, so one missing member no longer empties the confident set (on by default). Test tests/models/test_confidence_estimates_ignore_nan_rows.py; 238 existing ensembling/predict tests pass. Commit below. |
| 18-F2 | DEFERRED | Multiclass (N, K) member predictions are flattened to N*K and skipped by the length gate, so their NNLS/Caruana weights never get fitted. A fix needs a scoring choice for a multiclass member (true-class probability or multiclass log-loss) and a test through run_stacking_aware_gate, which takes a dozen arguments. Deferred for that design decision. |
| 18-F4 | DEFERRED | Level-1 ensembles are appended as (entry, train_df, val_df, test_df) tuples and level 2+ reads val_probs/oof_probs from the tuple. Unwrapping at append is the natural fix, but downstream readers of the level tuple (the stacking and gate paths) assume its shape, and verifying the change needs an end-to-end score_ensemble run with max_ensembling_level=2. Deferred until that fixture exists. |
| 18-F5 | RESOLVED | batch_numaggs passes whiten_means=False as the serial path does. Test tests/models/test_batch_numaggs_matches_serial_whitening.py is a source-level guard (weak: it checks the call text, not the output); 74 existing numagg tests pass. Commit below. |
| 18-F3 | RESOLVED | Target-MAE gate reads a binary (n, 2) predict_proba as its positive column (_as_gate_vector), not as 2n flattened values, so the MAE is no longer NaN and the block is no longer skipped. Test tests/models/test_gate_vector_uses_positive_column.py; 82 existing gate/stacking tests pass. Commit below. |
| 20-F2 | RESOLVED | Implied AUC of a Fisher J is Phi(sqrt(J)/sqrt(2)) for J as the squared Mahalanobis distance; the earlier Phi(sqrt(J)/2) understated every separability figure. Test tests/reporting/test_implied_auc_formula.py; 25 existing separability tests pass. Commit below. |
| 20-F4 | RESOLVED | Calendar encodings are used as computed on the non-missing rows; the second full-length mask index (IndexError on any NaT, swallowed by the caller) is gone. Test tests/reporting/test_calendar_features_with_nat.py; 4 existing calendar tests pass. Commit below. |
| 20-F1 | DEFERRED | The current labels pass the existing tests: a lognormal (long positive residual tail) reads RIGHT-SKEW, and the swap the audit proposes breaks that test. The asymmetry metric (tail medians of the detrended QQ) is itself unreliable: an upper-tail mixture reads LEFT, so the finding is a metric defect, not a label inversion. Fixing the metric changes the verdicts of a published panel, so it waits for a decision. |
| 20-F6 | RESOLVED | Adversarial validation on a frame with no scalar column returns an annotated skip instead of fitting a 0-feature model. Test tests/reporting/test_adversarial_all_embedding_frame.py; 150 existing adversarial/drift tests pass. Commit below. |
| 20-F3 | RESOLVED | NDCG bootstrap builds its resample gather in row blocks (bounded memory) with the same seeded draws. Test tests/reporting/test_bootstrap_ndcg_ci_blocks.py matches a row-by-row reference; 115 existing LTR/bootstrap tests pass. Commit below. |
| 20-F5 | RESOLVED | Target drift verdict does not run the binary class-1 proportion test for multiclass targets; it reports the mean shift and states that no drift flag applies. Test tests/reporting/test_target_drift_multiclass_no_false_flag.py; 217 existing drift/error-analysis tests pass. Commit below. |
| 21-F1 | RESOLVED | Empty label arrays map to an empty index instead of raising at classes.min(), so the designed no-data path is reached. Test tests/reporting/test_risk_coverage_empty_labels.py; 28 existing risk-coverage tests pass. Commit below. |
| 21-F4 | RESOLVED | Scatter subsample past the cap slices symmetric and asymmetric per-point error arrays with the same indices (on a copy of the spec). Test tests/reporting/test_scatter_subsample_keeps_error_bars_aligned.py; 61 existing scatter tests pass. Commit below. |
| 21-F2 | RESOLVED | Slice finder feeds the 3-way combo builder the feature rows that match the finite errors (mat filtered with row_mask on the NaN path). Test tests/reporting/test_slice_finder_nan_errors_arity3.py; 50 existing slice tests pass. Commit below. |
| 21-F3 | RESOLVED | Per-instance and interaction SHAP builders skip a multi-output CatBoost model before shap's TreeExplainer, as shap_panels already does. Test tests/reporting/test_shap_multioutput_catboost_is_skipped.py (monkeypatched; it would not catch a renamed explainer call); 91 existing SHAP tests pass. Commit below. |
| 22-F6 | RESOLVED | replay_cv_results returns the loaded mapping; the loop variable no longer shadows it. The existing replay assertions had encoded the bug (they checked for the inner run's key) and now assert the loaded structure. Tests: tests/core/test_core_infra_a_fixes.py (34 pass). Commit below. |
| 22-F7 | RESOLVED | Probability generator names only the n_informative columns that reach X, so n_informative < n_classes no longer crashes. Test tests/data/test_synthetic_probability_fewer_informative_than_classes.py; 16 existing synthetic tests pass. Commit below. |
| 22-F5 | RESOLVED | Band correction clips only the corrected in-band values; out-of-band predictions are unchanged, as the docstring states. Test tests/calibration/test_band_correction_leaves_out_of_band_unchanged.py; 7 existing band tests pass. Commit below. |
| 22-F4 | RESOLVED | Confidence shrinkage fully shrinks an output with no confidence (weight 0), including when the confidence map is empty; the early return that gave raw predictions is removed, and a segment without a confidence is also fully shrunk. Test tests/calibration/test_confidence_shrinkage_missing_output_is_shrunk.py; 9 existing shrinkage tests pass. Commit below. |
| 22-F8 | RESOLVED | Stratified bootstrap groups are keyed by the label value itself (a list of per-label index arrays), so fractional labels no longer merge. Test tests/evaluation/test_stratified_bootstrap_fractional_labels.py; 72 existing bootstrap tests pass. Commit below. |
| 22-F1 | RESOLVED | Independence summary reads 1.0 (fully redundant) when every member is constant, instead of 0.0 (independent), so odds_ratio_combine warns. Test tests/calibration/test_independence_all_constant_members_is_redundant.py; 30 existing independence/odds tests pass. Commit below. |
| 22-F2 | RESOLVED | A calibrator whose OOF outputs are non-finite is excluded from candidate scoring (logged), instead of being scored over its finite rows. Test tests/calibration/test_candidate_with_nan_outputs_is_excluded.py; 340 existing calibration policy tests pass. Commit below. |
| 22-F3 | DEFERRED | get_postcalibrators imports netcal, pycalib, ml_insights, betacal and venn_abers unguarded, and the calibrator list references their attributes directly. Making each optional means restructuring the whole zoo list per library. The change is mechanical but touches every entry; deferred to a dedicated pass with a test that blocks each module in turn. |
| 23-F4 | RESOLVED | rank_agreement counts only concordant pairs (both orderings equal); a tie in either ranking is not agreement. Test tests/evaluation/test_rank_agreement_ignores_ties.py; 23 existing blend-selection tests pass. Commit below. |
| 23-F3 | RESOLVED | Missing group ids (NaN/None) are one group in the leakage guard, so a shared missing id across a split raises. Test tests/evaluation/test_group_leakage_nan_ids.py; 43 existing leakage tests pass. Commit below. |
| 23-F2 | RESOLVED | Leaking folds are decided with the same tolerance as the verdict (the caller's leak_tolerance when given), not a fixed 0.02, so leaky_row_positions agree with leak_detected. Regression coverage: the existing 50 expanding/leak tests pass; no new fixture with a real leak was built, so the change is verified by the existing suite and by reading. Commit below. |
| 23-F1 | RESOLVED | When more than a quarter of resamples fail, the bootstrap CI spans the surviving range (as the docstring documents) and the result carries degraded=True. Test tests/evaluation/test_bootstrap_degraded_ci_spans_survivors.py; 73 existing bootstrap tests pass. Commit below. |
| 24-F2 | FIXED | timeseries walk kernels now return exclusive end including the crossing element (forward) and inclusive start for the backward walk; regression tests in tests/feature_engineering/test_cumsum_walk_window_bounds.py; test_coverage_fill end-index expectation updated to the corrected contract |
| 24-F3 | FIXED | backward walk now visits index 0 (range covers right_index-1..0); covered by test_backward_walk_reaches_index_zero |
| 24-F4 | FIXED | integer input promoted to float64 in rolling_moving_average; test_rolling_moving_average_integer_input.py |
| 24-F5 | FIXED | adjust=True now equals pandas ewm(adjust=True) via zero-seed numerator over 1-(1-alpha)^(t+1); test_ewma_adjust_is_causal.py; test_ewma_residual_recurrence_njit reference updated |
| 24-F6 | FIXED | CUSUM reference mean is the expanding mean of prior rows; auto threshold is MAD of the first 30 rows; test_cusum_is_causal.py; test_cusum_walk_njit reference updated to causal form |
| 24-F7 | FIXED | NaN/None states no longer count as possession; test_state_duration_and_sentinel_guards.py |
| 24-F8 | FIXED | sentinel detection is numeric-only, non-bool, and needs at least 3 other levels; same test file |
| 24-F9 | FIXED | IDW, label-dispersion and gradient drop the query point itself when query is the reference pool (shared _drop_self_match); tests/feature_engineering/test_spatial_query_is_ref_and_nan_queries.py |
| 24-F10 | FIXED | knn_within_bucket_aggregate drops the self-match when query is reference; same test file |
| 24-F11 | FIXED | knn_gradient uses zero-filled coords for the design matrix and marks non-finite query rows singular, so no NaN reaches the batched SVD/solve; same test file |
| 24-F12 | FIXED | rolling_longest_monotone_run returns NaN for windows containing a non-finite value; tests/feature_engineering/test_feature_edge_cases_24_f12_to_f15.py |
| 24-F13 | FIXED | nearest_past_join indexes unresolved rows through the positional copy; same test file |
| 24-F14 | FIXED | create_ts_features_parallel returns (None, None) for an empty span before dividing; same test file |
| 24-F15 | FIXED | mps shift fills the vacated tail from the last shifted position; same test file |
| 24-F18 | FIXED | NaN entity kept as its own group (dropna=False); test_tfidf_entity_nan_and_case.py |
| 24-F19 | FIXED | OOV check compares lowercased tokens when the vectorizer lowercases; same test file |
| 24-F20 | FIXED | with fit_on_train_only the member stage receives the train partition, matching MRMR |
| 24-F21 | FIXED | corr_clusters divides Z.T@Z by n (population sd); test_hybrid_ddof_and_raw_labels.py |
| 24-F22 | FIXED | cohort helper returns raw column labels, so non-string labels match; same test file |
| 25-F1 | FIXED | weighted quantile above the last midpoint returns the group max; tests/inference/test_entity_collapse_quantile_above_last_midpoint.py |
| 25-F2 | FIXED | monotonicity check compares gids[1:] >= gids[:-1], safe for unsigned ids |
| 25-F3 | FIXED | label_ranking_average_precision scores degenerate rows 1.0 and keeps them in the mean (sklearn); tests/metrics/test_multilabel_extras.py compares full arrays |
| 25-F4 | FIXED | label_ranking_loss keeps degenerate rows with loss 0 in the mean (sklearn) |
| 25-F5 | FIXED | multilabel inference remap also for two columns (len(probs) >= 2); tests/metrics/test_multiclass_labels_fairness_bins.py |
| 25-F6 | FIXED | isinstance(str) before the special-name membership test; same test file |
| 25-F7 | FIXED | excluded rare bins become NaN in bins; same test file |
| 25-F8 | FIXED | bootstrap distribution and resampler factory check length and return NaN for NaN scores (sibling convention); tests/metrics/test_auc_entry_checks_and_thread_safe_resampler.py |
| 25-F9 | FIXED | resampler scratch buffers are thread-local; concurrency regression in the same file |
| 25-F10 | FIXED | fast_aucs_per_group_optimized checks length, refuses NaN with ValueError and normalises labels; same file |
| 25-F11 | FIXED | binary label normaliser raises for more than two distinct labels; same file |
| 25-F12 | FIXED | non-finite values are still dropped, now logged with counts (warn option of the finding); tests/metrics/test_cluster25_entry_checks_and_fold_fits.py |
| 25-F13 | FIXED | MAPE checks length before the kernels; same test file |
| 25-F14 | FIXED | XGBoost interaction path uses native_contrib_kwargs for DMatrix and predict; same test file |
| 25-F15 | FIXED | eval_fraction splits within the training fold; the test fold never reaches fit; same test file |
| 25-F16 | FIXED | ETR no longer receives the CatBoost iterations key; same test file |
| 25-F17 | FIXED | isotonic refit slices sorted contiguous group segments (one sort); same test file checks per-group monotonicity |
| 25-F18 | FIXED | a column is a drop candidate only if unimportant in every training fold, importances on training rows only. Residual: the fold's held-out rows still shape the other folds' masks; a fully nested variant is a design option. Test: same test file |
| 25-F19 | FIXED | NaN epochs are skipped before the baseline and weights; a RuntimeWarning fires when no epoch is finite; same test file |
| 25-F18 | FIXED | correction: the drop mask is the mean of per-training-fold importances (each computed on training rows only) at or below the threshold; an all-folds AND broke biz_val pruning (26 of 43 survivors). Residual: held-out rows of one fold still shape other folds' masks |
| 26-F1 | FIXED | non-mutating cleaning path assigns per column (no assign(**kwargs)); tests/preprocessing/test_cluster26_preprocessing_fixes.py |
| 26-F2 | FIXED | dedup gate tests only float NaN, so pd.NA no longer hits bool(); same file |
| 26-F3 | FIXED | tree branch scores against supplied bounds as the non-tree branch does; same file |
| 26-F4 | FIXED | expanding denominator is the per-feature observed count; same file (weak test: only augmented rows are exposed) |
| 26-F5 | FIXED | auto-discovered non-numeric columns take mode under median/mean; same file |
| 27-F1 | FIXED | edge blend reads a snapshot of the hard predictions; tests/training/test_direct_horizon_edge_blend_from_hard.py |
| 27-F2 | FIXED | polars persist path fits a real SimpleImputer, so sklearn's dtype state is set; covered by tests/inference/test_predict_nan_guard_polars_native.py and the schema drift matrix |
| 27-F3 | FIXED | fingerprint digests __slots__ and __dict__ state for objects without get_params; tests/training/test_cluster27_fingerprint_threshold_backends.py |
| 27-F4 | FIXED | positive class is the larger label (y == max) in _counts_at and the minority count; same file |
| 27-F5 | FIXED | duplicated buckets read member slices from one stable argsort; covered by test_biz_val_training_pseudo_group_reconstruction.py |
| 27-F6 | FIXED | cb/lgb RFECV built only when the class is installed and the backend is requested, else None; same file tests the helper |
| 28-F1 | FIXED | NaT rows dropped before binning, order maps back to original rows; tests/training/test_cluster28_suite_fixes.py |
| 28-F2 | FIXED | Multilabel decided from array shape only; string class labels stay MULTICLASS. Test tests/training/test_cluster28_multiclass_string_labels_not_multilabel.py. |
| 28-F3 | FIXED | single-class y_true uses label 1 as the positive class, matching probs column 1; no separate test (internal branch of the exploss metric) |
| 28-F4 | FIXED | sort mode returns NaN rows verbatim with a RuntimeWarning; same test file |
| 28-F5 | FIXED | TTA jitter uses nanstd (NaN-safe) at all three sites; same test file |
| 28-F6 | FIXED | the fail-loud inf check runs on the batched path when fix_infinities is False; no separate test (guard branch in preprocess) |
| 28-F7 | FIXED | split artifacts are always rewritten, not skipped when the name exists; no separate test |
| 28-F8 | FIXED | dotted components checked against denylist prefixes in the older loader; same test file |
| 28-F9 | FIXED | write_ added to denied prefixes; same test file |
| 28-F10 | FIXED | automl_target_label defaults to None and the run target name is used; same test file |
| 28-F11 | FIXED | behaviour flag threaded from the suite into control config and read there; no separate test |
| 28-F12 | FIXED | class_weight keys absent from the training labels are skipped; no separate test |
| 28-F13 | FIXED | fallback retry goes through _train_model_with_f_phase; no separate test |
| 28-F14 | FIXED | an existing bundle that cannot be read raises instead of an unverified reopen; no separate test |
| 28-F15 | FIXED | drift cache key digests the full content (blake2b over row hashes); same test file |
| 28-F16 | FIXED | Val cache hit re-applies init_score; LightGBM cannot clear an init_score once set, so a val Dataset carrying one is rebuilt for a call without offsets (lgb_shim.py). Test tests/training/test_cluster28_val_cache_hit_reapplies_init_score.py. |
| 28-F17 | FIXED | set_label encodes through the fitted label encoder; no separate test |
| 28-F18 | FIXED | the unimplemented carrier stamp is replaced by a logged warning; no separate test |
| 29-F1 | FIXED | -P only on Python 3.11+ (empty flag tuple on older interpreters); tests/utils/test_native_import_probe.py and tests/utils/test_cluster29_utils_fixes.py |
| 29-F2 | FIXED | strided sample taken before the dense copy; n keeps the true row count; same test file |
| 29-F3 | FIXED | unmeasurable dimensions bucket to the string marker nan, not 0.0; same test file |
| 29-F4 | FIXED | structured/void dtype hashes the whole buffer; same test file |
| 29-F5 | FIXED | preprocess_glue NaN guard matches preprocess_sglue; no separate test (votenrank suite passed) |
| 29-F6 | FIXED | array-likes are written as a frame, other types raise TypeError; same test file |
| 31-F1 | FIXED | -P gated on Python 3.11+; PYTHONPATH still carries the import root; no separate test (subprocess command construction) |
| 31-F2 | FIXED | an ok sweep that persisted 0 regions counts as failed; cmd_refresh returns non-zero for 0; tests/system/test_cluster31_kernel_tuning_fixes.py |
| 31-F3 | FIXED | lock records pid:start-token; a live pid with a mismatching token is stale; same test file; test_kernel_tuning_lock_race.py assertion updated to the pid field |
| 32-F1 | FIXED | an empty rows list returns an empty frame before set_index; no separate test (LTR skip path) |
| 32-F2 | FIXED | a probability matrix takes the true-class probability and its bootstrap CI; tests/training/test_cluster32_baseline_fixes.py |
| 32-F3 | FIXED | one validation shard falls back to single-validation early stopping with one warning; no separate test |
| 32-F4 | FIXED | a non-finite global mean returns the minimum pseudocount (finite); tests/training/test_cluster32_baseline_fixes.py. Residual: the finding's preferred option, dropping non-finite y before the estimate, is not implemented; the caller still computes the mean from NaN-bearing y |
| 32-F5 | FIXED | nanmedian and nanquantile for the quantile baselines; no separate test |
| 32-F6 | FIXED | nanstd for the cross-output scale, non-finite scales skipped, non-finite normalized means excluded before the pick; no separate test |
| 33-F1 | FIXED | the CatBoost categorical NaN fill failure is logged at WARNING with the error type; the except is kept broad (not narrowed) so the fill never raises into predict; no separate test |
| 33-F2 | FIXED | label-stacking failure logged at WARNING; no separate test |
| 34-F1 | FIXED | trailing mean skips NaN entries and divides by the finite count; tests/training/test_cluster34_trailing_mean_nan.py |
| 34-F2 | FIXED | hints dropped by the finite-cell filter are removed from the hint slots on both paths; no separate test |
| 34-F3 | FIXED | demoted candidates are kept out of the MRMR relevance scale and ranked after the MRMR pass; no separate test |
| 34-F4 | FIXED | once any borderline column exists the collinear replay re-checks every column exactly. Cost: exact replay is O(n_cols * kept) in that case; no separate test |
| 35-F3 | FIXED | a base must take more than one value to set the early time-aware flag; tests/training/composite/test_cluster35_digest_and_flags.py |
| 35-F4 | FIXED | a non-finite raw baseline fails closed (empty result) when require_beats_raw_baseline is set, with a WARNING; no separate test |
| 35-F5 | FIXED | each fold keeps one slot (NaN when unusable); the paired test already skips non-finite pairs; no separate test |
| 35-F6 | FIXED | the stacked target is float64 before the residual; no separate test |
| 35-F7 | FIXED | time_aware with groups raises instead of silently dropping time order; no separate test |
| 35-F8 | FIXED | string digest length-prefixes each element; tests/training/composite/test_cluster35_digest_and_flags.py |
| 35-F9 | FIXED | external holdout key includes a frame content digest and a target digest; no separate test |
| 35-F10 | FIXED | meta model copied by the no-op cap and dropped explicitly on a trim; no separate test |
| 35-F11 | FIXED | the early logratio raise is removed so the domain mask routes base<=0 rows to the fallback, as the docstring says; no separate test |
| 35-F12 | FIXED | point-mass match uses rtol=0; no separate test |
| 35-F13 | FIXED | per-fold catch removed; a failing fold raises instead of NaN-filling; no separate test |
| 35-F14 | FIXED | predict_cdf sorts quantile rows before the binary search; no separate test |
| 36-F1 | FIXED | time_ordering is restricted to the covered rows with a length check before the nested selection; no separate test |
| 36-F2 | FIXED | sample_weight to the raw estimator is gated by _model_fit_accepts_sample_weight on both fits; no separate test |
| 36-F3 | FIXED | stage-2 sample_weight gated like stage 1; no separate test |
| 36-F4 | FIXED | the proba wrapper refuses non-binary classifiers at fit; tests/training/composite/test_cluster36_stacking_guards.py |
| 37-F1 | DEFERRED | not changed in this pass: masking the fabricated median in composite_t_full is a labelling-policy decision (what a training row with an out-of-domain base should carry). Needs owner decision |
| 37-F2 | DEFERRED | not changed in this pass: offering supplied lag values as a candidate base changes which bases the ceiling considers; needs a design decision on the candidate set |
| 37-F3 | FIXED | NaN scores get NaN intervals instead of the top bin; tests/training/composite/test_cluster37_venn_refit.py |
| 37-F4 | FIXED | one score source for all members (val only when every member has it, else OOF); tests/training/core/test_cluster37_diversity_single_score_scale.py |
| 37-F5 | FIXED | per-row sample weights are passed regardless of column pruning; no separate test beyond the suite run |
| 37-F6 | FIXED | refit_capped_stack takes and uses sample_weight, and the call site passes the OOF weights; tests/training/composite/test_cluster37_venn_refit.py |
| 37-F7 | FIXED | group sizes come from one np.unique pass; no separate test |
| 37-F8 | FIXED | a failed fingerprint cross-check distrusts the precomputed bundle and logs at WARNING; tests/training/core/test_cluster37_fingerprint_exception_distrusts.py |
| 38-F1 | FIXED | the suite-wide ensemble is not built when members span more than one target; tests/training/test_cluster38_suite_ensemble_targets.py |
| 38-F2 | FIXED | calib_preds are no longer re-stamped by the recalibrator, so conformal residuals are not in-sample for it; no separate test |
| 38-F3 | FIXED | a detector without predict() skips the val-side filter explicitly with a WARNING; no separate test |
| 39-F1 | FIXED | object targets are range-checked (non-integer or non-finite refused) before the int8 cast; tests/training/test_cluster39_encoders_and_casts.py |
| 39-F2 | FIXED | a batch missing a key raises instead of concatenating misaligned rows; no separate test |
| 39-F3 | FIXED | a failed concatenate raises with the shapes instead of truncating to batch 0; no separate test |
| 39-F4 | FIXED | the single-entry fallback applies only when no target was asked for; no separate test |
| 39-F5 | FIXED | a configured group column missing from the frame logs a WARNING; no separate test |
| 39-F6 | DEFERRED | not changed: the recency weighting's NaN policy for missing timestamps needs a decision (drop, neutral, or raise) |
| 39-F17 | FIXED | fit_transform applies the binary target check for woe, as fit does; no separate test |
| 39-F18 | FIXED | a float NaN text becomes the empty string, as the docstring says; no separate test |
| 39-F19 | DEFERRED | not changed: the memory-pressure eviction policy (what low RAM should evict and how far) needs an owner decision |
| 39-F20 | DEFERRED | not changed: the disk tier is unreachable (no caller builds DiskKey). Wire it in or remove it: an owner decision |
| 39-F32 | DEFERRED | not changed: the polars NaN-string rebrand needs a closer read of the cast path before a safe change |
| 39-F33 | FIXED | null categories group with dropna=False; tests/training/test_cluster39_encoders_and_casts.py |
| 39-F34 | FIXED | zero smoothing falls back to the global prior; tests/training/test_cluster39_encoders_and_casts.py |
| 39-F35 | FIXED | the 'no terms remain' pruning error is handled like empty vocabulary; no separate test |
| 39-F36 | FIXED | the entry is pinned in the strong tier while it loads (RLock, safe to reenter); no separate test |
| 39-F37 | DEFERRED | not changed: NaN input to PolynomialFeatures needs a NaN policy decision (refuse, or NaN-propagate per row) |
| 39-F53 | FIXED | multilabel heads use per-label sigmoid; tests/training/test_cluster39_encoders_and_casts.py |
| 39-F54 | FIXED | orthogonal_ and sparse_ fall back on 1-D tensors instead of raising; no separate test |
| 39-F67 | FIXED | the L1 exclusion ids are computed per step, not cached and pickled; no separate test |
| 40-F6 | FIXED | unseen validation labels raise instead of mapping to a neighbouring class; no separate test beyond the suite run |
| 40-F12 | FIXED | kwargs are filtered by signature and the fit is called once, so a TypeError inside a fit propagates; tests/training/test_cluster40_pipeline_fixes.py |
| 40-F14 | FIXED | duplicated index labels no longer multiply rows: both FE joins are positional; tests/training/test_cluster40_pipeline_fixes.py |
| 40-F19 | FIXED | array-valued parameters are keyed by shape, dtype and content digest; tests/training/test_cluster40_pipeline_fixes.py |
| 40-F3 | DEFERRED | not changed in this pass: needs a targeted reading and a design decision (see the cluster report); no fix applied yet |
| 40-F4 | DEFERRED | not changed in this pass: needs a targeted reading and a design decision (see the cluster report); no fix applied yet |
| 40-F5 | DEFERRED | not changed in this pass: needs a targeted reading and a design decision (see the cluster report); no fix applied yet |
| 40-F7 | DEFERRED | not changed in this pass: needs a targeted reading and a design decision (see the cluster report); no fix applied yet |
| 40-F8 | DEFERRED | not changed in this pass: needs a targeted reading and a design decision (see the cluster report); no fix applied yet |
| 40-F9 | DEFERRED | not changed in this pass: needs a targeted reading and a design decision (see the cluster report); no fix applied yet |
| 40-F10 | DEFERRED | not changed in this pass: needs a targeted reading and a design decision (see the cluster report); no fix applied yet |
| 40-F11 | DEFERRED | not changed in this pass: needs a targeted reading and a design decision (see the cluster report); no fix applied yet |
| 40-F13 | DEFERRED | not changed in this pass: needs a targeted reading and a design decision (see the cluster report); no fix applied yet |
| 40-F15 | DEFERRED | not changed in this pass: needs a targeted reading and a design decision (see the cluster report); no fix applied yet |
| 40-F16 | DEFERRED | not changed in this pass: needs a targeted reading and a design decision (see the cluster report); no fix applied yet |
| 40-F17 | DEFERRED | not changed in this pass: needs a targeted reading and a design decision (see the cluster report); no fix applied yet |
| 40-F18 | DEFERRED | not changed in this pass: needs a targeted reading and a design decision (see the cluster report); no fix applied yet |
| 41-F2 | FIXED | verbose NDCG logging uses .get so an eval_at without 10 cannot raise; no separate test |
| 41-F3 | FIXED | NaN temporal labels count as missing in the polars path, matching the pandas twin; no separate test |
| 41-F6 | FIXED | GroupKFold with fewer groups than k returns no shards, as the docstring promises; no separate test |
| 41-F1 | DEFERRED | not changed: the NaN categorical predict mapping needs the sentinel contract confirmed with the fit side |
| 41-F4 | DEFERRED | not changed: the early branches skip the skew check; reordering changes the rationale text, needs a decision on the intended output |
| 41-F5 | DEFERRED | not changed: per-group masks are an O(G*n) performance issue; a groupby rewrite needs a benchmark decision |
| 42-F1 | FIXED | tie-number keys use the tolerant PRETTY_NAMES lookup like the result rename; no separate test |
| 42-F2 | FIXED | sub-leaderboards inherit allow_partial and partial_fill; no separate test |
| 04-F13 | FIXED | graded-cardinality weights scaled to 1/(levels-1) so each cardinality carries equal signal range; existing dataset tests pass (49) |
| 01-F1 | FIXED | bench bias indexed by each row's own group code; benchmark-only, no test |
| 01-F2 | FIXED | bench seeds use zlib.crc32 instead of salted hash(); benchmark-only, no test |
| 01-F9 | FIXED | sort timer excludes RNG generation; benchmark-only, no test |
| 01-F12 | FIXED | module docstrings moved above the imports; benchmark-only |
| 01-F4 | DEFERRED | not changed in the P2/P3 pass yet; benchmark-only, low impact |
| 01-F5 | DEFERRED | not changed in the P2/P3 pass yet; benchmark-only, low impact |
| 01-F6 | DEFERRED | not changed in the P2/P3 pass yet; benchmark-only, low impact |
| 01-F7 | DEFERRED | not changed in the P2/P3 pass yet; benchmark-only, low impact |
| 01-F8 | DEFERRED | not changed in the P2/P3 pass yet; benchmark-only, low impact |
| 01-F10 | DEFERRED | not changed in the P2/P3 pass yet; benchmark-only, low impact |
| 01-F11 | DEFERRED | not changed in the P2/P3 pass yet; benchmark-only, low impact |
| 01-F13 | DEFERRED | not changed in the P2/P3 pass yet; benchmark-only, low impact |
| 02-F1 | FIXED | wall time measured on an un-profiled call; benchmark-only |
| 02-F2 | DEFERRED | benchmark-script cleanup not done in this pass; low impact |
| 02-F3 | DEFERRED | benchmark-script cleanup not done in this pass; low impact |
| 02-F4 | DEFERRED | benchmark-script cleanup not done in this pass; low impact |
| 02-F5 | DEFERRED | benchmark-script cleanup not done in this pass; low impact |
| 02-F6 | DEFERRED | benchmark-script cleanup not done in this pass; low impact |
| 03-F3 | FIXED | coverage gap closed: parallel and serial robust means agree at n above the dispatch threshold; tests/core/test_robust_mean_parallel_matches_serial.py |
| 03-F1 | DEFERRED | benchmark-script change not made in this pass; low impact |
| 03-F2 | DEFERRED | benchmark-script change not made in this pass; low impact |
| 03-F4 | DEFERRED | benchmark-script change not made in this pass; low impact |
| 03-F5 | DEFERRED | benchmark-script change not made in this pass; low impact |
| 03-F6 | DEFERRED | benchmark-script change not made in this pass; low impact |
| 04-F25 | FIXED | scipy rvs gets shape parameters first, loc and scale as keywords; tests/data/test_cluster04_sapp_args_and_coerce.py |
| 04-F23 | FIXED | non-numeric values coerced to NaN are counted in the notes; same test file |
| 04-F15 | DEFERRED | not changed in this pass; needs a targeted design or plumbing decision (see cluster report) |
| 04-F16 | DEFERRED | not changed in this pass; needs a targeted design or plumbing decision (see cluster report) |
| 04-F17 | DEFERRED | not changed in this pass; needs a targeted design or plumbing decision (see cluster report) |
| 04-F18 | DEFERRED | not changed in this pass; needs a targeted design or plumbing decision (see cluster report) |
| 04-F19 | DEFERRED | not changed in this pass; needs a targeted design or plumbing decision (see cluster report) |
| 04-F20 | DEFERRED | not changed in this pass; needs a targeted design or plumbing decision (see cluster report) |
| 04-F21 | DEFERRED | not changed in this pass; needs a targeted design or plumbing decision (see cluster report) |
| 04-F22 | DEFERRED | not changed in this pass; needs a targeted design or plumbing decision (see cluster report) |
| 04-F24 | DEFERRED | not changed in this pass; needs a targeted design or plumbing decision (see cluster report) |
| 04-F26 | DEFERRED | not changed in this pass; needs a targeted design or plumbing decision (see cluster report) |
| 05-F2 | FIXED | the duplicated fixture delegates to the shared builder, with the same draws; benchmark-only |
| 05-F6 | FIXED | tests/data_valuation/test_dv_bench_shared_fixture.py covers the shared builder |
| 05-F1 | DEFERRED | benchmark-script cleanup not done in this pass; low impact |
| 05-F3 | DEFERRED | benchmark-script cleanup not done in this pass; low impact |
| 05-F4 | DEFERRED | benchmark-script cleanup not done in this pass; low impact |
| 05-F5 | DEFERRED | benchmark-script cleanup not done in this pass; low impact |
| 06-F1 | DEFERRED | benchmark-script finding; low impact, deferred as a group pending an owner decision on benchmark cleanup |
| 06-F2 | DEFERRED | benchmark-script finding; low impact, deferred as a group pending an owner decision on benchmark cleanup |
| 06-F3 | DEFERRED | benchmark-script finding; low impact, deferred as a group pending an owner decision on benchmark cleanup |
| 06-F4 | DEFERRED | benchmark-script finding; low impact, deferred as a group pending an owner decision on benchmark cleanup |
| 06-F5 | DEFERRED | benchmark-script finding; low impact, deferred as a group pending an owner decision on benchmark cleanup |
| 07-F1 | DEFERRED | benchmark-script finding; low impact, deferred as a group pending an owner decision on benchmark cleanup |
| 07-F2 | DEFERRED | benchmark-script finding; low impact, deferred as a group pending an owner decision on benchmark cleanup |
| 07-F3 | DEFERRED | benchmark-script finding; low impact, deferred as a group pending an owner decision on benchmark cleanup |
| 07-F4 | DEFERRED | benchmark-script finding; low impact, deferred as a group pending an owner decision on benchmark cleanup |
| 07-F5 | DEFERRED | benchmark-script finding; low impact, deferred as a group pending an owner decision on benchmark cleanup |
| 07-F6 | DEFERRED | benchmark-script finding; low impact, deferred as a group pending an owner decision on benchmark cleanup |
| 07-F7 | DEFERRED | benchmark-script finding; low impact, deferred as a group pending an owner decision on benchmark cleanup |
| 07-F8 | DEFERRED | benchmark-script finding; low impact, deferred as a group pending an owner decision on benchmark cleanup |
| 07-F9 | DEFERRED | benchmark-script finding; low impact, deferred as a group pending an owner decision on benchmark cleanup |
| 07-F10 | DEFERRED | benchmark-script finding; low impact, deferred as a group pending an owner decision on benchmark cleanup |
| 07-F11 | DEFERRED | benchmark-script finding; low impact, deferred as a group pending an owner decision on benchmark cleanup |
| 07-F12 | DEFERRED | benchmark-script finding; low impact, deferred as a group pending an owner decision on benchmark cleanup |
| 07-F13 | DEFERRED | benchmark-script finding; low impact, deferred as a group pending an owner decision on benchmark cleanup |
| 07-F14 | DEFERRED | benchmark-script finding; low impact, deferred as a group pending an owner decision on benchmark cleanup |
| 08-F3 | FIXED | a single-class fold's one-column proba maps to the seen class instead of crashing on column 1; tests/feature_engineering/transformer/test_cluster08_edges.py covers the sibling edges, not this path directly |
| 08-F5 | FIXED | both SMOTE siblings return (0, d) on an empty minority; tests/feature_engineering/transformer/test_cluster08_edges.py |
| 08-F10 | FIXED | magnitude check uses max(/min/,/max/) with Python ints, so the most negative int64 is flagged; tests/feature_engineering/transformer/test_cluster08_edges.py |
| 08-F11 | FIXED | the silent temperature override now warns (RuntimeWarning); no separate test |
| 08-F7 | DEFERRED | not changed in this pass; needs a design or behaviour decision (see cluster report) |
| 09-F15 | DEFERRED | tried an early return in the underdetermined branch; it broke the contract that the linear column stays finite. The real defect is the per-row exception path zeroing whole rows; needs a column-aware fix |
| 09-F24 | FIXED | binary local-lift labels are normalised to {0,1} for both the global mean and the neighbour sums; tests/feature_engineering/transformer/test_cluster09_edges.py covers the KMeans path only, lift path covered by existing lift tests |
| 09-F26 | FIXED | anchor count is capped by the row count, so a one-row fold yields one anchor; tests/feature_engineering/transformer/test_cluster09_edges.py |
| 09-F28 | DEFERRED | not changed in this pass; needs a design or performance decision (see cluster report) |
| 09-F29 | DEFERRED | not changed in this pass; needs a design or performance decision (see cluster report) |
| 09-F30 | DEFERRED | not changed in this pass; needs a design or performance decision (see cluster report) |
| 09-F31 | DEFERRED | not changed in this pass; needs a design or performance decision (see cluster report) |
| 12-F2 | DEFERRED | P2 located under _benchmarks (benchmark or research harness); deferred under the benchmark rule |
| 12-F3 | DEFERRED | P2 located under _benchmarks (benchmark or research harness); deferred under the benchmark rule |
| 12-F4 | DEFERRED | P2 located under _benchmarks (benchmark or research harness); deferred under the benchmark rule |
| 12-F5 | DEFERRED | P2 located under _benchmarks (benchmark or research harness); deferred under the benchmark rule |
| 12-F6 | DEFERRED | P2 located under _benchmarks (benchmark or research harness); deferred under the benchmark rule |
| 12-F7 | DEFERRED | P2 located under _benchmarks (benchmark or research harness); deferred under the benchmark rule |
| 12-F9 | DEFERRED | P2 located under _benchmarks (benchmark or research harness); deferred under the benchmark rule |
| 12-F10 | DEFERRED | P2 located under _benchmarks (benchmark or research harness); deferred under the benchmark rule |
| 12-F11 | DEFERRED | P2 located under _benchmarks (benchmark or research harness); deferred under the benchmark rule |
| 12-F12 | DEFERRED | P2 located under _benchmarks (benchmark or research harness); deferred under the benchmark rule |
| 12-F13 | DEFERRED | P2 located under _benchmarks (benchmark or research harness); deferred under the benchmark rule |
| 12-F14 | DEFERRED | P2 located under _benchmarks (benchmark or research harness); deferred under the benchmark rule |
| 12-F15 | DEFERRED | P2 located under _benchmarks (benchmark or research harness); deferred under the benchmark rule |
| 12-F17 | DEFERRED | P2 located under _benchmarks (benchmark or research harness); deferred under the benchmark rule |
| 12-F40 | DEFERRED | P2 located under _benchmarks (benchmark or research harness); deferred under the benchmark rule |
| 12-F41 | DEFERRED | P2 located under _benchmarks (benchmark or research harness); deferred under the benchmark rule |
| 12-F42 | DEFERRED | P2 located under _benchmarks (benchmark or research harness); deferred under the benchmark rule |
| 12-F43 | DEFERRED | P2 located under _benchmarks (benchmark or research harness); deferred under the benchmark rule |
| 12-F44 | DEFERRED | P2 located under _benchmarks (benchmark or research harness); deferred under the benchmark rule |
| 12-F45 | DEFERRED | P2 located under _benchmarks (benchmark or research harness); deferred under the benchmark rule |
| 12-F46 | DEFERRED | P2 located under _benchmarks (benchmark or research harness); deferred under the benchmark rule |
| 12-F47 | DEFERRED | P2 located under _benchmarks (benchmark or research harness); deferred under the benchmark rule |
| 12-F48 | DEFERRED | P2 located under _benchmarks (benchmark or research harness); deferred under the benchmark rule |
| 12-F49 | DEFERRED | P2 located under _benchmarks (benchmark or research harness); deferred under the benchmark rule |
| 12-F50 | DEFERRED | P2 located under _benchmarks (benchmark or research harness); deferred under the benchmark rule |
| 12-F51 | DEFERRED | P2 located under _benchmarks (benchmark or research harness); deferred under the benchmark rule |
| 12-F52 | DEFERRED | P2 located under _benchmarks (benchmark or research harness); deferred under the benchmark rule |
| 12-F53 | DEFERRED | P2 located under _benchmarks (benchmark or research harness); deferred under the benchmark rule |
| 12-F54 | DEFERRED | P2 located under _benchmarks (benchmark or research harness); deferred under the benchmark rule |
| 12-F55 | DEFERRED | P2 located under _benchmarks (benchmark or research harness); deferred under the benchmark rule |
| 12-F56 | DEFERRED | P2 located under _benchmarks (benchmark or research harness); deferred under the benchmark rule |
| 12-F57 | DEFERRED | P2 located under _benchmarks (benchmark or research harness); deferred under the benchmark rule |
| 12-F58 | DEFERRED | P2 located under _benchmarks (benchmark or research harness); deferred under the benchmark rule |
| 12-F59 | DEFERRED | P2 located under _benchmarks (benchmark or research harness); deferred under the benchmark rule |
| 12-F60 | DEFERRED | P2 located under _benchmarks (benchmark or research harness); deferred under the benchmark rule |
| 12-F61 | DEFERRED | P2 located under _benchmarks (benchmark or research harness); deferred under the benchmark rule |
| 12-F62 | DEFERRED | P2 located under _benchmarks (benchmark or research harness); deferred under the benchmark rule |
| 12-F63 | DEFERRED | P2 located under _benchmarks (benchmark or research harness); deferred under the benchmark rule |
| 12-F64 | DEFERRED | P2 located under _benchmarks (benchmark or research harness); deferred under the benchmark rule |
| 12-F65 | DEFERRED | P2 located under _benchmarks (benchmark or research harness); deferred under the benchmark rule |
| 12-F66 | DEFERRED | P2 located under _benchmarks (benchmark or research harness); deferred under the benchmark rule |
| 12-F67 | DEFERRED | P2 located under _benchmarks (benchmark or research harness); deferred under the benchmark rule |
| 12-F68 | DEFERRED | P2 located under _benchmarks (benchmark or research harness); deferred under the benchmark rule |
| 12-F69 | DEFERRED | P2 located under _benchmarks (benchmark or research harness); deferred under the benchmark rule |
| 12-F70 | DEFERRED | P2 located under _benchmarks (benchmark or research harness); deferred under the benchmark rule |
| 12-F71 | DEFERRED | P2 located under _benchmarks (benchmark or research harness); deferred under the benchmark rule |
| 12-F72 | DEFERRED | P2 located under _benchmarks (benchmark or research harness); deferred under the benchmark rule |
| 12-F73 | DEFERRED | P2 located under _benchmarks (benchmark or research harness); deferred under the benchmark rule |
| 12-F74 | DEFERRED | P2 located under _benchmarks (benchmark or research harness); deferred under the benchmark rule |
| 12-F75 | DEFERRED | P2 located under _benchmarks (benchmark or research harness); deferred under the benchmark rule |
| 12-F76 | DEFERRED | P2 located under _benchmarks (benchmark or research harness); deferred under the benchmark rule |
| 12-F77 | DEFERRED | P2 located under _benchmarks (benchmark or research harness); deferred under the benchmark rule |
| 12-F78 | DEFERRED | P2 located under _benchmarks (benchmark or research harness); deferred under the benchmark rule |
| 12-F79 | DEFERRED | P2 located under _benchmarks (benchmark or research harness); deferred under the benchmark rule |
| 12-F80 | DEFERRED | P2 located under _benchmarks (benchmark or research harness); deferred under the benchmark rule |
| 12-F81 | DEFERRED | P2 located under _benchmarks (benchmark or research harness); deferred under the benchmark rule |
| 12-F84 | DEFERRED | P2 located under _benchmarks (benchmark or research harness); deferred under the benchmark rule |
| 13-F3 | FIXED | sub-fits run with premerge_clusters=False, since the parent has already collapsed X; covered by the existing Boruta premerge and stability tests (run in progress) |
| 13-F4 | DEFERRED | needs the held-out split indices available at the auto-measure probe; the probe currently reads all rows, so the routing decision is not yet leak-free |
| 13-F5 | DEFERRED | P2 under _benchmarks (benchmark harness); deferred under the benchmark rule |
| 13-F6 | DEFERRED | P2 under _benchmarks (benchmark harness); deferred under the benchmark rule |
| 13-F7 | DEFERRED | P2 under _benchmarks (benchmark harness); deferred under the benchmark rule |
| 13-F8 | DEFERRED | P2 under _benchmarks (benchmark harness); deferred under the benchmark rule |
| 13-F9 | DEFERRED | P2 under _benchmarks (benchmark harness); deferred under the benchmark rule |
| 13-F10 | DEFERRED | P2 under _benchmarks (benchmark harness); deferred under the benchmark rule |
| 13-F11 | DEFERRED | P2 under _benchmarks (benchmark harness); deferred under the benchmark rule |
| 13-F12 | DEFERRED | P2 under _benchmarks (benchmark harness); deferred under the benchmark rule |
| 13-F13 | DEFERRED | P2 under _benchmarks (benchmark harness); deferred under the benchmark rule |
| 13-F14 | DEFERRED | P2 under _benchmarks (benchmark harness); deferred under the benchmark rule |
| 13-F3 | FIXED | note: tests/feature_selection/boruta_shap/test_explain_basis_is_the_fitted_slice.py fails both before and after this change (pre-existing, verified by restoring the prior file); not caused by the fix, needs a separate look |
| 14-F5 | FIXED | a failed seed or n_jobs set_params is logged at WARNING; no separate test |
| 14-F6 | FIXED | an unreadable xgboost class count fails closed; tests/feature_selection/shap_proxied/test_cluster14_xgb_eligibility_fails_closed.py |
| 14-F3 | DEFERRED | string handling on the array path needs a design decision on supported dtypes |
| 14-F7 | DEFERRED | duplicated per-column correlation loops: performance refactor, not a correctness fix |
| 14-F5 | FIXED | note: tests/feature_selection/shap_proxied/test_biz_val_shap_proxied_core_refine.py::test_biz_val_auto_matches_core_on_non_saturated_bed fails on the exact pre-change file too (pre-existing, verified) |
| 15-F8 | FIXED | a callable prescreen's result is filtered to the candidate names the search offered; no direct test (the prescreen path needs a full RFECV fixture) |
| 15-F9 | FIXED | the importance callable is called with target only when its signature accepts it, so a TypeError inside it runs once; tests/feature_selection/test_cluster15_importance_getter_and_prescreen.py |
| 15-F10 | DEFERRED | not changed in this pass; checkpoint-state or performance refactor needing a design decision (see cluster report) |
| 15-F11 | DEFERRED | not changed in this pass; checkpoint-state or performance refactor needing a design decision (see cluster report) |
| 16-F1 | DEFERRED | P2/P3 located under _benchmarks (benchmark harness); deferred under the benchmark rule |
| 16-F2 | DEFERRED | P2/P3 located under _benchmarks (benchmark harness); deferred under the benchmark rule |
| 16-F3 | DEFERRED | P2/P3 located under _benchmarks (benchmark harness); deferred under the benchmark rule |
| 16-F4 | DEFERRED | P2/P3 located under _benchmarks (benchmark harness); deferred under the benchmark rule |
| 16-F5 | DEFERRED | P2/P3 located under _benchmarks (benchmark harness); deferred under the benchmark rule |
| 16-F6 | DEFERRED | P2/P3 located under _benchmarks (benchmark harness); deferred under the benchmark rule |
| 16-F7 | DEFERRED | P2/P3 located under _benchmarks (benchmark harness); deferred under the benchmark rule |
| 16-F8 | DEFERRED | P2/P3 located under _benchmarks (benchmark harness); deferred under the benchmark rule |
| 16-F9 | DEFERRED | P2/P3 located under _benchmarks (benchmark harness); deferred under the benchmark rule |
| 16-F10 | DEFERRED | P2/P3 located under _benchmarks (benchmark harness); deferred under the benchmark rule |
| 16-F11 | DEFERRED | P2/P3 located under _benchmarks (benchmark harness); deferred under the benchmark rule |
| 17-F13 | FIXED | the quantile calibration branch refuses non-integer labels instead of truncating them; tests/metrics/test_cluster17_kappa_and_label_guard.py |
| 17-F16 | FIXED | single-class perfect agreement gives kappa 1, as sklearn; tests/metrics/test_cluster17_kappa_and_label_guard.py |
| 17-F15 | FIXED | a failed title-token metric is logged at WARNING, not DEBUG; no separate test |
| 17-F17 | DEFERRED | not changed in this pass; clamping semantics, performance, or test-coverage work needing a decision (see cluster report) |
| 17-F18 | DEFERRED | not changed in this pass; clamping semantics, performance, or test-coverage work needing a decision (see cluster report) |
| 17-F19 | DEFERRED | not changed in this pass; clamping semantics, performance, or test-coverage work needing a decision (see cluster report) |
| 17-F20 | DEFERRED | not changed in this pass; clamping semantics, performance, or test-coverage work needing a decision (see cluster report) |
| 18-F9 | FIXED | gate weighted means return NaN when every weight is zero instead of raising; tests/models/test_cluster18_zero_weight_gate.py; existing quality-gate and ensembling tests: 231 passed |
| 18-F12 | DEFERRED | not changed in this pass; needs a targeted design or performance decision (see cluster report) |
| 19-F1 | DEFERRED | P2/P3 located under _benchmarks (benchmark harness); deferred under the benchmark rule |
| 19-F2 | DEFERRED | P2/P3 located under _benchmarks (benchmark harness); deferred under the benchmark rule |
| 19-F3 | DEFERRED | P2/P3 located under _benchmarks (benchmark harness); deferred under the benchmark rule |
| 19-F4 | DEFERRED | P2/P3 located under _benchmarks (benchmark harness); deferred under the benchmark rule |
| 19-F5 | DEFERRED | P2/P3 located under _benchmarks (benchmark harness); deferred under the benchmark rule |
| 19-F6 | DEFERRED | P2/P3 located under _benchmarks (benchmark harness); deferred under the benchmark rule |
| 19-F7 | DEFERRED | P2/P3 located under _benchmarks (benchmark harness); deferred under the benchmark rule |
| 19-F8 | DEFERRED | P2/P3 located under _benchmarks (benchmark harness); deferred under the benchmark rule |
| 19-F9 | DEFERRED | P2/P3 located under _benchmarks (benchmark harness); deferred under the benchmark rule |
| 20-F9 | FIXED | the DeLong warning now says labels outside {0, 1} are excluded from both classes, matching the code; no separate test |
| 20-F17 | FIXED | the per-alpha pinball panel drops non-finite rows first, as its sibling panels do; tests/reporting/test_cluster20_pinball_panel_nan.py |
| 20-F14 | DEFERRED | not changed in this pass; chart-semantics or benchmark finding needing individual judgment (see cluster report) |
| 20-F18 | DEFERRED | not changed in this pass; chart-semantics or benchmark finding needing individual judgment (see cluster report) |
| 20-F20 | DEFERRED | not changed in this pass; chart-semantics or benchmark finding needing individual judgment (see cluster report) |
| 20-F21 | DEFERRED | not changed in this pass; chart-semantics or benchmark finding needing individual judgment (see cluster report) |
| 20-F22 | DEFERRED | not changed in this pass; chart-semantics or benchmark finding needing individual judgment (see cluster report) |
| 20-F24 | DEFERRED | not changed in this pass; chart-semantics or benchmark finding needing individual judgment (see cluster report) |
| 21-F5 | FIXED | an all-missing slice column takes the missingness code; tests/reporting/test_cluster21_slice_missing_and_metric_direction.py |
| 21-F16 | FIXED | metric direction uses the family name before any @k cutoff; same test file |
| 21-F8 | DEFERRED | not changed in this pass; renderer or chart semantics needing individual judgment (see cluster report) |
| 21-F9 | DEFERRED | not changed in this pass; renderer or chart semantics needing individual judgment (see cluster report) |
| 21-F14 | DEFERRED | not changed in this pass; renderer or chart semantics needing individual judgment (see cluster report) |
| 22-F13 | FIXED | the PIT chi-square's expected total is the binned count, so NaN is consistent; tests/calibration/test_cluster22_chi_square_nan.py; 52 calibration quality tests pass |
| 22-F20 | FIXED | the binary predict_proba clips the positive probability to [0,1] like the multiclass branch; no separate test |
| 22-F17 | DEFERRED | not changed in this pass; needs individual judgment or a behaviour decision (see cluster report) |
| 22-F18 | DEFERRED | not changed in this pass; needs individual judgment or a behaviour decision (see cluster report) |
| 22-F23 | DEFERRED | not changed in this pass; needs individual judgment or a behaviour decision (see cluster report) |
| 22-F26 | DEFERRED | not changed in this pass; needs individual judgment or a behaviour decision (see cluster report) |
| 23-F5 | FIXED | the leakage p-value is computed whether or not a tolerance decided the verdict; tests/evaluation/test_cluster23_leak_pvalue_and_event_days.py |
| 23-F8 | FIXED | event distances use fractional days (total seconds / 86400) instead of the floored .dt.days; same test file; 95 related tests pass |
| 23-F7 | DEFERRED | not changed in this pass; low-impact judgment or cleanup (see cluster report) |
| 23-F12 | DEFERRED | not changed in this pass; low-impact judgment or cleanup (see cluster report) |
| 24-F48 | FIXED | validation-only class labels get the probability floor instead of a neighbouring class's column; tests/feature_selection/test_cluster24_unseen_validation_label.py |
| 24-F23 | DEFERRED | not reviewed in this pass; P2/P3 still open (see cluster report) |
| 24-F25 | DEFERRED | not reviewed in this pass; P2/P3 still open (see cluster report) |
| 24-F26 | DEFERRED | not reviewed in this pass; P2/P3 still open (see cluster report) |
| 24-F28 | DEFERRED | not reviewed in this pass; P2/P3 still open (see cluster report) |
| 24-F30 | DEFERRED | not reviewed in this pass; P2/P3 still open (see cluster report) |
| 24-F31 | DEFERRED | not reviewed in this pass; P2/P3 still open (see cluster report) |
| 24-F36 | DEFERRED | not reviewed in this pass; P2/P3 still open (see cluster report) |
| 24-F37 | DEFERRED | not reviewed in this pass; P2/P3 still open (see cluster report) |
| 24-F38 | DEFERRED | not reviewed in this pass; P2/P3 still open (see cluster report) |
| 24-F39 | DEFERRED | not reviewed in this pass; P2/P3 still open (see cluster report) |
| 24-F47 | DEFERRED | not reviewed in this pass; P2/P3 still open (see cluster report) |
| 24-F50 | DEFERRED | not reviewed in this pass; P2/P3 still open (see cluster report) |
| 24-F25 | FIXED | empty support returns NaN per query point; tests/feature_engineering/test_cluster24_empty_and_alignment.py |
| 24-F31 | FIXED | chunks that return (None, None) are skipped and an all-empty run returns (None, None); no separate test |
| 24-F36 | FIXED | the long MA is reindexed to the short one's index before differencing; tests/feature_engineering/test_cluster24_empty_and_alignment.py |
| 24-F23 | DEFERRED | not reviewed in this pass; P2 still open |
| 24-F26 | DEFERRED | not reviewed in this pass; P2 still open |
| 24-F28 | DEFERRED | not reviewed in this pass; P2 still open |
| 24-F30 | DEFERRED | not reviewed in this pass; P2 still open |
| 24-F31 | FIXED | empty parallel chunks are skipped and an all-empty run returns (None, None); tests/feature_engineering/test_timeseries.py expectation updated to the corrected cumsum end contract; 129 related tests pass |
| 24-F23 | FIXED | local density drops the self-match when the query is the reference, so dist_to_kth is the k-th other point; tests/feature_engineering/test_cluster24_min_periods_and_self_match.py; 57 related tests pass |
| 24-F26 | FIXED | rolling min_periods is clamped to the window; same test file |
| 24-F28 | DEFERRED | read this pass; judgement or performance item deferred (see cluster report) |
| 35-F1 | RESOLVED | from_fitted_inner now stamps smearing_quantiles from the train features (X_train) and the reconstructed T, as fit() does; call sites pass the raw train rows. Tests tests/training/composite/test_smearing_fit_from_fitted_parity.py. |
| 09-F1 | RESOLVED | Random directions are unscaled; the d^{-1/4} scale moved to the inputs so the kernel is exp(q.k/sqrt(d)) as documented. Tests tests/feature_engineering/transformer/test_performer_kernel_identity.py; reference in test_performer_attention_key_side_stabilization.py updated. |
| 24-F30 | DOCUMENTED | The complement fallback is pinned by tests/feature_engineering/test_aggregated_view_based.py; the subset_direct feature flags it in the feature names. No behaviour change. |
| 01-F3 | WON'T FIX | Production keeps the np.unique + per-bin scan. probabilities.py:292-298 records a bench of this exact argsort/contiguous variant: bit-identical but 0.43x (n=1e6, n_bins=10: 96ms old vs 220-223ms). Porting it would slow the default path. |
| 35-F10 | FIXED | The copy branch already carried the meta state. The trim branch now carries the kept components' col means (so a capped linear_stack keeps its dropout stand-in) and logs a WARNING when a meta model is dropped. Test tests/training/composite/ensemble/test_cluster35_cap_keeps_component_means.py. |
| 35-F12 | FIXED | _is_point_mass uses rtol=0. Test tests/training/composite/test_cluster35_point_mass_tolerance.py. |
| 35-F13 | FIXED | Per-fold catch removed; a failing fold raises. Test tests/training/composite/test_cluster35_group_oof_no_silent_fold_nan.py. |
| 35-F14 | FIXED | predict_cdf sorts each quantile row before the binary search. Test tests/training/composite/test_cluster35_cdf_unsorted_quantile_rows.py. |
| 35-F11 | FIXED | The early logratio raise is removed; the domain mask routes base<=0 rows to the fallback. Test tests/training/composite/test_cluster35_quantile_logratio_nonpositive_base.py. |
| 35-F7 | FIXED | time_aware with groups raises. Test tests/training/composite/test_cluster35_stacking_time_and_groups_refused.py. |
| 35-F9 | FIXED | External holdout key includes a frame content digest. Test tests/training/composite/test_cluster35_holdout_digest_distinguishes_content.py. |
| 12-F38 | FIXED | downstream returns NaN for an empty selection, as downstream_on_cols does, instead of crashing the run. Test tests/feature_selection/test_fs_hybrid_empty_selection_downstream.py. |
| 12-F20 | FIXED | An empty RFECV selection is recorded as empty_selection and not scored; the first-column fallback is removed. No dedicated test yet: the module uses relative imports, so it cannot be loaded by path. |
| 35-F5 | FIXED | Each fold keeps one slot (NaN when unusable), so the paired test zips by fold. Test tests/training/composite/test_cluster35_stepwise_fold_slots.py. |
| 12-F20 | FIXED | Scoring moved to score_selection; an empty RFECV selection is recorded as empty_selection and not scored. Tests tests/feature_selection/test_fs_hybrid_empty_rfecv_selection_not_scored.py. |
| 12-F26 | FIXED | The default flips only when the new arm wins most cells and its mean delta is positive (decide_flip). Tests tests/feature_selection/test_fs_hybrid_flip_needs_positive_mean.py. |
| 12-F27 | FIXED | The proxy Brier is computed out-of-fold (oof_proxy_brier), so it is comparable with the honest out-of-fold Brier. Test tests/feature_selection/test_fs_hybrid_s2_proxy_is_out_of_fold.py. |
| 12-F28 | FIXED | A size with any failed cell gets no family mean instead of a mean over its survivors. Tests tests/feature_selection/test_fs_hybrid_family_curve_failed_cells.py. |
| 12-F29 | FIXED | The designed-bed verdict reads the pre-specified right operator (designed_recovery, DESIGNED_RIGHT_OP) instead of the max over test AUC. Tests tests/feature_selection/test_fs_hybrid_designed_verdict_uses_prespecified_op.py. |
| 12-F30 | FIXED | run_bed takes its split from split_indices (fixed seed), so every preset is fit on the same rows. Tests tests/feature_selection/test_fs_hybrid_cost_split_is_fixed.py. |
| 12-F32 | FIXED | Failures are logged at WARNING and the summary prints completed and failed run counts per config next to the means (config_run_counts). Test tests/feature_selection/test_fs_hybrid_shap_run_counts.py. |
| 12-F33 | FIXED | The flip verdict uses paired strict-minus-default deltas over 5 splits, holding only when the lower 95% bound is at least -0.003 (flip_verdict). Tests tests/feature_selection/test_fs_hybrid_flip_verdict_paired.py. |
| 12-F35 | FIXED | The five RFECV comparison benches budget by refits only (max_runtime_mins=None), so results no longer depend on wall-clock time. Test tests/feature_selection/test_fs_hybrid_rfecv_benches_have_no_wall_clock_cap.py. |
| 12-F36 | FIXED | A configuration with any failed run is reported INCOMPLETE and left out of the ranking; the NaN no longer reaches best/worst/spread/noise. Test tests/feature_selection/test_fs_hybrid_ablation_failed_runs_excluded.py. |
| 12-F37 | FIXED | Failed runs are excluded from the selection sizes (kept share). Test tests/feature_selection/test_fs_hybrid_ablation_failed_runs_excluded.py. |
| 12-F39 | FIXED | Errored rows are logged at WARNING and the per-config completed and failed counts print next to the means (run_counts_by_key in _downstream_shared). Tests tests/feature_selection/test_fs_hybrid_run_counts_by_key.py. |
| 12-F23 | FIXED (test pending) | The baseline and relay now use h.transform output, so engineered columns in raw_selected_ are kept as the sibling bench does. No dedicated test: a regression test needs a full HybridSelector fit on a bed with selected engineered columns. |
| 12-F25 | FIXED | The prior-protected RFECV bench budgets by refits only (max_runtime_mins=None), like the other RFECV benches (12-F35). Test extended in tests/feature_selection/test_fs_hybrid_rfecv_benches_have_no_wall_clock_cap.py. |
| 39-F2 | FIXED | A batch missing an ensemble key raises instead of concatenating misaligned rows. Test tests/training/test_cluster39_run_batched_alignment.py. |
| 39-F3 | FIXED | A failed concatenate raises with the shapes instead of truncating. Test tests/training/test_cluster39_run_batched_alignment.py. |
| 39-F4 | FIXED | The named-target lookup no longer borrows another target's params: the single-entry fallback and the suite-wide fallback apply only when no target was asked for. Test tests/training/test_cluster39_chosen_params_target_isolation.py. |
| 39-F5 | FIXED | A configured group column missing from the frame logs a WARNING. Test tests/training/test_cluster39_missing_group_column_logged.py. |
| 39-F17 | FIXED | fit_transform applies the binary {0,1} check for woe, as fit does. Test tests/training/test_cluster39_woe_fit_transform_binary_check.py. |
| 39-F18 | FIXED | A float NaN text is embedded as the empty string, as the docstring says. Test tests/training/test_cluster39_hf_nan_text_is_empty.py. |
| 39-F35 | FIXED | 'No terms remain' after min_df pruning is handled like empty vocabulary (zero-width matrix). Test tests/training/test_cluster39_tfidf_no_terms_remain.py. |
| 39-F54 | FIXED | orthogonal_ and sparse_ fall back on 1-D tensors (BatchNorm gamma -> 1.0, bias -> 0). Test tests/training/test_cluster39_orthogonal_init_on_batchnorm.py. |
| 39-F67 | FIXED | Correction: the earlier 'computed per step' claim was incomplete. The id set was still stored on self (and so pickled); it is now a local of training_step and never stored. Test tests/training/neural/test_cluster39_l1_exclusion_not_pickled.py. |
| 39-F36 | FIXED | Prewarm pins the entry in the strong tier while it loads. Test tests/training/test_cluster39_prewarm_pins_entry_during_load.py. |
| 39-F37 | FIXED | Decision (default taken): NaN/inf input is refused in fit and transform with a message naming the count and the remedy, instead of sklearn's opaque error. Tests tests/training/test_cluster39_polynomial_refuses_nonfinite.py. |
| 39-F6 | FIXED | Decision (default taken): a missing timestamp gets the floor weight min_weight (treated as the oldest, least influential row) so every training weight is finite. Test tests/training/test_cluster39_recency_missing_timestamp.py. |
| 39-F32 | FIXED | Only float NaN cells become the null sentinel; the string 'NaN' in a string column is kept as data. Test tests/training/test_cluster39_polars_string_nan_is_data.py. |
| 39-F19 | FIXED | Decision (default taken): low system RAM alone evicts one entry per check and yields; a cache over its size limit still evicts until under it. Tests tests/training/test_cluster39_low_ram_evicts_one_entry.py. |
| 04-F11 | FIXED | Decision (default taken): the beds are approximations, not published functions; provenance now says so instead of claiming the published formula. Tests tests/data/test_cluster04_friedman_provenance_is_honest.py. Exact Friedman forms need new basis kinds (separate work). |
| 04-F13 | FIXED | Graded-cardinality weights are 1/(levels-1). Test tests/data/test_cluster04_graded_cardinality_weights.py. |
| 22-F3 | DEFERRED | Still deferred: get_postcalibrators imports six optional libraries unguarded and builds each calibrator inline. Guarding each entry means restructuring the zoo list per library; deferred to a dedicated pass with a test that blocks each module in turn. |
| 23-F2 | FIXED | Explicit leak_tolerance decides the verdict (no fixed default). Test tests/evaluation/test_cluster23_leak_verdict_uses_caller_tolerance.py. |
| 40-F13 | FIXED | A pre-pipeline cache hit stamps model._mlframe_pre_pipeline before returning, as the fit path does. Test tests/training/test_cluster40_cache_hit_stamps_pre_pipeline.py. |
| 40-F18 | FIXED | A split with group ids but no aligned row index is refused (ValueError naming the split) instead of averaged across entities. Test tests/training/test_cluster40_ma_crossover_refuses_unaligned_groups.py; existing ma_crossover tests (40) pass. |
| 40-F3 | FIXED | Multi-target regression no longer takes the ICE branch: _metric_branch_for returns RMSE unless num_classes>1 and not multi-target regression. Tests tests/training/neural/test_cluster40_multitarget_regression_uses_rmse.py; existing multi-target tests (6) pass. |
| 40-F4 | FIXED | Multilabel predict returns the thresholded indicator matrix instead of argmax over classes_ (which multilabel fits never set). Test tests/training/neural/test_cluster40_multilabel_predict_returns_indicators.py. |
| 40-F5 | FIXED | class_weight is read through classes_ (original labels), so each label gets its own weight (_class_weight_vector). Test tests/training/neural/test_cluster40_class_weight_follows_original_labels.py. |
| 40-F8 | FIXED | class_weight is a non-persistent buffer (no state_dict key); load drops a stale key from older checkpoints. Test tests/training/neural/test_cluster40_class_weight_not_persisted.py; existing save/load tests (4) pass. |
| 40-F7 | FIXED | Feature scaling copies the array before the block write; the caller's array is no longer scaled in place. Test tests/training/neural/test_cluster40_scaling_does_not_mutate_input.py. |
| 40-F9 | FIXED | A categorical that was all-null at fit maps predict rows to the reserved unknown code instead of indexing an empty code array. Test tests/training/neural/test_cluster40_all_null_categorical_predicts.py. |
| 40-F10 | FIXED | Decision (default taken): MLPRanker refuses hidden_layers with differing sizes at construction, since the builder uses one width for every layer; equal sizes are unchanged. Test tests/training/neural/test_cluster40_ranker_refuses_unequal_hidden_layers.py. |
| 40-F15 | PARTIAL | Fixed for the entity-time module (refuses an unaligned split naming it; test tests/training/test_cluster40_entity_time_refuses_unaligned_split.py). Still open for latent-interaction SVD, MA crossover's no-group-id path, and nearest-past-join (same pattern). |
| 40-F15 | FIXED | All four modules refuse an inconsistent split: entity-time and latent-SVD (unaligned index), nearest-past-join (missing join columns), MA crossover (via 40-F18). Tests: test_cluster40_entity_time_refuses_unaligned_split.py, test_cluster40_latent_svd_join_refuses_unaligned_split.py, test_cluster40_nearest_past_join_refuses_missing_columns.py, test_cluster40_ma_crossover_refuses_unaligned_groups.py. 57 existing related tests pass. |
| 40-F16 | FIXED | Powerset composites are computed for all splits first and attached only when all succeed; a split missing the source columns skips the step for every split. Test tests/training/test_cluster40_powerset_all_or_nothing.py; 25 related tests pass. |
| 40-F17 | FIXED | Replay restores the fitted effective_k (persisted in metadata) instead of recomputing it from the serving batch. Tests tests/training/test_cluster40_replay_uses_fitted_effective_k.py; related existing tests pass. |
| 40-F11 | FIXED | The precomputed pair cache is capped per query like the loss (_cap_pair_indices, deterministic). Tests tests/training/neural/test_cluster40_ranker_pairs_are_capped.py; related existing ranker tests pass (5 run, 37 skipped). |
| 25-F2 | FIXED | Monotonicity is checked by comparison, not np.diff (which wraps on unsigned ids). Test tests/metrics/test_cluster25_unsigned_group_ids_unsorted.py. |
| 26-F4 | FIXED | Expanding mean and std use the observed count per feature. The earlier test only checked finiteness; a strict test now pins the value (tests/preprocessing/test_cluster26_expanding_uses_observed_count.py). |
| 28-F12 | FIXED | A class_weight key absent from the training labels is skipped at fit. Test tests/training/test_cluster28_class_weight_absent_key_skipped.py. |
| 28-F2 | FIXED | Multilabel decided from array shape only; string class labels stay MULTICLASS. Test tests/training/test_cluster28_multiclass_string_labels_not_multilabel.py. |
| 28-F6 | FIXED | The fail-loud inf check also runs on the batched path when fix_infinities is False. Test tests/training/test_cluster28_polars_inf_guard_runs_when_not_fixing.py. |
| 28-F17 | FIXED | set_label applies the fitted label encoder, so the cached Dataset matches what fit trained on. Test tests/training/test_cluster28_set_label_encodes_like_fit.py. |
| 28-F7 | FIXED | save_split_artifacts rewrites every artifact (the existing-name skip is gone). Test tests/training/test_cluster28_split_artifacts_rewritten.py. |
| 28-F11 | FIXED | The auto-wrap switch is carried on the control config the training loop reads (set by _build_control_config), so False is honoured. Test tests/training/test_cluster28_wrap_flag_reaches_control.py. |
| 28-F14 | FIXED | An existing bundle that cannot be read raises instead of falling back to an unverified reopen. Test tests/training/test_cluster28_unreadable_bundle_raises.py. |
| 28-F16 | FIXED | Val cache hit re-applies init_score; LightGBM cannot clear an init_score once set, so a val Dataset carrying one is rebuilt for a call without offsets (lgb_shim.py). Test tests/training/test_cluster28_val_cache_hit_reapplies_init_score.py. |
| 01-F1 | FIXED | Bench groups are derived from the same codes that index the bias multiplier (bench_group_bias_correction). Bench-only; verified by reading, no test. |
| 01-F2 | FIXED | Bench seeds use zlib.crc32 instead of salted hash() (bench_ece_nbins). Bench-only; verified by reading, no test. |
| 01-F4 | DEFERRED | Bench re-implements the ECI histogram and entropy instead of calling the shipped function. Bench-only rewrite; deferred. |
| 01-F6 | DEFERRED | Bench draws labels with a per-row Python loop (performance only). Bench-only; deferred. |
| 01-F8 | DEFERRED | Bench verdict rests on a parallel-vs-serial comparison that needs re-measuring. Bench-only; deferred. |
| 02-F1 | FIXED | The reported wall time comes from an un-profiled call (_cprofile_bench_shared). Bench-only; verified by reading. |
| 02-F2 | DEFERRED | Bench-only group of scripts (seeding and setup); not read in this pass. Deferred to a bench pass. |
| 02-F3 | DEFERRED | Bench-only group of scripts; not read in this pass. Deferred to a bench pass. |
| 03-F1 | DEFERRED | Bench-only perf: the cProfile section reruns the full workload. Deferred. |
| 03-F2 | DEFERRED | Bench-only: the harness times but does not assert equality. Production equality is covered by tests/core/test_robust_mean_parallel_matches_serial.py. |
| 03-F3 | FIXED | Production parallel-vs-serial robust mean agreement is tested (tests/core/test_robust_mean_parallel_matches_serial.py, passing). |
| 04-F25 | FIXED | SAPP passes shape parameters positionally and loc/scale by keyword. Test tests/data/test_cluster04_sapp_shape_then_loc_scale.py. |
| 04-F20 | FIXED | id_trap row_id uses the permutation (id) family, so every row is distinct; the test now asserts exact uniqueness. Test tests/data/test_datasets_observation_layers.py::test_the_id_trap_column_is_unique_and_declared_irrelevant. |
| 04-F23 | FIXED | Non-numeric values coerced to NaN are counted in the notes. Test in tests/data/test_cluster04_sapp_args_and_coerce.py (passing). |
| 04-F17 | FIXED | Removed the no-op threshold line: sign(score)*/score/ equals score, so behaviour is unchanged. Existing link tests pass (test_link_region_reaches_feature_truth.py, 2). |
| 04-F16 | FIXED | The shortfall caveat uses the spec's CeilingTarget.tolerance instead of a hard-coded 0.01. Test tests/data/test_cluster04_ceiling_tolerance_drives_caveat.py. |
| 04-F18 | FIXED | Renamed the coverage-adjusted plug-in estimator to coverage_adjusted_plugin (it is not Chao-Shen entropy). Test tests/data/test_cluster04_oracle_estimator_names_are_honest.py. |
| 04-F15 | DEFERRED | _probability_only is an allowlisted exploration helper (tests/test_meta/test_shared_uncalled_functions.py) with a wrong body per the audit. Fixing it needs the difficulty-sweep intent confirmed; deferred. |
| 04-F21 | FIXED | Probe resampling keeps the donor's missing values, so probes match the donor's missingness. Test tests/data/test_cluster04_probe_keeps_donor_missingness.py. |
| 04-F22 | FIXED | Probes built into one dict and appended with a single concat; donors converted once. Values unchanged (same per-probe streams). Test tests/data/test_cluster04_probes_built_in_one_pass.py. |
| 04-F19 | FIXED | Removed the three never-populated FeatureTruth fields (marginal_mi, conditional_mi_given_rest, r2_to_source); the generator fills only role, region, source, cost. Test tests/data/test_cluster04_feature_truth_fields.py. |
| 04-F24 | FIXED | replay(with_data=False) compares the spec hash and skips generation; replay_report uses it, so one line per scenario no longer regenerates every bed. Test tests/data/test_cluster04_replay_status_skips_generation.py. |
| 04-F26 | FIXED | _feature_parents raises naming a parent that is neither observed, latent, nor the selected target, instead of dropping its effect while the truth graph keeps the edge. Test tests/data/test_cluster04_derived_refuses_dropped_parent.py. |
| 05-F1 | FIXED | The recursion benchmark's second arm is labelled interpreted_py and the docstring says it is the CPython body of the njit kernel, not a numpy backend (no such backend exists). Test tests/data_valuation/test_cluster05_recursion_bench_labels_the_interpreted_arm.py. |
| 05-F3 | FIXED | The shared fixture's docstring is rewritten to name its consumers. Docs only. |
| 05-F5 | FIXED | functools moved into the stdlib import group. Import order only. |
| 05-F6 | FIXED | Smoke test imports each benchmark script and builds its fixtures at tiny sizes, so signature drift surfaces in the test run. Test tests/data_valuation/test_cluster05_benchmark_helpers_smoke.py. |
| 06-F1 | FIXED | The bench's NEW side is the production MyDecorrelator.fit and its OLD side a labelled replica; the git show call that decoded nothing is removed. |
| 06-F2 | FIXED | The bench times and gates the production class instead of a copy of its logic. |
| 06-F3 | FIXED | The dead git show subprocess is removed with the 06-F1 fix. |
| 06-F4 | FIXED | The production drop set is one reduction over the upper triangle, same positions (custom.py). Covered by test_cpx39_decorrelator_identity.py. |
| 06-F5 | FIXED | Constant and all-NaN columns are checked against the loop rule. Test tests/estimators/test_cluster06_decorrelator_constant_and_nan_columns.py. |
| 07-F1 | FIXED | The docstring names _two_sided_t and its lru_cache thrash, not the nonexistent uncached norm.ppf. Docs only; the bench is cost-only. |
| 07-F2 | FIXED | The dead argsort and empty rank array are removed from _fast_auc; ranks are still rankdata. Test tests/evaluation/test_cluster07_bench_fixes.py. |
| 07-F3 | FIXED | Failed bootstrap trials are logged at WARNING and counted, and a run with more than 5% failures raises instead of reporting coverage over the survivors. Test tests/evaluation/test_cluster07_bench_fixes.py. |
| 07-F4 | FIXED | Only the constructed leak is swallowed; any other ValueError from the guard propagates. Test tests/evaluation/test_cluster07_bench_fixes.py. |
| 07-F6 | FIXED | The parallel bench raises SystemExit when a parallel run drifts or reproducibility fails, and derives its tolerance from four alternate seeds instead of one. Script-level check; the exit path is exercised by the bench, not a unit test. |
| 07-F5 | FIXED | The stratify bench times the production _stratified_class_book (the same helper bootstrap_metric calls). Test tests/evaluation/test_cluster07_bench_fixes.py::test_stratify_bench_times_the_production_book. |
| 07-F7 | FIXED | The docstring names the real iteration log and drops the flip instruction, which is already in place. Docs only. |
| 07-F8 | FIXED | The DeLong bench reads samples directly, so a missing key raises instead of silently crediting bootstrap. Bench-only, no unit test. |
| 07-F9 | FIXED | One condition feeds both win counters. Bench-only, no unit test. |
| 07-F10 | PARTIAL | The no-op numba import is removed. The cupy pin is kept on purpose to force the CPU bootstrap path, so moving it is a behaviour change to decide separately. |
| 07-F11 | FIXED | The unused ties branch of bench_rank is removed. Bench-only, no unit test. |
| 07-F12 | FIXED | The bench is labelled cost-only in the code that builds the identical variants. |
| 07-F13 | FIXED | run_once takes kwargs built once outside the timer, so the median excludes resampler setup. Bench-only, no unit test. |
| 07-F14 | FIXED | The unverifiable timings in the code comments are replaced with a note that they are not stored and need a re-run. Comments only. |
| 25-F23 | FIXED | The unanimous prune keeps the eligible feature whose permutation helped least when every feature is eligible, so the selection is never empty. Test tests/feature_selection/test_cluster25_prune_keeps_one_feature.py. |
| 25-F28 | FIXED | The warm start drops non-finite evaluations, with their candidates, before the surrogate sees them. Test tests/models/test_cluster25_known_nonfinite_and_title.py. |
| 25-F32 | FIXED | The plot title formats the best candidate only when it exists. Test tests/models/test_cluster25_known_nonfinite_and_title.py::test_plot_title_survives_no_best_candidate. |
| 25-F42 | FIXED | fast_rmse returns NaN for no rows, matching rmse_loss. Test tests/metrics/test_cluster25_empty_and_degenerate_guards.py. |
| 25-F43 | FIXED | hamming_loss returns NaN when there are no labels (K == 0). Test tests/metrics/test_cluster25_empty_and_degenerate_guards.py. |
| 25-F21 | FIXED | Each structure family runs under its own guard; a failed family is named in the report's skipped field and the others still run. Test tests/feature_selection/test_cluster25_structure_families_isolated.py. |
| 25-F26 | FIXED | Exhausting the start-run retries raises a scrubbed RuntimeError instead of returning None. Scope: only already-active errors are retried; other failures already raised at once. Behaviour change for any caller that expected None. Test tests/integrations/test_cluster25_mlflow_exhaustion_raises.py. |
| 25-F29 | FIXED | The unweighted fallback when ML scoring is skipped is logged at WARNING. Log-only change; no unit test, since the path needs a trained surrogate. |
| 25-F41 | FIXED | One GPU argsort failure logs a WARNING and latches the GPU path off for the process. Test tests/metrics/test_cluster25_gpu_argsort_latches.py. |
| 25-F48 | FIXED | set_params failures during CatBoost clone are logged at WARNING, and the excepts are narrowed. No unit test: it needs a CatBoost build that rejects set_params. |
| 25-F27 | FIXED | The docstring now says hard mode repeats to a fixpoint and soft mode is one ordered pass. The code is unchanged. Test tests/inference/test_cluster25_soft_constraints_single_pass.py. |
| 25-F33 | FIXED | fast_roc_curve uses the same positive-label rule as fast_roc_auc and refuses a NaN score as sklearn does. Note: fast_roc_auc returns NaN for NaN scores; the curve raises. Test tests/metrics/test_cluster25_roc_labels_and_err_bounds.py. |
| 25-F46 | FIXED | expected_reciprocal_rank rejects a negative label. Test tests/metrics/test_cluster25_roc_labels_and_err_bounds.py. |
| 25-F30 | FIXED | A CatBoost fold no longer receives its scored test fold as eval_set, so best-iteration selection cannot use it. Behaviour change: CatBoost folds now run to their full iteration count. Test tests/inference/test_cluster25_catboost_fold_never_uses_test_as_eval.py. |
| 25-F24 | FIXED | A width mismatch between data and ranked features raises ValueError instead of turning diversification off. Test tests/metrics/test_cluster25_contract_fixes.py::test_width_mismatch_refuses_rather_than_disabling_diversification. |
| 25-F38 | FIXED | compute_fairness_metrics returns an empty DataFrame with the documented index when no subgroup produced a row. Test tests/metrics/test_cluster25_contract_fixes.py. |
| 25-F45 | FIXED | fast_log_loss_binary converts its inputs with np.asarray before reading attributes, so lists work. Test tests/metrics/test_cluster25_contract_fixes.py. |
| 25-F54 | FIXED | The return annotation of prepare_trials_dataset is the tuple it returns. Test tests/metrics/test_cluster25_contract_fixes.py. |
| 25-F25 | DEFERRED | Needs an owner decision on the run lifecycle: _start_run_with_retries ends the run before returning it, so the caller's handle is already closed. Removing end_run changes when runs close for every caller. |
| 25-F49 | FIXED | compute_all_metrics returns {} for learning_to_rank rather than regression metrics under ranking keys. Test tests/metrics/test_cluster25_ranking_round_has_no_regression_block.py. |
| 25-F40 | FIXED | Prewarm imports shap without replacing is_transformers_lm on the shared module; the SHAP use site keeps its own scoped patch. No unit test: the cold shap import is too slow for the unit suite. |
| 25-F47 | FIXED | The precision method reports 1 - precision, an error like the other methods, so the aggregate moves the right way. Test tests/metrics/test_cluster25_ice_precision_is_an_error.py. |
| 25-F34 | FIXED | The GPU multi-column ROC primitive returns NaN for a column whose scores contain NaN, matching the CPU batch path. Test tests/metrics/test_cluster25_gpu_roc_masks_nan_columns.py (cupy-gated). |
| 25-F31 | FIXED | GroupTimeSeriesSplit builds its per-group indices with one stable sort instead of a per-sample loop; the folds are unchanged. Test tests/models/test_cluster25_group_time_split_vectorised.py; 21 existing group-split tests pass. |
| 25-F37 | FIXED | The NaN fill is one dict construction and the NaN count only walks the valid groups; output unchanged. Test tests/metrics/test_cluster25_group_aucs_nan_fill_vectorised.py. |
| 25-F44 | FIXED | Integer multilabel input is validated by its min and max instead of a float64 copy; the 0/1 rule and the uint8 result are unchanged. Test tests/metrics/test_cluster25_multilabel_coerce_integer_range.py. |
| 25-F20 | FIXED | The three remaining cache=True numba kernels in mi.py are cache=False, as the module comment requires. Test tests/feature_selection/test_cluster25_structure_op_symbols.py. |
| 25-F22 | FIXED | Both the live modular report and the recipe report use one op-to-symbol helper, so diff reads as a minus. Test tests/feature_selection/test_cluster25_structure_op_symbols.py. |
| 25-F39 | FIXED | numba_warmup guards each kernel separately, so one signature mismatch no longer skips the other compiles. The function is public and stays. Test tests/metrics/test_cluster25_numba_warmup_per_kernel.py. |
| 25-F36 | FIXED | The kernel tuning cache is loaded once per process; a failed load is retried, as before. Test tests/metrics/test_cluster25_kernel_tuning_cache_loaded_once.py. |
| 25-F50 | DEFERRED | Duplicated bodies in _ice_metric.py: merging them needs an equivalence test per copy first. Deferred to a dedicated refactor pass. |
| 25-F51 | DEFERRED | Duplicated bodies in ranking.py: merging them needs an equivalence test per copy first. Deferred to a dedicated refactor pass. |
| 25-F52 | DEFERRED | Duplicated bodies in _multilabel_extras.py: merging them needs an equivalence test per copy first. Deferred to a dedicated refactor pass. |
| 25-F53 | DEFERRED | Duplicated logical-constraint loops: merging them needs an equivalence test per copy first. Deferred to a dedicated refactor pass. |
| 25-F35 | DEFERRED | Keeping labels on the device changes cupy semantics for every caller and needs a GPU benchmark on real hardware before it can be called safe. |
| 26-F7 | FIXED | The fractional-digit probe refuses max_fract_digits below 2 with a ValueError, not an UnboundLocalError. Test tests/training/test_cluster26_aggregate_nan_and_fraction_probe.py. |
| 26-F11 | FIXED | float16 input is widened to float32 before the numba outlier kernels. Test tests/signal/test_cluster26_gp_and_outlier_dtypes.py. |
| 26-F23 | FIXED | The PDP panels order importances with NaN last through one helper, the same key _ranked_top_features uses. Test tests/reporting/test_cluster26_pdp_nan_importance_last.py. |
| 26-F28 | FIXED | Non-finite (time, value) pairs are dropped per entity before the GP fit; an entity with nothing finite is NaN. Test tests/signal/test_cluster26_gp_and_outlier_dtypes.py. |
| 26-F30 | FIXED | The aggregate early-stopping selection is NaN-aware and raises when every round is NaN. Test tests/training/test_cluster26_aggregate_nan_and_fraction_probe.py. |
| 26-F10 | FIXED | Nearest-mode imputation places its replacement by position, so a duplicated index no longer raises. Test tests/preprocessing/test_cluster26_unseen_nearest_duplicate_index.py. |
| 26-F6 | FIXED | When every candidate fails the target-retention guard and identity is not a candidate, the search evaluates identity as its documented fallback instead of taking min of an empty set. Test tests/preprocessing/test_cluster26_power_search_identity_fallback.py. |
| 26-F31 | FIXED | A multiclass calibration whose labels cannot be aligned to its columns leaves those columns identity, as the code comment says, instead of calibrating against a guessed label. Test tests/training/test_cluster26_calibration_misaligned_labels_identity.py. |
| 26-F24 | FIXED | The prediction-stability chart is refused, with a warning naming both row counts, when the target and member predictions differ in length, instead of pairing rows by position after truncation. Test tests/reporting/test_cluster26_stability_refuses_mismatched_rows.py. |
| 26-F9 | FIXED | The docstring now states the caller contract: last non-null is in row order, so the result is for all-null groups only. Behaviour is pinned by a test. Test tests/preprocessing/test_cluster26_cold_start_partial_group_contract.py. |
| 26-F8 | FIXED | The binary-column fill restores the column's current dtype rather than the head snapshot taken before the category conversion. The test checks the recorded dtype on a plain object column; the category-conversion path is not exercised by the test. Test tests/preprocessing/test_cluster26_binary_fill_keeps_current_dtype.py. |
| 26-F13 | FIXED | The opt-in registry lists the fit/apply halves that exist (outlier cap, rare collapse, missing indicator, regime median, outlier bounds) and UnseenCategoryImputer. Entries that do not exist as named in the audit were not invented; is_tree_based_model is a utility, not opt-in. Test tests/preprocessing/test_cluster26_opt_in_registry_resolves.py. |
| 26-F20 | FIXED | An auto-keyed share() compares the content of the reused segment; a rewritten buffer gets its own segment and key, so pending tasks keep their snapshot. Tests tests/reporting/test_cluster26_shared_array_content_check.py; existing async render queue tests pass (26 passed, 1 skipped). |
| 26-F20 | FIXED | An auto-keyed share() compares the content of the reused segment; a rewritten buffer gets its own segment and key, so pending tasks keep their snapshot. Tests tests/reporting/test_cluster26_shared_array_content_check.py; existing async render queue tests pass (26 passed, 1 skipped). |
| 26-F21 | DEFERRED | Sharing nested arrays inside a spec through SharedArrayPool needs a spec-tree walk and a descriptor format for nested fields. That is a design change to the process backend, deferred for its own pass. |
| 27-F10 | FIXED | A non-finite value in any seed curve raises naming the seed, instead of producing a stability verdict. Test tests/training/test_cluster27_cv_stability_rejects_nan_curve.py. |
| 27-F15 | FIXED | The LightGBM factories import the plain class when the shim is off, instead of returning the unbound module name. Test tests/training/test_cluster27_lgb_factory_without_shim.py. |
| 27-F16 | FIXED | The MSLE to rmse substitution for LightGBM is logged at WARNING. Test tests/training/test_cluster27_lgb_msle_substitution_warns.py. |
| 27-F10 | FIXED | A non-finite value in any seed curve raises naming the seed, instead of producing a stability verdict. Test tests/training/test_cluster27_cv_stability_rejects_nan_curve.py. |
| 27-F15 | FIXED | The LightGBM factories return the lightgbm class when the shim is off, instead of the unbound module name. Test tests/training/test_cluster27_lgb_factory_without_shim.py. |
| 27-F16 | FIXED | The MSLE to rmse substitution for LightGBM is logged at WARNING. Test tests/training/test_cluster27_lgb_msle_substitution_warns.py. |
| 27-F11 | DEFERRED | Hashing every row needs a content hash that keeps the documented pandas and polars equality invariant, and a per-row Python hash is too slow on large frames. A design pass on the fingerprint is needed; the 64-row sample stays until then. |
| 27-F9 | FIXED | The pandas NaN fill works on a shallow copy, so the caller's frame keeps its NaNs, as the polars branch already did. Test tests/training/test_cluster27_nan_fill_leaves_caller_frame.py. |
| 27-F7 | FIXED | A weighted fit's permutation importances skip the unweighted CUDA kernel and use the weighted CPU path. Test tests/training/test_cluster27_weighted_fi_skips_cuda.py. |
| 27-F8 | DEFERRED | Owner decision: signed or magnitude coef_ importances. The two branches disagree today; choosing either changes the reported numbers for one of them. |
| 27-F13 | DEFERRED | Owner decision: refit on the full train, or correct the documented contract. The dichotomic path keeps the 90 percent model today, and history has mixed types. |
| 28-F42 | FIXED | A failed text-feature probe logs at WARNING with its traceback. Test tests/training/test_cluster28_probe_and_names_warn.py (covers the names path; the probe path shares the same change). |
| 28-F49 | FIXED | An unreadable feature-name list logs at WARNING, and the attribute read itself is inside the guard. Test tests/training/test_cluster28_probe_and_names_warn.py. |
| 28-F40 | FIXED | The logging proxy raises AttributeError for its unset _inner slot instead of recursing. Test tests/training/test_cluster28_logging_proxy_and_rss.py. |
| 28-F39 | FIXED | A failed post-call RSS re-read reports rss_mb and d_rss_mb as None, not as zero. Verified by reading the change; no unit test, since it needs a psutil failure mid-call. |
| 28-F28 | FIXED | A digest mismatch on read logs at WARNING and deletes the entry. Test tests/training/test_cluster28_suite_cache_keys_and_tamper.py. |
| 28-F29 | FIXED | The default key builder digests array and frame arguments from their bytes; repr elided the middle of large arrays. Test tests/training/test_cluster28_suite_cache_keys_and_tamper.py. |
| 28-F41 | FIXED | Every grid label is planned and checked before the first variant runs, so a duplicate refuses the grid with nothing trained. Test tests/training/test_cluster28_grid_labels_checked_before_runs.py. |
| 28-F43 | FIXED | A rejected CatBoost set_params is logged at WARNING before the private _init_params fallback is written. Test tests/training/test_cluster28_cb_set_params_fallback_logged.py. |
| 28-F35 | FIXED | The drift helper converts nullable pandas columns to float with NaN for pd.NA, so the sensor is not dropped. Test tests/training/test_cluster28_drift_nullable_column.py. |
| 28-F33 | FIXED | The binary default is applied whenever no task key is given, after the time budget is merged, so a budget no longer drops it. No local test: lightautoml is not installed here; the test skips without it. Verified by reading the change. |
| 28-F32 | FIXED | The test AUC is computed whatever the verbosity, and only its log line is gated. Verified by reading; the AutoML path needs lightautoml or autogluon, which are not installed here, so no unit test. |
| 28-F46 | FIXED | Category alignment always takes a shallow copy before reassigning a column; the old flag shortcut is gone. Test tests/training/test_cluster28_category_alignment_leaves_caller_frame.py; 12 related alignment tests pass. |
| 28-F45 | FIXED | The text-column decategorisation writes into a shallow copy, so the caller's eval-set frame keeps its categories and NaN. Test tests/training/test_cluster28_decategorize_leaves_caller_frame.py. |
| 28-F34 | FIXED | The OOF target is read with the same inner-model fallback as the OOF probabilities, through one helper used at both sites. Test tests/training/test_cluster28_oof_target_found_like_probs.py. |
| 28-F44 | FIXED | Both CatBoost metric sites keep a user-set eval_metric and set only the loss when one exists; the MTR default is used only when none is set. Test tests/training/test_cluster28_cb_keeps_user_eval_metric.py covers the MTR site; the multilabel site uses the same guard. |
| 28-F38 | DEFERRED | Owner decision: scale the Huber floor to the target's robust scale, or document it as a raw-unit floor. Either changes every booster's loss on small-scale targets. |
| 28-F25 | FIXED | A bundle that cannot be digested fails the save with a WARNING, so no unverifiable sidecar is written and a later load does not raise. Test tests/training/test_cluster28_save_fails_without_digest.py; 5 related sidecar tests pass. |
| 28-F27 | DEFERRED | The whole loader body sits inside _LOAD_MODEL_CACHE_LOCK, so narrowing the lock to the cache get and put means restructuring its control flow. A dedicated change, deferred. |
| 28-F26 | DEFERRED | Streaming the hash and the unpickle from one handle changes the verified-open contract the loader relies on. Deferred to a dedicated change to the bundle reader. |
| 28-F31 | FIXED | The multiclass split summary drops non-finite labels before counting and reports them as n_missing, as the binary summary does. Test tests/training/test_cluster28_multiclass_summary_drops_missing_labels.py. |
| 28-F36 | FIXED | A host-RAM fallback from a failed GPU memory probe is returned but not cached under the cuda:N key. Test tests/training/test_cluster28_mem_probe_fallback_not_cached_as_gpu.py; 19 tests pass with the existing memory-probe tests. |
| 28-F21 | FIXED | The pandas inf-to-NaN replacement writes into a shallow copy, so the caller's frame keeps its infinities. Test tests/training/test_cluster28_special_values_leave_caller_frame.py. |
| 28-F19 | FIXED | When both refit paths fail, the saturated-best-iteration refit restores the model's parameters, as the degenerate path does. Verified by reading; no unit test, since it needs a CatBoost-shaped loss fallback. |
| 28-F30 | FIXED | Eviction first adopts .pkl entries on disk that this process's ledger does not know, as the oldest, so they count against and can be evicted from the byte budget. Test tests/training/test_cluster28_cache_adopts_other_process_entries.py; 5 cache tests pass. |
| 29-F21 | FIXED | agreement_rate refuses k below 1 with a ValueError. Test tests/votenrank/test_cluster29_agreement_k_validation.py. |
| 29-F24 | FIXED | create_experiment raises NotImplementedError instead of silently storing nothing; its stub test now expects the refusal. Tests tests/utils/test_experiments.py (23 pass with the votenrank test). |
| 29-F18 | FIXED | A blank GloVe line is skipped instead of aborting the file, and skipped lines are counted in a WARNING. Test tests/utils/test_cluster29_glove_blank_and_bad_lines.py. |
| 29-F11 | FIXED | An unpickling failure from a renamed or removed class (AttributeError, ImportError and similar) is a miss that drops the entry, instead of raising out of get(). Test tests/utils/test_cluster29_disk_cache_stale_class_is_a_miss.py. |
| 29-F9 | FIXED | A hash-seed request logs a WARNING that PYTHONHASHSEED only reaches child processes started afterwards and leaves this process's string hashing unchanged. Test tests/utils/test_cluster29_hash_seed_warns.py. |
| 29-F26 | FIXED | The fingerprint width is the reshaped matrix width, so a 3-D input reports the width its statistics use. Test tests/utils/test_cluster29_param_oracle_width_is_the_stat_width.py. |
| 29-F10 | FIXED | put catches any pickling failure, removes its temp file and logs at WARNING, so an unpicklable value neither raises nor leaves an unreclaimable tmp_ file. Test tests/utils/test_cluster29_disk_cache_unpicklable_value.py. |
| 29-F17 | FIXED | The numba cache dependency scan reads 'from . import b' as a dependency on b, as well as 'from .b import x'. Test tests/utils/test_cluster29_numba_guard_sees_bare_package_imports.py. |
| 29-F16 | FIXED | The cuBLAS probe failure is logged once per process, so the library lookup no longer runs on every matmul check after a failed probe. Test tests/utils/test_cluster29_cublas_failure_logs_once.py. |
| 29-F23 | FIXED | preprocess_glue and preprocess_sglue copy the frame before renaming duplicate models, so the caller's Model column is unchanged. Test tests/votenrank/test_cluster29_glue_preprocess_keeps_caller_frame.py (glue path; the sglue path is the same change). |
| 29-F19 | FIXED | The bare asserts in geometric_weight_blend are explicit ValueErrors naming the cause (all restarts non-finite, or alpha missing). Test tests/votenrank/test_cluster29_geometric_blend_refuses_nan_loss.py. |
| 29-F15 | DEFERRED | The only caller uses the default k=3, so the parameter is effectively a constant. A k-neighbour vote changes every recommendation, and removing the parameter changes the signature; both need a decision on the recommender's behaviour. |
| 29-F12 | FIXED | The array content hash feeds the header and the array buffer to xxhash incrementally, avoiding two full copies per lookup. The digest equals the one-shot digest, so existing keys are unchanged. Test tests/utils/test_cluster29_array_hash_matches_oneshot.py. |
| 29-F22 | FIXED | The tracker table reads only run 0 and lists the other runs in a WARNING, instead of reading a later run as run 0. Test tests/votenrank/test_cluster29_tracker_reads_only_run_zero.py; it skips here because experiment_impact_tracker is not installed, so the change is verified by reading only. |
| 29-F20 | FIXED | Smoothed rank percentiles clamp to the same [0.5/n, (n-0.5)/n] band as the hard mode, so the stacker never sees 0 or 1 it was not trained on. The O(n_query*n_oof) cost is unchanged and is not addressed here. Test tests/votenrank/test_cluster29_smoothed_percentiles_stay_in_band.py. |
| 29-F13 | DEFERRED | Each record rewrites the whole parquet store under a lock, and each recommend reads all of it. Fixing that means a change to the store's storage design (append-only or indexed), which needs its own design pass. |
| 29-F7 | FIXED | The stability figure passes the x-axis range once, not repeated 100 times. Test tests/votenrank/test_cluster29_stability_axis_and_reference.py. |
| 29-F8 | FIXED | The stability reference ranking comes from the unmodified table's rank_all, not from the first noised entry of exp_range. Verified by reading; no unit test, since the leaderboard setup is heavy. Test file tests/votenrank/test_cluster29_stability_axis_and_reference.py covers F7 only. |
| 29-F25 | DEFERRED | The shared DMatrix must be mutated and trained inside one lock. That wraps the xgb.train call, and a threaded regression test needs a dedicated harness, so it is deferred to its own change. |
| 30-F1 | FIXED | The DTW bench prints measured pool bytes; the analytic float32 overwrite is removed. Bench output; verified by reading, not by a unit test (needs a GPU). |
| 30-F2 | FIXED | The analytic overwrite of the measured memory is removed; the printed values are the measured pool numbers. Bench output; verified by reading. |
| 30-F4 | FIXED | The GP bench builds its dataset before the timer, so the wall time covers smoothing only. Bench output; verified by compiling and reading. |
| 31-F6 | FIXED | A failed cold-cache check records the registry keys as checked, so no new thread starts per fit, and logs at WARNING. Test tests/system/test_cluster31_cold_check_failure_is_recorded.py. |
| 31-F9 | FIXED | refresh-all prints the count retune_all actually tuned and fails (exit 1) when nothing was tuned. Test tests/system/test_cluster31_refresh_all_reports_what_it_tuned.py. |
| 31-F4 | FIXED | An unwritable location for the tuning lock's pid file logs a WARNING and takes no lock, instead of raising out of ensure. The _lock_path mkdir is unchanged. Test tests/system/test_cluster31_lock_write_failure_is_not_raised.py. |
| 31-F7 | FIXED | Discovery captures a guarded module's state without importing it, as its docstring requires; an unloaded module stays unloaded. Test tests/system/test_cluster31_discovery_does_not_import_guarded.py. |
| 31-F5 | DEFERRED | A fresh cache load after each sweep is needed for correctness: each sweep runs in a child process that writes the cache to disk, and the parent must reload to count the regions. Reusing one object would read stale data. The fix is a reload method on KernelTuningCache, which lives outside this module. |
| 31-F8 | DEFERRED | The module walk repeats in each sweep child, which is a separate process, so a per-process cache does not help. Passing the discovered module list into the children changes how sweeps launch, so it is deferred. |
| 28-F20 | FIXED | With balance_proxy the elkan_noto prior is NaN instead of the balanced proxy's mean scaled by c, which is not a prior. Test tests/training/test_cluster28_pu_prior_nan_when_balanced.py. |
| 28-F22 | FIXED | Lasso and elasticnet classifiers refuse alpha <= 0 with a ValueError naming the unpenalised alternative, instead of silently using C=1.0. Test tests/training/test_cluster28_linear_classifier_alpha_positive.py. |
| 28-F23 | DEFERRED | Uncalled function: project rule is to wire it in or classify it before deleting. Classification is deferred to the uncalled-functions pass. |
| 28-F24 | DEFERRED | The TTA duplication needs the fused function kept and the others made thin wrappers, or removed; that touches a public API and the bench that calls it. Deferred. |
| 28-F47 | DEFERRED | Duplicated GTC cache state in the helpers module: deletion needs a check that nothing imports it. Deferred to the uncalled-functions pass. |
| 28-F48 | DEFERRED | report_title is built and never emitted: wiring it in changes the train report, so it is an owner decision (emit or delete). |
| 32-F10 | FIXED | Integer timestamps go through the same unit detector as object timestamps, so epoch seconds are no longer read as nanoseconds. Test tests/training/test_cluster32_integer_epoch_seconds_unit.py. |
| 32-F26 | FIXED | The smoke script exits 1 when the suite raises instead of printing a pass line and returning 0. Verified by reading; the smoke run is a manual end-to-end script, not a unit test. |
| 33-F3 | FIXED | The conformal rank rule lives in one helper (conformal_rank), used by the split, signed and online calibrations. Two further copies in conformal_classification.py and conformal_glm.py are outside the finding and are not yet changed. Tests: tests/training/composite/test_biz_val_conformal_ood.py (16 pass). |
| 36-F5 | FIXED | A failed candidate fold logs at WARNING with the exception class, not DEBUG. Verified by reading; no unit test (the optimiser path needs a full composite fit). |
| 36-F7 | FIXED | Imputed NaN meta-feature rows in the stacker are counted and logged at WARNING. Verified by reading; no unit test. |
| 36-F6 | FIXED | More CV folds than rows raises ValueError naming both numbers, instead of an empty-fold search returning an arbitrary winner. Test tests/training/composite/test_cluster36_hpo_refuses_more_folds_than_rows.py. |
| 34-F7 | FIXED | Under 50 valid rows the spec still passes unscored, but now logs a WARNING naming the spec and row count (test_cluster34_discovery_silent_skips_are_logged) |
| 34-F8 | FIXED | Residual-std probe failure logged at WARNING with base and transform (test_cluster34_discovery_silent_skips_are_logged) |
| 34-F5 | FIXED | Negative train/val/test positions are wrapped by len(df) and bounds-checked before the overlap test (test_cluster34_negative_indices_are_positions) |
| 34-F6 | FIXED | Docstring corrected: the holdout X_remaining is not the in-screen dedup/carry-masked set, so the two stamped numbers are not directly comparable. Feature-set alignment not implemented (design change). |
| 34-F1 | FIXED | Trailing mean skips NaN and divides by the finite count (commit 9c32cf0b1). Test tests/training/test_cluster34_trailing_mean_nan.py. |
| 34-F2 | FIXED | Hints dropped by the finite filter are removed before slots and before the return (commit 9c32cf0b1). Test test_cluster34_auto_base_mrmr_and_collinear.py. |
| 34-F3 | FIXED | Demoted candidates ranked after the MRMR pass, outside the relevance scale (commit 9c32cf0b1). Test test_cluster34_auto_base_mrmr_and_collinear.py. |
| 34-F4 | FIXED | Collinear replay re-checks every column exactly (commit 9c32cf0b1). Test test_cluster34_auto_base_mrmr_and_collinear.py. |
| 34-F9 | PARTIAL | Per-fold refit failure logged at WARNING with exception class (test_cluster34_fold_refit_failure_is_visible). Not done: marking the spec as leaked-param-scored; unreachable from default discovery since grouped transforms are rejected upstream. |
| 34-F10 | FIXED | Upgraded multi-base spec scored with the final kept set's forward-stepwise CV RMSE; seed score only when no base accepted (test_cluster34_multibase_upgrade_score) |
| 34-F11 | FIXED | Auto block search fits folds with sample_weight and pools a weighted fold RMSE; unweighted path unchanged (test_cluster34_direct_multi_horizon_weighted_search) |
| 34-F12 | FIXED | Column set built once per spec in the rescore; call-count regression test (test_cluster34_incremental_column_set_built_once) |
| 34-F18 | FIXED | Degenerate fits (disabled, or under 50 rows) publish specs_, report_, val/test idx, elapsed_seconds_ and signature bookkeeping via _fit_publish_empty (test_cluster34_degenerate_fit_publishes_attributes) |
| 34-F24 | FIXED | domain_check output cast to bool at both unguarded eval sites (test_cluster34_domain_mask_is_boolean) |
| 34-F22 | FIXED | Duplicate logger and _UNARY_BASE_SENTINEL definitions removed; single-definition test (test_cluster34_fit_steps_single_definitions) |
| 34-F13 | FIXED | Per-group loops use one stable argsort and contiguous slices; routing extracts each base column once (test_cluster34_route_extracts_base_once) |
| 34-F31 | FIXED | Cap checked before appending, so a full hint list logs 0 MI candidates (test_cluster34_auto_base_log_and_wording) |
| 34-F25 | FIXED | Messages now say skipping spec. Wording only; no behaviour change, so no separate test. |
| 34-F23 | FIXED | Comment corrected: np.argsort places non-finite keys last. Comment only; no behaviour change, so no separate test. |
| 34-F19 | FIXED | Null permutes the base over the scored rows only (per column), so NaN base entries cannot enter the null (test_cluster34_base_carry_null_rows; the test is behavioural and may not fail on the old code) |
| 34-F27 | FIXED | Shared-fold cache keys use a blake2b digest of the rows instead of 64-bit hash() (test_cluster34_fold_cache_row_digest; 13 existing shared-fold tests pass) |
| 34-F29 | FIXED | MatrixKeyedCache implements its own clear(); no borrowed ContentMemo method (test_cluster34_matrix_cache_clear; 4 existing matrix-cache tests pass) |
| 34-F30 | FIXED | Attached target kept float64 on the val side (keep_float64); engineered bases stay float32 by design (test_cluster34_attach_keeps_target_float64) |
| 34-F15 | FIXED | Tie ranking vectorised for finite input (NaN keeps the walk); PAVA compiled via njit_or_passthrough (test_cluster34_leakage_rank_and_pava; 62 existing leakage tests pass) |
| 34-F17 | DEFERRED | Batched null kernel shuffles blocks of binned codes; the carry null shuffles raw base values and bins inside _mi_pair_bin. Switching changes the null distribution, so it needs a benchmark and owner sign-off on numerics. |
| 34-F26 | FIXED | Helper renamed to _run_folds_serial_early_stop (rename only; no behaviour change, so no separate test) |
| 34-F28 | FIXED | BoundedMemo builds outside the lock; first stored value wins on a race. Re-entrancy and single-build tests (test_cluster34_bounded_memo_build_outside_lock) |
| 34-F34 | FIXED | Eviction test added: LRU order kept, evicted key rebuilt (test_cluster34_bounded_memo_eviction) |
| 34-F35 | FIXED | Bit-identity-to-reference test on float32 strided input (test_cluster34_abs_corr_no_copy_matches_reference) |
| 34-F32 | FIXED | NEW arm built explicitly as old pool plus widened transforms via _full_new_pool (test_cluster34_bench_new_pool_is_explicit) |
| 34-F33 | FIXED | Strict-win counts computed per scenario and per side in _strict_win_counts (test_cluster34_bench_strict_wins_per_scenario) |
| 35-F6 | FIXED | Stacked target float64 at both fit entry points; AST guard test (test_cluster35_stacked_target_float64_guard) |
| 35-F15 | FIXED | _kwargs bound before the try; non-finite weights refused; both fallbacks logged at WARNING (test_cluster35_sample_weight_fallback; 69 related tests pass) |
| 35-F16 | FIXED | Group counts from one np.unique pass via _group_counts; equivalence test (test_cluster35_group_counts_one_pass) |
| 35-F17 | FIXED | predict logs a WARNING naming the component and the count of non-finite rows; blend still NaN on those rows (test_cluster35_predict_nonfinite_component_warns) |
| 35-F18 | FIXED | GPU probe failures (xgboost, catboost) and the refit-groups fallback now log at WARNING; xgboost probe path tested (test_cluster35_gpu_probe_failure_warns); refit-groups path not separately tested |
| 35-F20 | FIXED | GBM stacker honours group_ids and time_ordered: row-random early stopping is off when either is set; dispatcher forwards both (test_cluster35_gbm_stacker_group_aware, 3 pass) |
| 35-F21 | FIXED | OOF cache key now includes a full digest of sample_weight and of the component specs (arrays digested in full, nested dicts walked); tests test_cluster35_oof_cache_key_weights_and_specs.py; 48 existing cache-key tests pass |
| 35-F22 | FIXED | Auto gate measures residual correlation (y - prediction), as its protocol defines; test test_cluster35_auto_gate_measures_residuals.py; 4 existing auto-aware gate tests pass |
| 35-F23 | FIXED | Requested calibration method stored at construction and restored at the start of each fit; a degenerate fit no longer changes later fits (test_cluster35_calibrator_method_restored_per_fit; 27 existing calibration tests pass) |
| 35-F24 | FIXED | Failed lag-column extraction logged at WARNING via log_throttle, naming the column and exception (test_cluster35_lag_extraction_failure_warns; 27 existing soft-shrink tests pass) |
| 35-F25 | FIXED | A base fit range stamped on the spec is kept; capture runs only when the key is absent (test_cluster35_from_fitted_keeps_stamped_range; 48 related tests pass) |
| 35-F27 | FIXED | predict_quantile resolves the causal lag once per batch and passes it to apply_smart_fallback; default path unchanged (test_cluster35_smart_fallback_resolves_lag_once; 22 soft-shrink and parity tests pass) |
| 35-F29 | FIXED | Mixture decides sample_weight support from the fit signature; unrelated TypeError propagates; unweighted regressors logged at WARNING (test_cluster35_mixture_sample_weight_support; 44 existing mixture tests pass) |
| 35-F30 | FIXED | discover_and_wrap coerces a dict config before reading the seed, so the seed and report use the dict's random_state (test_cluster35_highlevel_dict_config_seed; 201 existing highlevel tests pass) |
| 35-F31 | FIXED | Group broadcast is one vectorised reindex; NaN group ids are counted and logged at WARNING (test_cluster35_group_broadcast_vectorised; 6 existing group-aggregate tests pass) |
| 35-F32 | FIXED | One numeric column returns that column as one subset instead of raising in linkage (test_cluster35_subset_bagging_single_column; 17 existing bagging tests pass) |
| 35-F34 | FIXED | Worst-candidate choice uses the judge's fill (fit rows that the judge uses) and treats a NaN score as worst, so it no longer depends on order (test_cluster35_worst_candidate_nan_order; 142 existing y-scale gate tests pass) |
| 35-F35 | PARTIAL | Spec with fewer than 50 in-domain fit rows is now rejected with a ledger entry instead of passing unchecked. No direct test: that branch needs a full gate fixture (a transform whose domain keeps under 50 rows). 142 existing y-scale tests pass. |
| 35-F36 | PARTIAL | Multi-base specs still pass the structural fragility gate unjudged; the bypass is now logged at WARNING. Gating on the primary base changes which specs ship, so it is left as an owner decision. Test test_cluster35_structural_gate_multibase_logged.py; existing y-scale gate tests pass. |
| 35-F37 | FIXED | WAIC scoring failures (transform lookup and forward) logged at WARNING with the spec name and exception (test_cluster35_waic_failure_logged; existing WAIC tests pass) |
| 35-F38 | PARTIAL | The kNN MI pair mask is present in the kNN loop of _mi_to_target (origin's version). The de-duplication into _mi_per_feature_knn was not carried over: origin already changed that loop, so the shared-helper refactor and its test are not on master. Left as a follow-up. |
| 35-F39 | FIXED | Both stacking passes share one ranking helper (_rank_pass1_specs) and one group-id derivation (_train_group_ids); the pass-specific cap, warnings and aggregation stay in each caller (test_cluster35_stacked_shared_helpers; existing stacking tests pass) |
| 35-F40 | FIXED | Group path and default path use one helper (_polars_default_cat_candidates), so the two candidate sets cannot drift; temporal columns are excluded on both (test_cluster35_polars_candidates_one_helper; 67 existing auto-detect tests pass) |
| 35-F41 | FIXED | Non-finite (y, base) pairs dropped before the posterior, counted in n_dropped_nonfinite, logged at WARNING (test_cluster35_bayesian_nonfinite_dropped; 27 existing bayesian tests pass) |
| 35-F42 | DEFERRED | A broken process pool serially rescores every task, not only the unfinished ones. Documented as deliberate, and correctness is unaffected. Rescoring only the unfinished tasks needs partial results from the broken pool, which is a design change; owner decision on whether the time cost is worth it. |
| 35-F43 | FIXED | Smoke tests for predict_cdf with enforce_non_crossing=False, mixture soft_routing=True, and gated outlier calibrate_classifier=True (test_cluster35_untested_paths_smoke; 3 pass; test-only change) |
| 35-F45 | FIXED | Bootstrap posterior result carries posterior_kind, n_samples, degrees_of_freedom, sigma2_posterior_mean and n_dropped_nonfinite, like the conjugate one (test_cluster35_bootstrap_result_keys; 27 existing bayesian tests pass; the one failure is the pre-existing uncalled-function gate) |
| 35-F47 | FIXED | Dead _groups_screen = None removed (passed as a literal None) and blank line added; the null-canary gate needed a reasoned exemption for _select_worst_candidate (F34). 231 tiny-rerank tests pass. |
| 35-F48 | FIXED | Unused Ridge import removed; __all__ rebuilt as the public surface; cache-key block moved to ensemble/_oof_cache_keys.py so the facade is under the 1k module gate (955 lines), with re-exports (test_cluster35_ensemble_public_surface; module-split and 13 digest/cache tests pass) |
| 35-F49 | FIXED | Uncached transform-pair path forwards refit_supervised like the memoised path (test_cluster35_uncached_pair_forwards_refit; 331 existing ensemble tests pass) |
| 35-F50 | FIXED | Write-only _sig_degenerate attribute deleted; no reader anywhere. Existing 27 output-calibration tests pass; removal only, so no separate test. |
| 35-F51 | FIXED | n_oof_rows counts rows finite in y AND every OOF column, the rows the fit used (test_cluster35_meta_stack_row_count; 59 existing meta-stacker tests pass; the one failure is the pre-existing uncalled-function gate) |
| 35-F52 | PARTIAL | Subsample bench fits weights against the returned holdout target, row for row (test_cluster35_bench_subsample_targets_align). Calibration bench already uses crc32 (no salted hash() left in the ensemble benches). Lasso/elasticnet bench duplication not consolidated. |
| 35-F53 | FIXED | Dead getattr(self, 'smearing', True) gate removed: smearing always applies to smeared transforms (test_cluster35_smearing_always_on; existing smearing and estimator-edge tests pass) |
| 35-F54 | FIXED | predict before fit raises sklearn NotFittedError (a ValueError subclass, so existing catches still work) (test_cluster35_dual_direction_not_fitted). 31 existing dual-direction tests pass; 2 fail inside the numba metrics path at fit time (untyped global length_of in metrics/regression/_regression_metrics.py), unrelated to this change and not reproduced by it. |
| 35-F55 | PARTIAL | One shared subset_rows in _frame_ops for both gated models (test_cluster35_shared_subset_rows; 61 existing gated tests pass, one pre-existing numba failure confirmed on master). The glm _is_polars_df re-export is kept because a test imports it. |
| 35-F56 | WON'T FIX | Docstring cites the defining module, varying_size_top_k_subsets._cluster_anchors, which is correct. The import through feature_selection.shared is a re-export of that same function. No drift to fix. |
| 35-F57 | FIXED | Pointless try/except around list(names) removed (list() of an attribute does not raise here); 13 existing extremes tests pass; removal only, so no separate test. |
| 36-F9 | FIXED | Class factories are inspected for sample_weight from the fit signature without constructing an estimator; callables are still called (test_cluster36_factory_signature_no_construction; 44 existing multi-stage tests pass, one pre-existing numba failure) |
| 37-F1 | FIXED | Median fill for out-of-domain T applies only to rows outside the train split; a training row stays missing (test_cluster37_composite_t_no_train_label_fill). Existing test pinned the old fill and was updated to put the violating row outside train; 281 existing composite phase tests pass, 1 pre-existing numba failure. |
| 37-F2 | FIXED | A lag supplied as an array with no lag column in the frame is attached as a named column and offered as the causal base (test_cluster37_supplied_lag_is_a_base; 50 existing achievable-ceiling tests pass) |
| 37-F3 | FIXED | Calibration on non-finite positive-class scores is refused with ValueError; NaN no longer becomes a grid point. The lookup already maps NaN to NaN (test_cluster37_venn_abers_nan_refused; 152 existing Venn-Abers tests pass) |
| 37-F4 | FIXED | Member score uses one source for every member: val score only when all members have it, else OOF for all (test_cluster37_diversity_single_score_scale; existing diversity tests pass) |
| 37-F9 | FIXED | Render scope entered inside the try; cleanup guarded when entry raised (_main_train_suite.py) |
| 37-F10 | FIXED | Pandas leaderboard copied before inserting target columns (_main_train_suite_phases.py) |
| 37-F11 | FIXED | Knob overrides applied via model_copy only; caller config not mutated (test_cluster37_knob_overrides_no_caller_mutation.py) |
| 37-F12 | FIXED | Comment corrected: pandas df[list] copies; reorder only when columns are extra or out of order |
| 37-F13 | FIXED | Group subsample drops the last-added group when kept groups cover all rows; returns None if one group spans all (test_cluster37_subsample_groups_and_nan_gate.py) |
| 37-F14 | FIXED | Non-finite ensemble RMSE counts as a gate failure (test_cluster37_subsample_groups_and_nan_gate.py covers subsample; NaN branch in gate code) |
| 37-F15 | FIXED | OOF gate exception logged at WARNING with type and marked UNVALIDATED; metadata stamp not added |
| 37-F16 | FIXED | leaky RMSE pre-check failure logged at WARNING per component |
| 37-F17 | FIXED | NNLS RuntimeError caught per column; that column alone falls back to equal-mean |
| 37-F18 | PARTIAL | Each full-length base column built once per name across specs (cache). Still allocates full n per distinct base; index-subset build needs a benchmark |
| 37-F19 | FIXED | Pandas unseen-category drift via _compute_drift_rows_pandas (test_cluster37_pandas_unseen_category_drift.py) |
| 37-F20 | FIXED | Ignored plot_format_subfolders / calibration_colormap now logged at WARNING |
| 37-F21 | FIXED | Numba prewarm runs when dummy baselines OR feature selection will run |
| 37-F22 | FIXED | Label distribution drift wrapped best-effort with WARNING |
| 37-F23 | WON'T FIX | MTRPerColumnEqualMeanEnsemble.fit is the public E3 API, exercised by tests/training/test_mtr_nnls_ensemble.py; production path uses compute_mtr_oof_nnls_weights. Premise 'no caller' holds only for production; semantics differ (holdout fit, no exclusion), so routing would change behaviour |
| 37-F24 | FIXED | unpack_ctx_configs_to_locals is already classified as intentionally unused in tests/test_meta/test_shared_uncalled_functions.py allowlist with reason |
| 37-F50 | FIXED | Redundant tier alias removed; no behaviour change |
| 37-F52 | FIXED | Binary-only error message no longer points at out-of-scope one-vs-rest; Venn-Abers tests pass |
| 38-F1 | FIXED | Suite-wide ensemble not built across targets (commit 9d0472dca; test_cluster38_suite_ensemble_targets.py) |
| 38-F2 | FIXED | Recalibrated calib predictions no longer re-stamped for conformal residuals (commit 9d0472dca) |
| 38-F3 | FIXED | fit_predict-only detector skips the val OD filter explicitly with WARNING (commit 9d0472dca) |
| 38-F4 | FIXED | Warning now states the caller keeps the prior ensemble, matching behaviour; no test |
| 38-F5 | FIXED | Bool-to-int8 cast runs on a shallow copy; test_cluster38_predict_and_fit_side_effects.py |
| 38-F6 | FIXED | Empty per-type dict skipped before taking the first item; test_cluster38_predict_and_fit_side_effects.py |
| 38-F7 | FIXED | PySR y-length mismatch and extraction failure logged at WARNING regardless of verbose; no test |
| 38-F8 | FIXED | Non-neural time reset moved out of the pre-pipeline loop (once per target); no test |
| 38-F11 | FIXED | Save removes the other-format metadata file so readers cannot load stale metadata; no test |
| 38-F12 | FIXED | Embedding detection requires a non-scalar shape; no test |
| 38-F13 | FIXED | Failed split extraction for a high-cardinality column logged at WARNING; failed split not recorded in metadata; no test |
| 38-F14 | FIXED | Composition record states described_entry=0 and entry_count; no test |
| 38-F15 | FIXED | No bare model_clone appended on all-split prediction failure; skip is the single visible outcome; no test |
| 38-F16 | FIXED | Temporal audit gets the real target type for regression and binary; other types skipped; no test |
| 38-F17 | FIXED | set() hoisted out of comprehensions (4 sites); no test |
| 38-F18 | FIXED | Dead metadata.get expression removed |
| 38-F19 | FIXED | Duplicate logger assignment removed |
| 38-F20 | FIXED | Docstring corrected: cache holds the ndarray, no strong-ref window; no test |
| 38-F23 | FIXED | Documented: only the first regression target drives bucket stratification |
| 38-F25 | FIXED | Narrowing signature suffix widened from 4 to 8 hex characters |
| 38-F26 | FIXED | AssertionError carries a message; pure-selector warning text corrected |
| 38-F27 | FIXED | Threshold tuning skip on filtered subset logged and stamped default_0.5; no test |
| 39-F20 | FIXED | Persistence on with no DiskKey logs a throttled WARNING that the disk tier is unused (feature_handling/cache.py); tests/training/test_cluster39_disk_tier_key_warning.py |
| 41-F4 | FIXED | Skew branch now evaluated before the mild-kurtosis and platykurtic returns, so strong skew is no longer masked (residual_audit _diagnose); tests/training/targets/test_cluster41_skew_not_masked_by_kurtosis_band.py |
| 41-F1 | FIXED | Predict maps categorical-dtype NaN to the fit-time sentinel, as fit does (training/ranking/ranking.py predict_ranker_scores); tests/training/ranking/test_cluster41_catboost_predict_nan_categorical.py |
| 41-F5 | FIXED | Per-group loops use one stable sort and contiguous slices instead of G full-length masks (_target_distribution_analyzer_modes.py, _target_distribution_analyzer_stats.py); tests/training/targets/test_cluster41_group_stats_match_reference.py |
| 20-F1 | WON'T FIX | Not reproduced: the label swap was tried and reverted. The WORM verdict already labels a long positive-tail residual RIGHT-SKEW (tests/reporting/test_worm_plot_verdict_is_operative.py pins it, 13 pass on HEAD), so the finding's premise fails |
| 22-F3 | FIXED | Each optional calibration library is imported separately; an absent one skips only its candidates (calibration/post.py get_postcalibrators); tests/calibration/test_cluster22_missing_calibrator_library_skipped.py |
| 22-F3 | FIXED | Each optional calibration library is imported on its own; an absent one skips only its candidates (calibration/post.py get_postcalibrators); tests/calibration/test_cluster22_missing_calibrator_library_skipped.py |
| 18-F2 | PARTIAL | Multiclass (N,K>2) member probabilities now produce an explicit warning that the NNLS/Caruana weight fit does not support them (score_flavours.py run_stacking_aware_gate), instead of blaming the row count. Per-class weighting is not built; tests/models/test_cluster18_multiclass_gate_skip_is_explicit.py |
| 40-F27 | FIXED | Per-column float32 fallback rebuilds the frame from its column labels instead of assign(**kwargs), which needed string keys (training/neural/data.py); tests/training/neural/test_cluster40_int_column_labels_convert.py |
| 14-F1 | FIXED | su_seeded_snr_z defaults to None (unset); an explicit value always wins, so an explicit 3.0 under proxy_mode=auto is no longer replaced by 4.5 (_shap_proxied_methods.py _su_screen_snr_z); tests/feature_selection/shap_proxied/test_cluster14_explicit_snr_z_pin_wins.py |
| 18-F4 | FIXED | Level-1 member tuples (entry, train_df, val_df, test_df) are unwrapped to the entry namespace in the probability accessors (models/ensembling/process_method.py _entry_of); tests/models/test_cluster18_level_member_tuple_unwrapped.py |
| 40-F28 | FIXED | PySRTransformer.transform copies the frame (shallow) before adding PySR columns, so the caller's frame is not mutated (training/pipeline/__init__.py); tests/training/pipeline/test_cluster40_pysr_transform_does_not_mutate_input.py |
| 40-F23 | FIXED | Embedding join looks up each unique id once, with normalised keys (3 matches 3.0), zero-fills unknown and NaN ids (training/pipeline/_latent_interaction_svd_composite_fe.py); tests/training/pipeline/test_cluster40_embedding_join_normalised_keys.py |
| 40-F30 | FIXED | Weighted classification loss divides by class-times-sample weights when class weights are set, so the scale matches the unweighted CrossEntropy mean (neural/_recurrent_torch_model.py _compute_weighted_loss); tests/training/neural/test_cluster40_class_weighted_loss_scale.py |
| 40-F25 | DEFERRED | Fit-time val/test lookup uses raw keys while replay normalises. Not changed: no discriminating case found; the two-step encoder returns the global prior for the entities tested, so a regression test cannot tell the versions apart. Needs the terminal-encoding semantics before a fix. |
| 40-F20 | FIXED | Frame content hash hashes the per-row hash bytes in row order instead of their sum, so row-permuted frames no longer share a cache key (training/pipeline/_pipeline_cache.py); tests/training/pipeline/test_cluster40_frame_hash_is_row_order_sensitive.py |
| 40-F21 | FIXED | TrunkResidualMLPRegressor and FieldGroupedMLPRegressor refuse NaN/inf X at fit (and at predict), as Tabular1DCNN does; helper _refuse_nonfinite in each module; tests/training/neural/test_cluster40_residual_regressors_refuse_nonfinite.py |
| 40-F24 | FIXED | With keep_cols pinned, the nullable-bool scan covers only the pinned columns; the dropped object columns are no longer scanned (training/pipeline/_pipeline_extensions.py); tests/training/pipeline/test_cluster40_filter_numeric_keep_path.py pins the behaviour |
| 41-F13 | FIXED | A failing normality test is logged at WARNING and the verdict falls back to the moment heuristic visibly (training/targets/regression_residual_audit.py); tests/training/targets/test_cluster41_normality_fallback_warns.py |
| 41-F27 | FIXED | A failing collapse sensor (outer probe and envelope probe) is logged at WARNING, so a broken sensor is not mistaken for a healthy run (training/reporting/_reporting_regression/_sensors.py); tests/training/reporting/test_cluster41_collapse_sensor_failure_warns.py |
| 41-F8 | FIXED | Timestamp bin starts are differenced as datetime64 (real unequal ages for monthly bins), and the even-spacing fallback is a WARNING (training/targets/_target_temporal_audit_from_agg.py); tests/training/targets/test_cluster41_bin_ages_from_timestamps.py |
| 41-F11 | DEFERRED | Not reproduced as stated: for a regression target with one NaN the clustered-target check still runs and records the ratio. The classification path never records within_between_group_var_ratio (clean or NaN alike), a separate gap that needs a design decision on whether classification should run it. No code changed. |
| 41-F20 | FIXED | is_catboost_model and is_neural_model answer quietly for estimator instances (no dispatch WARNING for unknown estimators); training/strategies/__init__.py; tests/training/test_cluster41_predicates_do_not_warn.py |
| 41-F23 | FIXED | Pareto artifact returns None when there is no shard history, before any indexing (training/slicing/_slice_pareto_plot.py); tests/training/slicing/test_cluster41_pareto_empty_history.py |
| 41-F15 | FIXED | Multilabel runs with two labels get per-label chart names (the caller passes st.is_multilabel to _class_plot_paths); binary naming unchanged (training/reporting/_reporting_probabilistic.py); tests/training/reporting/test_cluster41_multilabel_two_labels_distinct_plots.py |
| 41-F16 | FIXED | Non-numeric feature values coerce to NaN (pd.to_numeric errors=coerce) in the calibration-by-feature and heatmap renderers, as the comment promised (training/reporting/_reporting_probabilistic_calib.py); tests/training/reporting/test_cluster41_feature_values_coerce_nonnumeric.py |
| 41-F18 | FIXED | A numeric transform that changes width without feature names raises instead of positionally guessing (training/strategies/base.py _NumericOnlyTransformer); tests/training/test_cluster41_numeric_transform_no_positional_guess.py |
| 41-F19 | FIXED | Cache budget no longer floored at 2 GiB above the free-RAM cap; test test_cluster41_cache_budget_respects_free_ram.py |
| 41-F7 | FIXED | Quantile and multi-target regression use the regression name; the single-class warning runs only for binary targets. Test test_cluster41_target_name_by_task_type.py. |
| 41-F22 | FIXED | Integer-valued float targets with few classes are treated as classification, so shards are stratified. Test test_cluster41_float_binary_target_stratified.py. |
| 41-F12 | FIXED | Spearman ranks average tied values (scipy rankdata semantics), so hetero_spearman no longer depends on row order. Test test_cluster41_ranks_average_ties.py. |
| 41-F9 | FIXED | Single-target temporal aggregation counts labelled rows in n_obs (polars and pandas), matching the multi-target path. Test test_cluster41_single_target_counts_labelled_rows.py. |
| 41-F25 | FIXED | Group-aware MRMR documents its numeric-only scope and logs the number of skipped non-numeric columns at WARNING. Test test_cluster41_mrmr_numeric_only_logged.py. |
| 41-F24 | FIXED | NaN gate decides membership on validation scores only; NaN test scores are floored at scoring time. Test test_cluster41_nan_gate_is_val_only.py. |
| 41-F21 | FIXED | String columns absent from the XGBoost category map are dropped with a WARNING instead of getting a per-frame Enum. Test test_cluster41_xgb_unmapped_string_column_dropped.py. |
| 41-F17 | FIXED | Confusion-derived metrics (accuracy, MCC, F-beta, specificity, FPR/FNR, G-mean) use the tuned threshold when one exists, the same as precision/recall/F1. Test test_cluster41_confusion_block_uses_tuned_threshold.py. |
| 41-F26 | DEFERRED | Dead code per the uncalled-function rule: rrf_fuse/borda_fuse duplicate ranker_suite ensemble_ranker_scores, _binned_mi and _group_features_mi_njit have no src caller. Wiring them in or deleting them is an owner call (wire vs remove the duplicate); deferred to avoid deleting code the tests exercise. |
| 39-F7 | FIXED | Categorical cast failures in predict_from_models log at WARNING with the column name. |
| 39-F8 | FIXED | Volatility-router val-predict failure logs at WARNING with traceback. Test test_cluster39_logging_levels_visible.py. |
| 39-F10 | FIXED | allowed_targets typo warning is emitted at any verbosity. Test test_cluster39_logging_levels_visible.py. |
| 39-F56 | FIXED | Lookahead slow-weight commit failure logs at WARNING, stating predictions use fast weights. |
| 39-F69 | FIXED | Lookahead commit failure logs at WARNING with exc_info (same site as 39-F56). |
| 39-F70 | FIXED | Optimizer and scheduler names read with getattr fallback, so partial/non-class objects no longer raise AttributeError. |
| 37-F26 | FIXED | Honest-gain floor drops specs whose gain is not strictly above the floor, so a NaN gain is dropped. Test test_cluster37_nan_gain_dropped_and_merge_logged.py. |
| 37-F29 | FIXED | A back-merge skipped on row-count change logs a WARNING naming the dropped engineered columns. Test test_cluster37_nan_gain_dropped_and_merge_logged.py. |
| 37-F25 | FIXED | Watchdog skips for a missing group column or absent base columns log a WARNING. Test test_cluster37_watchdog_skip_logged.py. |
| 41-F10 | FIXED | Single-target polars granularity picker receives (min, max), as the multi path does. Test test_cluster41_single_picker_gets_span.py. |
| 39-F27 | FIXED | E5 prefix detection also matches names starting with e5- / e5_ / e5/. Test test_cluster39_hf_e5_unc_target_coerce.py. |
| 39-F40 | FIXED | long_path_safe emits the UNC long form for \server shares. Test test_cluster39_hf_e5_unc_target_coerce.py. |
| 39-F43 | FIXED | Multi-column DataFrame targets raise instead of being raveled. Test test_cluster39_hf_e5_unc_target_coerce.py. |
| 39-F42 | FIXED | Tokenizer role names (pad/eos/bos/unk/sep/cls/mask), cache_key and token_type are not masked as secrets. Test test_cluster39_secret_name_masking.py. |
| 39-F57 | FIXED | Negative and NaN categorical codes map to the reserved row instead of category 0. Tests updated in test_categorical_embeddings.py. |
| 39-F21 | FIXED | Object-column fingerprints hash by value (hash_pandas_object), not PyObject pointers. Test test_cluster39_object_column_hash_by_value.py. |
| 39-F22 | FIXED | fit() retries without y only for a y-signature mismatch, the same narrowing fit_transform uses; genuine TypeErrors surface. Test test_cluster39_custom_fit_narrow_retry.py. |
| 39-F39 | FIXED | CUDA runtime failure in keep-n auto sizing falls back to 2 with a WARNING instead of breaking every acquire. Test test_cluster39_keep_n_cuda_failure.py. |
| 39-F45 | FIXED | Character entropy counts with a single Counter over chained characters. Test test_cluster39_char_entropy_counter.py. |
| 39-F55 | FIXED | set_params raises ValueError for unknown keys and method names instead of warning or overwriting. Test test_cluster39_set_params_unknown_raises.py. |
| 39-F25 | FIXED | James-Stein pooled-variance residuals use one pandas map per fit instead of a dict lookup per row. Test test_cluster39_js_variance_vectorised.py. |
| 39-F44 | PARTIAL | Datetime token construction is vectorised (np.char.add), pinned to the per-element formula. The object-token loop (_canonical_cat_token per value) and the _objectwise_isnull per-row loop are still Python loops. Test test_cluster39_datetime_tokens_vectorised.py. |
| 39-F29 | FIXED | Concurrent misses on one cache key elect a leader; other callers wait and then read the memory tier. Test test_cluster39_cache_single_flight.py. |
| 42-F4 | FIXED | Leaderboard refuses duplicate task column names. Test test_cluster42_duplicate_task_and_missing_task.py. |
| 42-F7 | FIXED | Geometric mean names an all-NaN task column as missing, not as a zero score. Test test_cluster42_duplicate_task_and_missing_task.py. |
| 42-F3 | FIXED | Two-step group sub-boards inherit allow_partial from the parent (already in code, line 281); pinned by test_partial_opt_in_reaches_two_step_group_rankings. |
| 42-F5 | FIXED | linprog solver failures (status other than 2, infeasible) log a WARNING instead of being reported as infeasible. Test test_cluster42_gamma_and_solver_status.py. |
| 42-F6 | FIXED | optimality_gap without gamma raises a ValueError naming the missing gamma at every elect/rank call site. Test test_cluster42_gamma_and_solver_status.py. |
| 42-F8 | FIXED | Removed the no-op full-table reindex in Leaderboard.__init__. Test test_cluster42_majority_graph_and_table.py. |
| 42-F9 | FIXED | Majority graph computes the below/above weighted sums once per model and reuses them for wins and ties. Values pinned by test_cluster42_majority_graph_and_table.py. |
| 42-F11 | FIXED | Votenrank bench imports sigmoid from the family's _bench_shared module; the package-root copy stays for the other benches. Test test_cluster42_bench_sigmoid_shared.py. |
| 42-F10 | FIXED | Minimax benchmark times and checks the shipped Leaderboard.minimax_ranking; the duplicate _new_minimax copy is removed; the old formulation stays as a fixture. Test test_cluster42_minimax_bench_uses_shipped.py. |
| 42-F12 | DEFERRED | Repeated cProfile/perf_counter boilerplate across about a dozen bench scripts: a cross-file cleanup with no behaviour impact; deferred so the benches are not churned in this pass. |
| 32-F8 | FIXED | Polars group-mean join matches null keys (nulls_equal=True), as the pandas path does. Test test_cluster32_polars_null_group_matches_pandas.py. |
| 32-F17 | FIXED | Threshold benchmark seed uses zlib.crc32 (was already in code); pinned by test_cluster32_bench_seed_is_stable.py. |
| 32-F14 | DEFERRED | Per-drop X.loc copy is held per concurrent worker. Removing it needs _fit_quick_and_score to accept a column mask instead of a frame (API change across the baseline diagnostics). Owner to schedule. |
| 32-F13 | FIXED | Baseline numba kernels use the NaN-safe fastmath flag set (no nnan/ninf), so NaN predictions propagate. Test test_cluster32_kernels_nan_propagates.py. |
| 32-F27 | DEFERRED | Perf-only: the first full-size frame is built before the extra-target and ts knobs are known, then rebuilt. Avoiding it means reordering the benchmark's setup, which needs checking every use of df in between. Deferred. |
| 32-F18 | FIXED | Cached benchmark arm uses one key across splits so it can hit; per-split keys never hit. Test test_cluster32_cache_arm_shares_key.py. |
| 32-F19 | FIXED | TF-IDF arm B fills NA before casting, so both arms vectorise the same tokens. Test test_cluster32_tfidf_arms_equivalent.py. |
| 32-F20 | FIXED | Ordered-encoder bench builds inputs outside the timers. Test test_cluster32_ordered_bench_prebuilt_inputs.py. |
| 32-F21 | FIXED | The auto early-stopping rule is one function (resolve_early_stopping_rounds) used by get_training_configs and by the benchmark's auto arm. Test test_cluster32_early_stopping_rule_shared.py. |
| 32-F25 | FIXED | Benchmark model names include scenario, mode, val size and seed, so cells sharing a workdir cannot collide. Test test_cluster32_val_size_model_names_unique.py. |
| 32-F22 | FIXED | Corrected the benchmark docstring and comment: components are fit on train only with no early stopping on val, so the val arm does not exercise a double-dip. Docstring-only correction; no test (no behaviour changed). |
| 32-F23 | DEFERRED | Verdict follows from the fixture's construction (model A strong on class 0, B on minorities). A real verdict needs real or independently generated multiclass data; owner to supply a dataset or accept a labelled synthetic claim. |
| 32-F24 | DEFERRED | The near-duplicate pool builds the dedup verdict in by design. A recommendation needs a non-duplicated or real-data scenario; owner to decide the scenario. |
| 22-F33 | FIXED | Subset draws with fewer than two members raise a clear ValueError instead of an empty-choice crash. Test test_cluster22_synthetic_subset_guard.py. |
| 22-F32 | FIXED | Both sd_sigma and nonoutlying_dist_percentage unset raises a ValueError naming the parameter, instead of a TypeError. Test test_cluster22_tukey_both_none_is_valueerror.py. |
| 14-F12 | FIXED | split_into_train_test refuses an empty features_indices with a ValueError instead of indexing [0]. Test test_cluster14_split_empty_features_raises.py. |
| 15-F17 | FIXED | All-NaN column gets mean 0 and std 1 in the knockoff standardisation, so its knockoff is finite. Test test_cluster15_knockoff_all_nan_column.py. |
| 13-F22 | DEFERRED | Dead branch: the row-alignment check is always true (both arrays have n rows). Removing it or asserting the invariant touches the residual pipeline's None contract, and no cheap direct test reaches it. Deferred with the shap_proxied dead-branch cleanup. |
| 04-F30 | FIXED | replay_from_record refuses a record without dataset_seed instead of defaulting to 0. Test test_cluster04_replay_requires_seed.py. |
| 14-F13 | FIXED | A failed thread pin logs a WARNING naming the parameters instead of a debug line. Test test_cluster14_thread_pin_failure_warns.py. |
| 22-F41 | FIXED | Bootstrap resamples whose metric raises are counted and reported in one WARNING with the failed and total counts. Test test_cluster22_failed_resamples_warn.py. |
| 22-F37 | FIXED | Dtype family prefixes match only when followed by a bit width, so interval dtypes are not mapped to int codes. Test test_cluster22_dtype_canon_interval.py. |
| 22-F38 | FIXED | module_available docstring now states that a dotted name imports its parent package to locate the child. Docstring-only correction; no behaviour change, no test. |
| 22-F40 | FIXED | Infinite group labels get their own string keys; the integer branch requires finite values. Test test_cluster22_group_keys_inf.py. |
| 22-F34 | FIXED | Non-finite valuations raise a ValueError naming the count, instead of the internal-bug RuntimeError. Test test_cluster22_weights_nonfinite_valueerror.py. |
| 22-F39 | FIXED | ensure_no_infinity_pd works on a copy, so the caller's DataFrame keeps its infinities. Test test_cluster22_ensure_no_inf_copies.py. |
| 22-F42 | FIXED | A cupy measurement failure is a WARNING that says cupy is excluded; a missing cupy stays at debug. Test test_cluster22_cupy_tuning_failure_warns.py. |
| 22-F44 | FIXED | trimmed_mean and robust_mean_mestimator refuse NaN/inf with a ValueError instead of a biased or NaN result. Test test_cluster22_robust_location_nonfinite.py. |
| 22-F43 | FIXED | Throttle key is the stable name post_train_calib_target_coerce_failed instead of a line number. Test test_cluster22_throttle_key_stable.py. |
| 22-F45 | WON'T FIX | The per-improvement deepcopy is required for correctness: a snapshot taken only at stop would be a live reference that keeps mutating, so best_model_ would hold the final degraded weights (the bug the code comment records). Performance cost is accepted; the suggested fix reintroduces that bug. |
| 22-F35 | FIXED | Docstring corrected: the exact-own-value claim holds only for k=1; for k>1 a subsample row is the mean of its k nearest subsample rows. Docstring-only; no behaviour change, no test. |
| 22-F36 | FIXED | NaN check in PdOrdinalEncoder.transform reads the native dtype (pd.isna) instead of copying the output to float64. Test test_cluster22_ordinal_nan_check.py. |
| 25-F57 | FIXED | Prefilter scans candidate sizes ascending so the smallest qualifying size wins for unsorted input. Test test_cluster25_prefilter_unsorted_sizes.py. |
| 25-F56 | FIXED | Dead marginals array removed from the MI kernel; output unchanged. Regression: existing MI parity and range tests (test_mi_cross_parity.py, test_mi_kernels_reject_out_of_range_codes.py), 28 passing. |
| 25-F58 | FIXED | Anchor de-duplication uses a set for membership (O(1)) and keeps the list for rank order; output unchanged. Regression: test_biz_val_varying_size_top_k_subsets.py, test_varying_size_cluster_anchors_nan_and_offset.py, 6 passing. |
| 25-F65 | FIXED | Brier score coerces y_prob to an array, so a list input no longer reaches numba untyped. Test test_cluster25_brier_list_probs.py. |
| 26-F32 | FIXED | A negative MLFRAME_MAX_DECOMPRESSED_BYTES falls back to the default ceiling with a warning; 0 still disables explicitly. Test test_cluster26_decompress_cap_negative.py. |
| 26-F14 | FIXED | batch_univariate_auc refuses multi-class targets instead of scoring label 1 one-vs-rest. Test test_cluster26_auc_multiclass_refused.py. |
| 26-F15 | FIXED | Silhouette in clusterize excludes DBSCAN noise via _silhouette_without_noise; undefined with fewer than two real clusters. Tests test_cluster26_silhouette_excludes_noise.py; existing cluster tests (15) pass. |
| 26-F29 | FIXED | A failing cupy or cuda DTW path falls back to the CPU result with a WARNING saying it is slower, not a debug line. Test test_cluster26_dtw_gpu_fallback_warns.py. |
| 25-F63 | FIXED | rf_proximity_matrix checks the row count of X before applying the forest, so an oversized request is refused without the apply cost. Test test_cluster25_proximity_guard_before_apply.py; existing proximity tests (8) pass. |
| 25-F67 | FIXED | Miller-Madow occupancy counts histogram bins only; the non-finite slot is excluded. Test test_cluster25_kl_mm_excludes_nonfinite_slot.py; existing drift tests (43) pass. |
| 25-F70 | FIXED | pinball_loss_per_alpha refuses duplicate alphas (the alpha-keyed result would drop a column's loss). Test test_cluster25_pinball_duplicate_alphas.py; 56 existing pinball/quantile tests pass. |
| 25-F66 | DEFERRED | The 500_000 MAPE threshold may be a measured parallel crossover; switching it to the shared 100_000 constant changes performance without a benchmark. Needs a crossover benchmark before the constant is shared. |
| 25-F69 | FIXED | Docstring corrected: multilabel_auc_per_label always loops per column and does not route to the batched GPU path, which is compute_batch_aucs. Docstring-only; no test. |
| 25-F71 | DEFERRED | The MAPE warmup diagnostic is an active CI investigation hook (env-gated) for a failure the code comments say is unresolved. Removing it discards the instrument; retire it once the CI issue is closed. Owner to decide. |
| 28-F73 | FIXED | A bare model-name string is keyed as one model, not iterated per character. Test test_cluster28_suite_key_string_models.py; 26 existing artefact-cache tests pass. |
| 28-F57 | FIXED | Failed categorical casts in the LightGBM shim (fit-time val and predict-time) log a WARNING naming the column. Test test_cluster28_lgb_cat_cast_warns.py; 5 existing shim tests pass. |
| 28-F66 | FIXED | Every RMSE recommendation carries lgb/xgb/cb_extra_params keys (empty), matching the other branches. Test test_cluster28_rmse_recommendation_keys.py; 15 existing loss-recommendation tests pass. |
| 28-F62 | FIXED | Collapsed the no-op l1_ratio conditional: both branches returned config.l1_ratio, so behaviour is unchanged. Regression: 43 existing linear-model tests pass. |
| 28-F67 | FIXED | parse_catboost_devices returns GPUs in the requested order, each once. Test test_cluster28_catboost_devices_requested_order.py; 37 existing device-parsing tests pass. |
| 28-F60 | FIXED | Removed the unreachable '(none)' description branch: _bad_desc is only built inside if _bad. Regression: 14 existing CatBoost text-feature probe tests pass. |
| 28-F71 | FIXED | The categorical column set is built once per call rather than once per feature. Output unchanged. Regression: 53 existing drift-report tests pass. |
| 27-F22 | FIXED | Title positive rate counts the larger label as positive, matching the decision-threshold block (labels {1, 2} no longer read as 0%). Test test_cluster27_title_rate_positive_larger_label.py. |
| 27-F21 | FIXED | recalibrate on an unfitted DistributionalRecalibrator raises RuntimeError instead of returning the identity. Test test_cluster27_unfitted_recalibrator_raises.py; 19 existing calibration tests pass. |
| 27-F18 | FIXED | select_from_pareto logs a WARNING when no frontier point is scored and the first iteration is returned by default. Test test_cluster27_pareto_default_warns.py; 45 existing Pareto/aggregation tests pass. |
| 27-F17 | FIXED | The iterations-to-max_iter validator works on a copy, so the caller's dict is not popped or written. Test test_cluster27_iterations_mapping_no_mutation.py; 17 existing model-config tests pass. |
| 27-F26 | FIXED | CUDA permutation importance restores the caller's device and train/eval mode instead of forcing cpu and eval. Test test_cluster27_cuda_fi_restores_net_state.py covers the CPU-only fallback; the CUDA restore branch is not exercised on this machine. |
| 27-F24 | FIXED | CUDA_VISIBLE_DEVICES default moved from import time to the script entry point; importing the module no longer changes the process environment. Test test_cluster27_profile_import_keeps_env.py. |
| 27-F23 | WON'T FIX | The two precompute stubs raise NotImplementedError on purpose: their docstrings explain the split-time dependency, precompute_all does not call them, and tests/training/test_caching_caching.py pins the raise. Removing them would break the public re-exports and meta-tests. |
| 27-F25 | FIXED | Backend selection uses the same absolute-PATH nvidia-smi lookup as the probe, not a plain shutil.which on the full PATH. Test test_cluster27_gpu_backend_absolute_path.py; 7 existing GPU probe tests pass. |
| 27-F20 | FIXED | format_metric docstring example corrected to the actual output (0.000012); test pins every documented example. Test test_cluster27_format_metric_docstring_examples.py. |
| 29-F29 | FIXED | A raising post-import callback is logged with its traceback and the remaining callbacks still run. Test test_cluster29_post_import_callbacks_all_run.py; 3 existing post-import tests pass. |
| 29-F32 | FIXED | Plain signed numbers are not formula-prefixed; real formulas still are. Test test_cluster29_csv_safe_numbers_untouched.py (8 passing); existing formula-injection tests (3) pass. |
| 29-F38 | FIXED | A failing KTC cache lookup logs a WARNING naming the fallback backend, instead of DEBUG. Test test_cluster29_ktc_lookup_failure_warns.py; 13 existing KTC/gated-blend tests pass. |
| 29-F37 | FIXED | A refit without iteration-metric capture clears the previous fit's iteration_metrics_ on the XGB shim. Test test_cluster29_xgb_refit_clears_iteration_metrics.py; the existing GBDT cache test passes. |
| 29-F34 | FIXED | Names mismatch in diversity ablation raises ValueError listing both lists, instead of an empty AssertionError. Test test_cluster29_diversity_names_mismatch_message.py; 8 existing diversity/correlation tests pass. |
| 31-F10 | FIXED | The sweep timeout log says the kernel timed out instead of describing a fallback that never ran. Test test_cluster31_timeout_log_is_accurate.py; 3 existing kernel-tuning fix tests pass. |
| 31-F11 | FIXED | The source-root check in _run_sweep is an explicit RuntimeError instead of an assert that python -O strips. Test test_cluster31_timeout_log_is_accurate.py (2 passing, incl. the missing-__file__ case). |
| 31-F12 | FIXED | Removed the dead hasattr check on args.dims: the explain subparser always defines --dims. Regression: 24 existing kernel-tuning CLI tests pass. |
| 31-F13 | DEFERRED | Attribute calls named kernel_tuner match too widely, but real registrations are often written registry.kernel_tuner(...). Restricting matches risks skipping real tuners, so the fix needs module-alias resolution first. Deferred. |
| 30-F5 | FIXED | Changepoint benchmark builds its series outside the timers; _run accepts a prebuilt y. Test test_cluster30_changepoint_bench_prebuilt_series.py. |
| 30-F6 | FIXED | Unverified speed claims in the changepoint benchmark comments replaced with neutral wording. Comment-only; no behaviour change. |
| 30-F3 | FIXED | Unused logger and logging import removed from the DTW GPU banded benchmark. Import check passes; no behaviour change. |
| 28-F50 | FIXED | Removed the duplicate SimpleNamespace import (aliased copy) and used the plain name. The repeated _pre_pipeline_groups reset is left in place: it is only redundant if the attribute is already None, which this change does not establish. Regression: module import check and a related trainer test pass. |
| 28-F55 | FIXED | Removed the no-op del of a parameter (it freed nothing while the caller held the frame); _maybe_clean_ram still runs. Import check passes; the removed branch had no observable effect, so no test. |
| 26-F17 | FIXED | Removed the second copy of logger, NDIGITS, DATEFRACTS_CODES and DATEFRACTS_MULTIPLIERS in the cleaning helpers; values unchanged. Regression: 17 existing cleaning tests pass. |
| 26-F33 | FIXED | Tuples are tagged in the canonical tree, so a tuple and a list with the same items no longer collide. Note: digests of tuple inputs change once, so existing tuple-keyed cache entries miss. Test test_cluster26_canonical_tuple_vs_list.py; 14 existing canonical/cache tests pass. |
| 26-F18 | PARTIAL | Removed the dead bare-string block (commented-out loop) in cleaning.py. The bare string in _cleaning_helpers.py is explanatory prose with an example, left in place. Regression: 16 existing cleaning tests pass. |
| 26-F25 | FIXED | Docstring corrected: above DIAG_ROW_CAP worst-K is computed on the bounded subsample, not the full arrays. Docstring-only; no behaviour change, no test. |
| 26-F16 | PARTIAL | LOF novelty=True crash fixed: the LOF branch fits then reads negative_outlier_factor_ (same value as fit_predict for novelty=False). Still open: the docstring's claim that tie-breaking is stable (np.argpartition is not). Test test_cluster26_lof_novelty_scores.py (2 passing); 14 existing outlier tests pass. |
| 26-F16 | FIXED | LOF novelty crash fixed and the contamination cutoff uses a stable sort, so ties resolve deterministically and the docstring is accurate. Test test_cluster26_lof_novelty_scores.py (3 passing); existing outlier tests pass. |
| 26-F27 | PARTIAL | ScatterPanelSpec validates x/y/point_color/point_size lengths at construction. HistogramPanelSpec.bar_colors not checked: its length contract is ambiguous for pre-binned data and no consumer pins it. Test test_cluster26_scatter_spec_lengths.py (3 passing); 47 existing spec tests pass. |
| 26-F34 | DEFERRED | Import-time env write and catboost.Pool monkeypatch in mlframe.training are relied on by callers that never enter the suite entry point; moving them changes behaviour for every importer. Owner decision on where the shim must install. |
| 26-F19 | FIXED | The set_params side effect for XGBoost is documented in the docstring: XGBoost takes early stopping as a constructor parameter, so the mutation is needed. Docstring-only; no behaviour change, no test. |
| 28-F61 | FIXED | Removed the unused optional matplotlib import in the trainer-configure helpers (no name use, no importer). Regression: the two importing tests pass (37). |
| 28-F50 | FIXED | The reset-then-read pair on _pre_pipeline_groups is replaced by passing None directly; the attribute is None on entry, so behaviour is unchanged. Regression: module import and a related trainer test pass. |
| 26-F27 | DEFERRED | Histogram bar_colors must match the drawn bar count, which is len(values) for pre-binned data but bins for raw data. The spec does not record which mode applies, so a single __post_init__ check would be wrong for one of them. Needs a spec field for the mode, an owner-approved API change. |
| 08-F18 | FIXED | fit_bgmm_and_sample raises a clear ValueError naming the caller when there are no minority rows, instead of numpy's integers(0, 0) error. Test test_cluster08_bgmm_empty_minority.py; 3 existing sampler tests pass. |
| 08-F19 | FIXED | kth_nearest_dists refuses k < 1, which silently indexed the farthest column. The documented clipping above the subset size is kept. Test test_cluster08_kth_nearest_k_positive.py (2 passing); 47 existing caller tests pass. |
| 18-F17 | FIXED | Removed the unreachable except NameError around the gate-stats lookup: _gate_stats is a parameter, always bound. Behaviour unchanged; 71 existing ensembling tests pass. |
| 18-F18 | FIXED | Docstring corrected: probabilistic outputs are averaged across per-class correlations, not collapsed to the last column. Docstring-only; no behaviour change, no test. |
| 18-F16 | FIXED | Removed the never-read ENSEMBLE_STREAMING_THRESHOLD_BYTES constant; the streaming choice is driven by EnsemblingConfig.quantile_budget_bytes. Regression: 7 existing ensembling tests pass. |
| 14-F9 | WON'T FIX | np.asarray(np.asarray(base).copy()) performs one copy; the outer asarray is a no-op on an ndarray, so rewriting it saves nothing. Cosmetic only. |
| 14-F10 | FIXED | Removed the orphan comment about a TreeSHAP cell threshold that is not defined in the file. Comment-only; no behaviour change. |
| 14-F8 | FIXED | Permutation buffer built with one C-order copy instead of ascontiguousarray().copy(), which copied twice for a non-contiguous input. Equivalence checked on a non-contiguous view (same values, C-contiguous, writable, no shared memory). Regression: 5 existing SHAP-proxy cluster tests pass. |
| 14-F11 | FIXED | Stale docstrings citing the non-existent kernel_tuning_cache.get API replaced with accurate statements (wrappers are live and kept). Docstring-only; module imports; no behaviour change. |
| 08-F20 | FIXED | prefer_hnsw_at_n default is now None (the sentinel), so an explicit 50_000 is honoured instead of deferring to the tuned choice. Test test_cluster08_knn_explicit_hnsw_threshold.py (2 passing); 17 existing knn helper tests pass. |
| 08-F21 | FIXED | The duplicated LightGBM baseline is kept by design (it also predicts on a query set); a drift-guard test pins its out-of-fold predictions to the shared helper's for binary and regression. Test test_cluster08_baseline_oof_paths_agree.py (2 passing). |
| 18-F20 | DEFERRED | The benchmark mirrors the ranking because the shipped _choose_ensemble_flavour takes only an ensembles dict, with no probe-order parameter. Calling the shipped logic for each probe order needs an API change in the chooser, owner decision. |
| 09-F42 | FIXED | Rebrand now replaces only the leading attn_ prefix; regression test tests/feature_engineering/transformer/test_cluster09_row_attention_prefix_only.py. |
| 26-F18 | FIXED | Both dead bare-string blocks removed (cleaning.py earlier, _cleaning_helpers.py now). Existing cleaning tests pass (17). No behaviour change, so no new test. |
| 39-F9 | FIXED | Persisted datetime dtype names resolve via np.dtype; unknown or non-numeric names raise ValueError instead of int8. Test tests/training/core/test_cluster39_persisted_dtype_strict.py. |
| 39-F23 | FIXED | _extract_column refuses ndarray input with TypeError instead of passing the whole matrix; polars and pandas selection unchanged. Test tests/training/feature_handling/test_cluster39_custom_extract_column_refuses_ndarray.py. |
| 39-F24 | DEFERRED | Owner decision. The rows never held out by a forward-chaining splitter have no past to encode from. Excluding only their own label from the prior still uses later labels (sibling time-aware test forbids that), so the fallback needs a policy: drop these rows, or use a documented constant. Not changed. |
| 37-F28 | FIXED | Composite train mean is computed once in the caller and passed to the per-entry scorer (was recomputed per entry and per split). Pinned by tests/training/core/test_cluster37_train_mean_computed_once.py (source-level; no behavioural test). |
| 37-F27 | FIXED | Full-length base columns are built once per base name within the domain-violation check (per-call cache _full_base). Pinned by tests/training/core/test_cluster37_domain_violation_base_cache.py (source-level). |
| 04-F34 | FIXED | sample_categorical clamps the drawn index to K-1 when rounding leaves the cumulative sum under 1. Test tests/data/test_cluster04_sample_categorical_in_range.py. |
| 04-F32 | FIXED | Scenario name index refuses duplicate names (_index_by_name raises ValueError). Tests tests/data/test_cluster04_scenario_names_unique.py. |
| 04-F31 | FIXED | rare_class_spec keeps the RARE_MIN_ROWS floor but logs a WARNING when it raises the caller's n_samples, and records requested_n_samples in provenance. Tests tests/data/test_cluster04_rare_class_floor_is_visible.py. |
| 04-F29 | FIXED | tail_gate_term builds only the quantile masks its direction uses; direction is validated first. Tests tests/data/test_cluster04_tail_gate_builds_needed_masks.py. |
| 04-F33 | FIXED | MCAR draws an exact count (matching the documented contract); the caveat reports the column's total hidden share. Note: MCAR output for seeded beds changes once. Test tests/data/test_cluster04_mcar_exact_share.py. |
| 04-F27 | FIXED | Derived ordering tests membership against a per-round set and drops ready names via a set; realize_derived builds the edge-weight map once (first edge wins, as before). Test tests/data/test_cluster04_derived_order_uses_sets.py. |
| 11-F14 | FIXED | Deleted the no-op bare expression nbins + 1 in variant a. No behaviour change; covered by the 9 existing tests in tests/feature_selection/fe/factory/test_offset_kernels.py (pass). |
| 11-F18 | FIXED | rel_improvement keeps NaN for a non-positive baseline but logs a WARNING naming the cause. Test tests/feature_selection/fe/factory/test_cluster11_rel_improvement_logs_nan.py. |
| 11-F16 | FIXED | Rank-aggregation arm refuses a feature missing from the aggregate with KeyError instead of padding it with the minimum score. Test tests/feature_selection/test_cluster11_rank_aggregate_missing_refused.py. |
| 11-F17 | DEFERRED | Empty trace gives selection_score None while the kept set has a score (current_score in greedy_backward_elimination). Fixing needs the final score exported from greedy_backward_elimination, which changes a shared function's return contract. Design decision, not changed. |
| 11-F19 | FIXED | All MRMR arm and wrapper constructions draw from one bare_mrmr_kwargs helper in _arms.py; the variant arm overrides only its declared keys. Test tests/feature_selection/test_cluster11_mrmr_kwargs_single_source.py. |
| 12-F18 | FIXED | _quantile_block returns an explicit 'No finite tau' line instead of crashing on an empty tau list. Test tests/feature_selection/test_cluster12_quantile_block_empty.py. |
| 12-F85 | FIXED | main returns the count of failed runs and the entry point exits 1 when any failed (was len(rows), so the else branch was unreachable). Test tests/feature_selection/test_cluster12_ablation_main_exit_count.py. |
| 12-F99 | FIXED | Three run scripts (run_phase0, run_scm_beds, run_synth_control) exit 1 when no cell executed; was count >= 0, so always 0. Test tests/feature_selection/test_cluster12_run_scripts_zero_cells_fail.py (source-level pin; the mains need real beds). |
| 12-F86 | FIXED | Per-estimator summary prints 'no runs (n=0)' when the subset is empty instead of min/max on an empty array (--seeds 0). Test tests/feature_selection/test_cluster12_parity_zero_seeds_summary.py. |
| 12-F88 | FIXED | Quadratic truth key is 'quadratic_operands' in hard_synth, synth and the round4 diagnostic reader; hard_synth2 docstring corrected to 1970 noise (2000 - 15 linear - 15 redundant). Test tests/feature_selection/test_cluster12_truth_key_plural.py. |
| 12-F89 | FIXED | Golden fingerprint refuses a selected column with no FI score (was 0.0); MODE=baseline refuses to overwrite an existing golden unless ALLOW_REBASELINE=1. Test tests/feature_selection/test_cluster12_golden_fingerprint_fi_complete.py. |
| 12-F90 | FIXED | Round-3 hard-bed print divides by len(strong) and len(weak) instead of /4 and /8; noise_floor drops a discarded LGBMClassifier fit. Source-level pins in tests/feature_selection/test_cluster12_bench_denominators_and_fit.py. |
| 12-F92 | PARTIAL | Verdict prints 'undefined' when no relay variant ran (NaN best) instead of 'no gain / KILL'. The per-column np.corrcoef loop at lines 50-51 is not changed. Test tests/feature_selection/test_cluster12_relay_verdict_undefined.py. |
| 12-F98 | FIXED | fs_hybrid bench scripts write results through bench_out(name) (new _bench_out.py): MLFRAME_BENCH_OUT if set, else _results next to the package. 46 code-level D:/Temp sites in 22 files rewritten; docstring mentions reworded. Tests tests/feature_selection/test_cluster12_bench_out_location.py include a source guard against new code-level D:/Temp literals. Outside fs_hybrid (filters, other _benchmarks) not changed. |
| 12-F100 | PARTIAL | Tie-break seed in run_experiment is a SeedSequence over (dataset_seed, cv_seed); the old dataset*1000+cv collided for cv_seed >= 1000. Test tests/feature_selection/test_cluster12_tie_break_seed_no_collision.py. The slow full-bed prewidth read in run_phase0 (perf half of the finding) is not changed. |
| 12-F101 | PARTIAL | Unknown scenario raises KeyError naming the request and listing valid names (scenarios.make). Still open: sweep_recovery_vs_n json.dump writes NaN, which is not strict JSON. Test tests/feature_selection/test_cluster12_scenario_unknown_name_lists_valid.py. |
| 12-F101 | FIXED | Unknown scenario lists valid names (scenarios.make); sweep_recovery_vs_n writes strict JSON: non-finite floats become null, dump uses allow_nan=False. Tests tests/feature_selection/test_cluster12_scenario_unknown_name_lists_valid.py and tests/feature_selection/test_cluster12_sweep_strict_json.py. |
| 12-F102 | PARTIAL | Module-level warnings.filterwarnings('ignore') removed; LightGBM test uses importorskip. Not changed: the single-seed 0.01 AUC tolerance and the 303-column fit. Tests tests/feature_selection/test_cluster12_tree_member_test_hygiene.py (source pins) and the file's own gate tests. |
| 12-F91 | PARTIAL | Bootstrap RFECV seed is 1000*sd+b, unique per outer seed. Not changed: the per-config m.transform recompute in round3_mrmr_bench. Test tests/feature_selection/test_cluster12_bootstrap_rfecv_seed.py (source pin). |
| 12-F93 | PARTIAL | Rich-tree bench transforms the test split once and reuses it for AUC and width. Not changed: the ~12 other per-row or repeated transform sites across the round-4/5 benches named in the finding. Test tests/feature_selection/test_cluster12_rich_tree_single_transform.py (source pin). |
| 12-F94 | PARTIAL | Both FE acceptance benches print a labelled WARNING when the base falls back to the first 25 columns by position. Not changed: the unlabeled base selection itself and the unused n_jobs=1 in the frugal bench. Test tests/feature_selection/test_cluster12_fe_accept_fallback_warns.py (source pin). |
| 12-F95 | FIXED | Cap cost bench repeats each (n, cap) fit REPEATS times (CAP_REPEATS, default 3) and stores the median. Test tests/feature_selection/test_cluster12_cap_bench_median_repeats.py (source pin). |
| 12-F96 | PARTIAL | Confirm bench: the non-regression bar is printed as a one-sided floor (d >= -0.005) and the test transform is computed once. Not changed: the duplicated transform in round4_hybrid_mrmrcap_bench and round4_jmim_bench named in the finding. Test tests/feature_selection/test_cluster12_hybrid_confirm_guard_label.py (source pins). |
| 12-F97 | PARTIAL | Knockoff cut prints SKIPPED when fewer than 5 features instead of returning silently. Not changed: verdicts are still prose (KILL/CONFIRMED rules not computed in code) and round4_mdlp_collapse_diag is untouched. Test tests/feature_selection/test_cluster12_knockoff_skip_logged.py (source pin). |
| 13-F25 | FIXED | 'pd' removed from shap_proxied_fs __all__ so a star import no longer exports the pandas module. Test tests/feature_selection/test_cluster13_shap_proxied_exports_and_imports.py. |
| 13-F23 | FIXED | Redundant inner import of compute_phi_rank_stability removed from the hardcoded-ladder branch of _shap_proxied_fit.py (the import at the top of the function covers it; the finding's fit_steps line is not where the duplicate was). Test tests/feature_selection/test_cluster13_shap_proxied_exports_and_imports.py. |
| 13-F24 | FIXED | Both core-refine None guards are explicit RuntimeError checks instead of asserts that python -O strips. Test tests/feature_selection/test_cluster13_no_production_assert.py (source pin). |
| 13-F17 | FIXED | Train_model retries without verbose only when the TypeError names the verbose keyword; any other TypeError raised inside fit propagates. Test tests/feature_selection/test_cluster13_boruta_train_model_retry.py. |
| 13-F27 | PARTIAL | New-registry refresh returns 1 when tune_spec yields zero regions (was 0). Not changed: the cross-function return-code convention the finding also names. Test tests/feature_selection/test_cluster13_refresh_zero_regions_fails.py. |
| 13-F33 | FIXED | summarise returns an explicit line when there are no utilisation samples (was an IndexError on times[-1]). All three cited sites are inside summarise and covered by the same early return. Test tests/feature_selection/test_cluster13_utilization_summary_empty.py. |
| 13-F36 | FIXED | Bit-identity check keeps a per-seed flag; the seed line reports its own result and the overall result is the conjunction. Test tests/feature_selection/test_cluster13_bitident_per_seed_flag.py (source pin). |
| 13-F35 | FIXED | Stage sampler slices the leaves to 12 before printing; the list-comprehension-for-side-effects is replaced by a loop. Test tests/feature_selection/test_cluster13_stage_leaves_capped.py (source pin). |
| 13-F26 | FIXED | Both interaction-proxy fallbacks catch (ValueError, TypeError, RuntimeError, LinAlgError) instead of Exception, so programmer errors surface. Test tests/feature_selection/test_cluster13_interaction_fallback_narrow.py plus the existing shap_proxy interaction-proxy tests (8 pass). |
| 13-F31 | FIXED | Regime summary F1 is the mean of per-cell F1 values. Test tests/feature_selection/test_cluster13_regime_summary_f1.py (source pin). |
| 13-F32 | FIXED | Wide-scaling progress log path honours MLFRAME_BENCH_OUT and defaults under _benchmarks/_results; no hard-coded D: drive. Test tests/feature_selection/test_cluster13_wide_progress_path.py. |
| 13-F30 | PARTIAL | Package docstring names ~/.pyutilz/kernel_tuning and says hw_fingerprint lives in pyutilz. Not changed: the dispatch.py 75k/10k CUDA crossover claim (the fallback always returns njit), which is a separate code-vs-doc question. Test tests/feature_selection/test_cluster13_kernel_cache_doc_paths.py. |
| 13-F29 | FIXED | par_threshold falls back to swept_max*100 (never-wins sentinel, same as cuda_threshold) instead of 50_000. Test tests/feature_selection/test_cluster13_par_threshold_sentinel.py (source pin). |
| 42-F13 | FIXED | Tie count uses nunique(dropna=False) at both sites, so NaN ranks count as one shared value. Test tests/votenrank/test_cluster42_tie_count_nan.py. |
| 42-F14 | FIXED | Optimality-gap ranking drops the redundant DataFrame rebuild; np.minimum already returns a DataFrame with the same index and columns (verified equal). Covered by the existing gamma/optimality tests in tests/votenrank/test_cluster42_gamma_and_solver_status.py (4 pass). |
| 42-F15 | FIXED | Dated timing numbers and the 2026-07-10 claim removed from the confidence-gated bench docstring; the run command and dispatch pointer are kept. Test tests/votenrank/test_cluster42_bench_docstring_no_numbers.py. |
| 15-F12 | FIXED | Deleted the unused nested _is_discrete (superseded by _is_discrete_v2, which is kept and used). Tests tests/feature_selection/test_cluster15_unused_is_discrete_removed.py plus tests/feature_selection/contracts/test_importance.py. |
| 15-F14 | FIXED | _scaled_coef logs a throttled WARNING when std scaling is skipped for non-numeric data (was a silent pass); coefficients stay unscaled. Test tests/feature_selection/test_cluster15_scaled_coef_warns.py. |
| 15-F18 | FIXED | Univariate prescreen treats a falsy prescreen_top_k as no cap, matching the mRMR prescreen (0 kept nothing before). Tests tests/feature_selection/test_cluster15_prescreen_topk_zero.py (behavioural). |
| 15-F19 | FIXED | Knockoff helper docstring corrected: knockoffs come from the raw values and only the real slot is mean-imputed afterwards. Tests tests/feature_selection/test_cluster15_knockoff_doc_matches_code.py plus the existing all-NaN knockoff test. |
| 15-F15 | FIXED | get_signed_linear_coef takes an optional trailing data argument used as the scaling fallback when train_data is None, matching the legacy path; train_data still wins. Tests tests/feature_selection/test_cluster15_signed_coef_scaling_fallback.py. |
| 15-F16 | FIXED | Swap pass snapshots selected_features_per_nfeatures, evaluated_scores_mean and evaluated_scores_std before running and restores them in place on failure, so the argmax never reads a half-updated state. Tests tests/feature_selection/test_cluster15_swap_rollback.py; existing swap_top_k test files also pass. |
| 09-F39 | FIXED | Scale tag uses round(ns*100) so 0.29 is n29 (int truncation gave n28). Test tests/feature_engineering/transformer/test_cluster09_diffusion_scale_tag.py. |
| 09-F45 | FIXED | Empty quintile stratum logs a WARNING naming the stratum before its features fall back to the whole fold. Test tests/feature_engineering/transformer/test_cluster09_empty_stratum_warns.py (source pin). |
| 09-F44 | FIXED | Local logistic solver penalises only the slopes (intercept index 0 unpenalised); gradient and Hessian use the same penalty vector. Behaviour change: outputs of the local classifier move for any ridge>0. Tests tests/feature_engineering/transformer/test_cluster09_local_logreg_intercept_unpenalised.py; 9 existing local-classifier tests pass (397 skipped by their gates). |
| 09-F33 | DEFERRED | Accepted-but-unused task/standardize/dists parameters in six transformer modules. Removing or wiring them changes public signatures and every caller that passes them; owner decision (remove vs wire through). Not changed. |
| 09-F38 | FIXED | Silverman bandwidth uses the multivariate rule (4/(d+2))^(1/(d+4)) n^(-1/(d+4)); equals the 1-D rule at d=1, changes d>1 outputs. Tests tests/feature_engineering/transformer/test_cluster09_silverman_multivariate.py; 8 existing density-ratio tests pass (397 skipped by gates). |
| 09-F41 | DEFERRED | Empty-band convention differs across band-attention modules. Masking empty bands (the suggested fix) changes attention outputs in residual_band, signed_residual_band and prediction_band; the convention needs owner confirmation first. Not changed. |
| 09-F37 | PARTIAL | Removed the stale comment about the removed _GPU_AUTO_CACHE in random_features.py. Not changed: the unreachable padding branches in denoising_autoencoder, hard_row_attention and related modules named in the finding. Test tests/feature_engineering/transformer/test_cluster09_no_stale_cache_comment.py (source pin). |
| 09-F36 | DEFERRED | Mode A RobustScaler is fitted on the whole X_train passed in. A per-fold refit needs the OOF caller restructured; the finding itself rates the unsupervised leakage negligible. Not changed; owner to decide whether the discipline change is worth the restructure. |
| 09-F34 | DEFERRED | Stub parameters (n_bits, top_k_pairs) and per-module Literal/dtype gates differ across siblings. Each stub needs an implement-or-remove decision, and widening a Literal without implementing the value would only defer the error. Owner decision. Not changed. |
| 40-F33 | WON'T FIX | Polars cannot represent N rows with zero columns (verified: select([]) has height 0), so the (N, 0) contract is unattainable for polars frames. Callers that need the row count must read it from the input. Not changed. |
| 40-F34 | DEFERRED | Bench keeps its own _binned_mi variants (addat, bincount) for A/B timing, not a copy of production. Both agree with production _ranker_fs._binned_mi, but on a strongly dependent sample all three return 0.0, which needs a review of the expected y/x contract before a sanity test can be written. Not changed. |
| 17-F23 | FIXED | Module docstring of _classification_extras_blocks.py rewritten to describe its three block kernels and point to _classification_extras.py for the public metrics. Test tests/metrics/test_cluster17_blocks_docstring.py. |
| 17-F24 | FIXED | top_k_accuracy emits a UserWarning when k >= K, as its docstring promises; the NaN return is unchanged. Test tests/metrics/test_cluster17_topk_warns.py. |
| 17-F31 | FIXED | Deleted the no-op placeholder expression in the Welford bench. Test tests/metrics/test_cluster17_no_placeholder_expression.py (source pin). |
| 17-F30 | FIXED | KS bench restores the production gate in a finally clause, so an exception inside a timed run cannot leave it disabled. Test tests/metrics/test_cluster17_ks_bench_restores_gate.py (source pin). |
| 17-F29 | FIXED | CRPS bench verifies the shipped function with one computation and one comparison. Test tests/metrics/test_cluster17_crps_bench_single_check.py (source pin). |
| 17-F28 | FIXED | Adjusted-R2 bench helpers delegate to fast_r2_score and fast_adjusted_r2_score; they agree on a sample. Test tests/metrics/test_cluster17_adjr2_bench_uses_shipped.py. |
| 17-F27 | DEFERRED | Bench kernel copies with fastmath=True are the A/B variants under test; replacing them with production kernels removes the comparison. Which arm the bit-identical claim describes, and whether the R2=-inf discrepancy is a bench or production bug, needs an owner decision. Not changed. |
| 17-F26 | WON'T FIX | Welford kernels are exercised by tests/metrics/regression/test_fused_regression_metrics_block.py and classified as deliberate alternatives in the shared uncalled-functions allowlist (tests/test_meta/test_shared_uncalled_functions.py). The classification the finding asks for already exists. Not changed. |
| 17-F22 | DEFERRED | Default aggregation and grid differ across metric siblings (max_error raw_values vs uniform_average; calibration nbins/use_weights). Aligning defaults changes what existing callers receive; owner decision (align or document). Not changed. |
| 17-F25 | DEFERRED | Per-class ICE path repeats binning and calibration MAE/STD across three modules. A dedup refactor across the calibration and report modules; not started in this pass. Not changed. |
| 21-F18 | FIXED | Perfect model gets a 'deferral cannot reduce error' verdict instead of 'insufficient data' (new branch for err_at_100 == 0). Test tests/reporting/test_cluster21_perfect_model_verdict.py drives build_risk_coverage_spec with a perfect model that has varying confidence. |
| 21-F17 | FIXED | Training-curve alignment no longer holds the last value flat past a series' last logged sample: such series take the gap path, and _interp_with_gaps returns NaN beyond its last position. Test tests/reporting/test_cluster21_training_curve_tail_gap.py; existing training-curve tests (4 files) pass. |
| 21-F20 | FIXED | A failed os.replace of a rendered chart now abandons (removes) its scratch file, as the other branch does. Test tests/reporting/test_cluster21_kaleido_scratch_on_move_failure.py (source pin). |
| 21-F19 | FIXED | Removed the early apply_heatmap_tick_budget call inside _heatmap; the renderer's post-layout loop applies the budget once to every heatmap panel. Test tests/reporting/test_cluster21_single_tick_budget.py (source pins). |
| 21-F23 | FIXED | Removed the no-op 'if not p.equal_aspect: pass' branch in the scatter renderer by guarding both aspect cases on equal_aspect; behaviour unchanged. Test tests/reporting/test_cluster21_scatter_aspect_chain.py (source pins); 35 existing scatter-related reporting tests pass. |
| 21-F24 | FIXED | multi_output is optional (default False) in resolve_output_path; the save path no longer computes or passes it. Existing callers that pass it still work. Test tests/reporting/test_cluster21_resolve_output_no_multi.py; 23 existing save and output-path tests pass. |
| 21-F27 | FIXED | Split-name canonicalisation raises ValueError when two keys normalise to the same name, instead of silently overwriting. Test tests/reporting/test_cluster21_split_names_collision.py; 31 existing split-comparison tests pass. |
| 21-F28 | FIXED | Binary split takes the positive column of a (n, 2) predict_proba and refuses other multi-column shapes instead of raveling them against an n-length label vector. Test tests/reporting/test_cluster21_binary_score_shape.py; 31 existing split-comparison tests pass. |
| 21-F25 | PARTIAL | Confusion-margins heatmap truncates its tick labels as _heatmap does. Not changed: the duplicated cell-text loop (the two copies differ in pinned colour bounds, so a shared helper needs that parameter). Test tests/reporting/test_cluster21_confusion_labels_truncated.py; 23 existing heatmap tests pass. |
| 21-F22 | FIXED | Unknown-colormap warning throttled per name with max_count=1 (was unthrottled, per trace). Test tests/reporting/test_cluster21_unknown_colormap_once.py. |
| 21-F21 | FIXED | Timing chart-type grouper strips a hex disambiguator only when the token contains a digit (lookahead), so decade/accede/facade keep their names. Test tests/reporting/test_cluster21_timings_hex_suffix.py (6 pass). |
| 21-F26 | PARTIAL | SHAP per-instance result docstring now says the figure is closed on return (the finally closes it) and that n_background counts explained rows. Not changed: the same closed-figure pattern in shap_panels.py:811-812 named in the finding, and the field itself is not renamed (API). Test tests/reporting/test_cluster21_shap_result_docs.py (docstring pins). |
| 20-F30 | WON'T FIX | _per_query_groups has no production caller, but it is the parity reference for the live group slices: two tests import it, and the shared uncalled-functions allowlist classifies it as a deliberately kept reference. Not deleted. |
| 20-F29 | FIXED | Confusion-matrix hidden-values note is joined to the title only when a title is set, so a missing title no longer becomes the literal 'None'. Test tests/reporting/test_cluster20_confusion_title_none.py (source pin); existing confusion-matrix tests pass. |
| 20-F28 | FIXED | Binary composer docstring corrected to the real default panel constant (ROC PR SCORE_DIST THRESHOLD); KS and GAIN stated as opt-in. Test tests/reporting/test_cluster20_binary_default_panels_doc.py. |
| 20-F31 | FIXED | Deleted the unreachable duplicate empty-row guard in _interval_band_panel (the first guard returns before it). Test tests/reporting/test_cluster20_interval_band_single_guard.py (source pin); 37 existing quantile-chart tests pass. |
| 20-F32 | FIXED | Error-bias binning warning reworded to describe a skipped feature, not a fallback. Test tests/reporting/test_cluster20_error_bias_warning_text.py (source pin). |
| 20-F33 | FIXED | Model-comparison bar colours use the loop index (was an O(K^2) names.index lookup, equal to i for unique names); the duplicated Spearman comment block is reduced to one copy. Test tests/reporting/test_cluster20_model_comparison_index.py; 58 existing model-comparison tests pass. |
| 20-F34 | FIXED | Multilabel ROC and calibration legend entries shorten their class labels through the multiclass helper. Test tests/reporting/test_cluster20_multilabel_legend_names.py; 46 existing multilabel tests pass. |
| 20-F35 | FIXED | worst_k_table logs a WARNING when feature_importances does not match the feature count, before falling back to the first columns. Test tests/reporting/test_cluster20_worst_k_importance_warns.py; 78 existing error-analysis tests pass. |
| 20-F36 | FIXED | separability_panel refuses fewer than two features with a ValueError naming the count, instead of a raw IndexError from features[1]; longer lists keep their first-two behaviour. Test tests/reporting/test_cluster20_separability_needs_two.py; 11 separability tests pass. |
| 20-F37 | FIXED | Decision-curve _finite_binary logs a WARNING with the dropped row count when it removes non-finite or off-{0,1} rows. Test tests/reporting/test_cluster20_decision_curve_drop_logged.py; 46 existing decision-curve tests pass. |
| 20-F38 | FIXED | Calibration heatmap docstring corrected: each cell shows its binned ECE (support-weighted per-score-bin gaps), with the [0, 1] score assumption stated. Test tests/reporting/test_cluster20_calib_heatmap_doc.py (docstring pins). |
| 20-F39 | FIXED | Multiclass sampler derives class starts from change points in the already-sorted labels instead of a second np.unique sort; equal to np.unique's classes and first indices. Test tests/reporting/test_cluster20_multiclass_change_points.py; 46 existing multiclass tests pass. |
| 20-F40 | FIXED | 2-D PDP counts columns from shape_of(X) (1-D counts as one) instead of copying the whole frame. Test tests/reporting/test_cluster20_pdp2d_column_count.py; 42 existing 2-D PDP tests pass. |
| 20-F41 | FIXED | Grouping helper logs a WARNING with the count of rows that have a missing label and so belong to no group; fully labelled input is silent. Test tests/reporting/test_cluster20_grouping_missing_logged.py; 9 existing grouping tests pass. |
| 22-F28 | FIXED | The secondary-ECE None is passed straight into the holdout helper; the redundant assignment is removed (the helper's parameter is the starting value, so the assignment was not dead). Covered by the calibration policy tests (4 pass). |
| 22-F29 | DEFERRED | true_outcomes is unused in generate_similar_probs_logit_space and _random_walk, but these are public names called positionally by tests and the calibration API. Removing or documenting the parameter changes signatures: owner decision. Not changed. |
| 22-F30 | DEFERRED | Lazy resolution cannot be limited to __all__: 50 public names exported by the submodules (for example generate_similar_probs, fast_roc_auc) are not in __all__ but are reachable as attributes. Fixing the perf needs an owner decision on the public surface. Not changed. |
| 22-F31 | FIXED | Recency-weight module docstring corrected: i counts back from the most recent (i = 1), matching the code, where the newest observation carries the largest weight. Test tests/core/test_cluster22_recency_index_convention.py. |
| 22-F32 | FIXED | Tukey multiplier: lru_cache moved onto a helper that receives only hashable arguments; extra dist kwargs (possibly unhashable, e.g. array loc) bypass the cache instead of raising TypeError. The both-None guard was fixed earlier. Tests tests/core/test_cluster22_tukey_unhashable_kwargs.py (2) plus existing Tukey tests (8 total) and stats doctests (4) pass. |
| 22-F27 | DEFERRED | Timing helpers in _bench_timing_shared.py differ in warmup semantics and are called by about a dozen bench scripts; consolidating to one implementation per shape needs a decision on the kept signatures. Not changed. |
| 25-F56 | FIXED | Unused marginals array removed from the MI kernel in an earlier commit (995050561); verified by the MI kernel tests (28 pass). Recorded here, where the disposition was missing. |
| 25-F55 | FIXED | MI docstrings corrected from the fixed 0-14 range to the validated [0, 127] / [0, n_bins) contract (three sites). Test tests/feature_selection/test_cluster25_mi_bin_range_doc.py (source pins). |
| 25-F57 | FIXED | Ridge prefilter scans candidate sizes ascending, so the smallest qualifying size wins regardless of input order (fixed earlier in 7565c18fd). Covered by tests/feature_selection/test_cluster25_prefilter_unsorted_sizes.py and the nested-CV prefilter test (3 pass). |
| 25-F58 | FIXED | Varying-size anchor de-duplication uses a set for membership while keeping the rank order (fixed earlier in 7a797751d). Covered by the existing varying-size subset tests (exit 0). |
| 25-F59 | FIXED | Removed the unused logger, its import and the stray module-level string from optbinning. Test tests/feature_selection/test_cluster25_optbinning_cleanup.py (source pins; module imports). |
| 25-F62 | FIXED | Tuning-rule comment corrected to OR semantics; expected_fitness assigned in both search modes so the exploration plot is not stale. Test tests/models/test_cluster25_expected_fitness_both_modes.py; existing tuning tests (5) pass. |
| 25-F64 | WON'T FIX | _resample_desc_order_counting is re-exported by _core_auc_brier, referenced by the fused bootstrap benchmark, and listed in the shared uncalled-functions allowlist as a deliberately kept alternative (the classification the rule asks for). The function has moved to _bootstrap_auc_resample.py. Not deleted. |
| 28-F52 | FIXED | Bare raise AssertionError in load_and_prepare_dataframe replaced with a TypeError naming the function and the received type. Test tests/training/test_cluster28_preprocessing_typed_error.py; 5 loader tests in test_preprocessing pass. |
| 28-F51 | FIXED | Narrowing helper docstring now says non-finite values (NaN features) also yield None, matching the behaviour. Test tests/training/test_cluster28_narrow_numeric_docstring.py (3 pass). |
| 28-F53 | FIXED | Object-label stack uses arr.tolist() (np.stack converts each element itself, so the result is identical; checked on array and list elements). Test tests/training/test_cluster28_object_stack_equivalence.py (2 pass). |
| 28-F55 | FIXED | The no-op del of a parameter and its unreachable NameError handler were removed earlier (93739ee10); related trainer tests pass (11). Recorded here, where the disposition was missing. |
| 28-F58 | FIXED | LightGBM shim fit uses the resolved cpu count only while building params and restores the caller's n_jobs, so get_params()/clone are unchanged by fit. Test tests/training/test_cluster28_lgb_njobs_not_mutated.py; 6 existing shim tests pass. |
| 28-F59 | FIXED | Recurrent configure step rebuilds RecurrentConfig with dataclasses.replace, so a field not in the old hand-copy list can no longer be silently dropped; __post_init__ validation still runs. Test tests/training/test_cluster28_recurrent_config_carries_all_fields.py (2 pass). The broader test_core_coverage run exceeded the foreground time limit and was not completed. |
| 28-F63 | FIXED | train_eval defines its logger once; the select_target comment now says the import re-exports it for existing callers. Test tests/training/test_cluster28_train_eval_single_logger.py (2 pass). |
| 28-F72 | FIXED | Target/model key built once by _entry_key (same value as the three inline copies). Test tests/training/test_cluster28_honest_entry_key.py (3 pass). |
| 28-F75 | FIXED | Test coverage added for heartbeat_interval_from_env: unset gives the default, '0' disables, invalid falls back with a WARNING, negative clamps to 0. Test tests/training/test_cluster28_heartbeat_env_parsing.py (4 pass). |
| 28-F74 | FIXED | all_active_phases tests added. Found while testing: phase() registered the thread's stack only on creation, so a cleared cross-thread map (test isolation) hid a live thread; registration now happens on every entry. The pop statement is an explicit if block. Tests tests/training/test_cluster28_all_active_phases.py (2 pass); 22 existing phase-related meta tests pass. |
| 28-F65 | DEFERRED | Several parts: force_windows and cpu_count are kept on the public signature deliberately for tests (removing them changes the signature), the CUDA probe block is repeated three times and needs a careful extraction, and the workers docstring needs a per-platform check. Owner decision on the signature; not changed. |
| 29-F28 | FIXED | Deleted the unused _RELATIVE_IMPORT pattern in numba_cache_guard. Test tests/utils/test_cluster29_numba_guard_no_dead_pattern.py (source pin). |
| 29-F27 | FIXED | Time-aware OOF skip for predict_proba is logged at WARNING (the INFO path is only for non-probability methods); covered by tests/training/composite/eval/test_composite_oof_time_aware_logged.py (3 pass). Recorded here, where the disposition was missing. |
| 29-F29 | FIXED | Post-import callbacks all run even when one raises (fixed earlier in 2b08fdc4a). Covered by tests/utils/test_cluster29_post_import_callbacks_all_run.py (1 pass). Recorded here, where the disposition was missing. |
| 29-F30 | DEFERRED | The docstring already states the filter is installed once and never restored. The remaining issue is the misleading block-scoped name; renaming changes a helper used across several modules, so it is an owner decision. Not changed. |
| 29-F31 | FIXED | Benchmark sweep logs a WARNING when no combo gives a finite score (the decorated call returns None). Test tests/utils/test_cluster29_param_oracle_no_finite_warns.py (2 pass). |
| 29-F34 | FIXED | Names-mismatch guard raises ValueError naming both lists (fixed earlier in 83b648677). Covered by tests/votenrank/test_biz_val_correlation_diversity_ablation.py and the diversity recommendations test (11 pass). Recorded here, where the disposition was missing. |
| 29-F33 | WON'T FIX | n_jobs is a documented no-op (coalition scoring is serial), marked unused-ok, and kept so every estimator in the votenrank API shares one signature. Removing it would break the uniform signature. Not changed. |
| 29-F35 | FIXED | Dual blend: zero restarts already raises a clear error (from the solver, naming restarts); the internal bare assert is replaced by an explicit ValueError; zero trials raises a ValueError naming n_optuna_trials instead of a raw best_params error. Test tests/votenrank/test_cluster29_dual_blend_zero_counts.py (2 pass); existing dual-blend tests (4) pass. |
| 29-F36 | FIXED | Legend entries are matched by label (optimality_gap / mean) instead of by position; pin test test_cluster29_stability_legend_by_name.py. |
| 29-F39 | FIXED | disk_cache docstring now states the summary key reads every element (no SHA pass, not sub-linear). |
| 29-F40 | FIXED | Protect set drops the bogus <key>.sha256 name; real <key>.pkl.sha256 sidecar kept. Test test_cluster29_disk_cache_protect_names.py. |
| 29-F41 | FIXED | Bagging without randomize_order/randomize_start logs a WARNING that all bags are identical. Test test_cluster29_hill_climb_bagging_warns.py. |
| 39-F11 | FIXED | Deleted the dead if-pass branch with a stale clean_ram comment in the extractor base. |
| 39-F12 | FIXED | Tuned-threshold report logger uses __name__ instead of another module's name. Test tests/training/test_cluster39_logger_name_and_dead_pass.py. |
| 39-F15 | FIXED | Facade drops unused typing/numpy/pandas/polars/pyutilz imports and the unused private helpers, and no longer exports np/pd/pl in __all__. Test tests/training/test_cluster39_extractors_facade_exports.py. |
| 39-F16 | FIXED | matplotlib.pyplot is imported inside _showcase_target_distributions, not at module load. Test tests/training/test_cluster39_showcase_lazy_matplotlib.py. |
| 39-F46 | FIXED | Preset factories merge caller kwargs over their defaults via _build_config, so tfidf_only(default_cat=...) no longer raises a duplicate-keyword TypeError. Test tests/training/test_cluster39_presets_keyword_override.py; 37 preset tests pass. |
| 39-F52 | FIXED | MLFRAME_MEMORY_BUDGET_GB must be a positive finite number; 0, negative, non-finite or unparsable values log a WARNING and fall back to psutil/cgroup. Test tests/training/test_cluster39_memory_budget_env_positive.py. |
| 39-F63 | FIXED | to_tensor_any converts a bare pl.Series via to_numpy instead of raising AttributeError. Test tests/training/test_cluster39_to_tensor_any_polars_series.py. |
| 39-F64 | FIXED | to_numpy_safe defaults to cpu=True so a CUDA tensor converts, matching the docstring's 'moves to CPU if needed'. Test tests/training/test_cluster39_to_numpy_safe_default_cpu.py. |
| 39-F65 | WON'T FIX | render() is a public single-module helper of the generator, uncalled inside the package by design; project rule keeps uncalled helpers and classifies them rather than deleting. LIGHT_MODULES re-export is documented as intentionally importable from here. No code change. |
| 39-F66 | FIXED | mi_normalization and group_mi_aggregate tuples now live in MRMR param_constants; the fit validator and the fs_params spec both read them. Test tests/training/test_cluster39_fs_params_enums_from_constants.py; test_fs_params_in_sync.py passes. |
| 39-F71 | FIXED | Mixup categorical-prefix lookup moved to MLPTorchModel._mixup_cat_prefix, memoised per network object, so the module walk no longer runs every training step. Test tests/training/test_cluster39_mixup_cat_prefix_cache.py. |
| 39-F49 | FIXED | Column-iterator docstring now names the TF-IDF path's list() materialisation instead of claiming full avoidance. Test tests/training/test_cluster39_text_encoder_docstring_accuracy.py. |
| 39-F51 | FIXED | Smoothing bench verdict relabelled historical: the live TargetEncodeParams default is 3.0, the bench compares against the former 10.0 incumbent. Test tests/training/test_cluster39_smoothing_bench_verdict_current.py. |
| 39-F50 | FIXED | iter113 benchmark NEW arm calls the production _float_canonical_tokens instead of an inline copy; the OLD np.unique arm stays inline as baseline. Test tests/training/test_cluster39_bench_uses_production_tokens.py. |
| 39-F48 | WON'T FIX | text_encoder._column_to_string_list is an uncalled helper; the live function of the same name is text_detection's, which the production path and tests use. Name clash is a readability hazard with no behavioural effect; project rule keeps uncalled helpers and classifies them rather than deleting. No code change. |
| 39-F59 | FIXED | AggregatingValidationCallback skips the metric with a WARNING when an epoch has no validation batches, instead of torch.concat raising on an empty list. The class has no src caller; it is kept (uncalled helpers are classified, not deleted). Test tests/training/test_cluster39_aggregating_callback_empty_epoch.py. |
| 39-F60 | FIXED | Periodic LR-finder condition drops the redundant 'or current_epoch == 0' clause (epoch 0 is already a multiple of any validated positive period); behaviour unchanged. Test tests/training/test_cluster39_periodic_lr_finder_epochs.py. |
| 39-F61 | DEFERRED | A non-linear output_activation is ignored for num_classes != 1. Raising would break classification: the default in _trainer_config_builders sets tanh_train_range for all runs. A warning would fire on every classification run. Needs a decision on whether the builder default should be regression-only. No code change. |
| 39-F14 | FIXED | Learning-curve scorer falls back to estimator.score only when the scorer cannot take (estimator, X, y) by signature; a TypeError raised inside a real scorer now propagates. Fallback logged at WARNING. Test tests/training/test_cluster39_learning_curve_score_fallback.py; 20 tests pass with the existing learning-curve suite. |
| 39-F47 | FIXED | CustomParams forbids undeclared keys like the other params models, so a misspelled field fails loudly; every in-repo caller passes declared fields only. Test tests/training/test_cluster39_custom_params_forbids_extra.py; 26 custom-handler tests pass. |
| 41-F29 | FIXED | Removed the never-read boundaries accumulator from find_change_points_pelt; the returned inner change-point pairs are unchanged. Test tests/training/test_cluster41_changepoint_no_dead_boundaries.py. |
| 41-F43 | FIXED | Predict kwargs are dropped only when the regressor's predict signature cannot accept them (_predict_accepts); a TypeError raised inside a regressor that accepts them now propagates instead of being retried without kwargs. Test tests/training/test_cluster41_predict_kwargs_by_signature.py; 45 related tests pass. |
| 41-F30 | FIXED | coerce_timestamps_for_audit detects tz on a plain parse before forcing UTC, so the tz-stripping warning fires only for tz-aware input. Test tests/training/test_cluster41_coerce_tz_warning_only_for_tz_input.py; 24 related audit tests pass. |
| 41-F28 | FIXED | Envelope-clip docstring now states that predictions are not clipped without train statistics (an eval-derived envelope would leak the scored labels), matching the code. Test tests/training/test_cluster41_envelope_clip_docstring_matches_code.py. |
| 41-F34 | FIXED | MTR verbose warning uses <...> placeholders instead of literal {plot_file}/{k}; each metric falls back to str() when non-numeric, so one bad value no longer drops the whole block. Test tests/training/test_cluster41_mtr_log_hygiene.py. |
| 41-F39 | FIXED | Ranker FS builds the input column set once per pipeline output (input_cols) instead of once per output column. Test tests/training/test_cluster41_ranker_fs_column_set_hoisted.py. |
| 39-F13 | RESOLVED | Already correct: the warm-start probe is inside a conditional that short-circuits when warm_start is False, so no estimator is built. Pinned by tests/training/test_cluster39_learning_curve_no_probe_without_warm_start.py. |
| 39-F58 | FIXED | Lookahead slow/fast sync issues one torch._foreach_lerp_ and one torch._foreach_copy_ instead of a per-parameter loop; numerically identical. Test tests/training/test_cluster39_lookahead_batched_sync.py; 17 lookahead tests pass. |
| 41-F33 | FIXED | scale_pos_weight is set only for two-class targets; multiclass XGBoost ignores it. class_weight and auto_class_weights are unchanged. Test tests/training/test_cluster41_scale_pos_weight_binary_only.py. |
| 41-F37 | FIXED | Pandas inf cleaning copies the frame only when infinities are present, so the caller's data is no longer mutated. Test tests/training/test_cluster41_ranker_inf_clean_no_caller_mutation.py. |
| 41-F38 | DEFERRED | Ranker categorical vocabulary includes val and test while strategies/_cat_levels_shared excludes test. Aligning them changes how test-only categories are encoded (unseen levels), so it is a behaviour decision needing an owner call. No label leak is involved. No code change. |
| 37-F33 | FIXED | Removed the unused duplicate _WATCHDOG_RELATIVE_THRESHOLD from _phase_composite_post; the live definition in _composite_wrap_watchdog now carries the rationale comment. Test tests/training/test_cluster37_watchdog_threshold_single_source.py. |
| 37-F34 | DEFERRED | Fitted-domain adapters for cbrt, yj, qn and sp are built but only log_y passes its fitted domain. Wiring the others in changes domain screening for those transforms, so it needs a behaviour decision. Removing the bindings alone would leave the tuple shape to rename. No code change. |
| 37-F37 | FIXED | Bounded-zoo check matches 'ols' and 'lstsq' as whole tokens and keeps 'linear' as a substring, so lgb_colsample stays bounded. Test tests/training/test_cluster37_bounded_zoo_token_match.py; 15 AR-skip tests pass. |
| 37-F46 | FIXED | Representative-target fallback no longer labels LEARNING_TO_RANK (and other non-supervised keys) as classification; they are skipped. Test tests/training/test_cluster37_pick_representative_target_ltr.py. |
| 37-F47 | FIXED | The float64 cast of a split column sits inside the materialisation guard, so a non-numeric column is skipped instead of raising ValueError. Test tests/training/test_cluster37_build_column_non_numeric_skipped.py. |
| 37-F40 | FIXED | Removed the dead NameError guard around the getattr for plot_outputs; the read is unchanged. Test tests/training/test_cluster37_plot_outputs_no_dead_guard.py. |
| 37-F41 | FIXED | Numba warm-up failure is logged at INFO (was DEBUG), so a slow first fit is explained. Test tests/training/test_cluster37_prewarm_failure_logs_info.py. |
| 37-F43 | FIXED | Preprocessing-config guard is an explicit RuntimeError, not an assert, so it survives python -O. Test tests/training/test_cluster37_preprocessing_guard_not_assert.py. |
| 37-F44 | FIXED | Target-distribution module logs under its own __name__ so per-module log levels apply; the two caplog expectations that pinned the parent name are updated. Test tests/training/test_cluster37_target_distribution_logger_name.py. |
| 37-F45 | FIXED | Knob provenance recorder failure is logged at WARNING (was DEBUG) and training continues. Test tests/training/test_cluster37_knob_provenance_failure_warns.py. |
| 37-F32 | FIXED | Dedup priority builds set(ranked) once (ranked_set) instead of once per comprehension element. Test tests/training/test_cluster37_dedup_ranked_set_hoisted.py. |
| 37-F31 | DEFERRED | _run_composite_targ_split_cfg_overrides_td takes about 30 positional parameters. Splitting it is a multi-call-site refactor that needs its own pass with call-site tests; a partial change here would add risk without a behaviour fix. No code change. |
| 37-F38 | FIXED | XT-ensemble keep-mask builds the survivor set once (_survivor_set) instead of once per OOF name. Test tests/training/test_cluster37_xt_survivor_set_hoisted.py. |
| 37-F39 | DEFERRED | Component-shim loop is duplicated across _xt_ensemble_helpers and _phase_dummy_baselines, with two emit helpers. Consolidating needs a shared helper and call-site tests in both paths, so it gets its own pass. No code change. |
| 37-F30 | FIXED | Honest-gain floor filter compares specs by identity, so dict equality on ndarray values cannot raise or mismatch. Test tests/training/test_cluster37_honest_gain_floor_identity.py. |
| 37-F49 | FIXED | _entry_metric reads the positive class (1) first for a {0,1} class-indexed layout, as its comment states; multiclass keeps ascending order. Test tests/training/test_cluster37_entry_metric_positive_class.py. |
| 20-F45 | FIXED | Calibration chart subsample indexes s and t directly with the drawn indices instead of a one-off lambda; same values. Test tests/training/test_cluster20_calibration_subsample_direct_index.py. |
| 20-F43 | FIXED | Fuzzy-membership docstring and comment now state the fit draw is bounded and with replacement, matching rng.integers. Test tests/training/test_cluster20_fuzzy_subsample_wording.py. |
| 20-F42 | DEFERRED | The panel's CI base rate uses the full y while the table's point estimate uses its own subsample base (row_idx). Aligning them needs category_discriminability_table to return the base it used, which changes its return shape. Deferred to a pass that changes that API. No code change. |
| 20-F44 | FIXED | __all__ moved to the end of _error_analysis_shared; blank-line gap added before _frame_rows in _drift_shared. Test tests/training/test_cluster20_shared_module_layout.py. |
| 26-F38 | FIXED | Extra dispatch module docstring describes its own renderers (model comparison, decile, model card, stability, split and target overlays) instead of the copied error-analysis/drift text. Test tests/training/test_cluster26_extra_dispatch_docstring.py. |
| 26-F37 | FIXED | A target_type given as an enum value resolves to the member (value first, then name), instead of being ignored. Test tests/training/test_cluster26_target_type_by_value.py. |
| 26-F36 | FIXED | SHAP confidence plot returns with a visible WARNING when matplotlib is missing (plt is None), instead of raising on plt.get_fignums. Test tests/training/test_cluster26_shap_plot_without_matplotlib.py. |
| 32-F16 | FIXED | Group-target cast is one assignment; the two identical branches are collapsed. Test tests/training/test_cluster32_group_target_single_cast.py. |
| 39-F30 | WON'T FIX | Uncalled helpers are classified, not deleted (project rule): fingerprint_df and CustomHandler.signature have no production caller and are kept as public API. purge_by_df_token is NOT dead: the dataset-cache phase calls it through getattr (_phase_train_one_target_dataset_cache.py:193). No code change. |
| 41-F31 | FIXED | Censoring verdict requires a non-negative zero-window level (0 <= ratio <= threshold); a negative level is declining_level. Test tests/training/test_cluster41_maturity_negative_intercept.py; 12 existing maturity/temporal tests pass. |
| 41-F32 | FIXED | Low-variance detection at zero mean is scale-free: only an exactly constant column is flagged; a non-zero mean keeps the relative-std test. Test tests/training/test_cluster41_feature_low_variance_scale_free.py; 40 analyzer tests pass. |
| 41-F35 | DEFERRED | interaction_strength keeps its own 20 s cap, against the shared diagnostics budget design. Making it budget-driven changes how the cap is set for every diagnostic, so it needs a design decision. No code change. |
| 41-F36 | DEFERRED | Split label is inferred from the last underscore token of plot_file. Fixing it means passing the split explicitly through the chart call, an API change across the diagnostics callers. No code change. |
| 41-F44 | DEFERRED | The DSL render branch ignores show and save_path. Honouring them needs a decision on the DSL's display contract across callers. No code change. |
| 13-F16 | FIXED | Removed nine module-level imports used only by their own import line from the BorutaSHAP package; the classes it uses stay. Test tests/feature_selection/test_cluster13_boruta_unused_imports_removed.py. |
| 13-F15 | WON'T FIX | TentativeRoughFix, missing_values_y, check_missing_values, average_of_list and hasNumbers have no production caller. Uncalled helpers are classified and kept, not deleted (project rule). No code change. |
| 13-F21 | FIXED | fold_rank_consistency builds each fold's top mask from the top indices with a boolean array instead of a Python loop over every column; same result. Test tests/feature_selection/test_cluster13_fold_rank_consistency_vectorised.py. |
| 12-F87 | PARTIAL | Removed the unused sd_a/sd_b fit fields and the unreachable ratio_eps branch from fe_richops_bench. Not changed: _apply_op skips a pair with a missing operand without recording it; output names carry the pair index, so the gaps are identifiable. Test tests/feature_selection/test_cluster12_richops_dead_code_removed.py. |
| 12-F16 | FIXED | Unblinding replaces whole labels in one pass, longest first, so arm_b no longer rewrites inside arm_bx; a non-object mapping JSON means no mapping instead of a crash. Test tests/feature_selection/test_cluster12_blinding_boundaries_and_json.py; 16 blinding tests pass. |
| 32-F15 | FIXED | Config sufficiency test uses the signed delta, so a negative delta (init_score better than raw) counts as sufficient, as the module note says; the abs() had made it high_potential. Test tests/training/test_cluster32_init_score_sign_consistent.py; 55 baseline diagnostics tests pass. |
| 13-F28 | DEFERRED | kernel_arg_dtype_audit maps long to int32 by Windows LLP64. A portable fix needs a platform-aware type table and a decision on which platform the audit must catch; deferred to the benchmarks owner. No code change. |
| 26-F35 | DEFERRED | parametric.py calls settings.load_profile at import, changing global Hypothesis settings for any importer. Moving it to an explicit call changes what test suites that import the module get. Needs a decision. No code change. |
| 37-F48 | DEFERRED | The OOD lag router uses the validation split for both the route-or-keep decision and the reported score, so the reported score is optimistic. Fixing that is an ML evaluation-design decision (which split reports the estimate). No code change. |
| 09-F43 | DEFERRED | Pseudo-SMOTE regression confidence threshold is an in-sample quantile. Using out-of-fold predictions changes the aux model cost and the threshold semantics, which needs an ML design decision. No code change. |
| 32-F28 | FIXED | Skipped-target results in the profile harness return the same 16-field tuple the caller unpacks (one helper, _skipped_profile_result); the six-field early returns would have raised on unpack. Test tests/training/test_cluster32_profile_skip_arity.py. |
| 13-F19 | FIXED | explain() raises a clear RuntimeError at its top when the fitted frames were released for a large fit, instead of passing None into the explainer. Test tests/feature_selection/test_cluster13_explain_after_frame_release.py. |
| 12-F103 | DEFERRED | downstream is reimplemented in several bench files with different return types. Consolidating means a shared evaluator with one return contract, which needs an owner decision on the contract. No code change. |
| 12-F8 | DEFERRED | Three near-identical downstream evaluators differ in rounding and empty-input handling. Unifying them changes results at the rounding edge, so it needs a decision on the canonical behaviour. No code change. |
| 13-F18 | DEFERRED | Per-trial Python loops in update_importance_history and calculate_hits could be vectorised, but the change needs a perf benchmark and equivalence check of the importance history, deferred to a perf pass. No code change. |
| 27-F19 | DEFERRED | Two suffix-stripping implementations differ (only one strips WithFastpath). Unifying them changes which names are stripped in one path, so it needs a decision. No code change. |
| 28-F54 | DEFERRED | The two CatBoost loss-ensuring helpers duplicate about 60 lines. Extracting a shared helper is a refactor that needs call-site tests in both paths; deferred to a refactor pass. No code change. |
| 33-F4 | DEFERRED | Binary and multiclass margin paths clip probabilities to different epsilons. Picking one changes numerical output at the clip boundary, so it needs an owner decision on the clip. No code change. |
| 37-F42 | DEFERRED | _apply_plot_style_overrides changes process-wide matplotlib and plotly state and is not restored on failure. Restoring it needs a context-manager design across the setup path; deferred. No code change. |
| 37-F36 | DEFERRED | The unary transforms logger name differs from the composite_transforms name used elsewhere; changing it affects log filters configured by users, so it needs a naming decision. No code change. |
| 39-F72 | DEFERRED | The torch from_numpy benchmark's Strategy C measures a path the code does not produce, and the 0.014 ms figure needs re-measurement on the real path before the doc can be corrected. Deferred to a benchmark re-run. No code change. |
| 15-F13 | DEFERRED | Three near-identical unwrap-to-inner-estimator loops with different attribute sets across the wrappers files. Consolidating them needs one attribute-set contract across the three call sites, deferred to a refactor pass. No code change. |

| 39-F38 | FIXED | HGB TF-IDF cap is now applied on the feature-handling path: TF-IDF max_features is capped for hgb via _tfidf_params_for_model (was exported but never called). Test tests/training/test_cluster39_hgb_tfidf_cap_wired.py; 38 feature-handling tests pass. |

| 17-F9 | FIXED | Degenerate calibration report and fast_ice_only on empty input use ICE_UNCOMPUTABLE (1e6, finite) instead of 1.0, so a real model with ICE above 1.0 can no longer tie it. Behaviour change: degenerate ICE value changes. Test tests/metrics/test_cluster17_degenerate_ice_sentinel.py; existing degenerate-report test updated to the sentinel. |



| 09-F7 | PARTIAL | stacked_qnn Mode A is nested: each outer fold rebuilds layer 1 on its training part only, so a row's label never shapes its own layer-2 feature. Behaviour change: Mode A values change. Not done: stacked_attention.py (same leak, and its docstring still claims leak-free) is unchanged. Test tests/feature_engineering/test_cluster09_stacked_mode_a_nested_no_self_label.py; 100 shortlist tests pass. |


| 12-F34 | DEFERRED | H35 and H32 pick the config by OOF AUC on rows the members were fit on. A real fix refits the members on a training split and scores the augmented features of held-out rows, and HybridSelector has no transform path that builds those features. Needs that transform first (a feature-engine change). Benchmark-only, not production. No code change. |

| 08-F17 | FIXED | Removed the unused chunk parameter from _median_pairwise_chunked and its caller argument; the median is unchanged. Test tests/feature_engineering/test_cluster08_median_pairwise_no_chunk.py. |


| 08-F9 | FIXED | The per-fold float32 cast of the training target is computed once before the head loop, not once per head. Values unchanged. Test tests/feature_engineering/test_cluster08_oof_cast_hoisted.py; 9 OOF tests pass. |

| 08-F6 | FIXED | Adversarial flip docstring states the implemented regression rule: a move of the prediction by more than the 10th percentile of train absolute residuals. Docstring-only fix. Test tests/feature_engineering/test_cluster08_flip_docstring_matches_rule.py. |

| 08-F14 | FIXED | ShortlistTransformerAdapter keeps random_state exactly as given (sklearn get_params/clone contract); the effective RNG seed is resolved by _rng_seed() (seed when given, else random_state). Test tests/feature_engineering/test_cluster08_adapter_random_state_kept.py; 7 existing adapter tests pass. |

| 08-F8 | FIXED | The key-bank cache fingerprint includes the ANN backend (build_key_bank passes its backend through to the fingerprint and save), so indices from one backend are never reused for another. Behaviour change: every cache key changes once (cache invalidation). Test tests/feature_engineering/test_cluster08_key_bank_fingerprint_backend.py; 8 key-bank cache tests pass. |

| 08-F12 | WON'T FIX | batch_weighted_mean/std/vector_mean are uncalled outside benchmarks. Uncalled helpers are classified and kept per the project rule; the docstring claim about _oof is corrected in its own pass. No code change. |

| 08-F13 | WON'T FIX | _project_single_head_njit is a test-parity fallback with no production caller; kept as classified test-only helper per the project rule. No code change. |

| 08-F15 | DEFERRED | Four near-identical SMOTE loops remain despite the module header. Consolidating them changes the kernel call contract across the synthesis paths; deferred to a refactor pass with call-site tests. No code change. |

| 08-F16 | DEFERRED | The package eagerly imports about 130 transformer modules. Lazy imports change the package's public import surface and the import-time behaviour of every caller; deferred as a design decision. No code change. |

| 09-F16 | FIXED | FCA lattice-construction failure logs at WARNING (was INFO): the zero features are no longer indistinguishable from a real no-match. Test tests/feature_engineering/test_cluster09_fca_fallback_warns.py. |

| 09-F18 | FIXED | NCA fit records warnings and logs a ConvergenceWarning at WARNING naming max_iter; other warnings stay suppressed as before. Test tests/feature_engineering/test_cluster09_nca_convergence_surfaced.py. |

| 09-F17 | FIXED | Tree path extraction failure now yields NaN for the path and match-count columns and logs a WARNING; it used to emit constant-1.0 columns, which made every path look matched. Behaviour change: failed folds produce NaN. Test tests/feature_engineering/test_cluster09_tree_path_failure_nan.py. |

| 09-F14 | FIXED | H0 mean_birth was constant zero; replaced by std_lifespan (std of bar lifespans), which varies. Schema change: the column is renamed pd_mean_birth -> pd_std_lifespan. Test tests/feature_engineering/test_cluster09_persistence_no_constant_birth.py. |

| 09-F22 | FIXED | A failed gradient member's cosines are NaN (were 0, which read as orthogonal disagreement); mean and min use the cosines that exist. Behaviour change: those rows change. Test tests/feature_engineering/test_cluster09_failed_member_cos_nan.py. |

| 09-F23 | FIXED | Degenerate folds in the LDA projection and the per-class spectral attention return NaN instead of zeros, so they cannot be confused with real rows at the origin. Behaviour change: degenerate folds yield NaN. Test tests/feature_engineering/test_cluster09_degenerate_fold_is_nan.py. |

| 09-F25 | DEFERRED | The MDL binning is a single top-level split by design: its docstring pins it bit-identical to a reference in tests/feature_engineering/transformer/test_mdl_binning_split_kernel.py. Recursive Fayyad-Irani splitting changes the bins of every feature and breaks that pin, so it needs an owner decision on the reference before code changes. No code change. |

| 09-F19 | FIXED | For binary targets the sign-residual baseline emits NaN bias columns: with y in {0,1} the sign target y > mu_hat is the label itself, so the second model only re-learned y. The OOF baseline column is kept. Behaviour change: binary output changes. Test tests/feature_engineering/test_cluster09_sign_residual_binary_nan.py. |

| 09-F20 | FIXED | The ordinal monotonicity count (n_violations) is NaN for binary targets: their sub-population thresholds are not nested, so counting adjacent increases has no ordinal meaning. Behaviour change: binary n_violations is NaN. Regression and other targets are unchanged. Test tests/feature_engineering/test_cluster09_ordinal_violations_binary_nan.py. |

| 09-F21 | FIXED | Distributional moments are NaN for binary targets: the reweighted classifiers' outputs are not quantiles of one distribution, so sorting them as quantiles had no meaning. The binary quantile block is NaN and the moment features propagate it; those classifiers are no longer trained. Behaviour change: binary output is NaN. Test tests/feature_engineering/test_cluster09_distributional_binary_nan.py. |

| 10-F6 | FIXED | Benchmark harness failures (MRMR fit/transform, AUC scoring, per-config failures) are logged at WARNING instead of DEBUG, so NaN rows have a visible reason at the default level. Test tests/feature_selection/test_cluster10_bench_failures_warn.py. |

| 11-F8 | FIXED | catboost availability logs at INFO when not installed (ImportError) and at WARNING when installed but broken, so a broken install is no longer indistinguishable from a missing one. Test tests/feature_selection/test_cluster11_catboost_broken_warns.py. |


| 11-F7 | FIXED | A counted estimator module that fails to import with a non-ImportError is reported as missing and logged at WARNING; it used to abort every arm's FitCounter. Test tests/feature_selection/test_cluster11_counted_import_failure_listed.py. |

| 11-F12 | FIXED | A subnormal-width candidate range (NB / width overflows to inf) takes the flat-range branch instead of binning with a non-finite scale, so no NaN bins are produced. Test tests/feature_selection/test_cluster11_subnormal_range_no_nan_edges.py. |

| 11-F9 | FIXED | RFECV ranking_ numeric ranks of a different length than the names fall back to kept-vs-dropped with a WARNING, instead of being stringified and matched against names. Test tests/feature_selection/test_cluster11_rfecv_numeric_rank_mismatch.py. |

| 11-F11 | FIXED | A BorutaShap column with no finite trial maps to -inf (ranked below every scored column) instead of 0.0, which tied with a real zero score. Behaviour change: unscored scores are -inf. Test tests/feature_selection/test_cluster11_boruta_unscored_not_zero.py. |

| 11-F13 | FIXED | The wiring CLI takes repeatable --param NAME:TYPE and forwards them to plan and apply, so a generated operator keeps its constructor parameters. Test tests/feature_selection/test_cluster11_wiring_params_passed.py. |

| 11-F4 | FIXED | qbin ranks only the finite values; NaN rows get the dedicated code k instead of the top bins. Behaviour change: NaN rows return code k (fixed-bin callers must allow k). Finite-only input is unchanged. Test tests/feature_selection/test_cluster11_qbin_nan_own_code.py. |


| 11-F10 | DEFERRED | InformationGreedyArm calls the njit kernel once per (candidate, chosen) pair from Python. Vectorising needs an equivalence check of the selected sets across JMIM, RELAX and CMIM, and the finding is performance-only with unchanged output. Deferred to a benchmark-validated pass. No code change. |

| 10-F7 | FIXED | _check_versions reports a capture older than MAX_VERSION_AGE_DAYS (90) as a warning, using capture_ts; the constant is no longer dead. Test tests/feature_selection/test_cluster10_aggregate_version_policy.py. |

| 10-F5 | FIXED | A version check with one missing capture logs that the comparison was skipped. Test tests/feature_selection/test_cluster10_aggregate_version_policy.py. |

| 10-F8 | FIXED | Redundant columns without informative ones raise a clear ValueError. Test tests/feature_selection/test_cluster10_bench_utility_edges.py. |

| 10-F9 | FIXED | A failed log-file write is reported once on stderr (was on every failure); the console keeps printing. Test tests/feature_selection/test_cluster10_bench_utility_edges.py. |

| 10-F4 | RESOLVED | Each local copy of a shared bench helper now carries a marker naming its intentional divergence (iter31 extra stage, iter33 own stage order, iter54 name/label header, stability orchestration build_selector). The 25 inline ShapProxiedFS constructions are per-experiment configs and are kept inline. Test tests/feature_selection/test_cluster10_documented_divergence.py. |

| 12-F31 | RESOLVED | All 60 fs_hybrid bench files that installed a bare catch-all ignore now silence only FutureWarning and DeprecationWarning; other warnings surface. Each file compiles. Test tests/feature_selection/test_cluster12_no_catch_all_warning_filter.py. |

| 35-F44 | RESOLVED | Repeated integer-valuedness check removed from the grouping call site (level-count helper). The two production-dead helpers stay as test-facing references, allowlisted in the uncalled-functions gate. Test tests/training/composite/discovery/test_cluster35_low_card_check_once.py. |

| 35-F46 | RESOLVED | _residualise uses a magnitude-relative constant test. The screening re-exports of the leaf modules and shape_of moved to the top of screening.py (no circular import: the leaves do not import screening). Tests test_cluster35_residualise_relative_constant.py, test_cluster35_low_card_check_once.py (4 passed). |

| 40-F29 | FIXED | RecurrentConfig refuses bad batch_size, sequence_preprocessing, ema_decay, num_classes and now mixup_alpha (negative or non-finite) at construction; 0 still disables mixup. Test tests/training/neural/test_cluster40_recurrent_config_validates_fields.py (8 passed). |

| 35-F33 | FIXED | TailCompositeEstimator.fit cross-fits the body by default (residual_oof_folds=5), so the tail threshold and GPD are fitted on out-of-fold residuals. Costs residual_oof_folds extra body fits. residual_oof_folds below 2 keeps the in-sample path with its WARNING. Behaviour change for default callers. Tests test_cluster35_tail_oof_default.py and test_cluster35_tail_in_sample_residuals_warn.py. |

| 35-F26 | WON'T FIX | Discovery fits its tiny models without sample weights (model.fit(x_fit[valid], t_fit) in every call path), so passing weights into smearing would describe a weighted residual distribution that the fitted model never saw. smeared_prediction already accepts sample_weight for when a weighted fit is introduced. Existing test test_cluster35_smeared_prediction_matches_fit.py covers the unweighted match. No code change. |

| 36-F8 | DOCUMENTED | Known train/serve skew is documented at _build_meta_features_at_predict: stage 2 trains on OOF meta-features, predict uses the full-refit stage-1 model. Fold-model averaging would change serve output for all callers, so it is not done. No behaviour change. |

| 34-F16 | WON'T FIX | calibration_adjusted_score has no caller inside the package but is a public lazy export (mlframe.training.__init__). Wiring it in would change the calibration path; removing the export breaks the public API. Classified as an intentional public helper, per the uncalled-functions rule. No code change. |

| 28-F70 | FIXED | A failed grid variant with stop_on_error=False is a typed GridVariantError (label, exc_type, message) instead of a plain error dict. No in-repo consumer read the old dict. Test tests/training/test_cluster28_grid_typed_error.py (2 passed). |

| 38-F22 | FIXED | A pre-OD category union also takes the validation categories, so the strict val cast cannot fail on a val-only category. Behaviour change: such categories now join the Enum instead of crashing. Test tests/training/core/test_cluster38_precomputed_union_covers_val.py (plus existing test_fe_preod_enum_domain.py, 2 passed). |

| 34-F14 | DEFERRED | RAM is already bounded: the Gram is accumulated over 8192-row blocks, so one block of 500 float64 columns is about 32 MB. The audit's full-buffer fix (about 6.4 GB) would raise RAM, and column chunking re-reads columns per block pair, so it trades RAM for more reads. Reducing the 195k column reads is a speed change that needs a benchmark on real data. No code change. |

| 38-F10 | RESOLVED | Deleted the dead weight-iteration module (_phase_train_one_target_weight_iteration.py) and its test; the live inline loop in _phase_train_one_target_steps.py is the only copy. Its allowlist entry in test_shared_uncalled_functions.py is removed. Note: the removed test pinned the identity-equivalent dedup ordering on the dead copy; the live path has no unit test for that ordering yet. |




| 24-F17 | PARTIAL | Causal KDE lookback (opt-in, default exact). Dominant error found: the grid argmax jumps to another density peak on about 0.6% of rows, giving max gaps near 0.36 while the mean is 0.006. The auto criterion now uses the MEAN probe gap, default lookback_tol=1e-2 (chosen by me; owner to confirm). Measured on synthetic AR(1) groups (20 x 1500, exp 0.95): fixed L=256 7.1x faster, mean gap 6.0e-3, 0.84% of rows gap above 0.05; auto (selects a candidate) 18.3x faster, mean gap 1.1e-2, 2.1% of rows gap above 0.05. Owner decision still open: accept this tail or tighten tol. Tests tests/feature_engineering/test_cluster24_causal_kde_lookback.py (6 passed). Bench: scratchpad/bench_kde_lookback.py. |

| 40-F31 | FIXED | The stratified inverse-frequency sampler is one function, build_stratified_sampler in _recurrent_data.py, used by the wrapper (seeded by random_state) and by RecurrentDataModule (unseeded, as before). RecurrentDataModule stays exported. Tests tests/training/neural/test_cluster40_stratified_sampler_single_source.py and test_recurrent_workers_default.py (17 passed). |

| 39-F41 | FIXED | Stale-lock reclaim is documented as intentionally absent: a PID-check removal races with a holder whose PID was reused, so a SIGKILLed holder still blocks until timeout and the .locks directory must be cleared by hand. StaleLockReclaimed is kept as a reserved public name and never raised; hf_provider comment corrected. Test tests/training/feature_handling/test_cluster39_no_lock_reclaim_documented.py (2 passed). |

| 40-F26 | DOCUMENTED | Per-split entity history is a documented choice at both sites (entity-time and MA-crossover FE): eval splits restart their entity history so no eval row sees training-era state; carrying train history forward would shift the scored distributions. Test tests/training/pipeline/test_cluster40_per_split_history_documented.py (2 passed). |

| 39-F26 | FIXED | The CUDA-lost CPU fallback stays float32 (fp16 on CPU is slow or unsupported). The fallback warning now states that the precision changes mid-run and that earlier batches were fp16, and the comment no longer claims identical embeddings. Test tests/training/feature_handling/test_cluster39_cpu_fallback_precision_documented.py (2 passed). |

| 41-F14 | DEFERRED | Probability metrics fed by one-hot predict() when predict_proba is missing. The fix needs the probabilistic report to skip ROC, PR, Brier, calibration and plots on that path. report_probabilistic_model_perf takes about 50 parameters and computes all of them in one pass, so gating each branch needs a dedicated pass with a report-level test. Plan: return a 'probabilities unavailable' marker from _predict_probs_if_missing, short-circuit the metric block, and keep label metrics. No code change in this pass. |

| 18-F15 | FIXED | The streaming harm tracks a zero member per cell and returns exactly 0 there, matching the materialised zero rule, instead of about 3e-150 from the 1e150 surrogate. Other cells keep the plain harmonic mean. Observation outside this finding: on non-zero cells the materialised path with default parameters gave different values for this input, not checked here. Tests tests/models/ensembling/test_cluster18_streaming_harm_zero.py (2 passed); existing streaming tests 31 passed. |

| 39-F68 | FIXED | The all-zero-weight check in the weighted MLP loss runs on every 50th batch (_ZERO_WEIGHT_CHECK_EVERY), not on every step, so the host-device sync is paid 1/50 as often. A zero-weight batch is reported at the next sampled batch rather than on the batch itself; the loss value is unchanged. Test tests/training/neural/test_cluster39_zero_weight_check_sampled.py (2 passed). |

| 28-F69 | FIXED | apply_to_target_types is a frozenset in both BaselineDiagnosticsConfig and DummyBaselinesConfig; pydantic coerces list and tuple input. Test tests/training/test_cluster28_apply_types_one_type.py and the updated tests/training/baselines/test_baseline_diagnostics.py (35 passed). |


| 26-F26 | DOCUMENTED | Queued figures are recorded as saved at submit (documented choice at the submit site): the worker owns the figure from there, and a later worker failure is counted in the queue summary but not reconciled into charts['failed']. Reconciling needs a worker-to-dict update path, not done. Test tests/training/test_cluster26_34_documented_choices.py (2 passed). |

| 34-F20 | DOCUMENTED | The warm-start signature is computed on the full frame, not a sample: a sampled signature can match a frame whose rows changed outside the sample and reuse stale specs. The cost is paid per call and is documented at the check. Test tests/training/test_cluster26_34_documented_choices.py (2 passed). |

| 38-F9 | PARTIAL | The pre-screen skip now tests the group column's own protection (group_ids_raw carries its name), not the whole protected set, which target names always filled. Not done: the explicit group-name field from the extractor, which needs a new TrainingContext slot. Test tests/training/core/test_cluster38_group_guard_specific.py. |

| 38-F24 | FIXED | A CV policy without a target_type is refused with a ValueError instead of being silently wired as regression. Test tests/training/core/test_cluster38_cv_policy_needs_target_type.py (2 passed). |


| 35-F28 | DOCUMENTED | The default quantile grid (0.05..0.95) truncates the tails: sample() holds draws flat outside Q(0.05)..Q(0.95) and crps() omits the outer 5% per side. Documented at the grid; extending it with 0.001/0.999 costs two inner fits and is noisy on small samples, so the default is kept and an explicit tail grid is the way out. Test tests/training/composite/test_cluster35_grid_truncation_documented.py (2 passed). |

| 35-F19 | DOCUMENTED | The three composite-refit copies (ensemble/__init__.py plain fold, ensemble/__init__.py oof holdout, ensemble/_oof_external.py) each state their group semantics at the function. Behaviour is unchanged; the copies still differ, and a fix to one must be checked in the others. Consolidating them needs the owner's group semantics. Test tests/training/composite/test_cluster35_refit_copies_documented.py (2 passed). |

| 32-F9 | FIXED | CBIterationMetricsCallback.flush() captures the final round when the stride skipped it. It is now called by the fit wrapper (_train_model_with_fallback) after a successful fit; a failed fit does not flush. Tests tests/training/test_cluster32_fit_wrapper_flushes_cb.py and tests/training/callbacks/test_cluster32_cb_flush_last_round.py (5 passed). |

| 32-F12 | FIXED | The baseline diagnostic skips when fewer than 50 rows remain after sampling, with a reason naming the in-sample problem, instead of scoring the training rows as its own holdout. Tests tests/training/baselines/test_cluster32_diagnostic_min_rows.py and the existing test_baseline_diagnostics.py (35 passed). |


| 18-F10 | FIXED | The regression train-split path is chosen by any member instead of member 0: each member is probed once, and members without predictions are None in the list, as the classification path already does. Before, a member-0 miss dropped the whole train-split ensemble. Tests tests/models/ensembling/test_cluster18_train_path_any_member.py (2 passed); related ensembling tests 17 passed. |


| 18-F8 | FIXED | The default AUC in _score_blend drops rows with a NaN label before scoring, so they no longer count as negatives. Tests tests/models/ensembling/test_cluster18_score_drops_nan_labels.py (2 passed). |

| 15-F3 | FIXED | The RFECV checkpoint now saves and restores nofeatures_score, so a resumed run keeps its no-features baseline instead of NaN. Test tests/feature_selection/test_cluster15_checkpoint_round_trip.py (3 passed). |

| 15-F4 | FIXED | The RFECV checkpoint now saves and restores per_fold_scores, so the per-split columns survive a resume. Test tests/feature_selection/test_cluster15_checkpoint_round_trip.py. |

| 15-F5 | FIXED | The RFECV checkpoint now saves and restores the signed importances onto the estimator, so dispatched linear aggregation sees the same scale after a resume. Test tests/feature_selection/test_cluster15_checkpoint_round_trip.py. The heavy test_biz_val_wrappers_rfecv.py file was not confirmed in this pass. |

| 15-F7 | FIXED | Stability selection passes the bootstrap's fitted rows as train_data, so the coefficient scale comes from the rows the model was fit on rather than its out-of-bag rows. Test tests/feature_selection/test_cluster15_coef_scale_uses_fit_rows.py (2 passed). |

| 21-F6 | FIXED | Multiclass SHAP interactions rank each pair by the class it moves most (largest mean /interaction/), as the main-effects panel does; binary keeps the positive-class convention. Tests tests/reporting/test_cluster21_multiclass_interaction_best_class.py (2 passed). |

| 21-F13 | FIXED | The trend fit cache stores a blake2b digest of each array's contents and checks it on every hit, so an in-place refill of the same buffer is refitted rather than served stale. Tests tests/reporting/test_cluster21_trend_cache_content.py (2 passed). |

| 26-F12 | FIXED | The changepoint penalty is dimensionless: it multiplies a robust variance of y (squared 1.4826*MAD), so rescaling the series keeps the breakpoints. For unit-variance y this reproduces the former absolute penalty. Tests tests/signal/test_cluster26_changepoint_scale_free.py (2 passed); existing changepoint tests pass. Note: the uncalled-functions meta-test fails on eight filters/ functions that come from the merged origin tree, not from this change. |

| 27-F12 | FIXED | Trainset aging without timestamps is refused with a ValueError in the no-timestamp row-based split, instead of slicing a shuffled split by position. Without the limit the split is unchanged. Test tests/training/test_cluster27_aging_needs_timestamps.py (2 passed). |

| 27-F14 | DOCUMENTED | cv_plus_intervals is a pooled symmetric band from out-of-fold absolute residuals, not the CV+ or Jackknife+ construction of Barber et al. 2021; its docstring now states that no coverage guarantee is proven. The name is kept for API compatibility. A real CV+ needs per-fold models and is not built. No behaviour change. |

| 28-F37 | FIXED | resolve_mlp_predict_batch_size takes a device_id and forwards it to the memory probe; the predict caller passes the current CUDA device, so the batch is sized for the device the model runs on. The default stays 0. Tests tests/training/neural/test_cluster28_predict_batch_device_id.py and the existing test_mlp_runtime_defaults.py (38 passed). |

| 29-F14 | DOCUMENTED | Aggregated oracle rows keep only the median and n_obs, so a later append takes a median of medians and the result depends on append batching. The exact fix stores (value, weight) pairs per metric, which changes the stored format and needs a schema bump and migration. Documented at the aggregation site; not changed here. |

| 18-F7 | FIXED | The DEGENERATE marker judges the actual label values: a single label is tagged, two labels are judged by their ratio, and a multiclass subset is not tagged by this binary rule. Labels {0, 2} balanced are no longer tagged. Tests tests/models/ensembling/test_cluster18_degenerate_actual_labels.py (4 passed) and the train-path tests (6 passed in total). |

| 17-F11 | FIXED | The comment claimed ECE used the same data-adaptive grid as CMAEW. The ECE kernel uses the fixed [0,1] equal-width grid, and CMAEW uses the data span, so the two partition the data differently. The comment is corrected; behaviour is unchanged (aligning the grids would change the headline ECE). |

| 32-F7 | DOCUMENTED | The class universe is the union of train, val and test labels, so eval-only labels stay encodable in 0..K-1. The priors are train-only, so an eval-only class gets zero prior mass and no eval split informs the baseline probabilities; only the (N, K) width depends on the union. Documented at the union in dummy.py; behaviour unchanged. |

| 11-F3 | DOCUMENTED | The searched arms in exp6_tables (preset_best and each mode) report the max over many in-sample candidates, while truth and als are single candidates, so the searched numbers are inflated even on noise. Documented at the comparison as an upper bound, not a like-for-like result. A null run (permuted y) would be the follow-up if a conclusion depends on it; no code or numbers changed. |

| 18-F6 | FIXED | The outlier gate keeps a row when any member is finite (the quality_gate rule), and uses nanmedian with NaN-aware MAE/STD when cells are missing, so disjoint NaN rows no longer switch the gate off. The finite path and the numba kernel are unchanged. Tests tests/models/ensembling/test_cluster18_gate_disjoint_nan.py (2 passed). Note: tests/models/test_ensemble_gate_does_not_freeze_caller_arrays.py::test_the_cache_does_not_retain_the_arrays already fails on HEAD without this change; not caused here. |

| 32-F11 | FIXED | The init_score combiner is fit on the quick model's training rows only (the same 80/20 split, via _quick_train_mask) and scores every row, so the holdout labels no longer shape the holdout init_score. Tests tests/training/baselines/test_cluster32_init_score_combiner_train_only.py and the existing test_baseline_diagnostics.py (36 passed). |

| 18-F19 | FIXED | A failed member test-prediction stacking is logged as a WARNING with its cause, so the missing member_test_preds has a visible reason; a successful stacking still stamps the attribute. Tests tests/models/ensembling/test_cluster18_stacking_failure_logged.py (2 passed). |

| 34-F21 | FIXED | The finite-row mask is computed once after the impute and reused; a test now pins it to the row-wise definition (target and every kept feature finite), including NaN and inf cells. Tests tests/training/composite/test_cluster34_finite_mask_matches_rows.py (2 passed). |

| 09-F32 | FIXED | The binary branch of _fit_3baselines_with_sigma predicts only on the query rows; a spy on LightGBM predict_proba pins that no call sees the training fold. The regression branch keeps its training predictions for the out-of-fold residuals. Test tests/feature_engineering/test_cluster09_binary_baselines_query_only.py (1 passed). |

| 09-F35 | FIXED | The quantile-neighbour graph is Euclidean (NearestNeighbors with the default metric) and the module docstring now says so; the cosine claim is gone. Test tests/feature_engineering/test_cluster09_quantile_neighbours_euclidean.py pins the requested metric (1 passed). The other docstrings listed in this finding are not re-checked here. |

| 40-F22 | FIXED | MLPRanker stores the fitted column names and predict refuses a different name set or order, so reordered columns of equal width no longer score silently. NaN labels are refused at fit and the count is checked at predict (already in place). Tests tests/training/neural/test_cluster40_ranker_column_names_order.py and the existing test_cluster40_ranker_refuses_nan_labels_and_bad_width.py (4 passed). |

| 24-F16 | DOCUMENTED | The exp scheme uses one recurrence pass per group (A_L = p*(A+v), B_L = p*(B+1)), equal to the prefix re-walk within 1e-9 (test_cluster24_exp_causal_recurrence.py, 2 passed). The poly scheme keeps the O(m^2) prefix re-walk: its weights depend on the window length, so prefix sums cannot be updated exactly. The choice is now stated at the poly branch. |


| 08-F4 | FIXED | The BGM sampler gained an opt-in return_source flag: (samples, 'bgmm') for a mixture sample and (samples, 'bootstrap') for any resample, including a too-small minority set. The default return is unchanged, so current callers are unaffected. The fallback is logged at WARNING (earlier change). Its inner warning filter is narrowed to Future/Deprecation. Tests tests/feature_engineering/test_cluster08_bgmm_source_flag.py (3 passed). |

| 09-F40 | FIXED | pure_pos_smote and smote_distance choose SMOTE candidates by excluding the source index, not by slicing column 1 onward, so a duplicate that sorts first cannot leave self among the candidates. density_weighted_smote already excluded by index. Test tests/feature_engineering/test_cluster09_smote_no_self_candidate.py forces self into the second neighbour slot: it fails on the old code and passes on the fix (2 passed). |

| 25-F60 | DOCUMENTED | The unused order/inv_order arrays in group_zero_sum_constraint are gone (grep is empty). interaction_values is filled only on the CatBoost native branch and interaction_base_values is never filled; both are now marked reserved at their definition and kept in the returned tuple so the public shape is unchanged. No behaviour change. |

| 25-F61 | DOCUMENTED | The CatBoost depth space is randint(1, 17), so depth 16 is reachable (scipy randint excludes the high bound; the comment at tuning_catboost.py:55 says so). The loss_function list at :257 is a deliberate exclusion that applies only when leaf_estimation_method is Newton, and it is commented there. No behaviour change beyond the depth fix already in place. |

| 25-F68 | FIXED | metrics.__all__ no longer exports internals (logger, logging, np, numba, annotations, core, Callable); every listed name resolves. Test tests/metrics/test_cluster25_metrics_all_no_internals.py (2 passed). |

| 26-F22 | FIXED | The adversarial fit, the shared error-prep build and the PSI heatmap all build outside their locks; each lock now guards only the cache read and insert, so a second target is no longer held for the first target's build. Tests tests/reporting/test_cluster26_prep_build_outside_lock.py (fails on HEAD, passes on the fix) and tests/reporting/test_cluster26_adversarial_fit_outside_lock.py (2 passed). |

| 28-F56 | FIXED | A failed train-envelope computation is logged at WARNING and says the reporter falls back to the per-split envelope, since the report content changes. The early-stopping strip failure was already WARNING. Test tests/training/test_cluster28_envelope_failure_warns.py (1 passed). |

| 39-F62 | FIXED | The probe that runs the degenerate-init check on every fit is renamed _generate_mlp_every_fit_init_probe (it was named as a bench-profiling helper); its failure is logged at WARNING and says the check was skipped. The old name is gone from the sources. Tests tests/training/neural/test_cluster39_init_probe_name.py (2 passed). |

| 39-F31 | FIXED | apply.py uses one named constant _SVD_DEFAULT_DIM (256, unchanged) at both SVD call sites instead of a repeated literal, and the assembler docstring names the real svd_default_dim argument. There is no config field for the width, so none is added. Tests tests/training/feature_handling/test_cluster39_svd_default_constant.py (2 passed). |

| 40-F32 | FIXED | Removed the dead xxhash probe and the empty TYPE_CHECKING blocks from _recurrent_torch_model.py and recurrent.py (the cache-key path has its own probe in _recurrent_cat_embeddings.py). The unused importlib.util import in the UMAP builder was removed earlier. get_num_classes is a public method with its own tests, so it stays. Tests tests/training/neural/test_cluster40_recurrent_dead_probe_removed.py (2 passed). |

| 37-F35 | FIXED | Removed the unused short_name parameter from both chain factories in nonlinear.py and its keyword at every call site (registry.py and the auto-chain factory call). The duplicate logger and TYPE_CHECKING were already removed. Test tests/training/composite/test_cluster37_chain_factory_no_short_name.py (2 passed); the auto-chain cache replay test passes (3 passed with it). |

| 17-F21 | FIXED | The RMSLE and MAPE warn-dedup sets in _regression_extras.py are cleared at a 4096-entry bound, the same bound the Tweedie cache uses, so a long run cannot grow them without limit. Test tests/metrics/test_cluster17_warn_seen_bounded.py (2 passed). |

| 11-F5 | FIXED | A failing operator in the exp2 study is reported at WARNING through the throttled logger, once per operator; the same change was already in core.py. Test tests/feature_selection/test_cluster11_exp2_failure_warns.py (1 passed). |

| 11-F6 | FIXED | No feature-selection benchmark module installs a process-wide catch-all warning filter: the whole _benchmarks tree (148 files, the fe_operator_factory and stat_study modules included) silences only FutureWarning and DeprecationWarning. Each changed file compiles. Test tests/feature_selection/test_cluster11_no_catch_all_filter_in_factory.py (1 passed) walks the whole tree. |

| 11-F15 | FIXED | The set(names) rebuilt inside membership comprehensions is hoisted once per call in the arms modules (_arms_it_family.py was the remaining site). Perf-only: the selected list is unchanged by construction, so no separate test is added. |

| 11-F20 | PARTIAL | Test gap closed for _integer_bins (test_cluster11_integer_bins_contract.py) and for als_rank1 (test_cluster11_als_rank1_smoke.py, 1 passed: finite per-row output, tracks the signal). Not done: family_eval and preset_matrix_best need the unary/binary transform fixtures, which are not built in this pass. |

| 13-F34 | FIXED | env_sampled_fit exits cleanly with a message when no samples were taken, before any statistics: the mean, the percentile and the page-fault arithmetic all come after that exit. The script runs its fit at import, so no unit test is added; the guard is verified by reading the code. |

| 28-F68 | FIXED | Removed the empty trailing 'Callback Classes for Training Monitoring' section header from helpers.py (no code under it). The device-gating comment the finding names is no longer in the file. Test tests/training/test_cluster28_helpers_no_empty_header.py (2 passed). |

| 37-F51 | FIXED | The group-slice helper is renamed _group_train_slice at its definition and at its lazy-proxy call site; the old name is gone from the core sources. Test tests/training/core/test_cluster37_group_slice_name.py (2 passed). |

| 38-F21 | DOCUMENTED | The fallback cache key is repr(pre_pipeline), not the object identity; the comment now says so. Keying on id() would need a strong reference that pins each pipeline in memory and changes cache hits. The collision is narrow: only un-introspectable pipelines with identical reprs share a slot, and only on the fallback path. Kept as is, documented. |

| 04-F14 | DEFERRED | bayes_auc's per-row tie-group loop is vectorised and matches the old loop exactly (test_cluster04_bayes_auc_vectorised_matches_loop.py). Left open: the prevalence bisection runs 60 passes per call. A warm start narrows the bracket from the previous result, which changes the numerics within tolerance, so it needs a benchmark on real call counts before it lands. |


| 04-F28 | FIXED | Declared kind roles built once per derivation (_declared_kind_roles, first edge wins) instead of rescanning all edges per column; M-bias collider check uses the precomputed ancestor set. Tests: tests/data/test_cluster04_declared_roles_single_pass.py, test_cluster04_m_collider_no_descendant_walk.py |

| 24-F40 | FIXED | Windowed edge aggregation: a NaN entity is its own group. Positions come from factorize codes (NaN kept), so the lookup no longer raises KeyError; grouping is one stable argsort, not a per-entity scan. Test tests/feature_engineering/test_cluster24_nan_entity_and_warmup_vote.py |

| 24-F35 | FIXED | MA crossover vote_sum is NaN on rows where every pair is still warming up (any_vote mask), instead of 0 that reads as neutral consensus. Test tests/feature_engineering/test_cluster24_nan_entity_and_warmup_vote.py |

| 24-F32 | FIXED | Spectral NaN fill is per window inside _spec_pow (the segment-wide mean let early windows read future values); _nan_impute_segment removed. Test tests/feature_engineering/test_cluster24_spectral_scale_and_nan_windows.py; test_fe_top_a_fixes.py::test_f20 revised |

| 24-F33 | FIXED | Spectral entropy normalises by the window's own power with p*log(p) only where p>0; flatness floor is relative to each window's mean power, and a zero-power window reports 0.0. Scale-invariance test in test_cluster24_spectral_scale_and_nan_windows.py |

| 24-F34 | FIXED | Rolling Hurst-family wrappers report NaN for every window that held a non-finite input (cumulative-count mask), instead of a finite value from the zero fill. Test test_cluster24_spectral_scale_and_nan_windows.py::test_hurst_window_with_nan |

| 24-F44 | FIXED | Verify path uses max(..., default=0.0) for the derived signal: a raw column with no present derived column keeps a zero retained fraction, so it is kept and recorded in safety_report instead of raising ValueError. Test tests/feature_selection/test_cluster24_drop_raw_no_derived_columns.py |

| 24-F43 | FIXED | Hetero vote importance reads coef_ / feature_importances_ from a Pipeline's final step, so the linear panel member votes with /coef_/ as documented instead of permutation importance on a subsample. Test tests/feature_selection/test_cluster24_hetero_pipeline_coef.py; stability/test_hetero_vote.py and test_degenerate_input_guards.py pass (10 passed). biz_val hetero tests left to CI |

| 24-F58 | FIXED | nearest_past_join fallback carries the row position in a private column instead of reset_index(), which created a column named index; the caller's index is restored. Test tests/feature_engineering/test_cluster24_nearest_past_fallback_index.py |

| 24-F57 | RESOLVED | Not reproduced: pandas sort_values keeps tied rows in input order for both the multi-key and single-key sorts used here, so the tie order is deterministic. No code change. |

| 24-F46 | FIXED | chronological_order warns (UserWarning) with the count of labels that could not be parsed; they still sort last as before. Test tests/feature_selection/test_cluster24_unparsed_time_and_append_index.py |

| 24-F49 | FIXED | append_columns raises ValueError when new_cols does not share X's index, instead of appending NaN columns silently; all three callers already pass index=X.index. Test tests/feature_selection/test_cluster24_unparsed_time_and_append_index.py |

| 24-F41 | FIXED | MRMR member abstains (no mrmr key, warning logged) when MRMR selects no relevant column, instead of voting for every relevant column. Tests: tests/feature_selection/test_cluster24_hybrid_vote_fallbacks.py; stability/test_hybrid_selector_deep.py::test_use_mrmr_false_zeroes_engineering_and_still_votes updated |

| 24-F42 | FIXED | No-consensus fallback keeps the clusters with the most votes (ties all kept) instead of the union of every pick. Tests: test_cluster24_hybrid_vote_fallbacks.py; stability/test_hybrid_selector_chunk48_fixes.py::test_combine_union_fallback_warns updated |

| 24-F59 | FIXED | _rep_member docstring names cluster_rep='first' as the constructor default (it said sum_fi). Test tests/feature_selection/test_cluster24_p3_docs_and_dead_code.py |

| 24-F61 | FIXED | Removed the features_usefulness zero array that was overwritten; the same zeros are now passed in place, so behaviour is unchanged. Existing MRMR tests cover the path |

| 24-F66 | FIXED | Removed the unused classes local in prune_conditional_twins (computed and deleted). Test tests/feature_selection/test_cluster24_p3_docs_and_dead_code.py; tests/feature_selection/test_hybrid_twin_prune*.py pass (12) |

| 24-F62 | FIXED | run_efs refuses duplicate target names before binning or any MI work, instead of after the full permutation loop. Test tests/feature_selection/test_cluster24_p3_guard_order_and_type.py |

| 24-F63 | FIXED | ForwardSelectSelector raises TypeError naming the returned type instead of a bare AssertionError. Test tests/feature_selection/test_cluster24_p3_guard_order_and_type.py |

| 24-F56 | FIXED | stack_relational_chain no longer zero-fills the new columns at each hop: a parent with no eligible history keeps NaN instead of a zero mean. Test tests/feature_engineering/test_cluster24_relational_no_history_nan.py; test_biz_val_relational_dfs.py and test_fe_top_b_fixes.py pass (17) |

| 24-F55 | FIXED | Feature std is the unbiased sample std (ddof=1) everywhere in the feature path: compute_numaggs and the moments kernel emit n-1 (NaN under two values), row_wise_summary_polars and the numpy reference row_wise_summary use ddof=1. Skew and kurt keep the population std they standardise with. Left as is on purpose: the weighted std (no single unbiased definition for weights), Welford in _numerical_stable (shared with metrics and ensembling). Tests: test_numerical.py, test_row_wise_summary_polars.py, test_row_wise_polars_fastpath.py, test_biz_val_row_wise_summary.py, feature_engineering changed-kernel set (426 passed, 10 skipped) |


| 24-F64 | FIXED | ACESelector and the four functional adapters (ForwardSelect, GreedyBackwardElimination, ZeroImportancePruning, CascadeSelect) are exported from the package and listed in __all__. Test tests/feature_selection/test_cluster24_p3_exports.py; the existing public-name tests pass (20) |

| 24-F51 | FIXED | The per-window Hurst docstring states that it returns log(R/S)/log(n), and that compute_hurst_rs returns the raw R/S ratio, so the two agree only after the same transform. Test tests/feature_selection/test_cluster24_p3_exports.py |

| 24-F60 | FIXED | One Benjamini-Hochberg implementation in the leaf module _fdr_bh; general.py and ace.py delegate (ace keeps its crit dtype). Test tests/feature_selection/test_cluster24_p3_dedup_bh_and_ace.py: shared, general and ace agree with a textbook BH on random p-values, ties and empty input; existing ACE/FDR/EFS tests pass (230) |

| 24-F65 | FIXED | The fit, read-importances and width-check block shared by the held-out and in-sample branches of ACE's replicate importances is one helper, _fit_and_split_importances. Tests in test_cluster24_p3_dedup_bh_and_ace.py; existing ACE tests pass (230) |

| 24-F52 | DOCUMENTED | pysr_operators comment now states which operators return NaN on bad input and that inv returns +/-Inf at 0 in Julia and sympy zoo at predict time, so train and predict disagree there. Behaviour not changed |

| 24-F53 | DOCUMENTED | Classified, not deleted: _wavedec_numba_typedlist and _waverec_numba_typedlist have no production caller and are referenced only by tests, and wavedec_dispatch is a public helper with no production caller; recorded in wavelet_dwt.py per the wire-in-or-classify rule |

| 24-F54 | DOCUMENTED | wavelet_dwt documents that it takes one numba-flags constant from mlframe.metrics.shared and that the fallback must mirror it; the silent fallback is by design |

| 24-F45 | FIXED | Forward chaining now covers both consumers. Stability vote: forward_chain_folds. SHAP OOF explainer: _oof_split_iter forward-chains a temporal policy, rows no fold explains (first block) are NaN in phi/base, and _fit_oof_shap drops them from phi, base, phi_var, y_phi, X_proxy, y_search, X_search and the search policy so every row-aligned array stays on the same rows; per_fold_mean is sized to the folds run. Cost: the search set loses its first block (1/n_splits of the rows) under a temporal policy. Tests: tests/feature_selection/test_cluster24_forward_chain_folds.py, tests/feature_selection/shap_proxied/test_cluster24_forward_chained_oof_shap.py (incl. a full ShapProxiedFS fit under a temporal policy); existing SHAP/cv_policy set 110 passed |

| 20-F10 | FIXED | Heterogeneity noise floor uses the prevalence of the finite labels, so one NaN label no longer makes it NaN. Test test_cluster20_nan_prevalence_bench_and_pdp_warn.py |

| 20-F16 | FIXED | The ndarray path of _resolve_feature_matrix refuses feature_names whose length differs from the column count, as the DataFrame path does. Test test_cluster20_axis_contracts_and_probes.py |

| 20-F19 | FIXED | Relevance-grade count uses the value range (a span of at most 12 grades needs no scan), else a full unique, instead of probing only the first 50k rows. Test test_cluster20_axis_contracts_and_probes.py |

| 20-F23 | FIXED | category_discriminability_table checks X against y length before sampling, so a longer X is refused. Test test_cluster20_axis_contracts_and_probes.py |

| 20-F25 | FIXED | The Spearman bench sets _rank_stats.SPEARMAN_NJIT_MIN_N (where spearman reads it) instead of the removed prediction_stability attribute. Test test_cluster20_nan_prevalence_bench_and_pdp_warn.py |

| 20-F26 | FIXED | metric_over_time marks the axis as time only for a datetime index; otherwise x_is_time is False and the label is bucket. Test test_cluster20_axis_contracts_and_probes.py |

| 20-F27 | FIXED | A failed SHAP pair choice in the 2-D PDP is logged at WARNING (was DEBUG), naming the exception. Test test_cluster20_nan_prevalence_bench_and_pdp_warn.py. tests/reporting/test_cluster20_nan_prevalence_bench_and_pdp_warn.py, tests/reporting/test_cluster20_axis_contracts_and_probes.py; existing chart tests 992 passed, one unrelated xgb dmatrix shim failure not checked on HEAD |

| 22-F9 | FIXED | _autoconfigure_cuda_home imports logging once at the top of the function; the two inner imports made logging a local name for the whole function, so the except handler raised UnboundLocalError when the probe failed first. Test tests/test_cluster22_cuda_autoconfig_logging.py |

| 22-F12 | FIXED | show_classifier_calibration cuts a full-length ranking down to each interval's rows (positions inside [lo, r), shifted by lo, order kept) before binning, instead of passing it unchanged to the unchecked njit kernel. Test tests/calibration/test_cluster22_interval_ranking_slices.py; existing calibration tests 170 passed, 2 unrelated fast_r2 numba-typing failures in test_audit_divide_by_zero.py (metrics.regression, not touched) |

| 20-F11 | FIXED | calibration_drift base rate is the mean of the finite labels, and the usable-window note counts windows with a finite ECE. Test tests/reporting/test_cluster20_drift_nan_label_base_rate.py |

| 22-F25 | FIXED | The fused bootstrap AUC/Brier/log-loss/ECE bundle returns None (not applicable) unless y is exactly 0/1, so a {1,2}, {-1,+1} or NaN-bearing encoding falls back to bootstrap_metrics instead of giving a meaningless AUC. Test tests/evaluation/test_cluster22_fused_bootstrap_binary_labels.py |

| 22-F11 | FIXED | leak_scan shifted overlap takes the union over the same aligned segments as the intersection (segment popcounts), instead of full-row popcounts; the now-unused row/block popcount variables are gone. Test test_cluster22_leak_scan_shifted_jaccard.py (the audit's worked example does not add up, so the test uses its own rows); existing leak_scan tests 77 passed |

| 22-F14 | FIXED | The persistence floor uses the row's actual mass beyond the active class as the rest mass, so every returned row sums to 1 even when the input row did not. The biz_val flicker test used rows summing to 0.9 and relied on the old arithmetic; its rows are normalised and the floor moved to 0.475 (inside the window that suppresses flicker without blocking real changes). Test test_cluster22_persistence_floor_row_sum.py |

| 22-F15 | FIXED | fit_group_bias_correction excludes rows with a non-finite y_true or y_pred (warned), so one NaN no longer makes the global mean NaN and every shrunk ratio fall back to 1.0. Test test_cluster22_group_bias_nonfinite_rows.py |

| 22-F19 | FIXED | IdentityEstimator.fit collects the label set with pd.unique for an object ndarray (no sort), so None next to strings no longer raises before the str-key fallback runs. Test tests/estimators/test_cluster22_estimator_contracts.py |

| 22-F21 | FIXED | MyDecorrelator.transform drops by the names recorded at fit when the DataFrame still has them all; position-based only when fit saw an ndarray. A reordered frame no longer loses the wrong columns. Test test_cluster22_estimator_contracts.py |

| 22-F22 | FIXED | get_best_dummy_score retries the scorer unweighted only when the TypeError mentions sample_weight; any other TypeError propagates. Test test_cluster22_estimator_contracts.py; existing estimator tests 334 passed |

| 22-F10 | DEFERRED | Tried the direct fix (walk every module that defines a callee name): the gate then reports 21 new paths on the real tree, 14 of them through the closure name _kernel that many modules define, so they are same-name collisions, not real calls. The gate resolves callees by bare name; a correct fix has to resolve them through the caller's imports (qualified names), which is a larger change to the scanner. Left as is so the CI gate stays meaningful |

| 13-F20 | FIXED | stability_n_jobs is now a BorutaShap constructor parameter (default -1, stored verbatim, documented), so get_params and clone keep it. Test tests/feature_selection/boruta_shap/test_cluster13_stability_n_jobs_param.py. Existing BorutaShap tests: 24 passed with no failure before my own time cap on the heavy ones; the rest left to CI |

| 17-F14 | FIXED | Optimizer failure now logs a WARNING with the exception type and traceback before the coordinate-scan fallback; a missing scipy logs INFO. Regression test added. |

| 17-F10 | FIXED | Fixed-[0,1] ECE/Brier kernels return NaN when a score lies outside [0,1] beyond 1e-9, matching fast_brier_score_loss and the fused block. Regression test added. Seven failures in test_metrics.py (fast_ice_only empty input, regression length_of numba typing) are in untouched code and not caused by this change. |

| 17-F12 | FIXED | Quantile binning kernel locates each pocket by binary search (O(log nbins)); NaN scores stay in pocket 0 as before. Regression test compares with the old linear scan, including ties and edge values. |

| 09-F27 | FIXED | Pairwise distances are built once per query chunk and class and reused for all bandwidths (_gaussian_kde_log_multi); results are bit-identical to the single-bandwidth calls. Test tests/feature_engineering/transformer/test_cluster09_kde_shared_distances.py. |

| 23-F6 | FIXED | Derived ratio keeps NaN for missing or zero denominators (inf mapped to NaN) so the finite mask and missing_correlation see it. Test tests/evaluation/test_cluster23_leak_scan_ratio_nan.py. |

| 23-F9 | FIXED | Docstring now says best_scheme is the smallest mean per-fold gap. |

| 23-F10 | DEFERRED | Docstring and code differ on the z-score reference set; choosing leave-one-out or fixing the text changes which variants are flagged, an owner call. |

| 23-F11 | RESOLVED | Not dead code: the next call takes jackknife as an argument, so the None initial value is used. Removing it raises UnboundLocalError (18 existing tests failed). Left as is. |

| 23-F13 | FIXED | Dropped the unused name parameter of _merge_parallel_chunks and its call argument. |

| 23-F14 | DOCUMENTED | Docstring now states the sign of cv_delta is ignored, so a loss on a shifted subgroup is flagged too; behaviour unchanged. |

| 15-F6 | FIXED | Stability selection counts failed estimator fits and importance computations and logs one WARNING when at least half failed (with an explicit note when none succeeded), regardless of verbose. Test tests/feature_selection/test_cluster15_stability_failed_fits_warn.py. |

| 18-F11 | FIXED | Streaming ensemble keeps the member shape for ndim>=2, so (N,K,C) members work as in the materialised path; uncertainty is the per-row mean of the std over all other axes; 0-d members raise a clear ValueError. Test tests/models/test_cluster18_streaming_ensemble_three_dim.py. |

| 21-F11 | FIXED | Split metrics take n_rows, n_pos and n_neg from the full split before subsampling, so the legend and the AUC noise bar use the true counts. Test tests/reporting/test_cluster21_split_comparison_true_counts.py. |

| 21-F7 | FIXED | Risk-coverage logs a WARNING naming the labels outside {0,1} that it drops, and the empty-figure annotation mentions the 0/1 requirement. Test tests/reporting/test_cluster21_risk_coverage_labels_and_shap_proxy_warn.py. |

| 21-F15 | FIXED | A failing decision_function/predict_proba/predict in the SHAP score proxy now logs a throttled WARNING stating that tail preservation is lost. Same test file. |

| 21-F10 | FIXED | Low-evidence indices are mapped through the downsample selection (remap_low_evidence_indices) in both the matplotlib and plotly scatter, so weak points stay hollow after capping. Test tests/reporting/test_cluster21_low_evidence_survives_downsample.py. |

| 21-F12 | FIXED | A static image replaced by an HTML file (burned kaleido or failing oneshot) or lost entirely is now counted in get_render_failure_stats(); the HTML file is still written and the figure handle kept. Test tests/reporting/test_cluster21_kaleido_html_fallback_is_counted.py. |

| 20-F13 | FIXED | Model-card heteroscedasticity Spearman uses the shared average-rank implementation, so tied predictions no longer make it depend on row order; local rank helper import removed. Test tests/reporting/test_cluster20_model_card_spearman_ties.py. |

| 20-F7 | FIXED | Non-finite values get a split-gain bin code of their own above every finite bin, so missing rows no longer dilute the top finite bin and a missingness-driven column is ranked. Test tests/reporting/test_cluster20_split_gain_missing_bin.py. |

| 20-F8 | PARTIAL | Multilabel legend ECE now uses each bin's mean prediction as the confidence (per label and pooled), and both axes say bin centre instead of claiming a mean prediction. The curves stay on the shared bin-centre grid because a line panel has one x for all series; per-series x would be a spec change. Test tests/reporting/test_cluster20_multilabel_ece_uses_mean_prediction.py. |

| 20-F12 | FIXED | bootstrap_ap_ci returns the distinct-threshold AP (sklearn definition, same as the PR panel) and shifts the interval by the gap to the per-row step sum, so the interval is centred on the AP the title shows under tied scores. Test tests/reporting/test_cluster20_ap_ci_matches_displayed_ap.py. |

| 20-F15 | DEFERRED | Re-read: the memo is keyed on frame identity with a weakref guard, so an in-place edit between two builder calls does return the old matrix. A sound fix is either passing the densified matrix explicitly from the dispatch to both builders (an API change across diagnostics_dispatch) or a content fingerprint, and a sampled fingerprint cannot detect every edit while a full one costs as much as the densify. Needs an owner choice between the two. |

| 24-F24 | FIXED | find_best_mps_sequence docstring now states profits has length n (sized to raw_prices) and that callers use profits[:-1] for per-interval values. |

| 24-F29 | FIXED | Windowed targets use at least float64 (np.promote_types(dtype, float64)); features keep the requested dtype. Test tests/feature_engineering/test_cluster24_dtype_docs_and_nan_pairs.py. |

| 24-F27 | FIXED | variance_gated_pairwise_diff logs a WARNING naming the columns whose NaN/inf values drop every pair involving them. Same test file. |

| 22-F24 | FIXED | cluster_resample_indices builds the resample with one np.repeat gather instead of a Python slice per drawn group; output and rng consumption are identical (test compares with the old form over 20 seeds). Test tests/evaluation/test_cluster22_cluster_resample_vectorised.py. |

| 22-F16 | RESOLVED | Not a defect: in the inner_cv path the full-OOF fit (cal_oof) is still used, for the non-finite check and as the returned calibrated_probs, so it cannot move into the same_oof branch. No change. |

| 41-F41 | FIXED | The pairwise redundancy scan selects the upper-triangle pairs with np.triu_indices and one mask; output order and values are identical to the nested loop (test compares both, including a constant column). Test tests/training/test_cluster41_pairwise_redundancy_vectorised.py. |

| 41-F40 | FIXED | The sample probe now takes the first non-null element via dropna().iloc[0]. Re-reading the loop found a worse defect: the branch tested 'dt is object', which is never true for a pandas dtype instance, so no object column was categorised or checked for nested embeddings. It now uses dt == object. Test tests/training/test_cluster41_ranker_object_columns_swept.py. |

| 41-F42 | PARTIAL | Removed the duplicate logger and _NOT_AGGREGATABLE definitions, the doubled @dataclass on TemporalAuditResult, the unused importlib.util import and the unused second _SLUG_PATTERN in strategies/base.py. Not done: the truncated comment at the regression __init__ and the shared MTRESID helper (two call sites with different inputs, a small refactor left for the regression-report owner). Test tests/training/test_cluster41_duplicate_definitions_removed.py. |

| 14-F4 | FIXED | fast_model now reads iterations, depth and rsm as well as n_estimators, max_depth and colsample_bytree, so a catboost template gets the quarter budget, depth 3 and column subsampling (helper _fast_model_params, keeps the cap as a minimum). A catboost template that does not set iterations explicitly still keeps its default budget because get_params omits defaults. Test tests/feature_selection/shap_proxied/test_cluster14_fast_model_catboost_params.py. |

| 18-F13 | PARTIAL | A failed per-member auto-tune now logs a WARNING with the error type and message (was DEBUG), and the docstring at member_metrics names the real module path. ensure_per_member_tuning stays: it has no production caller but is an uncalled-function entry on the project's kept list and a manual calibration entry point, so wiring it in or retiring it is an owner call. Test tests/models/test_cluster18_per_member_tuning_failure_visible.py. |

| 18-F14 | DEFERRED | Re-read: the _all_members_catastrophic sentinel has no reader and the K>2 path still builds a one-member ensemble, while the K=2 path returns early with a _reason. Making the K>2 gate behave like K=2 changes which models ship as ensembles for a whole class of runs (single survivor vs best single member vs dummy baseline), so it needs the owner to pick the policy; the warning already tells the operator in the meantime. |



| 39-F28 | DEFERRED | Re-read: the validator assigns the derived values onto the public memory.budget_gb, cache.ram_max_gb and cache.ram_reserve_gb fields, and the test suite and cache.py read them there. Moving them to private resolved fields changes what every reader and every dumped config sees, so it needs the readers migrated together with a way to dump the user's original (None) values; a partial change would leave the dump/re-validate case failing as before. |

| 28-F64 | DEFERRED | Re-read: the write is the mechanism that hands the first model's stats to every later model through the shared common_params, and process_model also returns them. Removing the write is safe only once every later call site is shown to pass the returned value explicitly; no wrong result was observed, so the audited-reader pass is left to the training-loop owner. |


| 28-F76 | PARTIAL | Tests now cover _oof_train_timestamps, retry_params_after_empty_text_dictionary, _open_verified_bundle, _realign_val_columns and _handle_fit_failure (KeyboardInterrupt text, GPU callbacks, CUDA tree learner, unsupported dtype re-raise, unexpected validation keyword, unrecognised error). Still without a test reference: _run_val_split_metrics and _run_test_split_metrics, which need a fixture trainer state. Tests in tests/training/test_cluster28_*.py. |
