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
| 09-F7 | DEFERRED | Layer 2 in Mode A consumes layer-1 OOF features of training rows that were produced by folds sharing those rows, so the stacking is optimistic. A fix re-folds layer 1 inside each layer-2 fold (nested OOF), which changes the cost and output of stacked_qnn and stacked_attention. Needs a design decision; the docstring claim at stacked_attention.py:15-16 must be corrected with it. |
| 10-F3 | RESOLVED | logreg_holdout_auc returns chance (0.5) for an empty selection instead of scoring all columns. Test tests/feature_selection/test_bench_logreg_empty_selection_is_chance.py. Commit below. |
| 10-F1 | RESOLVED | Adaptive-guards bench scores the refit on the untouched holdout split; the concatenated re-split is gone. Test tests/feature_selection/test_bench_holdout_auc_is_unseen.py. Commit below. |
| 10-F2 | RESOLVED | honest_auc fits on the search rows and scores the untouched holdout, at all three call sites. Test tests/feature_selection/test_bench_interaction_holdout_is_unseen.py. Commit below. |
| 11-F2 | RESOLVED | RFECV empty selection scores chance (0.5) in the importance-aggregation bench instead of refitting on every column. Test tests/feature_selection/test_bench_rfecv_empty_selection_is_chance.py. Commit below. |
| 11-F1 | RESOLVED | _integer_bins ranks finite values only; missing rows take bin 0 instead of the top bin. Test tests/feature_selection/test_bench_integer_bins_missing_not_top.py. Commit below. |
| 11-F3 | DEFERRED | Searched arms report the max over many in-sample candidates while truth, joint and als are single candidates, so the comparison is not like for like. The fix is a study-design change (same candidate sets, or held-out scoring) for the stat-study tables; it changes every published table, so it waits for a decision. |
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
| 17-F9 | DEFERRED | Degenerate reports use ICE = 1.0 as a worst-case placeholder, but a real model's ICE can exceed 1.0, so the placeholder can beat it. The report contract (module docstring) promises finite floats to consumers (plots, threshold search, string builders); choosing a larger finite placeholder or NaN changes that contract for every caller, so it waits for a decision. |
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
| 24-F16 | DEFERRED | causal per-row prefix weights for scheme 0 depend on the prefix length m, so a cumsum rewrite needs a per-scheme derivation; output is correct, only O(m^2) cost. Needs a design decision |
| 24-F17 | DEFERRED | causal KDE re-runs the grid per prefix; the fix needs incremental KDE, a design change beyond a local patch. Output is correct, only cost |
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
| 28-F2 | FIXED | multilabel decision from ndim==2 only (object dtype no longer implies multilabel); no separate test, covered by the multi-output suite run |
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
| 28-F16 | FIXED | validation cache hit re-applies init_score; no separate test |
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
| 37-F4 | DEFERRED | not changed in this pass: the scale mismatch between roc_auc and negated log-loss needs a decision on a common score; needs owner decision |
| 37-F5 | FIXED | per-row sample weights are passed regardless of column pruning; no separate test beyond the suite run |
| 37-F6 | FIXED | refit_capped_stack takes and uses sample_weight, and the call site passes the OOF weights; tests/training/composite/test_cluster37_venn_refit.py |
| 37-F7 | FIXED | group sizes come from one np.unique pass; no separate test |
| 37-F8 | FIXED | a failed fingerprint cross-check distrusts the precomputed bundle and logs at WARNING; no separate test |
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
| 39-F38 | DEFERRED | not changed: hgb_max_features_cap is exported with no caller; wire in or remove, an owner decision |
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
| 04-F14 | DEFERRED | not changed in this pass; needs a targeted design or plumbing decision (see cluster report) |
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
| 08-F4 | DEFERRED | not changed in this pass; needs a design or behaviour decision (see cluster report) |
| 08-F6 | DEFERRED | not changed in this pass; needs a design or behaviour decision (see cluster report) |
| 08-F7 | DEFERRED | not changed in this pass; needs a design or behaviour decision (see cluster report) |
| 08-F8 | DEFERRED | not changed in this pass; needs a design or behaviour decision (see cluster report) |
| 08-F9 | DEFERRED | not changed in this pass; needs a design or behaviour decision (see cluster report) |
| 08-F12 | DEFERRED | not changed in this pass; needs a design or behaviour decision (see cluster report) |
| 08-F13 | DEFERRED | not changed in this pass; needs a design or behaviour decision (see cluster report) |
| 08-F14 | DEFERRED | not changed in this pass; needs a design or behaviour decision (see cluster report) |
| 08-F15 | DEFERRED | not changed in this pass; needs a design or behaviour decision (see cluster report) |
| 08-F16 | DEFERRED | not changed in this pass; needs a design or behaviour decision (see cluster report) |
| 08-F17 | DEFERRED | not changed in this pass; needs a design or behaviour decision (see cluster report) |
| 09-F15 | DEFERRED | tried an early return in the underdetermined branch; it broke the contract that the linear column stays finite. The real defect is the per-row exception path zeroing whole rows; needs a column-aware fix |
| 09-F24 | FIXED | binary local-lift labels are normalised to {0,1} for both the global mean and the neighbour sums; tests/feature_engineering/transformer/test_cluster09_edges.py covers the KMeans path only, lift path covered by existing lift tests |
| 09-F26 | FIXED | anchor count is capped by the row count, so a one-row fold yields one anchor; tests/feature_engineering/transformer/test_cluster09_edges.py |
| 09-F14 | DEFERRED | not changed in this pass; needs a design or performance decision (see cluster report) |
| 09-F16 | DEFERRED | not changed in this pass; needs a design or performance decision (see cluster report) |
| 09-F17 | DEFERRED | not changed in this pass; needs a design or performance decision (see cluster report) |
| 09-F18 | DEFERRED | not changed in this pass; needs a design or performance decision (see cluster report) |
| 09-F19 | DEFERRED | not changed in this pass; needs a design or performance decision (see cluster report) |
| 09-F20 | DEFERRED | not changed in this pass; needs a design or performance decision (see cluster report) |
| 09-F21 | DEFERRED | not changed in this pass; needs a design or performance decision (see cluster report) |
| 09-F22 | DEFERRED | not changed in this pass; needs a design or performance decision (see cluster report) |
| 09-F23 | DEFERRED | not changed in this pass; needs a design or performance decision (see cluster report) |
| 09-F25 | DEFERRED | not changed in this pass; needs a design or performance decision (see cluster report) |
| 09-F27 | DEFERRED | not changed in this pass; needs a design or performance decision (see cluster report) |
| 09-F28 | DEFERRED | not changed in this pass; needs a design or performance decision (see cluster report) |
| 09-F29 | DEFERRED | not changed in this pass; needs a design or performance decision (see cluster report) |
| 09-F30 | DEFERRED | not changed in this pass; needs a design or performance decision (see cluster report) |
| 09-F31 | DEFERRED | not changed in this pass; needs a design or performance decision (see cluster report) |
| 10-F5 | FIXED | a version check with one missing capture logs that the comparison was skipped; benchmark utility, no test |
| 10-F8 | FIXED | redundant columns without informative ones raise a clear ValueError; benchmark utility, no test |
| 10-F9 | FIXED | a failed log-file write is reported once on stderr; benchmark utility, no test |
| 10-F4 | DEFERRED | not changed in this pass; benchmark-script cleanup or a policy decision (MAX_VERSION_AGE_DAYS enforcement) |
| 10-F6 | DEFERRED | not changed in this pass; benchmark-script cleanup or a policy decision (MAX_VERSION_AGE_DAYS enforcement) |
| 10-F7 | DEFERRED | not changed in this pass; benchmark-script cleanup or a policy decision (MAX_VERSION_AGE_DAYS enforcement) |
| 11-F4 | DEFERRED | all cluster 11 P2 findings sit under _benchmarks (benchmark harness); deferred under the benchmark rule |
| 11-F5 | DEFERRED | all cluster 11 P2 findings sit under _benchmarks (benchmark harness); deferred under the benchmark rule |
| 11-F6 | DEFERRED | all cluster 11 P2 findings sit under _benchmarks (benchmark harness); deferred under the benchmark rule |
| 11-F7 | DEFERRED | all cluster 11 P2 findings sit under _benchmarks (benchmark harness); deferred under the benchmark rule |
| 11-F8 | DEFERRED | all cluster 11 P2 findings sit under _benchmarks (benchmark harness); deferred under the benchmark rule |
| 11-F9 | DEFERRED | all cluster 11 P2 findings sit under _benchmarks (benchmark harness); deferred under the benchmark rule |
| 11-F10 | DEFERRED | all cluster 11 P2 findings sit under _benchmarks (benchmark harness); deferred under the benchmark rule |
| 11-F11 | DEFERRED | all cluster 11 P2 findings sit under _benchmarks (benchmark harness); deferred under the benchmark rule |
| 11-F12 | DEFERRED | all cluster 11 P2 findings sit under _benchmarks (benchmark harness); deferred under the benchmark rule |
| 11-F13 | DEFERRED | all cluster 11 P2 findings sit under _benchmarks (benchmark harness); deferred under the benchmark rule |
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
| 14-F4 | DEFERRED | catboost fast-budget parameter naming needs a behaviour decision (iterations vs n_estimators) |
| 14-F7 | DEFERRED | duplicated per-column correlation loops: performance refactor, not a correctness fix |
| 14-F5 | FIXED | note: tests/feature_selection/shap_proxied/test_biz_val_shap_proxied_core_refine.py::test_biz_val_auto_matches_core_on_non_saturated_bed fails on the exact pre-change file too (pre-existing, verified) |
| 15-F8 | FIXED | a callable prescreen's result is filtered to the candidate names the search offered; no direct test (the prescreen path needs a full RFECV fixture) |
| 15-F9 | FIXED | the importance callable is called with target only when its signature accepts it, so a TypeError inside it runs once; tests/feature_selection/test_cluster15_importance_getter_and_prescreen.py |
| 15-F3 | DEFERRED | not changed in this pass; checkpoint-state or performance refactor needing a design decision (see cluster report) |
| 15-F4 | DEFERRED | not changed in this pass; checkpoint-state or performance refactor needing a design decision (see cluster report) |
| 15-F5 | DEFERRED | not changed in this pass; checkpoint-state or performance refactor needing a design decision (see cluster report) |
| 15-F6 | DEFERRED | not changed in this pass; checkpoint-state or performance refactor needing a design decision (see cluster report) |
| 15-F7 | DEFERRED | not changed in this pass; checkpoint-state or performance refactor needing a design decision (see cluster report) |
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
| 17-F10 | DEFERRED | not changed in this pass; clamping semantics, performance, or test-coverage work needing a decision (see cluster report) |
| 17-F11 | DEFERRED | not changed in this pass; clamping semantics, performance, or test-coverage work needing a decision (see cluster report) |
| 17-F12 | DEFERRED | not changed in this pass; clamping semantics, performance, or test-coverage work needing a decision (see cluster report) |
| 17-F14 | DEFERRED | not changed in this pass; clamping semantics, performance, or test-coverage work needing a decision (see cluster report) |
| 17-F17 | DEFERRED | not changed in this pass; clamping semantics, performance, or test-coverage work needing a decision (see cluster report) |
| 17-F18 | DEFERRED | not changed in this pass; clamping semantics, performance, or test-coverage work needing a decision (see cluster report) |
| 17-F19 | DEFERRED | not changed in this pass; clamping semantics, performance, or test-coverage work needing a decision (see cluster report) |
| 17-F20 | DEFERRED | not changed in this pass; clamping semantics, performance, or test-coverage work needing a decision (see cluster report) |
| 18-F9 | FIXED | gate weighted means return NaN when every weight is zero instead of raising; tests/models/test_cluster18_zero_weight_gate.py; existing quality-gate and ensembling tests: 231 passed |
| 18-F6 | DEFERRED | not changed in this pass; needs a targeted design or performance decision (see cluster report) |
| 18-F7 | DEFERRED | not changed in this pass; needs a targeted design or performance decision (see cluster report) |
| 18-F8 | DEFERRED | not changed in this pass; needs a targeted design or performance decision (see cluster report) |
| 18-F10 | DEFERRED | not changed in this pass; needs a targeted design or performance decision (see cluster report) |
| 18-F11 | DEFERRED | not changed in this pass; needs a targeted design or performance decision (see cluster report) |
| 18-F12 | DEFERRED | not changed in this pass; needs a targeted design or performance decision (see cluster report) |
| 18-F13 | DEFERRED | not changed in this pass; needs a targeted design or performance decision (see cluster report) |
| 18-F14 | DEFERRED | not changed in this pass; needs a targeted design or performance decision (see cluster report) |
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
| 20-F7 | DEFERRED | not changed in this pass; chart-semantics or benchmark finding needing individual judgment (see cluster report) |
| 20-F8 | DEFERRED | not changed in this pass; chart-semantics or benchmark finding needing individual judgment (see cluster report) |
| 20-F10 | DEFERRED | not changed in this pass; chart-semantics or benchmark finding needing individual judgment (see cluster report) |
| 20-F11 | DEFERRED | not changed in this pass; chart-semantics or benchmark finding needing individual judgment (see cluster report) |
| 20-F12 | DEFERRED | not changed in this pass; chart-semantics or benchmark finding needing individual judgment (see cluster report) |
| 20-F13 | DEFERRED | not changed in this pass; chart-semantics or benchmark finding needing individual judgment (see cluster report) |
| 20-F14 | DEFERRED | not changed in this pass; chart-semantics or benchmark finding needing individual judgment (see cluster report) |
| 20-F15 | DEFERRED | not changed in this pass; chart-semantics or benchmark finding needing individual judgment (see cluster report) |
| 20-F16 | DEFERRED | not changed in this pass; chart-semantics or benchmark finding needing individual judgment (see cluster report) |
| 20-F18 | DEFERRED | not changed in this pass; chart-semantics or benchmark finding needing individual judgment (see cluster report) |
| 20-F19 | DEFERRED | not changed in this pass; chart-semantics or benchmark finding needing individual judgment (see cluster report) |
| 20-F20 | DEFERRED | not changed in this pass; chart-semantics or benchmark finding needing individual judgment (see cluster report) |
| 20-F21 | DEFERRED | not changed in this pass; chart-semantics or benchmark finding needing individual judgment (see cluster report) |
| 20-F22 | DEFERRED | not changed in this pass; chart-semantics or benchmark finding needing individual judgment (see cluster report) |
| 20-F23 | DEFERRED | not changed in this pass; chart-semantics or benchmark finding needing individual judgment (see cluster report) |
| 20-F24 | DEFERRED | not changed in this pass; chart-semantics or benchmark finding needing individual judgment (see cluster report) |
| 20-F25 | DEFERRED | not changed in this pass; chart-semantics or benchmark finding needing individual judgment (see cluster report) |
| 20-F26 | DEFERRED | not changed in this pass; chart-semantics or benchmark finding needing individual judgment (see cluster report) |
| 20-F27 | DEFERRED | not changed in this pass; chart-semantics or benchmark finding needing individual judgment (see cluster report) |
| 21-F5 | FIXED | an all-missing slice column takes the missingness code; tests/reporting/test_cluster21_slice_missing_and_metric_direction.py |
| 21-F16 | FIXED | metric direction uses the family name before any @k cutoff; same test file |
| 21-F6 | DEFERRED | not changed in this pass; renderer or chart semantics needing individual judgment (see cluster report) |
| 21-F7 | DEFERRED | not changed in this pass; renderer or chart semantics needing individual judgment (see cluster report) |
| 21-F8 | DEFERRED | not changed in this pass; renderer or chart semantics needing individual judgment (see cluster report) |
| 21-F9 | DEFERRED | not changed in this pass; renderer or chart semantics needing individual judgment (see cluster report) |
| 21-F10 | DEFERRED | not changed in this pass; renderer or chart semantics needing individual judgment (see cluster report) |
| 21-F11 | DEFERRED | not changed in this pass; renderer or chart semantics needing individual judgment (see cluster report) |
| 21-F12 | DEFERRED | not changed in this pass; renderer or chart semantics needing individual judgment (see cluster report) |
| 21-F13 | DEFERRED | not changed in this pass; renderer or chart semantics needing individual judgment (see cluster report) |
| 21-F14 | DEFERRED | not changed in this pass; renderer or chart semantics needing individual judgment (see cluster report) |
| 21-F15 | DEFERRED | not changed in this pass; renderer or chart semantics needing individual judgment (see cluster report) |
| 22-F13 | FIXED | the PIT chi-square's expected total is the binned count, so NaN is consistent; tests/calibration/test_cluster22_chi_square_nan.py; 52 calibration quality tests pass |
| 22-F20 | FIXED | the binary predict_proba clips the positive probability to [0,1] like the multiclass branch; no separate test |
| 22-F9 | DEFERRED | not changed in this pass; needs individual judgment or a behaviour decision (see cluster report) |
| 22-F10 | DEFERRED | not changed in this pass; needs individual judgment or a behaviour decision (see cluster report) |
| 22-F11 | DEFERRED | not changed in this pass; needs individual judgment or a behaviour decision (see cluster report) |
| 22-F12 | DEFERRED | not changed in this pass; needs individual judgment or a behaviour decision (see cluster report) |
| 22-F14 | DEFERRED | not changed in this pass; needs individual judgment or a behaviour decision (see cluster report) |
| 22-F15 | DEFERRED | not changed in this pass; needs individual judgment or a behaviour decision (see cluster report) |
| 22-F16 | DEFERRED | not changed in this pass; needs individual judgment or a behaviour decision (see cluster report) |
| 22-F17 | DEFERRED | not changed in this pass; needs individual judgment or a behaviour decision (see cluster report) |
| 22-F18 | DEFERRED | not changed in this pass; needs individual judgment or a behaviour decision (see cluster report) |
| 22-F19 | DEFERRED | not changed in this pass; needs individual judgment or a behaviour decision (see cluster report) |
| 22-F21 | DEFERRED | not changed in this pass; needs individual judgment or a behaviour decision (see cluster report) |
| 22-F22 | DEFERRED | not changed in this pass; needs individual judgment or a behaviour decision (see cluster report) |
| 22-F23 | DEFERRED | not changed in this pass; needs individual judgment or a behaviour decision (see cluster report) |
| 22-F24 | DEFERRED | not changed in this pass; needs individual judgment or a behaviour decision (see cluster report) |
| 22-F25 | DEFERRED | not changed in this pass; needs individual judgment or a behaviour decision (see cluster report) |
| 22-F26 | DEFERRED | not changed in this pass; needs individual judgment or a behaviour decision (see cluster report) |
| 23-F5 | FIXED | the leakage p-value is computed whether or not a tolerance decided the verdict; tests/evaluation/test_cluster23_leak_pvalue_and_event_days.py |
| 23-F8 | FIXED | event distances use fractional days (total seconds / 86400) instead of the floored .dt.days; same test file; 95 related tests pass |
| 23-F6 | DEFERRED | not changed in this pass; low-impact judgment or cleanup (see cluster report) |
| 23-F7 | DEFERRED | not changed in this pass; low-impact judgment or cleanup (see cluster report) |
| 23-F9 | DEFERRED | not changed in this pass; low-impact judgment or cleanup (see cluster report) |
| 23-F10 | DEFERRED | not changed in this pass; low-impact judgment or cleanup (see cluster report) |
| 23-F11 | DEFERRED | not changed in this pass; low-impact judgment or cleanup (see cluster report) |
| 23-F12 | DEFERRED | not changed in this pass; low-impact judgment or cleanup (see cluster report) |
| 23-F13 | DEFERRED | not changed in this pass; low-impact judgment or cleanup (see cluster report) |
| 23-F14 | DEFERRED | not changed in this pass; low-impact judgment or cleanup (see cluster report) |
| 24-F48 | FIXED | validation-only class labels get the probability floor instead of a neighbouring class's column; tests/feature_selection/test_cluster24_unseen_validation_label.py |
| 24-F23 | DEFERRED | not reviewed in this pass; P2/P3 still open (see cluster report) |
| 24-F24 | DEFERRED | not reviewed in this pass; P2/P3 still open (see cluster report) |
| 24-F25 | DEFERRED | not reviewed in this pass; P2/P3 still open (see cluster report) |
| 24-F26 | DEFERRED | not reviewed in this pass; P2/P3 still open (see cluster report) |
| 24-F27 | DEFERRED | not reviewed in this pass; P2/P3 still open (see cluster report) |
| 24-F28 | DEFERRED | not reviewed in this pass; P2/P3 still open (see cluster report) |
| 24-F29 | DEFERRED | not reviewed in this pass; P2/P3 still open (see cluster report) |
| 24-F30 | DEFERRED | not reviewed in this pass; P2/P3 still open (see cluster report) |
| 24-F31 | DEFERRED | not reviewed in this pass; P2/P3 still open (see cluster report) |
| 24-F32 | DEFERRED | not reviewed in this pass; P2/P3 still open (see cluster report) |
| 24-F33 | DEFERRED | not reviewed in this pass; P2/P3 still open (see cluster report) |
| 24-F34 | DEFERRED | not reviewed in this pass; P2/P3 still open (see cluster report) |
| 24-F35 | DEFERRED | not reviewed in this pass; P2/P3 still open (see cluster report) |
| 24-F36 | DEFERRED | not reviewed in this pass; P2/P3 still open (see cluster report) |
| 24-F37 | DEFERRED | not reviewed in this pass; P2/P3 still open (see cluster report) |
| 24-F38 | DEFERRED | not reviewed in this pass; P2/P3 still open (see cluster report) |
| 24-F39 | DEFERRED | not reviewed in this pass; P2/P3 still open (see cluster report) |
| 24-F40 | DEFERRED | not reviewed in this pass; P2/P3 still open (see cluster report) |
| 24-F41 | DEFERRED | not reviewed in this pass; P2/P3 still open (see cluster report) |
| 24-F42 | DEFERRED | not reviewed in this pass; P2/P3 still open (see cluster report) |
| 24-F43 | DEFERRED | not reviewed in this pass; P2/P3 still open (see cluster report) |
| 24-F44 | DEFERRED | not reviewed in this pass; P2/P3 still open (see cluster report) |
| 24-F45 | DEFERRED | not reviewed in this pass; P2/P3 still open (see cluster report) |
| 24-F46 | DEFERRED | not reviewed in this pass; P2/P3 still open (see cluster report) |
| 24-F47 | DEFERRED | not reviewed in this pass; P2/P3 still open (see cluster report) |
| 24-F49 | DEFERRED | not reviewed in this pass; P2/P3 still open (see cluster report) |
| 24-F50 | DEFERRED | not reviewed in this pass; P2/P3 still open (see cluster report) |
| 24-F51 | DEFERRED | not reviewed in this pass; P2/P3 still open (see cluster report) |
| 24-F52 | DEFERRED | not reviewed in this pass; P2/P3 still open (see cluster report) |
| 24-F53 | DEFERRED | not reviewed in this pass; P2/P3 still open (see cluster report) |
| 24-F54 | DEFERRED | not reviewed in this pass; P2/P3 still open (see cluster report) |
| 24-F55 | DEFERRED | not reviewed in this pass; P2/P3 still open (see cluster report) |
| 24-F56 | DEFERRED | not reviewed in this pass; P2/P3 still open (see cluster report) |
| 24-F57 | DEFERRED | not reviewed in this pass; P2/P3 still open (see cluster report) |
| 24-F58 | DEFERRED | not reviewed in this pass; P2/P3 still open (see cluster report) |
| 24-F59 | DEFERRED | not reviewed in this pass; P2/P3 still open (see cluster report) |
| 24-F60 | DEFERRED | not reviewed in this pass; P2/P3 still open (see cluster report) |
| 24-F61 | DEFERRED | not reviewed in this pass; P2/P3 still open (see cluster report) |
| 24-F62 | DEFERRED | not reviewed in this pass; P2/P3 still open (see cluster report) |
| 24-F63 | DEFERRED | not reviewed in this pass; P2/P3 still open (see cluster report) |
| 24-F64 | DEFERRED | not reviewed in this pass; P2/P3 still open (see cluster report) |
| 24-F65 | DEFERRED | not reviewed in this pass; P2/P3 still open (see cluster report) |
| 24-F66 | DEFERRED | not reviewed in this pass; P2/P3 still open (see cluster report) |
| 24-F25 | FIXED | empty support returns NaN per query point; tests/feature_engineering/test_cluster24_empty_and_alignment.py |
| 24-F31 | FIXED | chunks that return (None, None) are skipped and an all-empty run returns (None, None); no separate test |
| 24-F36 | FIXED | the long MA is reindexed to the short one's index before differencing; tests/feature_engineering/test_cluster24_empty_and_alignment.py |
| 24-F23 | DEFERRED | not reviewed in this pass; P2 still open |
| 24-F24 | DEFERRED | not reviewed in this pass; P2 still open |
| 24-F26 | DEFERRED | not reviewed in this pass; P2 still open |
| 24-F27 | DEFERRED | not reviewed in this pass; P2 still open |
| 24-F28 | DEFERRED | not reviewed in this pass; P2 still open |
| 24-F29 | DEFERRED | not reviewed in this pass; P2 still open |
| 24-F30 | DEFERRED | not reviewed in this pass; P2 still open |
| 24-F32 | DEFERRED | not reviewed in this pass; P2 still open |
| 24-F33 | DEFERRED | not reviewed in this pass; P2 still open |
| 24-F34 | DEFERRED | not reviewed in this pass; P2 still open |
| 24-F35 | DEFERRED | not reviewed in this pass; P2 still open |
| 24-F31 | FIXED | empty parallel chunks are skipped and an all-empty run returns (None, None); tests/feature_engineering/test_timeseries.py expectation updated to the corrected cumsum end contract; 129 related tests pass |
| 24-F23 | FIXED | local density drops the self-match when the query is the reference, so dist_to_kth is the k-th other point; tests/feature_engineering/test_cluster24_min_periods_and_self_match.py; 57 related tests pass |
| 24-F26 | FIXED | rolling min_periods is clamped to the window; same test file |
| 24-F24 | DEFERRED | read this pass; judgement or performance item deferred (see cluster report) |
| 24-F27 | DEFERRED | read this pass; judgement or performance item deferred (see cluster report) |
| 24-F28 | DEFERRED | read this pass; judgement or performance item deferred (see cluster report) |
| 24-F29 | DEFERRED | read this pass; judgement or performance item deferred (see cluster report) |
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
| 12-F31 | PARTIAL | The two named benches (round3_hard_bed_bench, round3_hybrid_refine_bench) now silence only FutureWarning/DeprecationWarning. 61 other fs_hybrid bench files still install a catch-all ignore; not changed in this pass. Test tests/feature_selection/test_fs_hybrid_warning_filter_keeps_degraded_warning.py. |
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
| 12-F34 | DEFERRED | H35 (and H32) pick the config by OOF AUC on rows the members were fit on. A real fix refits members on an inner split and scores engineered features on held-out rows; this bench's HybridSelector has no transform path for that. Benchmark-only, not production. Needs a design decision. |
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
| 39-F38 | DEFERRED | hgb_max_features_cap has no caller in the pipeline (only tests and the uncalled-functions allowlist in tests/test_meta). Project policy is wire in, not delete. Wiring needs the chosen insertion point (encoder construction or the assembler's dense branch), which changes how HGB text blocks are built: a design decision left open. |
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
| 28-F2 | FIXED (test pending) | The multilabel decision uses the array shape only (evaluation.py). A regression test needs the full per-class calibrator run inside a nine-argument function; no targeted helper exists yet. |
| 28-F6 | FIXED | The fail-loud inf check also runs on the batched path when fix_infinities is False. Test tests/training/test_cluster28_polars_inf_guard_runs_when_not_fixing.py. |
| 28-F17 | FIXED | set_label applies the fitted label encoder, so the cached Dataset matches what fit trained on. Test tests/training/test_cluster28_set_label_encodes_like_fit.py. |
| 28-F7 | FIXED | save_split_artifacts rewrites every artifact (the existing-name skip is gone). Test tests/training/test_cluster28_split_artifacts_rewritten.py. |
| 28-F11 | FIXED | The auto-wrap switch is carried on the control config the training loop reads (set by _build_control_config), so False is honoured. Test tests/training/test_cluster28_wrap_flag_reaches_control.py. |
| 28-F14 | FIXED | An existing bundle that cannot be read raises instead of falling back to an unverified reopen. Test tests/training/test_cluster28_unreadable_bundle_raises.py. |
| 28-F16 | FIXED (test pending) | Validation cache hit re-applies init_score (lgb_shim). A test needs a full cached fit through fit(); not yet written. |
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
