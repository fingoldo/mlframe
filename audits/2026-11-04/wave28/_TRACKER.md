# Wave 28 audit 2026-11-04 -- tracker

Three read-only reports by Opus 5.5 low-effort agents: the neural training package (about 6 of 60 modules read), the dynamic cluster discovery / recipe / composite-transform modules, and the transformer modules from `l` to `z` (about 40 findings, none reproduced by the agent). The agents could not write files, so the reports are their returned text, condensed; each finding was re-checked on HEAD before it was fixed. Dispositions: RESOLVED, FUTURE, DOC, REJECTED.

| Disposition | ID | Class | Severity | Location |
|---|---|---|---|---|
| **RESOLVED** | W28NN-1 | loss | Medium | the flat module's weighted multilabel loss summed B x K entries and divided by the sum of the B weights, so it was K times the unweighted mean; it now averages over the K labels |
| **RESOLVED** | W28NN-2 | precision | Medium | the recurrent module's zero-weight guard `1e-12` rounds to 0 in float16, so an all-zero-weight batch gave 0 / 0 = nan; the weighted mean is now taken in float32 (`_weighted_mean`) |
| **FUTURE** | W28RC-1 | replay | High | `_orth_basis_recipes.py:141-154,607-643`: NaN source values are filled with the mean of the batch being transformed instead of a frozen fit-time value; reason: the fit-time fill and the `tanh` std must be stored in the recipe and captured at about ten generator call sites that build `preprocess_params` (`_orthogonal_adaptive_arity_fe`, `_orthogonal_adaptive_degree_fe`, `_orthogonal_bootstrap_mi_fe`, ...); replay reads them when present and keeps the batch statistic for old pickles. One change for W28RC-1 and W28RC-2 |
| **FUTURE** | W28RC-2 | replay | High | `_orth_basis_recipes.py:82-84`: the `tanh` pre-transform std is recomputed from the replay batch and never stored in the recipe |
| **RESOLVED** | W28GP-1 | precision | Medium | `_pairs_abs_corr_gpu.py:28,34-39`: the target is reduced in float32 on the device, so a target with a large offset collapses to constant and the GPU returns 0.0 where the CPU twin returns the true |r; the target, the mask and the row count are float64 on the device (GPU returned 0.0 where the true |r| is 0.96) |
| **RESOLVED** | W28GP-2 | precision | Medium | `_pairs_abs_corr_gpu.py:84-94`: both operands are cast to float32 before centring, with the same collapse at large offsets; both operands are float64 on the device (resident key renamed `zerofill_operand_f64` so a cached float32 copy is not reused) |
| **RESOLVED** | W28DC-1 | cache | Medium | `_dcd_metrics.py:37-39`: `_is_foreign` treats any same-shape array sharing memory with the state matrix as the state's own, so another column view reuses the cached SU/VI/entropy; `_same_view` compares shape, strides and the first-element address, so another block or a reversed view of the same buffer is scored on its own data |
| **RESOLVED** | W28CT-1 | scale | Low | `composite/transforms/simple.py:321,331,452,468`: the eps floor `sign(base + 1e-300) * eps` is 0 when `base == -1e-300`; one `_floor_to_eps` helper keeps the sign of the divisor (+eps at exactly 0) at all four sites |
| **FUTURE** | W28CT-2 | ties | Low | `composite/transforms/unary.py:531-535`: tied knots map every tied y to the last duplicate position instead of the middle of the tie block |
| **RESOLVED** | W28TA-1 | metric | High | local_linear.py:107: neighbours come from a cosine index on median-centred data, so the local OLS is not local; the neighbour index uses the l2 space (local slope of x0^2 at the query 0.2 was 2.94, now near 0.4) |
| **RESOLVED** | W28TA-2 | fit | High | local_curvature.py:87-112: the quadratic design has more columns than k rows for d >= 8, lstsq interpolates exactly, resid_quad ~ 0; with fewer neighbours than quadratic coefficients plus two the curvature columns (trace, frobenius, residual difference, quadratic value) are NaN with a throttled warning and the linear value is kept; the older bit-identity test is reframed to this contract |
| **RESOLVED** | W28TA-3 | leakage | High | predictive_info_delta.py:110-119: H(y|bin) built from in-sample predictions, query rows use held-out ones; the bins are built from inner-KFold out-of-fold predictions (`fit_baseline_predict_oof`); mean abs delta on a noise target at n=300, d=20 was 0.40 and is 0.011 |
| **FUTURE** | W28TA-4 | replay | High | row_attention.py:202: apply_dedupe runs separately on the Mode A and Mode B outputs so their column sets can differ; reason: the dedupe keeps a data-dependent column subset, so Mode A and each Mode B batch can disagree; the fix is an additive `keep_columns` argument (names taken from the Mode A frame) that selects exactly those columns and skips the per-batch dedupe, to be wired through `compute_row_attention` and its wrappers together |
| **RESOLVED** | W28TA-5 | replay | High | spectral_attention.py:120-121: the Nystrom bandwidth is recomputed from the query batch instead of reusing the training graph sigma; the Nystrom step reuses the training graph's sigma; also found while testing: the eigenvector sign was random between two fits of the same data (ARPACK random start), so each eigenvector now has its largest-magnitude entry positive |
| **FUTURE** | W28TA-6 | replay | Medium | residual_attention.py:106,136: Mode A applies dedupe_threshold, Mode B hard-codes None; reason: `residual_attention` applies its dedupe in the shared tail of both modes, so the report's "hard-coded None" is not the cause; the batch-dependent subset is the same defect as W28TA-4 and gets the same `keep_columns` fix |
| **RESOLVED** | W28TA-7 | replay | Medium | local_classifier.py:162: the bandwidth is the median k-th distance over the query batch; the bandwidth is the median k-th neighbour distance of (up to 2000 evenly spaced) train rows, self excluded |
| **RESOLVED** | W28TA-8 | replay | Medium | mdl_binning_pairwise.py:226: n_unique_combos counted over the query batch; `unique_combos` counts the distinct combinations in the train rows |
| **FUTURE** | W28TA-9 | replay | Medium | performer_attention.py:65,121-122: the global-max shift is per batch and does not cancel |
| **FUTURE** | W28TA-10 | precision | Medium | performer_attention.py:61-66,114: float32 underflow hidden by the 1e-9 floor fallback (y_est = 0) |
| **RESOLVED** | W28TA-11 | leakage | Medium | nn_oof_target_mean.py:68-96,143-144: kNN index built on in-sample baseline predictions, queries use out-of-sample ones; the three members now come from `ib_baseline_codes._fit_3baselines_two` (same settings, out-of-fold train side); the logging-consistency test lists that function instead of the removed copy |
| **RESOLVED** | W28TA-12 | leakage | Medium | pairwise_kl_divergence.py:80-82: baseline sigma is the in-sample residual std; the regression sigmas come from inner-KFold(3) out-of-fold residuals of the same three members; on a noise target the deeper booster's sigma was 0.50 and is near 1 |
| **RESOLVED** | W28TA-13 | leakage | Medium | variance_baseline.py:79-85: mu_train is in-sample; mu_train is out-of-fold; the mean predicted sigma^2 on a noise target was 0.47 and is 1.18 (true 1) |
| **RESOLVED** | W28TA-14 | leakage | Medium | target_kmeans_codebook.py:405,409: train rows clustered on in-sample predictions, queries on held-out ones; the clustering coordinate is the out-of-fold baseline prediction |
| **RESOLVED** | W28TA-15 | scale | Medium | target_kmeans_codebook.py:420,428: cluster_size is a fold-sized count in Mode A and a full-data count in Mode B; `cluster_size` is the share of the train rows in the cluster (was a count, so the column changes scale: a downstream consumer reading counts sees values in [0, 1]) |
| **RESOLVED** | W28TA-16 | scale | Medium | local_lift.py:50: max(y_global_mean, 1e-9) breaks for a zero or negative mean; a global mean below 1e-9 in magnitude is floored with its sign kept |
| **RESOLVED** | W28TA-17 | ties | Medium | residual_stratified_distance.py:190-192: abs_r < median is empty when half or more of the residuals tie at the median; the easy band is the lower half by a stable rank, so tied residuals no longer empty it (constant target: log-ratio -16 before) |
| **FUTURE** | W28TA-18 | crash | Medium | stacked_attention.py:92-115: layer-0 NaN rows rejected by validate_numeric_input at layer 1 under TimeSeriesSplit |
| **FUTURE** | W28TA-19 | leakage | Medium | stacked_attention.py:115, stacked_qnn.py:225: layer 2 reuses the splitter on layer-1 out-of-fold features |
| **RESOLVED** | W28TA-20 | replay | Medium | row_attention.py:239: validate_projection_dims called without allow_overcomplete in Mode B; `build_key_bank` takes `allow_overcomplete` and the Mode B path forwards it |
| **RESOLVED** | W28TA-21 | crash | Medium | predict_proba(...)[:, 1] raises on a single-class training fold in twelve modules (multi_baseline_hard_row, aux_mlp, multi_aux_ensemble, nn_oof_target_mean, pairwise_kl_divergence, sign_residual_baseline, variance_baseline, target_kmeans_codebook, tree_path_boolean, trust_score_oof, _baseline_oof); the LightGBM baselines do not crash on a single-class fold (checked: `fit_baseline_predict_oof` returns finite values); the crash was real for sklearn's `MLPClassifier`, which returns a second column with a meaningless probability for a class it never saw; `positive_class_proba` handles it and `aux_mlp` uses it |
| **RESOLVED** | W28TA-22 | nan | Medium | Mode A output allocated with np.zeros so rows no fold validates read 0 in ten modules (sign_residual_baseline, signed_residual_band, smote_distance, spectral_attention, target_kmeans_codebook, target_quantile, tree_path_boolean, trust_score_oof, variance_baseline, y_quintile_baseline_knn); all ten allocate the Mode A output with NaN |
| **FUTURE** | W28TA-23 | crash | Low | spectral_attention.py:89,106,204: too few eigenvectors and NearestNeighbors failure on a small fold |
| **FUTURE** | W28TA-24 | scale | Low | spectral_attention.py:136: eigenvalues clamped at 1e-6 give embeddings of size ~1e6 |
| **RESOLVED** | W28TA-25 | crash | Low | target_kmeans_codebook.py:416: n_clusters_eff never capped at n; the cluster count is capped at the row count |
| **FUTURE** | W28TA-26 | validation | Low | row_attention.py:53, _oof.py:186: k checked against n_train, not the fold size; k_scales not validated |
| **FUTURE** | W28TA-27 | precision | Low | target_quantile.py:78-80, y_quintile_baseline_knn.py:251-253: the +1e-9 tie-break is below one ULP for |y| > 1e7 |
| **FUTURE** | W28TA-28 | ties | Low | multi_baseline_hard_row.py:169-175: rows with all-negative z-scores tie at 0 |
| **FUTURE** | W28TA-29 | ties | Low | multi_temp_cbhr.py:79-81: q1 == q4 picks the same anchors on both sides |
| **FUTURE** | W28TA-30 | weights | Low | multi_temp_residual_band.py:106-111: empty bands keep softmax weight |
| **FUTURE** | W28TA-31 | convention | Low | lda_projection, mixup_boundary, multiscale_smote, smote_distance, per_class_spectral: all-zero output when a class has fewer than 2 rows |
| **FUTURE** | W28TA-32 | validation | Low | per_class_spectral.py:88-102: y is not checked to be binary |
| **FUTURE** | W28TA-33 | naming | Low | column named after the requested k, not the clamped k (nn_oof_target_mean, neighbor_aggregate_features, multiscale_rate, persistence_diagram) |
| **RESOLVED** | W28TA-34 | labels | Low | focal_lgb.py:37, multi_aux_ensemble.py:57: {-1, 1} labels not remapped for the focal objective; the focal datasets take `y > 0.5` as the 0/1 label in `focal_lgb` and `multi_aux_ensemble` |
| **RESOLVED** | W28TA-35 | naming | Low | multi_temperature.py:78: temperatures 0.301 and 0.304 format to the same column name; labels fall back to six significant digits when the two-decimal labels collide (existing names unchanged) |
| **FUTURE** | W28TA-36 | precision | Low | per_column_rff.py:129-137: always float32, ignores dtype=float64 |
| **FUTURE** | W28TA-37 | leakage | Low | pseudo_smote.py:68: filter threshold is a quantile of in-sample predictions |
| **FUTURE** | W28TA-38 | doc | Low | persistence_diagram.py: y is never used although the docstring promises a joint X/y metric |
| **FUTURE** | W28TA-39 | leakage | Low | trust_score_oof.py:410,417: in-sample correctness mask, already stated in the module docstring |
| **FUTURE** | W28TA-40 | unverified | Low | quantile_spread_fan.py:78-86: negative-row weights ~3e-9 at gamma=5 may make the q90 fit degenerate (plausible, not verified) |

## Not covered by this wave

- training/neural: `data.py`, `base/_base_fit*.py`, `base/_cuda_fallback.py`, `base/_base_losses.py`, the training/validation step bodies of `_flat_torch_module/__init__.py`, `flat.py`, `ranker.py`, `_ranker_losses.py`, the recurrent data/wrapper/arch/embedding modules, the optimizers (`_sam`, `_lookahead`, `_muon`), `_mixup.py`, `_numerical_embeddings.py`, `feature_prep.py`, `field_grouped_mlp.py`, `trunk_residual_mlp.py`, `tabular_1dcnn.py`, `fixed_sparse_linear.py`.
- `_dynamic_cluster_discovery/__init__.py`, `_pairs_dispatch.py`, `_pairs_chunks.py`, `_pairs_emit_extval.py`, `_pairs_score_step4.py` lines 20-289.

## Implemented

| ID | Regression test |
|---|---|
| W28NN-1, 2 | `tests/training/test_neural_weighted_loss_reductions.py` |
| W28TA-21 | `tests/feature_engineering/transformer/test_single_class_fold_does_not_crash_the_aux_mlp.py` |
| W28TA-22 | `tests/feature_engineering/transformer/test_oof_rows_no_fold_validates_stay_nan.py` |
| W28TA-3 | `tests/feature_engineering/transformer/test_predictive_info_delta_bins_are_out_of_fold.py` |
| W28TA-12 | `tests/feature_engineering/transformer/test_pairwise_kl_sigmas_use_out_of_fold_residuals.py` |
| W28TA-13 | `tests/feature_engineering/transformer/test_variance_baseline_sigma_uses_out_of_fold_residuals.py` |
| W28GP-1, 2 | `tests/feature_selection/fe/test_pairs_abs_corr_gpu_keeps_float64_precision.py` |
| W28DC-1 | `tests/feature_selection/clustering/test_dcd_foreign_view_of_the_same_buffer_is_scored_on_its_own_data.py` |
| W28TA-20 | `tests/feature_engineering/transformer/test_row_attention_mode_b_accepts_overcomplete_heads.py` |
| W28TA-7, 8 | `tests/feature_engineering/transformer/test_query_features_do_not_depend_on_the_rest_of_the_batch.py` |
| W28TA-16, 25, 35 | `tests/feature_engineering/transformer/test_small_edge_cases_lift_clusters_and_temperature_names.py` |
| W28TA-17, 34 | `tests/feature_engineering/transformer/test_focal_labels_and_tied_residual_bands.py` |
| W28TA-2 | `tests/feature_engineering/transformer/test_local_curvature_is_nan_when_the_quadratic_fit_is_underdetermined.py` |
| W28TA-1 | `tests/feature_engineering/transformer/test_local_linear_neighbours_are_nearest_not_same_direction.py` |
| W28TA-5 | `tests/feature_engineering/transformer/test_spectral_attention_embedding_does_not_depend_on_the_query_batch.py` |
| W28TA-11 | `tests/feature_engineering/transformer/test_nn_oof_target_mean_train_embedding_is_out_of_fold.py` |
| W28TA-14, 15 | `tests/feature_engineering/transformer/test_target_kmeans_codebook_uses_out_of_fold_baseline_and_shares.py` |
| W28CT-1 | `tests/training/composite/test_ratio_floor_keeps_a_tiny_negative_base_finite.py` |
