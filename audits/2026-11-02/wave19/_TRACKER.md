# Wave 19 audit 2026-11-02 -- tracker

Six read-only reports by Opus 5.5 low-effort agents on the areas waves 17 and 18 left unread: the remaining metric cores, `estimators/` and `models/` outside ensembling, the evaluation leak / drift / bootstrap modules, composite discovery and ensemble, `training/feature_handling` and `training/neural`, and MRMR recipe replay with screening. The agents could not write files, so the reports are their returned text, condensed. Every claim was re-checked on HEAD before a fix. Dispositions: RESOLVED (fixed with a test), FUTURE, DOC, REJECTED.

| Disposition | ID | Severity | Location |
|---|---|---|---|
| **RESOLVED** | W19MET-1 | High | `create_fairness_subgroups` left `cats_to_use` unassigned (UnboundLocalError, or the previous feature's bins) when rare categories cannot be merged and exclusion is off |
| **RESOLVED** | W19MET-2 | Medium | Spearman numba kernels returned NaN for +-inf while the scipy path ranks it; they now treat only NaN as missing |
| **FUTURE** | W19MET-3 | Medium | `gpu_multiple_roc_auc_scores` ranks NaN scores as separate tie runs and returns a finite AUC; the CPU path's NaN handling was not read, so the target behaviour (NaN column) needs confirming first |
| **RESOLVED** | W19MET-4 | Medium | MBE, explained variance and wMAPE wrappers dropped the float64 cast and numba subtracted unsigned inputs in uint64 (-1 became 1.8e19) |
| **RESOLVED** | W19MET-5 | Low | Hosmer-Lemeshow silently dropped the group holding a NaN score |
| **RESOLVED** | W19MET-6 | Low | `fast_cv_rmse` documented NaN for mean <= 0 but guarded only mean == 0 |
| **RESOLVED** | W19EST-1 | High | `clip_to_quantiles(method="winsor_linear")` passed absolute values as the relative band, doubling the maximum instead of compressing it |
| **RESOLVED** | W19EST-2 | High | `soft_winsorize` raised on binary / count targets with ties at an extreme; `clip_to_quantiles` now skips a side with nothing beyond its quantile |
| **RESOLVED** | W19EST-3 | Medium | the signed relative band reversed the order of values beyond the quantile on negative targets |
| **RESOLVED** | W19EST-4 | Medium | a label without positives got class weight 1e6 and shrank every other label's weight to ~0 |
| **RESOLVED** | W19EST-5 | Medium | `init_evaluate_ascending/descending` ordered numeric candidates by their string form (9 before 100) |
| **RESOLVED** | W19EST-6 | Medium | a NaN evaluation reached the surrogate refit and crashed it |
| **RESOLVED** | W19EST-7 | Low | `rf_outlier_measure` cast `y` to int64, merging float labels 0.2 / 0.7 and rejecting strings; classes now come from `np.unique(return_inverse=True)` (`tests/models/test_rf_outlier_measure_class_labels.py`) |
| **RESOLVED** | W19EVA-1 | High | the temporal leak scan broke rank ties by row order, so a constant column correlated 0.87 with a time-ordered split and was flagged |
| **RESOLVED** | W19EVA-2 | Medium | NaNs ranked last in the leak scan; `correlation` now uses each column's finite rows and the report gains `missing_correlation` (the missingness indicator against the split label), either crossing the threshold sets `flagged` (`tests/evaluation/test_leak_scan_missingness_is_separate.py`) |
| **RESOLVED** | W19EVA-3 | Medium | `_stratified_class_book` keyed classes by `int(label)` (fractional labels merged, strings raised) |
| **RESOLVED** | W19CMP-1 | High | `fit_stacked_on_residual` dropped every spec with a NaN OOF row, so pass 2 never ran on temporal data |
| **RESOLVED** | W19CMP-2 | High | the same function built its OOF folds without group ids |
| **RESOLVED** | W19CMP-3 | Medium | the single-split group carve fell back to a group-blind row shuffle when the carve was degenerate; `_carve_holdout_groups` now skips groups that overshoot twice the request, logs a holdout outside [0.5x, 2x], and raises when no group-respecting split exists (`tests/training/composite/test_oof_group_carve_never_shuffles_rows.py`) |
| **RESOLVED** | W19CMP-4 | Medium | fewer groups than OOF folds fell back to a shuffled K-fold that splits groups |
| **RESOLVED** | W19CMP-5 | Medium | the OOF cache key hashed only 128 group labels and the mere presence of a time order |
| **RESOLVED** | W19CMP-6 | Low | the ridge / lasso / elastic-net meta-stackers take `group_ids` (GroupKFold) and `time_ordered` (forward-chaining) for their alpha search, threaded through `build_meta_stack_ensemble`; the only in-repo callers are benchmarks, so there is no suite call site to wire (`tests/training/composite/test_meta_stacker_alpha_cv_respects_groups.py`); the GBM stacker's internal early-stopping split is unchanged |
| **RESOLVED** | W19FHN-1 | High | the dense-only assembly refit `TruncatedSVD` per split; val/test now use `transform` of the train-fitted SVD (`AssembledMatrix.fitted_svds`) |
| **RESOLVED** | W19FHN-2 | High (use_mixup) | mixup interpolated integer category codes before the embedding lookup |
| **RESOLVED** | W19FHN-3 | Medium | unseen categories under WoE encoded to the prior log-odds instead of the neutral 0 |
| **RESOLVED** | W19FHN-4 | Medium | the neural estimator did not restore the fit-time column order at predict |
| **RESOLVED** | W19FHN-5 | Low | the text-encoder cache token fingerprinted five sampled rows; it now hashes every row of the column (polars `hash`, pandas `hash_pandas_object`, 8 bytes per row transient) so frames differing only in interior rows no longer share a cached encoder (`tests/training/feature_handling/test_text_column_token_sees_every_row.py`) |
| **RESOLVED** | W19MRM-1 | High | the chained recipe replay emitted a zero column for a missing raw input column instead of raising |
| **RESOLVED** | W19MRM-2 | Medium | the `smart_log` anchor was computed in float32 and replay shifts the float64 operand |
| **RESOLVED** | W19MRM-3 | Low | the RAM-footprint guard ran before a LazyFrame was collected and so saw no row count; it now runs again on the collected frame (`tests/feature_selection/filters/test_mrmr_footprint_guard_sees_collected_lazyframe.py`) |

## Not covered by this wave

`_ice_metric.py`, `classification/_ice_kernel_weighted`, `_classification_report`, `_gains_lift`, `regression/_regression_metrics`, `ranking.py`; `estimators/base`, `baselines`, `early_stopping*`, `pipelines`; `evaluation/reports.py`, `_bootstrap_fused_binary_bundle`, `adversarial_feature_audit`, `distribution_matching_subset_search`; about 60 composite discovery modules and `ensemble/_cross_target`, `_calibration`, `_oof_external`; the remaining `feature_handling` modules and neural `ranker`, `flat`, optimisers; the other `engineered_recipes` kinds (fill values and clip bounds at replay).

## Implemented

| ID | Regression test |
|---|---|
| W19MET-1 | `tests/metrics/test_fairness_subgroups_keep_rare_bins_when_not_excluded.py` |
| W19MET-2 | `tests/metrics/test_spearman_infinite_values_rank_like_scipy.py` (the NaN-precheck test in `test_rank_correlation.py` now injects a NaN) |
| W19MET-4 | `tests/metrics/test_regression_extras_unsigned_inputs.py` |
| W19MET-5, 6 | `tests/metrics/test_hl_nan_and_cv_rmse_nonpositive_mean.py` |
| W19EST-1, 2, 3 | `tests/estimators/test_clip_to_quantiles_tails.py` |
| W19EST-4, 5, 6 | `tests/models/test_optimization_and_label_weight_edge_cases.py` |
| W19EVA-1 | `tests/evaluation/test_leak_scan_constant_column_is_not_a_leak.py` |
| W19EVA-3 | `tests/evaluation/test_stratified_class_book_keeps_non_integer_classes.py` |
| W19CMP-1, 2 | `tests/training/composite/test_stacked_on_residual_groups_and_warmup.py` |
| W19CMP-4, 5 | `tests/training/composite/test_ensemble_oof_split_and_cache_key.py` |
| W19FHN-1 | `tests/training/feature_handling/test_svd_basis_is_shared_between_train_and_val.py` |
| W19FHN-2 | `tests/training/neural/test_mixup_keeps_categorical_codes_whole.py` |
| W19FHN-3 | `tests/training/feature_handling/test_woe_unseen_category_is_neutral.py` |
| W19FHN-4 | `tests/training/neural/test_predict_restores_fit_time_column_order.py` |
| W19MRM-1 | `tests/feature_selection/filters/test_transform_missing_raw_source_raises.py` |
| W19MRM-2 | `tests/feature_selection/filters/test_smart_log_anchor_uses_float64.py` |
