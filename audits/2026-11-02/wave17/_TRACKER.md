# Wave 17 audit 2026-11-02 -- tracker

Six read-only package-by-package correctness reports by Opus 5.5 low-effort agents (metrics/preprocessing, calibration/evaluation, training core, feature-selection wrappers, MRMR/filters, composite training). The agents could not write files, so the reports are their returned text, condensed. Every claim was re-checked on HEAD before a fix. Dispositions: RESOLVED (fixed with a test), OPEN (verified, not yet fixed), FUTURE, DOC, REJECTED.

| Disposition | ID | Severity | Location |
|---|---|---|---|
| **RESOLVED** | W17MET-1 | High | `expected_reciprocal_rank` accepted labels above `max_grade` and returned ERR above 1; now raises `ValueError` |
| **RESOLVED** | W17MET-2 | Medium | `inference/recursive_forecast.py` oracle must apply the same `update_features_fn` chain with true lags |
| **RESOLVED** | W17MET-3 | Medium | `preprocessing/rare_count_pruning.py` numerator and Woolf SE use different smoothed counts |
| **RESOLVED** | W17MET-4 | Medium | `_split_by_group` with string/object group ids |
| **RESOLVED** | W17TRN-1 | High | a cached model file that failed to load left an unfitted estimator (fit gate tested only that the file exists) |
| **RESOLVED** | W17TRN-2 | High | OOF folds of TTR-wrapped regressors and Pipelines were fit without `sample_weight` (`training/_oof_weight_param.py`) |
| **RESOLVED** | W17TRN-3 | Medium | val/OOF slices of Series `sample_weight` / `group_ids` index by label |
| **RESOLVED** | W17TRN-4 | Medium | `metadata["model_schemas"]` keys collide across pre-pipelines and targets |
| **RESOLVED** | W17TRN-5 | Low | `val_placement="backward"` moves split groups to the wrong side |
| **RESOLVED** | W17TRN-6 | Low | multilabel 3-way split uses `val_size` as a fraction of all rows |
| **RESOLVED** | W17FSW-1 | High | RFECV `must_include` and group members dropped on ndarray input (string fit-time names vs integer columns) |
| **RESOLVED** | W17FSW-2 | Medium | RFECV `ranking_` disagrees with `support_` after pins / group expansion |
| **RESOLVED** | W17FSW-3 | Medium | per-fold scores stored in completion order under `n_jobs > 1` |
| **RESOLVED** | W17FSW-4 | Medium | final voting / p>=n cap on ndarray input compares int labels with str names |
| **RESOLVED** | W17MRM-1 | High | MRMR same-inputs skip and shared fit cache ignored `groups` |
| **FUTURE** | W17MRM-2 | Medium | UAED elbow trim slices `support_` by greedy-round position |
| **FUTURE** | W17MRM-3 | Medium | operand retention re-grows support after the UAED trim |
| **RESOLVED** | W17CMP-1 | High | `predict_quantile` applied smearing, returning means instead of quantiles |
| **RESOLVED** | W17CMP-2 | Medium | `GroupedBlockStacker` ignored a numeric `invalid_row_fill` |
| **RESOLVED** | W17CMP-3 | Low | `_best_iteration_of` off by one for XGBoost / CatBoost |
| **PARTIAL** | W17CMP-4 | Low | the hurdle's smearing factor is now weighted (`tests/training/composite/test_hurdle_smearing_factor_is_weighted.py`); the composite estimator's `residual_quantiles` still ignores sample weights |
| **RESOLVED** | W17CMP-5 | Low | T-clip envelope differs between fit and drift refit when MAD = 0 |
| **FUTURE** | W17CMP-6 | Low | chain / quantile-normal transforms get no smearing |
| **RESOLVED** | W17CAL-1 | Medium | `generate_probs_from_outcomes` default `flip_percent=0.6` reverses the signal |
| **RESOLVED** | W17CAL-2 | Medium | "random chunks" in the same generator are contiguous slices |
| **RESOLVED** | W17CAL-3 | Medium | `compare_cv_schemes` winner and significance test use different criteria |
| **RESOLVED** | W17CAL-4 | Low | string labels crash the stratified resampler (`policy.py` `int(c)`) |
| **RESOLVED** | W17CAL-5 | Low | `generate_similar_probs_by_ranking` has n_bins+1 bins |

## Not covered by this wave

Metric cores (`_core_auc_brier`, `_ice_metric`, `_drift`, `_fairness_metrics`), `models/`, calibration `post`/`quality`/`ensembling`, evaluation leak/adversarial modules, HPO and most transforms of the composite package, MRMR `_mrmr_fe_step`, and the rest of the training core were not read.

## Implemented

| ID | Regression test |
|---|---|
| W17MET-1 | `tests/metrics/test_err_rejects_labels_above_max_grade.py` |
| W17TRN-1 | `tests/training/test_trainer_refits_when_cached_model_does_not_load.py` |
| W17TRN-2 | `tests/training/test_oof_weight_reaches_wrapped_estimators.py` |
| W17FSW-1 | `tests/feature_selection/wrappers/test_rfecv_must_include_on_ndarray_input.py` |
| W17MRM-1 | `tests/feature_selection/filters/test_mrmr_fit_signature_includes_groups.py` |
| W17CMP-1 | `tests/training/composite/estimator/test_predict_quantile_is_not_smeared.py` |
| W17CMP-2 | `tests/training/composite/test_grouped_block_stacker_numeric_invalid_row_fill.py` |
| W17FSW-3 | `tests/feature_selection/wrappers/test_rfecv_fold_scores_keep_fold_order.py` |
| W17CAL-1, 2, 5 | `tests/calibration/test_probability_generators_fixes.py` |
| W17CAL-4 | `tests/calibration/test_resample_indices_accept_non_integer_class_labels.py` |
| W17CAL-3 | `tests/evaluation/test_compare_cv_schemes_winner_matches_tested_quantity.py` |
| W17MET-4 | `tests/metrics/test_ranking_extras_string_group_ids.py` |
| W17TRN-3 | `tests/training/test_trainer_slices_series_weights_by_position.py` |
| W17MET-3 | `tests/preprocessing/test_rare_collapse_z_uses_one_smoothed_table.py` |
| W17MET-2 | `tests/inference/test_diagnose_error_accumulation_oracle_applies_feature_updates.py` |
| W17FSW-4 | `tests/feature_selection/wrappers/test_rfecv_support_from_ranking_on_ndarray.py` (helper-level; the ranking key type was not confirmed end to end, the string comparison is correct for both) |
| W17FSW-2 | `tests/feature_selection/wrappers/test_rfecv_must_include_on_ndarray_input.py` |
| W17CMP-3 | `tests/training/composite/ensemble/test_best_iteration_counts_trees_for_zero_based_boosters.py` |
| W17CMP-5 | `tests/training/composite/cache/test_drift_refit_keeps_no_t_clip_when_mad_is_zero.py` |
| W17TRN-6 | `tests/training/test_split_3way_and_cb_plot.py::test_3way_val_size_is_a_fraction_of_the_post_test_remainder` (reframed from the whole-fraction assertion) |
| W17TRN-5 | `tests/training/test_backward_layout_group_span_goes_to_train.py` |

## FUTURE reasons

- W17TRN-4: resolved in wave 18 as W18TRN-1 (first record keeps the bare key, later ones are qualified; readers only iterate the records)
- W17MRM-2: opt-in (`uaed_auto_size`, default off); the fix maps greedy rounds to features through `_predictors_log_`, which needs a parity check against the retention and rescue appends
- W17MRM-3: opt-in (`uaed_auto_size`, default off); needs the UAED-trimmed and p>=n-capped indices recorded and honoured by operand retention, designed together with W17MRM-2
- W17CMP-4: weighted smearing factor and weighted residual quantiles change every weighted composite's predictions; needs a measured bias comparison before the default moves
- W17CMP-6: adding a `curved_inverse` attribute to the chain and quantile-normal transforms changes their predictions and the discovery scoring; needs a discovery-gain comparison
