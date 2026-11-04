# Wave 31 audit 2026-11-04 -- tracker

Three read-only reports by Opus 5.5 low-effort agents on large packages (metrics; preprocessing / data / data_valuation; votenrank / competition / models / core / utils). Coverage was partial (about 6 of 94, 12 of 114 and 5 of 171 files). The agents could not write files, so the reports are their returned text, condensed; each finding is re-checked on HEAD before it is fixed. Dispositions: RESOLVED, FUTURE, DOC, REJECTED.

| Disposition | ID | Class | Severity | Location |
|---|---|---|---|---|
| **RESOLVED** | W31ME-1 | labels | High | `metrics/_auc_labels.py:16-17`: `_binary_labels_01` maps any label set outside [0,1] to `y == 1`, so labels {1,2} give 1 - AUC (sklearn uses the larger label as positive); {0,2} become all zeros; two distinct labels outside [0, 1] now take the larger as positive (the helper's docstring claimed this was sklearn's convention); any other label set keeps `== 1` |
| **FUTURE** | W31ME-2 | nan | Medium | `metrics/_core_auc_brier.py:176,594,762-778`: NaN scores sort last, the descending reversal puts them first, so NaN is ranked highest and the AUC is finite but wrong |
| **RESOLVED** | W31ME-3 | labels | Medium | `metrics/_log_loss_and_separation.py:115-119`: any `y_true` other than 1 (2, -1, NaN) is scored as a negative; `fast_log_loss_binary` (and `fast_log_loss` through it) returns NaN when `y_true` holds NaN and maps two labels outside {0, 1} with the larger as positive |
| **RESOLVED** | W31ME-4 | empty | Low | `metrics/calibration/_calibration_metrics.py:133,212,285`: empty input returns 1.0 (ECE, debiased ECE, Brier decompositions) where NaN is correct; the four kernels (plug-in and debiased ECE, both Brier decompositions, the fused one) return NaN in every field on empty input; five older tests that pinned the 1.0 sentinel are reframed |
| **RESOLVED** | W31PP-1 | asymmetry | Medium | `preprocessing/missing_indicator_pairing.py:60,85`: fit skips columns without NaN, so NaN in such a column at apply time gets no fill and no indicator; an explicit `columns` list is learned in full (auto-discovery still keeps only the columns with a NaN in the fit frame, as documented) |
| **DOC** | W31PP-2 | mutation | Medium | `preprocessing/transforms.py:166-266`: `prepare_df_for_catboost` rewrites the caller's pandas columns and grows the caller's `cat_features` list; in-place mutation is the documented design (never copy frames that can be 100+ GB) and `cat_features` is an out-parameter callers rely on; the contract is now written in the docstring |
| **DOC** | W31PP-3 | dtype | Medium | `preprocessing/transforms.py:184-197`: polars nullable Int/Boolean columns are cast to Float only when the frame has a null, so train and test dtypes differ; the widening exists only to clear nulls and CatBoost converts every numeric column to float32 internally; recorded in the docstring |
| **FUTURE** | W31PP-4 | crash | Low | `data_valuation/_adversarial_reweighting.py:157-188`: a 3-class `predict_proba` is written into a 1-D buffer |
| **RESOLVED** | W31UT-1 | cache | High | `utils/disk_cache.py:168-184`: `hash_array_summary` adds sums only for numeric dtypes, so bool and fixed-width string arrays differing in the middle rows get the same key; bool arrays add per-column sum/min/max over their 0/1 values, fixed-width string, bytes and datetime arrays hash the whole buffer |
| **FUTURE** | W31UT-2 | backend | Medium | `votenrank/confidence_gated_blend.py:157,47-65,202-205`: the numba kernels allocate 1-D output but the shape check accepts 2-D input that the numpy path blends |
| **FUTURE** | W31UT-3 | logging | Low | `votenrank/confidence_gated_blend.py:206-211`: a forced cupy backend failure falls back to numpy with a DEBUG log only |
| **FUTURE** | W31UT-4 | edge | Low | `competition/leak_scan.py:155-162`: `max_col_shift >= n_cols` gives a negative slice stop and a matmul shape error |

## Not covered by this wave

- metrics: all of `regression/`, `classification/` (except labels/log-loss parts above), `_fairness_metrics.py`, `quantile.py`, `_multilabel_*`, `_drift.py`, `_ranking_extras.py`, `core.py`, `iteration_metrics.py`, `rank_correlation.py`, `_core_brier.py`, `_gpu_metrics.py`, `ranking.py` after line 260.
- preprocessing: `cleaning.py`, `_cleaning_helpers.py`, `auto_transform_select.py`, `align_feature_direction.py`, `gaussian_power_transform_search.py`, `temporal_drift_augment.py`, `outlier_detector_zoo.py`, `outlier_policy.py`; data: all of `data/`; data_valuation: all but two modules.
- models/ (49 files), core/ (19), most of utils/, most of votenrank/ and competition/.

## Implemented

| ID | Regression test |
|---|---|
| W31ME-1 | `tests/metrics/test_auc_labels_use_the_larger_label_as_positive.py` |
| W31UT-1 | `tests/utils/test_disk_cache_array_key_sees_middle_rows_of_bool_and_string_arrays.py` |
| W31PP-1 | `tests/preprocessing/test_missing_indicator_explicit_columns_are_learned_even_without_train_nans.py` |
| W31ME-4 | `tests/metrics/classification/test_calibration_kernels_return_nan_on_empty_input.py` |
| W31ME-3 | `tests/metrics/test_log_loss_label_handling.py` |
