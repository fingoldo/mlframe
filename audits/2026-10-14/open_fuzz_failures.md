# Open fuzz failures (status: OPEN)

Recorded 2026-10-14 from `tests/training/_fuzz_results.jsonl`. Each entry is the latest failing row for the combo and its non-default configuration. Root cause is not yet established for any of the three.

## c0061_2d836be9 (NaN in X for MLP inside TTR)

- Error: `ValueError: X contains 9 NaN and 0 inf values ... PytorchLightningEstimator does NOT impute`
- Latest failing row (300 rows): `c0061_68d3d37c`, `apply_cross_sectional_composite_fe: split 'train' lacks snapshot column 'cat_0'`. Origin frame is `training/pipeline/_cross_sectional_composite_fe.py:80`. The NaN error is from the earlier run (`c0061_c1415acd`, before the cross-sectional fixes).
- Config (non-default fields):

```
cat_text_card_threshold_cfg: 300
categorical_encoding_cfg: ordinal
custom_prep: pca2
ensure_float32_cfg: true
fairness_col: cat_0
fix_infinities_cfg: true
honor_user_dtype: true
imbalance_ratio: rare_5pct
inject_degenerate_cols: true
inject_inf_nan: true
inject_rank_deficient: true
inject_test_drift: shifted_distribution
inject_zero_col: true
input_storage: parquet
input_type: polars_utf8
iterations: 3
master_seed: 20260422
models: [hgb, lgb, linear, mlp, xgb]
mrmr_nan_strategy_cfg: ffill_bfill
multilabel_chain_order_cfg: by_frequency
multilabel_cv_cfg: 2
multilabel_n_chains_cfg: 3
multilabel_strategy_cfg: chain
n_rows: 300
null_fraction_cats: 0.3
outlier_detection: lof
prefer_calibrated_classifiers: true
prep_ext_kbins_cfg: 5
prep_ext_nonlinear_cfg: RBFSampler
recurrent_model_cfg: lstm
rfecv_estimator_cfg: cb_rfecv
scaler_name_cfg: standard
seed: 61
target_carrier: native
target_type: multi_target_regression
test_size_cfg: 0.2
use_robust_eval_metric_cfg: true
use_text_features: true
val_placement_cfg: forward
val_sequential_fraction_cfg: 0.5
weight_schemas: [uniform]
weird_cat_content: unicode
wholeday_splitting_cfg: true
```

- Checked, not reproduced: 300-row run (`c0061_29ca0f68`, with budget) hit the MRMR/FS path; full-size run stopped by timeout (EXIT 124) after 5/5 models of the first loop, no NaN error.
- Ruled out in code: the TTR wrapper does not touch `X`; MLP input guard `_base_fit.py:94` / `_base_fit_helpers.py:80`; `_subset_dataframe` val_df; cached pre-pipeline frames.

## c0057_5755123b (missing load-bearing columns at predict)

- Error: `ValueError: Input DataFrame is missing 5 load-bearing feature column(s) ...: cat_0_concat_group__cat_0_text_0__target_*`
- Origin frame: `training/core/_misc_helpers.py:398 in _validate_input_columns_against_metadata`
- Config (non-default fields):

```
auto_detect_cats: true
cat_feature_count: 1
cat_text_card_threshold_cfg: 300
categorical_encoding_cfg: ordinal
continue_on_model_failure: true
ensure_float32_cfg: true
fairness_col: cat_0
honor_user_dtype: true
imbalance_ratio: rare_5pct
imputer_strategy_cfg: mean
inject_all_nan_col: true
inject_degenerate_cols: true
inject_label_leak: true
input_storage: parquet
input_type: polars_enum
iterations: 3
master_seed: 20260422
models: [cb, hgb, lgb, mlp, xgb]
mrmr_nan_strategy_cfg: ffill_bfill
multilabel_chain_order_cfg: by_frequency
multilabel_cv_cfg: 2
multilabel_n_chains_cfg: 3
multilabel_strategy_cfg: chain
n_rows: 300
null_fraction_cats: 0.3
outlier_detection: ocsvm
prep_ext_dim_reducer_cfg: TruncatedSVD
prep_ext_polynomial_degree_cfg: 2
prep_ext_pysr_enabled_cfg: true
prep_ext_scaler_cfg: StandardScaler
recurrent_model_cfg: gru
remove_constant_columns_cfg: true
rfecv_estimator_cfg: cb_rfecv
scaler_name_cfg: robust
seed: 57
shuffle_test_cfg: true
skip_categorical_encoding_cfg: true
target_carrier: native
target_type: multiclass_classification
test_size_cfg: 0.2
text_col_count: 1
trainset_aging_limit_cfg: 0.5
use_robust_eval_metric_cfg: true
val_placement_cfg: backward
weight_schemas: [recency]
weird_cat_content: empty
```

- Status: on the budgeted run (100 rows) the combo passes (`c0057_..` checked earlier); the failure is not reproduced on the current code.
- Checked: cross-sectional replay order (raw input before pre-pipeline); `_misc_helpers.py:398` is the validator on the predict input.

## c0021_68842ec8 (TypeError: float() from NoneType)

- Error: `TypeError: float() argument must be a string or a real number, not 'NoneType'`
- Origin frame: not recorded (row predates the origin field).
- Config (non-default fields):

```
auto_detect_cats: true
cat_feature_count: 8
cat_text_card_threshold_cfg: 300
categorical_encoding_cfg: ordinal
duration_s: 101.58
early_stopping_rounds_cfg: 10
embedding_col_count: 1
honor_user_dtype: true
imbalance_ratio: rare_1pct
imputer_strategy_cfg: most_frequent
include_confidence_analysis_cfg: true
inject_degenerate_cols: true
input_storage: memory
input_type: polars_nullable
iterations: 3
master_seed: 20260422
models: [cb, lgb, xgb]
mrmr_nan_strategy_cfg: separate_bin
multilabel_chain_order_cfg: random
multilabel_cv_cfg: 3
multilabel_n_chains_cfg: 2
multilabel_strategy_cfg: wrapper
n_rows: 300
null_fraction_cats: 0.1
outlier_detection: ocsvm
prep_ext_dim_reducer_cfg: PCA
prep_ext_nonlinear_cfg: RBFSampler
prep_ext_pysr_enabled_cfg: true
prep_ext_scaler_cfg: RobustScaler
recurrent_model_cfg: gru
rfecv_estimator_cfg: cb_rfecv
scaler_name_cfg: robust
seed: 21
shuffle_test_cfg: true
shuffle_val_cfg: true
target_carrier: native
target_type: multi_target_regression
test_size_cfg: 0.1
use_text_features: true
val_placement_cfg: backward
weight_schemas: [uniform]
weird_cat_content: empty
wholeday_splitting_cfg: true
```

- Checked: under the budget (100 rows) the reproduction `c0021_27ac93e8` (n1000) passes (14 min); the TypeError did not reproduce. Full-size not checked.
- Note: the most recent row for this combo in the jsonl is a different (`c0021_850e69a9`, `PytestNonZero`, n=300000) record, not the TypeError.
