# Wave 22 audit 2026-11-02 -- tracker

Five read-only reports by Opus 5.5 low-effort agents on areas no earlier wave read: the `shap_proxied_fs` fit / search modules and its TreeSHAP / interaction / clustering modules, `core` / `system` / `competition`, the remaining `reporting` charts, and the remaining info-theory / resident GPU / binner kernels. The agents could not write files, so the reports are their returned text, condensed. Every claim was re-checked on HEAD before a fix. Dispositions: RESOLVED (fixed with a test), FUTURE, DOC, REJECTED.

| Disposition | ID | Severity | Location |
|---|---|---|---|
| **RESOLVED** | W22SHP-1 | High | precomputed MRMR artifacts (SU to target) computed on all rows, holdout included, replace the search-rows-only prefilter; needs a row-count / index check or recomputation on the search rows |
| **RESOLVED** | W22SHP-2 | Medium | under a grouped CV policy the report slice is split row-wise from the holdout, sharing groups with the selection holdout |
| **RESOLVED** | W22SHP-3 | Medium | a small `report_holdout_fraction` of a small holdout crashed `train_test_split` (stratified over more classes than rows) |
| **RESOLVED** | W22SHP-4 | Medium | `per_fold_stability_select` gives NaN-fidelity folds weight 0 and no-trust folds weight 1 |
| **RESOLVED** | W22SHP-5 | Low-Medium | preflight class balance used `mean(y)`; now the rarest class's share |
| **DOC** | W22SHP-6 | Low | `brute_force_top_n` accumulates in the configured criterion dtype (float32 by default) while `brute_force_top_n_cpu_ref` is float64 like the GPU kernel; the reference's docstring now says only it is bit-identical to the GPU and that the two agree to float rounding (`tests/feature_selection/shap_proxied/test_brute_force_cpu_ref_agrees_with_default_to_float_rounding.py`) |
| **RESOLVED** | W22SHK-1 | Medium | the SHAP disk-cache key covered only a summary of X and y, so interior row reordering hit a stale entry |
| **RESOLVED** | W22KRN-1 | Medium | `PermutedTargetBinner.column` skipped the dtype-range guard its sibling paths have |
| **RESOLVED** | W22KRN-2 | Medium | `_native_kbins_quantile` put NaN into the top real bin; now code -1 |
| **RESOLVED** | W22COR-1 | High | `topk_by_partition` negated unsigned (wraps), bool (raises) and list inputs |
| **RESOLVED** | W22COR-2 | High | `ensure_no_infinity_pd` crashed on nullable Float64 with NA and inf and dropped the extension dtype |
| **RESOLVED** | W22COR-3 | Medium | `native_gpu_shap` builds `xgb.DMatrix(X)` without the model's `missing`, `enable_categorical` or `best_iteration`, so contributions sum to the full-tree margin |
| **DOC** | W22COR-4 | Low | the polars-to-pandas view keeps dtypes only for columns without nulls (integers with nulls become float64, booleans with nulls nullable Int8, as pyarrow converts); the `get_pandas_view_of_polars_df` notes now say so and `tests/training/test_polars_pandas_view_null_column_dtypes.py` pins the outcome; keeping the integer dtype would need pandas nullable extension types, which the zero-copy path deliberately avoids |
| **RESOLVED** | W22REP-1 | Medium | multiclass SHAP panels ranked by the last class only |
| **RESOLVED** | W22REP-2 | Low | the leaderboard tie warning compared the top-two gap with an absolute 5e-3; it is now 0.5% of the largest score magnitude, so RMSE-scale and 1e-4-scale metrics tie sensibly (`tests/reporting/test_leaderboard_tie_threshold_scales_with_the_metric.py`) |

## Not covered by this wave

`_shap_proxy_catboost`, `_shap_proxy_cluster`, `_shap_proxy_cluster_su` / `_joint`, `_shap_proxy_explain`, the Faith / interaction / GPU / gradient / heuristics modules, the LightGBM and CatBoost extractors and the CPU / GPU parity of the interaction kernels; `shap_proxied_fs` `_shap_proxied_fit_interactions` / `_residual` / `_resolvers` and the refine / core-stability / paired-parsimony revalidate modules; `core` `binning`, `ewma`, `matrix_seriation` and the other core modules; the other `competition` modules; `reporting` `split_comparison`, `diagnostics_dispatch` and the plotly renderers; the `info_theory` modules, `_gpu_resident_*`, `_batch_pair_mi_cuda_kernels`, the `_numba_utils_*` modules and five of the seven noise-gate files.

## Implemented

| ID | Regression test |
|---|---|
| W22SHP-3 | `tests/feature_selection/shap_proxied/test_report_slice_small_fraction.py` |
| W22SHP-5 | `tests/feature_selection/shap_proxied/test_shap_proxy_preflight_class_balance_multiclass.py` |
| W22SHK-1 | `tests/feature_selection/shap_proxied/test_explain_cache_key_sees_interior_row_order.py` |
| W22KRN-2 | `tests/feature_selection/test_wave22_binner_guards.py` (W22KRN-1 is the same one-line guard as its sibling paths and has no separate test) |
| W22COR-1 | `tests/core/test_topk_by_partition_unsigned_and_lists.py` |
| W22COR-2 | `tests/core/test_ensure_no_infinity_pd_nullable_float.py` |
| W22REP-1 | `tests/reporting/test_shap_panels_multiclass_feature_ranking.py` |
| W22SHP-1 | `tests/feature_selection/shap_proxied/test_precomputed_su_only_when_computed_on_search_rows.py` (the slow MRMR-to-ShapProxiedFS pipeline test is left to CI) |
| W22SHP-4 | `tests/feature_selection/shap_proxied/test_fold_vote_weights.py` |
| W22COR-3 | `tests/inference/test_native_gpu_shap_matches_the_served_model.py` (kwargs helper; the CUDA call itself needs a GPU and is covered by CI where one exists) |
| W22SHP-2 | `tests/feature_selection/shap_proxied/test_report_slice_small_fraction.py::test_grouped_report_slice_shares_no_group_with_the_rest_of_the_holdout` |
