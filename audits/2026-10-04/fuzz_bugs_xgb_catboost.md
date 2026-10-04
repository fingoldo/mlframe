# Fuzz bugs: xgb feature-count mismatch (c0001) and catboost Categorical (c0004)

Selection: `enumerate_combos(target=150, master_seed=20260422)` (FUZZ_SEED default), index 1 and 4; run with
`MLFRAME_FUZZ_PERF_MODE=1 pytest tests/training/fuzz/test_fuzz_suite.py -k "c0001 or c0004" --run-fuzz -n 0`
(ids `c0001_cd5ab15d-xgb-pandas-n1000`, `c0004_8c6d4030-linear-pl_utf8-n300000`; perf mode rescales n_rows to 1000, the id keeps the original size).

## Bug A: c0001 `Feature shape mismatch, expected: 25, got 20`
Reproduction: c0001 fails at the test-split metrics (MultiOutputClassifier over xgb), xgb_shim.py:817 -> xgboost inplace_predict.
Debug of the predict frame: train booster has 25 features; the failing frame lacks the 5 `xsnn_*` columns (xsnn_num_0_mean/std, xsnn_num_1_mean/std, xsnn_distance_ratio).
Root cause chain: the combo enables cross_sectional_neighbors (snapshot col `cat_0`, `inject_test_drift='unseen_category'`) ->
`core/_phase_helpers_fit_pipeline.py:443` -> `pipeline/_cross_sectional_composite_fe.py` `apply_cross_sectional_composite_fe`: a split with `nunique(snapshot) < 2`
(test rows all in one unseen category) hit `out[split_name] = df; continue`, silently leaving that split without the 5 columns while train/val had them.
xgb_shim is not at fault (it correctly rejects a narrower frame). A real user with a constant/unseen snapshot key in test would hit this.
Fix: a single-snapshot split is no longer skipped; `split_k = max(min(effective_k, n_snapshots-1), 1)` and `compute_cross_sectional_neighbor_features` already
yields NaN aggregates and distance_ratio 1.0 when no real neighbor exists, so the schema matches train.
Regression test: `tests/training/pipeline/test_cross_sectional_composite_fe.py::test_apply_cross_sectional_composite_fe_single_snapshot_split_keeps_train_schema`
(old code returns the test frame without xsnn columns). Result: c0001 passes in perf mode.

## Bug B: c0004 `CatBoostError: Unsupported data type Categorical for a numerical feature column`
Reproduction: c0004 in perf mode with `-n 0` (under xdist the worker died; after the first partial fix the same path produced a native access violation inside CatBoost Pool init).
Root cause chain: `_trainer_train_and_evaluate.py:573` -> `_trainer_train_and_evaluate_parts.py:_train_and_evaluate_score_ensemble_can_pick` ->
`trainer.py:_compute_oof_preds` clones the deployed CatBoost and calls `cross_val_predict(estimator, train_df, ...)`. The deployed model gets
`cat_features` only through `.fit(**fit_params)` (filtered by `_filter_categorical_features`), which cross_val_predict's per-fold `.fit(X, y)` never sees, so the
clone read the pandas Categorical columns (cat_0..cat_2) as numeric. The exception type (CatBoostError) is outside the `except (ValueError, TypeError, RuntimeError, NotImplementedError)`,
so it aborted the whole suite. Any user training CatBoost with categorical columns and OOF enabled hits this, not only polars_utf8 input.
Fix: `_oof_fold_policy.apply_feature_roles_to_clone` copies `cat_features`/`text_features`/`embedding_features` from the deployed `fit_params` onto the clone
(only for estimators whose `__init__` accepts them, only columns still in the frame); `fit_params` threaded `_train_and_evaluate_model` -> `_train_and_evaluate_score_ensemble_can_pick`
-> `_compute_oof_preds(fit_params=...)`. The clone cap path (`cap_early_stopping_clone`) is unchanged.
Regression test: `tests/training/test_oof_fold_policy.py::test_catboost_oof_clone_receives_cat_features_from_fit_params`
(old code: CatBoostError "column 'cat_0' has dtype 'category' but is not in cat_features"). Result: c0004 passes in perf mode (275 s).

## Gates
test_oof_fold_policy + test_cross_sectional_composite_fe: 12 passed; tests/test_meta selection: 12 passed; ruff and mypy clean on edited sources.
Not run: the full-size n=300000 c0004 (perf-mode scale only).
