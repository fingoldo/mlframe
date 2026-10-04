# BorutaShap bare-name aliases removed

BorutaShap kept its fitted/working state under trailing-underscore names and exposed the 28 historical bare names as read/write aliases. The owner confirmed
there are no external consumers, so every use was migrated to the underscore names and the aliases were deleted.

## Removed

- `_LegacyAlias` (the data descriptor), `install_legacy_aliases(cls)` and its call at the end of `boruta_shap/__init__.py`.
- `LEGACY_ALIASED_ATTRS` as an alias list. The tuple is kept as `PRE_RENAME_ATTRS` (same 28 names) because `__setstate__` and `SCRATCH_FITTED_ATTRS` use it.
- The vulture whitelist entry for `_LegacyAlias.__get__` (`objtype`).
- Kept: `__setstate__` migration of bare keys found in pickles from earlier releases, and the lean pickle (`TRANSIENT_ATTRS`).

## Renamed (receiver proven to be a BorutaShap instance or the wrapper that exposes its report contract)

Attribute accesses (AST based, receiver checked per file), 129 in total:

| file | count |
|---|---|
| tests/feature_selection/boruta_shap/test_boruta_shap_object_cat_and_axis_fixes.py | 22 |
| tests/feature_selection/boruta_shap/test_boruta_shap_margin_gated_trial_stop.py | 14 |
| tests/feature_selection/boruta_shap/test_boruta_shap_remove_features_single_drop.py | 11 |
| tests/feature_selection/boruta_shap/test_tentative_rough_fix_excludes_phantom_row.py | 9 |
| tests/feature_selection/regression/test_regression_shap_train_only_guard.py | 8 |
| tests/feature_selection/boruta_shap/test_tentative_rough_fix_clears_tentative.py | 8 |
| tests/feature_selection/cv_policy/test_cv_policy_selector_wiring.py | 7 |
| tests/feature_selection/test_feature_selection_nonmrmr_fixes.py | 7 |
| tests/feature_selection/boruta_shap/test_biz_val_filters_boruta_shap.py | 5 |
| tests/feature_selection/boruta_shap/test_boruta_newaxes_fixes.py | 5 |
| tests/feature_selection/boruta_shap/test_boruta_shap_subsample_stability.py | 5 |
| src/mlframe/feature_selection/_benchmarks/bench_boruta_early_stop_tentative.py | 5 |
| src/mlframe/feature_selection/_benchmarks/_bench_shared.py | 4 |
| tests/training/test_audit_dict_duplicate_keys.py | 3 |
| two each: test_boruta_shap_logger_warn_not_print, test_boruta_shap_shadow_fast_path, test_boruta_shap_shadow_tie_unbiased, round4_adaptive_n_trials_bench | 8 |
| one each: bench_boruta_shap_medoid, bench_shadow_min_pad_narrow_frames, round4_prior_protected_rfecv_bench, test_biz_val_wrappers_boruta_shap, test_boruta_shap_permutation_driver, test_explain_basis_is_the_fitted_slice, test_composite_x_cluster_reduced_selectors, test_correlated_features_real_rfecv_dropin | 8 |

String and keyword sites (getattr/hasattr names, stub objects, comments and docstrings that named the attributes):

- `boruta_shap/_fit_explain.py` (`getattr(sub, "accepted_")`), `training/core/_phase_train_one_target_helpers.py` (report builder reads `history_x_`,
  `accepted_`, `rejected_`, `tentative_`, `all_columns_`), `feature_selection/compare_selectors.py` (`accepted_` fallback accessor),
  `filters/correlated_features.py` (inner `accepted_`, and the wrapper's report-contract property is now `accepted_`).
- Benchmarks: `fs_hybrid/_arms.py`, `round4_synergy_combine_bench.py`, `round4_tentative_to_cmi_bench.py`, `round4_adaptive_n_trials_bench.py`.
- Tests with stub objects or getattr strings: `test_boruta_find_sample_terminates.py` (`stub.X_`), `test_boruta_shap_auto_dispatch.py`,
  `test_boruta_shap_permutation_driver.py`, `test_gini_accepts_unfitted_boosters.py`, `test_compare_selectors.py` (`_AcceptedSelector.accepted_`),
  `test_multiple_testing_corrections.py` (SimpleNamespace stand-in for `test_features`), `test_metadata_feature_selection_report_observability.py` (fake class).
- Deliberately not changed: `ace.py`'s own `accepted`, the optional third party `BorutaShap` package used in `wrappers/_importance_methods.py`, dict keys and
  string labels ("accepted"/"tentative"/"rejected" decisions, golden dicts in the profile benches), pandas/shap `columns`/`shap_values`, local X/y variables.
- Docs: no README or docs file named the bare attributes. CHANGELOG got a BREAKING entry under Unreleased.

## Tests

`tests/feature_selection/test_api_contract_estimator_protocol.py`: the alias assertions are replaced by a test that none of the 28 bare names is an attribute
(`vars` and `hasattr`), and the old-pickle test now builds a state keyed by every bare name and checks it loads into the underscore names with no bare attribute left.

Runs at `-n 1 --no-cov --timeout=0`:

- boruta_shap suite plus regression_shap_train_only_guard, nonmrmr_fixes, cv_policy wiring, api_contract, memory working_set, multiple_testing_corrections,
  compare_selectors, report observability, dict_duplicate_keys, isinstance_duck_typing, composite discovery selectors, correlated_features rfecv dropin:
  247 tests; 9 failed on the first pass (two loky worker deaths under memory pressure, one stub still reading `stub.X`, and files that another session
  reverted mid-run); after the re-apply the failed files pass (43 passed, then 7 passed for the permutation and auto-dispatch fits that had hit the worker deaths).
- Meta gates `tests/test_meta -k "setstate or fitted or long_functions or 1k_loc or c901 or fail_open or drifted or nondiscriminating or source_text or floorless
  or optional_numbers or uncalled or pydoclint or dead or vulture or docstring"`: 63 passed, 1 failed (`test_long_functions_do_not_grow`: the baseline entry for
  `_mrmr_fit_impl/_friend_graph_and_redundancy/_group1.py::_prefe_raw_sole_parent_pass` is stale after the concurrent MRMR work; unrelated to this change).
- mypy on `_estimator_protocol.py`, `boruta_shap/__init__.py`, `_fit_explain.py`, `compare_selectors.py`, `correlated_features.py`,
  `_phase_train_one_target_helpers.py`: no issues.
