# full audit 2026-11-16 (non-MRMR) -- master tracker

Twelve read-only reports by Sonnet 5.5 Low agents, one bug class each, MRMR excluded. Reports are in this folder (01 to 12). Method: grep-driven sweeps plus reading the hits; only targeted unit tests were allowed and none were run by the agents, so every TODO is still unverified by a run.

Statuses: RESOLVED (fixed, a test pins it), FUTURE (deferred with a trigger), DOC (documented, no code change), REJECTED (declined, reason given), TODO (open, fix pending, regression test named).

## Summary

| Report | RESOLVED | FUTURE | DOC | REJECTED | TODO | Coverage caveat |
|---|---|---|---|---|---|---|
| 01_security | 8 | 0 | 4 | 3 | 0 | footer in the report says RESOLVED 7; table has 8, table wins |
| 02_silent_correctness | 0 | 1 | 0 | 8 | 0 | shallow; ~190 `except: pass` sites unread (S-9) |
| 03_leakage_methodology | 2 | 1 | 3 | 0 | 1 | partial, several modules unread |
| 04_numerical_stability | 1 | 0 | 0 | 4 | 2 | grep-driven; softmax and float32 paths not traced |
| 05_concurrency_state | 1 | 1 | 2 | 3 | 2 | shallow; locked caches not fully traced |
| 06_memory_resources | 5 | 2 | 0 | 1 | 2 | grep-driven |
| 07_performance | 0 | 0 | 1 | 4 | 1 | shallow grep sweep of row loops |
| 08_api_contracts | 0 | 1 | 3 | 3 | 4 | one runtime check (clone+fit EarlyStoppingWrapper) |
| 09_test_quality | 0 | 2 | 2 | 3 | 8 | pattern greps only |
| 10_ci_supply_chain | 0 | 2 | 3 | 3 | 6 | configs only; ci.yml timeout count approximate |
| 11_persistence_reporting | 0 | 1 | 1 | 1 | 4 | writer sites and save path only; load paths not reviewed |
| 12_gpu_native | 0 | 1 | 1 | 3 | 4 | six modules read; other GPU modules only grepped |

TODO total: 34. Nothing is fixed yet; no code was changed in this audit.

## Open TODO rows (each needs a fix and a regression test before it is closed)

| Status and ID | Location | Finding | Proposed regression test |
|---|---|---|---|
| **RESOLVED** L1 (03) | training/composite/stacking_multi_stage.py:169-171 | fit takes groups and time_aware and forwards them to the stage-1 OOF split (previously plain KFold, leaking group structure) | tests/training/composite/test_multi_stage_stacker_forwards_groups.py 1/1 (fails on old code: fit has no groups kwarg); sample_weight and composite selection tests 3/3 |
| **RESOLVED** NS-01 (04) | data_valuation/_knn_shapley.py | standardisation guarded only sigma == 0.0; ulp-noise column changed Shapley values by 0.0107 | test_knn_shapley_ulp_noise_column.py::test_ulp_noise_column_does_not_change_standardised_values |
| **RESOLVED** NS-02 (04) | data_valuation/_training_weight_adapter.py:67 | same exact-zero sigma guard; shared helper _column_standardise_stats now used by both | same test (shared helper); test_biz_val_training_weight_adapter.py passes (3/3) |
| **RESOLVED** C3 (05) | metrics/_gpu_metrics.py | is_gpu_metrics_available runs the probe once under a lock with a re-check; the probe body is unchanged | tests/metrics/test_gpu_metrics_probe_runs_once_under_threads.py 1/1 (fails on old code: the monkeypatched probe is not reached); gpu probe tests 9/9 |
| **RESOLVED** C6 (05) | utils/log_sanitize.py | install_log_sanitizer's check-then-set now runs under a lock. Note: the race is timing-dependent, so the concurrent test may pass on old code too | tests/utils/test_install_log_sanitizer_concurrent.py 1/1; tests/utils/test_log_sanitize.py 8/8 |
| **RESOLVED** M1 (06) | competition/leak_scan.py | with a target, the frame is no longer copied: the target column is scored separately and stacked in front of the mask and numeric values | tests/competition/test_leak_scan_target_does_not_copy_frame.py 1/1 (fails on old code: DataFrame.copy is called); test_leak_scan_shift_larger_than_the_frame_width.py 1/1 |
| **RESOLVED** M2 (06) | metrics/_fairness_metrics.py | redundant `.copy()` before `.replace` removed. Note: the copy is not observable in results, so the pinning test passes on old code too; the fix is a perf-only change | test_fairness_merge_lowpop_keeps_input_intact.py + test_fairness_subgroups_keep_rare_bins_when_not_excluded.py 3/3 |
| **RESOLVED** P1 (07) | feature_engineering/transformer/apriori_itemsets.py:54 | itemset lifts use a plain loop over the itemsets column and an in-place AND per column (no fancy-index copy per itemset); same values as the reference | tests/feature_engineering/test_apriori_itemset_lifts_match_reference.py 1/1 (passes on old code too: perf-only change); transformer -k apriori 5/5 |
| **DOC** API-01 (08) | estimators/early_stopping.py:95,141 | max_iter=None is accepted by the constructor (sklearn clone convention) and rejected at fit with a ValueError; the class docstring now says max_iter must be set | existing fit-time rejection covered; no code change |
| **RESOLVED** API-02 (08) | feature_engineering/transformer/ib_baseline_codes.py:84-85 | n_bits other than the supported default 3 raises NotImplementedError instead of being ignored | tests/feature_engineering/transformer/test_transformer_unimplemented_params_fail_loudly.py::test_ib_baseline_rejects_a_non_default_n_bits; test_fe_transformer_b_fixes.py -k ib_baseline 1/1 |
| **RESOLVED** API-03 (08) | feature_engineering/transformer/apriori_itemsets.py:83 | standardize=False raises NotImplementedError; the default True path is unchanged (it has no effect, documented as such in the summary) | test_apriori_rejects_standardize_false |
| **RESOLVED** API-04 (08) | feature_engineering/transformer/mdl_binning_pairwise.py:159 | top_k_pairs other than the default raises NotImplementedError; standardize is documented as having no effect (existing caller test_mdl_binning_combo_count_vectorized passes standardize=False, so raising there would break a legitimate call) | test_mdl_binning_rejects_a_non_default_top_k_pairs; test_mdl_binning_combo_count_vectorized.py 9 passed with the new test file |
| **RESOLVED** TQ-01 (09) | tests/feature_selection/wrappers/rfecv/test_rfecv_fast_perm_scorer_identity.py | `or True` replaced with a real repeat-call equality | file 11/11 |
| **RESOLVED** TQ-02 (09) | tests/feature_selection/wrappers/test_wrappers_bug_fixes.py | `or True` replaced with `assert not hasattr(outer, "coef_")`; test passes | test_wrappers_bug_fixes.py -k outer 1/1 |
| **RESOLVED** TQ-03 (09) | tests/feature_engineering/test_coverage_fill.py | `or True` removed; asserts the rsi5 output name; test passes | test_apply_ta_indicator_with_window 1/1 |
| **RESOLVED** TQ-04 (09) | tests/preprocessing/test_biz_val_align_feature_direction.py | `or True` on a list replaced with a finiteness check | align/f10 selection 8/8 |
| **RESOLVED** TQ-05 (09) | tests/reporting/test_renderer_divergences_and_dead_paths.py | `!= viridis or True` (the default is CALIBRATION, not viridis); replaced with the override-cleared assertion | test_clearing_the_override_still_works 1/1 |
| **RESOLVED** TQ-06 (09) | tests/test_x_ml_correctness_meta_fixes.py | `or True` removed; asserts the attribute | f10 selection 8/8 |
| **RESOLVED** TQ-08 (09) | tests/feature_selection/fe/provenance/test_engineered_recipes.py | `assert r == r` replaced with value equality against an identical recipe and inequality against a different binary op | test_recipes_are_hashable 1/1 (passes on old code too: the reflexive check was the weak part) |
| **RESOLVED** TQ-10 (09) | tests/calibration/test_calibration_fixes.py | added finiteness assertion to the NaN-AUC fallback test | test_f5_n_iterations_returns_ndarray_when_score_is_nan_every_draw 1/1 |
| **RESOLVED** CI-05 (10) | .github/workflows/mypy-full.yml, black-filtered.yml | added a top-level `permissions: contents: read` block (job-level already present) | YAML parses; actionlint runs in CI on push |
| **REJECTED** CI-06 (10) | .github/workflows/mypy-full.yml:36-37 | blocking mypy gate has no timeout-minutes | Not added on the caller job. A job that calls a reusable workflow (`uses:`) does not take `timeout-minutes` in GitHub's workflow schema (unverified locally: the callee is not in this checkout, and no actionlint run was possible). The bound lives in the callee at the pinned SHA: `fingoldo/py-ci-shared` `mypy-full.yml` sets `timeout-minutes: 45` on its job (fetched via `gh api` at 5264e5d). |
| **REJECTED** CI-07 (10) | .github/workflows/docs.yml:33-34 | docs caller job without timeout-minutes | Same reason as CI-06. The callee `py-ci-shared` `docs.yml` at 5264e5d sets `timeout-minutes: 15` and `10` on its jobs, so the run is already bounded. |
| **REJECTED** CI-08 (10) | .github/workflows/black-filtered.yml:29-32 | reusable-workflow caller without timeout-minutes | Same reason as CI-06 (caller of a reusable workflow; the bound belongs to the callee at the pinned SHA). |
| **REJECTED** CI-09 (10) | .github/workflows/ci.yml | some jobs without timeout-minutes (count approximate) | Counted 6 jobs without the key in ci.yml (ruff-blocking, lint-blocking, lint-advisory, tests-ruff-blocking, tests-black-filtered, tests-lint-blocking). All 6 are `uses:` callers of reusable workflows, same reason as CI-06: no caller-level timeout is accepted, and the callee owns the bound. Jobs with steps all carry a timeout. |
| **REJECTED** CI-12 (10) | .pre-commit-config.yaml:126,200 | pinning the ruff-pre-commit revs by SHA would break tests/test_meta/test_ruff_pin_consistency.py, which parses the revs as version tags to check they agree with the ruff pin; the tag pin is the project's chosen form for this check | test_ruff_pin_consistency passes with the tag pins kept |
| **RESOLVED** P11-01 (11) | reporting/report_html.py:258 | combined report is written to a .partial file and swapped in with os.replace; a failed write keeps the previous report | tests/reporting/test_report_html_write_is_atomic.py 1/1 (fails on old code: the open('w') truncates the target); test_report_html.py 21/21 |
| **RESOLVED** P11-02 (11) | reporting/renderers/plotly.py (json branch) | json export goes through a temp file and os.replace | tests/reporting/test_plotly_json_write_is_atomic.py 1/1 (fails on old code: open('w') truncates the target) |
| **RESOLVED** (partial) P11-03 (11) | reporting/renderers/_kaleido.py:248 | fallback now calls ensure_parent_dir before writing. Note: the directory was already created by the first ensure_parent_dir call, so the test passes on old code too; the write itself is still not atomic (low) | tests/reporting/test_kaleido_fallback_creates_parent_dir.py 1/1 |
| **RESOLVED** P11-04 (11) | training/_io_save.py:453-455 | verbose=0 silently flipped lean (drops preds/probs/target); the flip is now always logged as a warning | tests/training/test_save_lean_flip_warns_when_quiet.py::test_lean_pre_flip_logs_warning_at_verbose_zero; test_io_save_size_sensor_and_lightning_bloat.py 9/9 |
| **RESOLVED** G1 (12) | feature_engineering/transformer/_kernels_cupy.py:256-261 | rff_matmul_cupy clamps batch_rows so the double-buffered device input and output fit in half the free VRAM (helper _clamp_rff_batch_rows) | tests/feature_engineering/transformer/test_rff_batch_rows_fit_free_vram.py 2/2 (fails on old code: helper absent); test_rff_cupy_batch_clamp.py + test_gpu_memory_released_on_failure.py 10/10 |
| **RESOLVED** G2 (12) | feature_engineering/transformer/_kernels_cupy.py:341-342 | a resident k_proj/y_train device array of a dtype other than float32 raises TypeError before any launch (the kernel reads float* and would reinterpret, not convert) | tests/feature_engineering/transformer/test_row_attention_stage4_empty_launch.py::test_resident_float64_bank_is_rejected_not_reinterpreted 3/3 with the other two |
| **RESOLVED** G3 (12) | feature_engineering/transformer/_kernels_cupy.py:320,352-354 | zero queries return before any launch; zero neighbours raise a clear ValueError instead of an invalid CUDA launch configuration | tests/feature_engineering/transformer/test_row_attention_stage4_empty_launch.py 2/2; test_hard_row_attention.py 8/8. Note: the zero-query test was not checked against the old code (it would launch a zero-sized grid there), so its failure-on-old-code is unmeasured |
| **RESOLVED** G5 (12) | calibration/ensembling.py | the two cupy fallback warnings go through the project's log_throttle with stable keys (per call site), so repeated fallbacks cannot flood the log | tests/calibration/test_ensembling_cupy_fallback_warns_once.py 1/1 (fails on old code: eight warnings instead of five plus a notice) |

## Plan

- One fix per /loop tick, in the order: correctness and silent-wrong-result first (NS-01, NS-02, G2, P11-04, L1, API-02/03/04, TQ-01..03), then the rest.
- Each fix: failing test first (verified to fail on current code), then the fix, then the test passes. Targeted test files only, never a suite.
- Commit with `-o` and the paths of that fix only. Push only after CI on master is green and no other session is pushing; the user asked to push only when CI is not pending.
- Update this tracker row to RESOLVED with the test name when a fix lands.
