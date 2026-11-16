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

| ID | Sev | Location | Finding | Proposed regression test |
|---|---|---|---|---|
| L1 (03) | med | training/composite/stacking_multi_stage.py:169-171 | MultiStageMetaFeatureStacker builds stage-1 OOF with plain KFold; fit() has no groups/time_aware | test_multi_stage_stacker_oof_respects_groups_no_group_in_both_train_and_val_fold |
| NS-01 (04) | RESOLVED | data_valuation/_knn_shapley.py | standardisation guarded only sigma == 0.0; ulp-noise column changed Shapley values by 0.0107 | test_knn_shapley_ulp_noise_column.py::test_ulp_noise_column_does_not_change_standardised_values |
| NS-02 (04) | RESOLVED | data_valuation/_training_weight_adapter.py:67 | same exact-zero sigma guard; shared helper _column_standardise_stats now used by both | same test (shared helper); test_biz_val_training_weight_adapter.py passes (3/3) |
| C3 (05) | RESOLVED | metrics/_gpu_metrics.py | is_gpu_metrics_available runs the probe once under a lock with a re-check; the probe body is unchanged | tests/metrics/test_gpu_metrics_probe_runs_once_under_threads.py 1/1 (fails on old code: the monkeypatched probe is not reached); gpu probe tests 9/9 |
| C6 (05) | RESOLVED | utils/log_sanitize.py | install_log_sanitizer's check-then-set now runs under a lock. Note: the race is timing-dependent, so the concurrent test may pass on old code too | tests/utils/test_install_log_sanitizer_concurrent.py 1/1; tests/utils/test_log_sanitize.py 8/8 |
| M1 (06) | med | competition/leak_scan.py:119 | full-frame df.copy() just to insert __target__; only a mask is needed | test_leak_scan_does_not_copy_full_frame |
| M2 (06) | RESOLVED | metrics/_fairness_metrics.py | redundant `.copy()` before `.replace` removed. Note: the copy is not observable in results, so the pinning test passes on old code too; the fix is a perf-only change | test_fairness_merge_lowpop_keeps_input_intact.py + test_fairness_subgroups_keep_rare_bins_when_not_excluded.py 3/3 |
| P1 (07) | med | feature_engineering/transformer/apriori_itemsets.py:54 | per-itemset iterrows loop with fancy-index np.all | test_apriori_itemset_lifts_matches_reference_after_vectorization |
| API-01 (08) | med | estimators/early_stopping.py:95, 141 | max_iter=None default cannot be fitted | test_early_stopping_wrapper_default_max_iter_fits_or_is_rejected_at_init |
| API-02 (08) | med | feature_engineering/transformer/ib_baseline_codes.py:84-85 | n_bits accepted and never read | test_ib_baseline_codes_n_bits_is_used_or_removed |
| API-03 (08) | med | feature_engineering/transformer/apriori_itemsets.py:83 | standardize accepted and never referenced | test_apriori_itemsets_standardize_not_silently_ignored |
| API-04 (08) | med | feature_engineering/transformer/mdl_binning_pairwise.py:159 | top_k_pairs and standardize accepted, unused | test_mdl_binning_pairwise_unused_params_documented_or_removed |
| TQ-01 (09) | RESOLVED | tests/feature_selection/wrappers/rfecv/test_rfecv_fast_perm_scorer_identity.py | `or True` replaced with a real repeat-call equality | file 11/11 |
| TQ-02 (09) | RESOLVED | tests/feature_selection/wrappers/test_wrappers_bug_fixes.py | `or True` replaced with `assert not hasattr(outer, "coef_")`; test passes | test_wrappers_bug_fixes.py -k outer 1/1 |
| TQ-03 (09) | RESOLVED | tests/feature_engineering/test_coverage_fill.py | `or True` removed; asserts the rsi5 output name; test passes | test_apply_ta_indicator_with_window 1/1 |
| TQ-04 (09) | RESOLVED | tests/preprocessing/test_biz_val_align_feature_direction.py | `or True` on a list replaced with a finiteness check | align/f10 selection 8/8 |
| TQ-05 (09) | RESOLVED | tests/reporting/test_renderer_divergences_and_dead_paths.py | `!= viridis or True` (the default is CALIBRATION, not viridis); replaced with the override-cleared assertion | test_clearing_the_override_still_works 1/1 |
| TQ-06 (09) | RESOLVED | tests/test_x_ml_correctness_meta_fixes.py | `or True` removed; asserts the attribute | f10 selection 8/8 |
| TQ-08 (09) | low | tests/feature_selection/fe/provenance/test_engineered_recipes.py:318 | `assert r == r` | test_engineered_recipe_equality_distinguishes_extr... (name to finalise with the fix) |
| TQ-10 (09) | RESOLVED | tests/calibration/test_calibration_fixes.py | added finiteness assertion to the NaN-AUC fallback test | test_f5_n_iterations_returns_ndarray_when_score_is_nan_every_draw 1/1 |
| CI-05 (10) | med | .github/workflows/black-filtered.yml, mypy-full.yml | no workflow-level permissions block in mypy-full | (CI config; verify by reading the workflow; no unit test) |
| CI-06 (10) | med | .github/workflows/mypy-full.yml:36-37 | blocking mypy gate has no timeout-minutes | (CI config) |
| CI-07 (10) | med | .github/workflows/docs.yml:33-34 | docs caller job without timeout-minutes | (CI config) |
| CI-08 (10) | low | .github/workflows/black-filtered.yml:29-32 | reusable-workflow caller without timeout-minutes | (CI config) |
| CI-09 (10) | low | .github/workflows/ci.yml | some jobs without timeout-minutes (count approximate) | (CI config; verify each job) |
| CI-12 (10) | low | .pre-commit-config.yaml:126, 200 | ruff-pre-commit pinned by tag, not SHA | (config; pin to SHA) |
| P11-01 (11) | RESOLVED | reporting/report_html.py:258 | combined report is written to a .partial file and swapped in with os.replace; a failed write keeps the previous report | tests/reporting/test_report_html_write_is_atomic.py 1/1 (fails on old code: the open('w') truncates the target); test_report_html.py 21/21 |
| P11-02 (11) | low | reporting/renderers/plotly.py:424 | plotly json branch non-atomic | test_plotly_json_write_is_atomic |
| P11-03 (11) | low | reporting/renderers/_kaleido.py:248 | calc_fig_sync fallback writes in place, no ensure_parent_dir | test_kaleido_fallback_creates_parent_dir |
| P11-04 (11) | RESOLVED | training/_io_save.py:453-455 | verbose=0 silently flipped lean (drops preds/probs/target); the flip is now always logged as a warning | tests/training/test_save_lean_flip_warns_when_quiet.py::test_lean_pre_flip_logs_warning_at_verbose_zero; test_io_save_size_sensor_and_lightning_bloat.py 9/9 |
| G1 (12) | med | feature_engineering/transformer/_kernels_cupy.py:256-261 | rff_matmul_cupy allocates without VRAM budget check | test_rff_matmul_cupy_refuses_or_chunks_over_vram_budget |
| G2 (12) | med | _kernels_cupy.py:341-342 | float64 device bank read as float32 by float* kernel | test_row_attention_rejects_or_converts_float64_device_bank |
| G3 (12) | low | _kernels_cupy.py:320, 352-354 | K==0 and n_queries==0 launches unguarded | test_row_attention_empty_k_and_queries_do_not_launch |
| G5 (12) | low | calibration/ensembling.py:161-162, 181-182 | repeated cupy-fallback warning floods log | test_ensembling_cupy_fallback_warns_once_per_process |

## Plan

- One fix per /loop tick, in the order: correctness and silent-wrong-result first (NS-01, NS-02, G2, P11-04, L1, API-02/03/04, TQ-01..03), then the rest.
- Each fix: failing test first (verified to fail on current code), then the fix, then the test passes. Targeted test files only, never a suite.
- Commit with `-o` and the paths of that fix only. Push only after CI on master is green and no other session is pushing; the user asked to push only when CI is not pending.
- Update this tracker row to RESOLVED with the test name when a fix lands.
