# full audit 2026-09-20 -- master tracker

Ten read-only reports over `src/mlframe/`, one per area, produced by one agent each. `src/mlframe/training/composite/`
is deliberately absent: the open [composite audit 2026-09-19](../composite_audit_2026-09-19/_TRACKER.md) owns it, and
this wave was scoped around it so the two do not fix the same code twice.

Reports: [training_core.md](training_core.md) (suite orchestration, splits, booster dataset reuse),
[feature_selection.md](feature_selection.md), [feature_engineering.md](feature_engineering.md) (plus preprocessing),
[metrics.md](metrics.md) (metrics and calibration), [predict_persistence.md](predict_persistence.md) (serving path and
artifact round-trip), [ensembling_models.md](ensembling_models.md) (model zoo, blends, thresholds, votenrank),
[evaluation_reporting.md](evaluation_reporting.md) (diagnostic verdicts, not chart aesthetics),
[performance.md](performance.md) (measured, not guessed), [concurrency_resources.md](concurrency_resources.md),
[config_contracts.md](config_contracts.md).

Statuses: **RESOLVED** (fixed in code; the note names the test that pins it), **PARTIAL** (part fixed; the note says
what remains), **TODO** (open), **REJECTED** (measured and declined; evidence in the note), **NOT A DEFECT** (the
claimed behaviour does not occur, or was already fixed before implementation started; the note names the commit).
Each report's own `- **Disposition**:` line is updated together with its row here.

## Summary

| File | Findings | RESOLVED | PARTIAL | TODO | REJECTED | NOT A DEFECT |
|---|---|---|---|---|---|---|
| `training_core.md` | 13 | 12 | 0 | 0 | 0 | 1 |
| `feature_selection.md` | 21 | 19 | 0 | 0 | 2 | 0 |
| `feature_engineering.md` | 14 | 14 | 0 | 0 | 0 | 0 |
| `metrics.md` | 18 | 16 | 0 | 0 | 2 | 0 |
| `predict_persistence.md` | 18 | 18 | 0 | 0 | 0 | 0 |
| `ensembling_models.md` | 14 | 14 | 0 | 0 | 0 | 0 |
| `evaluation_reporting.md` | 16 | 16 | 0 | 0 | 0 | 0 |
| `performance.md` | 7 | 7 | 0 | 0 | 0 | 0 |
| `concurrency_resources.md` | 12 | 12 | 0 | 0 | 0 | 0 |
| `config_contracts.md` | 32 | 30 | 0 | 0 | 0 | 2 |
| **Total** | **165** | **158** | **0** | **0** | **4** | **3** |

Queued after every finding above is implemented: [complexity_refactor.md](complexity_refactor.md) - 193 production
functions over a McCabe complexity of 25 (threshold confirmed by the project owner) plus the blocking ratchet gate
(CX-GATE). Counted separately from the 165 findings of this wave.

## Per-report status

One row per report, status first so a count can read it. The per-finding dispositions live in each report's own
table; this rolls them up to the coarsest status that is true of the whole report (a report is **TODO** until at
least one of its findings moves).

| Status | Report | Findings | Area |
|---|---|---|---|
| **CLOSED** | [training_core.md](training_core.md) | 13 | suite orchestration, splits, booster dataset reuse (12 fixed, TRC-01 not a defect) |
| **CLOSED** | [feature_selection.md](feature_selection.md) | 21 | feature selection (19 fixed; FS-07 and FS-16 rejected as defaults, measured, kept opt-in) |
| **CLOSED** | [feature_engineering.md](feature_engineering.md) | 14 | feature engineering and preprocessing (all 14 fixed) |
| **CLOSED** | [metrics.md](metrics.md) | 18 | metrics and calibration (16 fixed incl. MET-18 found during implementation; MET-08 and MET-16 rejected) |
| **CLOSED** | [predict_persistence.md](predict_persistence.md) | 18 | serving path and artifact round-trip (all 18 fixed) |
| **CLOSED** | [ensembling_models.md](ensembling_models.md) | 14 | model zoo, blends, thresholds, votenrank (all 14 fixed) |
| **CLOSED** | [evaluation_reporting.md](evaluation_reporting.md) | 16 | diagnostic verdicts (all 16 fixed) |
| **CLOSED** | [performance.md](performance.md) | 7 | measured performance (all 7 fixed) |
| **CLOSED** | [concurrency_resources.md](concurrency_resources.md) | 12 | concurrency and resources (all 12 fixed) |
| **CLOSED** | [config_contracts.md](config_contracts.md) | 32 | config contracts (30 fixed, CFG-05 and CFG-10 not defects) |

## What the wave is about

Three failure modes account for most of the list, and they cut across every area:

1. **Neutral substitution.** A value that could not be computed is replaced by one that wins: `0.0` into a signed
   importance array (`FS-02`, `FS-03`), a NaN loss term set to `0.0` in a lower-is-better metric (`MET-03`), a skip
   sentinel of `0` regardless of metric direction (`MET-04`), empty-input quantile losses returning `0.0` (`MET-11`),
   `0.0` pair-SU read as "maximally non-redundant" (`FS-12`), and five FE gates whose missing baseline makes the gate
   a no-op (`FS-19`, `FS-20`).
2. **Train/serve and train/report divergence.** State fitted on train never reaches predict (`FE-01`/`PRD-01`,
   `PRD-08`), blend weights and gate survivors are computed and then dropped (`ENS-01`, `ENS-02`, `ENS-03`), and the
   two predict entry points are each missing correctness steps the other has (`PRD-02`, `PRD-06`, `PRD-07`, `PRD-11`).
3. **Claims the code does not keep.** Three ICE paths advertise bit-exactness and return three different numbers
   (`MET-01`, `MET-02`), a memoisation that provably never hits is documented as the fix for a measured cost
   (`EVR-01`), a catalogue cites a test that does not exist (`EVR-05`), and roughly a dozen public config fields are
   validated and then ignored (`CFG-03`, `CFG-05`, `CFG-15`, `CFG-16`, `CFG-17`).

Highest-severity rows first: `MET-01` (P0: the metric driving early stopping is not the metric in the report, 38%
gap reproduced), then `ENS-01`/`ENS-04`, `PRD-02`/`PRD-04`/`PRD-05`, `CNC-01`, `TRC-01`/`TRC-02`, `FE-01`/`FE-02`,
`FS-01`/`FS-02`/`FS-03`, `EVR-02`/`EVR-03`, `CFG-01`, and the two import-cost items `PRF-01`/`PRF-02` (~25-30 s per
worker process and per notebook import, both import-graph edits).

## Per-report rows

Each report carries its own table with `| Sev | ID | Finding | Evidence (file:line + quote) | Failure scenario |
Suggested fix |`. Rows are dispositioned there and mirrored into the summary counts above as they are implemented.

## Found during implementation

Test failures met while verifying this wave that are NOT caused by it: each was re-run against the pre-wave source (`git archive 2769c85bc` on `PYTHONPATH`, current tests) and fails there with the same numbers. They are open and owed; none is deselected or marked.

| Test | Pre-wave result | Note |
|---|---|---|
| Default suite trained CatBoost without its categorical columns | failed on `origin/master` | FIXED here (found building TRC-10's end-to-end test): the default-on row-wise extension steps ran the numeric-only filter of `apply_preprocessing_extensions`, which DROPPED every string / categorical column, so they stayed listed in `cat_features` while no model saw them. Categorical-like columns now pass around the numeric steps and are re-attached at fit and at predict (`_mlframe_passthrough_columns_`); timedelta and other non-numeric leftovers are still dropped. Pinned by `tests/training/test_categoricals_reach_the_model.py` |
| Composite-FE columns replayed at predict were stripped before the model | failed on `origin/master` | FIXED here: the input validator allowed only raw input columns, so a replayed concat / target-encoding column was dropped as "extra" and the model trained on it could not predict. The fit phase records `metadata["composite_fe_emitted_columns"]` and the validator allows them. Same test file |
| A pandas `category` column promoted to text reached CatBoost as `category` | unreachable on master (the column was dropped) | FIXED here: text features are cast to strings after type detection (`_categorical_text_to_string`). Pinned by `test_high_cardinality_cat_column_trains_successfully` |
| `metadata["columns"]` on a polars run omitted the categorical columns the model is fitted on | wrong on master | FIXED here: the recorded schema lists the declared cat features. Pinned by `test_polars_and_pandas_paths_produce_close_metrics` |
| `charts["skipped"]` written as a list by the panel branch and as a dict by `_record_skipped` | TypeError on master when both fire | FIXED here: the panel branch writes name -> reason. Pinned by `tests/training/test_batch_e_registry_keys.py` (3 tests that crashed on it) |
| `TestConfidenceAnalysis::test_confidence_analysis_basic` | failed on `origin/master` | FIXED here: the confidence regressor inherited the main model's `callbacks`, which CatBoost refuses on GPU and which belong to the main model's eval set anyway; they are removed from its fit params |
| PySR columns drifted in the last float32 bit after a reload | failed intermittently | FIXED here: fit and predict evaluate every equation through one helper (`pysr_predict_column`) on the model's own feature columns in float64. Pinned by `tests/training/test_pysr_replay_parity.py` |
| `test_f1_degradation_augment_match_noise_level_shares_memory_for_non_numeric_cols` | failed on `origin/master` | FIXED here (test): an arrow-backed `string` column shares buffers even across a deep copy and builds a new array on every `to_numpy()`, so the check could not observe the copy; it now uses a numpy-backed object column and proves a deep copy would fail it |
| `tests/feature_selection/wrappers/test_wrappers_config_validation_and_stress.py::TestSkipRetrainingYContent` | failed on `origin/master` | FIXED here (stale fixture): on 3 columns both targets keep every column even on a fresh instance, so equal supports could not tell a refit from a replay. Now 6 columns, and the reused instance must reproduce a fresh fit's support for the second target |
| `test_discovery_algo_version_bumped` | failed on `origin/master` | FIXED here: discovery sources changed without a bump; `DISCOVERY_ALGO_VERSION` 3 -> 4 |
| Meta gates red on `origin/master` (15 tests) | same failures on `origin/master` 615fce57d | OPEN, owed as its own pass: new pyutilz / py-ci-shared checks landed without the code or baselines following. `test_code_audit_baseline` (22 src findings), `test_code_audit_tests_baseline` (24), `test_public_annotations` (27 functions), `test_no_value_bearing_asserts` (3 bench asserts), `test_no_underscore_imports_cross_package`, `test_no_new_nondiscriminating_assert`, `test_no_new_source_text_claims` (105 tests), `test_no_sklearn_metrics_in_production` (`preprocessing/auto_transform_select.py`), `test_no_new_unlocked_module_level_cache` (2), `test_no_unpaired_module_reload_in_tests` (1), `test_no_new_undocumented_public_symbols` (1), `test_no_new_pydoclint_findings` (1), `test_transform_name_tables_in_src_are_derived_or_explained` (1), `test_no_new_unguarded_scanning_gate` (7), `test_pipeline_extensions_sites_marked_best_effort` (count 1 != 3), `test_every_used_marker_is_registered` (`perf`), and two py-ci-shared checker bugs: `test_dev_requirements_git_dependencies_are_pinned_or_first_party` parses `requirements-dev.txt` as TOML, `test_version_consistent_across_sources` rejects a dynamic `[project].version` |
| `tests/feature_selection/shap_proxied/test_biz_val_shap_proxied_residual_passes.py::test_biz_val_residual_passes_no_noise_inflation` | fails, 23 selected vs 6 | residual pass 2 rescues noise columns on a pure-strong bed |
| `tests/feature_selection/shap_proxied/test_biz_val_shap_proxied_residual_passes.py::test_biz_val_residual_hard_vs_soft` | fails (2/6 vs 3/6), gate-independent | |
| `tests/feature_selection/shap_proxied/test_biz_val_shap_proxied_faith_interaction.py::test_biz_val_faith_interaction_beats_additive_on_xor` | fails, additive recovers 2/2 XOR operands | the bed's premise no longer holds |
| `tests/feature_selection/shap_proxied/test_biz_val_shap_proxied_banzhaf_ranking.py::test_biz_val_banzhaf_ranking_seed_stability_low_snr` | fails, gate-independent | banzhaf Jaccard 0.574 vs mean-abs-phi 0.468 |
| `tests/feature_selection/shap_proxied/test_shap_proxy_treeshap_interactions_gpu.py::test_biz_val_gpu_interaction_faster_than_numba` | 0.42x on a loaded host | a wall-clock ratio; re-measure on a quiet host before judging |
| `tests/feature_selection/biz_val/test_biz_val_wrappers_rfecv_stability.py::test_biz_val_rfecv_stability_beats_importance_on_many_steady` | fails, 0.8214 vs 0.8104 (+0.04 required) | identical numbers with and without FS-01 |
| `tests/feature_selection/fe/gates/test_fe_stability_vote.py::test_bizvalue_noise_survivor_reduction` | fails (2 <= 1) | |
| `tests/feature_selection/gpu/test_cmi_residency_traffic.py::test_pair_search_residency_no_nk_codes_bulk_d2h` | fails identically on `origin/master` | 17 bulk D2H transfers at (n,K) scale on the strict pair-search path; the codes/float buffer is not staying resident |
| `tests/feature_selection/gpu/test_gpu_cpu_mi_selection_equivalence.py::test_mrmr_gpu_cpu_selection_identical[clf_binary]` | fails identically on `origin/master` | the GPU path selects `add(qubed(c),rint(e))` where the CPU path does not |
| `tests/reporting/test_calibration_debiased_ece.py::test_biz_debiased_ece_bin_count_stable_on_perfectly_calibrated` | fails identically on `origin/master` | bin-count change 0.0123 against a 0.01 bound |
| `tests/feature_engineering/test_wavelet_dwt.py` (8 tests, order-dependent) | fails on `origin/master` under `--randomly-seed` 22/33/44 | FIXED here: the filter-cache test planted a zero-length db4 sentinel in the module's real cache and never restored it; it now uses a private cache, and the cached filter arrays are read-only |
| `BorutaShap(importance_measure="gini")` with any unfitted LightGBM / XGBoost / sklearn tree | raised on `origin/master` | FIXED here: `hasattr(model, 'feature_importances_')` is False before fit (the property raises NotFittedError), so every model but RandomForest was refused; the check now looks at the class. Pinned by `tests/feature_selection/boruta_shap/test_gini_accepts_unfitted_boosters.py` |
| `tests/feature_selection/mrmr/biz_val/test_biz_value_mrmr_regression_union/test_state_of_union_regression.py::TestCrossBasisHierarchyActivation::test_all_orth_arity_layers_contribute` | fails identically on `origin/master` | `hybrid_orth_features_` empty after the all-on fit |
| `tests/.../test_dcd_perf_bit_equivalence.py::TestLayer50_PerfBudget::test_dcd_all_auto_under_30s` | 127s on both `origin/master` and this tree on a loaded host | a wall-clock budget; re-measure on a quiet host |
| `tests/preprocessing/test_reject_outliers.py` (4 tests) | error on hosts without `imblearn` | FIXED here: the default pipeline needs the optional `imblearn`; those tests now `importorskip` it |
| `report_probabilistic_model_perf` cyclomatic complexity | 94 on `origin/master` (ruff limit 40) | OPEN, owed: brought to 84 by extracting the per-class aggregation, but a ~780-line function needs splitting into phase helpers - a refactor of its own, queued after this wave |
| `hybrid_orth_mi_fe` (+2) and `CompositeTargetEstimator.fit` (+1) over their length ceilings | introduced by `66fedbf83` (another active session) | left to that session to avoid colliding with work in progress |
| `tests/reporting/test_metric_over_time_direction.py::test_roc_auc_over_time_title_says_higher_is_better` | fails identically on `origin/master` | the line panel does not render, so the direction is never checked |
| `tests/test_meta/test_no_source_text_claims.py`, `test_no_single_shot_timing_assertion.py`, `test_no_nondiscriminating_assert.py`, `test_no_audit_metadata_in_comments.py`, `test_shared_uncalled_functions.py` | master-side entries only | each lists findings in files this wave did not touch (`data/datasets/*`, the ruff-pin and roster tests, six single-shot timing tests); the entries this wave introduced are fixed |
| Six source-text tests only py-ci-shared HEAD detects (`test_validator_wired_at_both_predict_entry_points`, `test_source_skip_path_present`, `test_metrics_core_uses_stable_argsort`, `test_codep110_fingerprint_call_outside_weight_loop`, `test_wave23_falls_back_to_source_default_when_cache_unavailable`, and one more in the same family) | pass under the pinned v1.17.0 | OPEN, owed before the py-ci-shared pin moves: each greps prod source; convert each to a behavioural test (call the entry point / spy the call) -- the v1.17.0 source-text baseline would otherwise go red on the bump |
| `shap_proxied_fs` GPU width thresholds (`GPU_MIN_FEATURES` = 2000 / 500) | hardcoded constants (were two resolvers returning their default) | OPEN: move onto `pyutilz.performance.kernel_tuning` (per-host sweep + measured fallback) like the other GPU dispatch thresholds |
