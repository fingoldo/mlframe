# Final verification audit (2026-05-25, post-Wave-10)

Pass-through of every finding from the 11 Wave-0 critic agents + the Wave-7/8/9/10 architectural-proposal slate. Status derived from (a) per-wave DONE_*.json manifest, (b) git log `da68ca6..HEAD` (~142 commits: 88 pre-Wave-8 + 8 Wave-8 + 23 Wave-9 + 23 Wave-10), (c) regression-sensor file existence under `tests/`, (d) targeted source greps for items not in any manifest.

Per `feedback_show_all_agent_findings` and `feedback_use_all_agent_findings`: every atomic finding gets a row. No round-number aggregation. No silent filtering. Low-tier surfaced alongside P0/P1/P2.

**Wave 8 landed (commit range `bafff1d2..ef82f687`)**: AP1 SuiteArtefactCache (55125d61) + 17 sensors, S68 bootstrap+DeLong (9a7e05a0, 578cb578), AP5 numba-coverage.yml workflow (06727c04), S66 categorical PSI (c2b741a6), S28 RNG fix (0bae104f), round5.5 composite_target_estimator+composite_ensemble bonus (d71506b7), misc helpers (fb80cbeb), bytes-budget+DeLong zero-variance sensor fix (578cb578); plus Wave-8-cleanup tail (406e7fca pre-commit wrapper bypassing the CUDA-init hang that blocked W8B/W8C/W8D/W8E + dead-helpers whitelist + general.py eager-format fix, and ef82f687 round5.5 test file rename per no_audit_wave_filenames rule).

**Wave 9 landed (commit range `ef82f687..242f7199`, 23 commits)**: AP12 calibration policy + AP13 honest_diagnostics aggregator (W9A); AP2 F15 finite_mask threading + AP4 fuzz axes F1/F2/F5/F6/F7 (W9B); S67 Enum bundle persistence full + D1 P2 #7/#8/#10 + D1 Low #11/#12/#13 (W9C); A1 FS deferred residue S30/S31/A1#6/A1#8/A1#9/A1#12/A1#13/A1#15 + A1#14/#16-#21 verified-already-fixed (W9D); A3 ensembling deferred cluster A3#6/#7/#8/#10/#11/#12 + Low #13/#14/#15/#16 (W9E). Pre-commit hook now passes via the W8-cleanup wrapper, so every W9 commit landed under standard hook flow.

**Wave 10 landed (commit range `c31b419d..04cfe69d`, 23 commits)**: W10A perf P2/Low residue (1 correction-only commit 27429ed3 - 9 A4 P2/Low items verified DONE in prior waves; FINAL_VERIFICATION mis-tracking of A4#5 corrected); W10B caching follow-ups (4 commits: b8083ef6 AP1 SuiteArtefactCache eviction sidecar accounting fix, 54d456ae MRMR x-hash extended to `_pre_pipeline_cache_key`, 65b3dded `_discovery_cache_bytes_total` helper for DiscoveryCache observability, c70c9317 A5#9 joint-stash architectural proposal deferred pending user OK); W10C tests infra residue (3 commits + manifest: 53d62fb8 T1 session-fixture immutability sensor extension, bb019e9e T5 BGM + RSD-kNN biz_value tests, 415d3e23 fast marker on bootstrap suite + strengthened dataset_cache_fingerprint assert; agent hung mid-sweep on fast-mode extension to remaining packages, fuzz dimension widening cross-axis, weak-assert W4-W8 cluster, numba bench-report script, pytest cache cleanup, RFECV monkey-patch upstream-fix); W10D code-arch residue (5 commits: faea57b3 CHANGELOG aggregate entry + 3 docs guides, 55e6b55d macOS+py3.13 CI matrix expansion + mypy strict expand to utils.safe_pickle, 8ab33fb9 cross-package underscore-import meta-test + top-level __init__ docstring + cupy SoftBan doc, b643da93 AP5 nightly run validation surfacing 2 fixes for Wave 11, d94313ab manifest); W10E monolith preventive splits (6 commits: 9d633121 strategies.py 956→316 with 4 siblings, 48626d64 extractors.py 937→326 with 3 siblings, 206ce81b train_eval.py 942→570 with 1 sibling, d61dd0cc helpers.py 994→267 with 1 sibling, ef9aec26 neural/flat.py 928→420 with 1 sibling, 2c2d8452 LOC-budget meta-test currently FAILS by design on 5 files >1k LOC). Plus d1820c55 numba-coverage report generator + meta-test extending AP5, and user direct commits aef36c5/d20f4b8b composite_cardinality_cap excluded from model_dump kwargs.

## Per-finding status table

### A1 - Feature selection (fs-critique.md, 21 findings)

| ID | Sev | Title | Status | Commit / Sensor / Reason |
|---|---|---|---|---|
| A1#1 / S01 | P0 | `_pre_pipeline_cache_key` 4-cell target fingerprint collision | DONE | f1d4212 + `tests/training/test_regression_S01_pipeline_cache_fp_collision.py` (w1a manifest); W10B 54d456ae extends key with full-X blake2b |
| A1#2 / S28 | P1 | `estimate_features_relevancy` `.copy()` + global `np.random.shuffle` | DONE | 0bae104f (W8E S28; seeded `default_rng(random_state)` + view-not-copy in general.py + importance.py) + `tests/feature_selection/test_regression_S28_efs_relevancy_rng.py` |
| A1#3 / S29 | P1 | `_rfecv_fit.cv_n` silent fallback to 3 for splitters w/o `n_splits` | DONE | 73aaac4d w5a "strengthen FS cache/sigs + add group-n_groups floor" (overlaps S65 RFECV gate) |
| A1#4 / S30 | P1 | MRMR `groups=` accepted but silently ignored (warn-only) | DONE | d9c9aba4 (W9D `fix(fs/mrmr): add strict_groups knob; raise instead of warn-only ignore`) |
| A1#5 / S31 | P1 | `np.random.shuffle` global RNG in fleuret / permutation njit kernels | DONE | 46db224b (W9D `fix(fs/njit-rng): seed inline LCG in parallel_mi + fleuret njit shuffles`) + ccfb45d2 (W9D `test(fs/permutation): tighten BC speedup floor for LCG-accelerated baseline`) |
| A1#6 / +1 | P1 | RFECV `cv_shuffle=True` silently swapped to TimeSeriesSplit on polars datetime hint | DONE | 1b4a2c32 (W9D `fix(fs/rfecv): respect explicit cv_shuffle=True over temporal auto-detect`) |
| A1#7 | P2 | `_mlframe_use_sample_weights_in_fs_` attr-marker not in pipeline signature | DONE | a96077da `fix(pipeline_cache): fold attribute-only seeds into signature` (w5b manifest A5_pipeline_cache_attribute_only_seeds_missing_from_sig=DONE) |
| A1#8 | P2 | `_mrmr_fingerprints._content_array_signature` 10-cell X sample | DONE | 7016379b (W9D `fix(fs): strengthen MRMR/RFECV cache keys + gate polars self_destruct`) |
| A1#9 | P2 | `_rfecv_fit` sample_weight length not checked vs y | DONE | 7016379b (W9D; bundled with cache-key strengthening) |
| A1#10 | P2 | `_mrmr_compute_y_fingerprint_sample` 6-decimal rounding | DONE | 73aaac4d (folded into A1_FS_mrmr_y_fingerprint_6decimal_rounding=DONE per w5a manifest) |
| A1#11 | P2 | `_rfecv_fit n_samples < 2*cv_n` floor ignores n_groups for GroupKFold | DONE | 73aaac4d (A1_FS_groupkfold_n_groups_vs_n_samples=DONE per w5a manifest) |
| A1#12 | P2 | RFECV polars `to_pandas(self_destruct=True)` risk if caller-owned | DONE | 7016379b (W9D; `gate polars self_destruct`) |
| A1#13 | P2 | RFECV `_x_hash` 10-row strided sample collision risk | DONE | 7016379b (W9D; cache-key strengthening covers x-hash sample); W10B 54d456ae extends pattern to `_pre_pipeline_cache_key` |
| A1#14 | Low | `benchmark_mi_algos` warmup arrays use global RNG, no seed | DONE | W9D verified-already-fixed in prior waves (seeded path in place) |
| A1#15 | Low | MRMR cache lookup signature build paid twice | DONE | 7ad8bd30 (W9D `perf(fs/mrmr): hoist _full_y_content_hash; reuse signature build (A1#15)`) |
| A1#16 | Low | `_rfecv_cv_setup` shallow-copy of splitter for early_stopping_val_nsplits | DONE | W9D verified-already-fixed in prior waves |
| A1#17 | Low | MRMR `max_nbytes` dead under `backend="threading"` | DONE | W9D verified-already-fixed in prior waves |
| A1#18 | Low | RFECV per-fold `_eval_fold` non-deterministic log ordering | DONE | W9D verified-already-fixed in prior waves |
| A1#19 | Low | MRMR `groups`/`sample_weight` asymmetric wrapper vs `_fit_impl` | DONE | W9D verified-already-fixed in prior waves (docs reconciled) |
| A1#20 | Low | RFECV `X.select_dtypes("number").to_numpy()` full-frame copy for Inf check | DONE | W9D verified-already-fixed in prior waves |
| A1#21 | Low | RFECV verbose `X.isna().to_numpy()` materialises bool N×M frame | DONE | W9D verified-already-fixed in prior waves |

### A2 - Feature engineering (fe-critique.md, 25 findings + W1-W8 weak-asserts)

| ID | Sev | Title | Status | Commit / Sensor / Reason |
|---|---|---|---|---|
| A2#1 / S02 | P0 | `_filter_to_numeric` full-frame `.copy()` on bool cols | DONE | a411d50/c4c7fdb (w1b manifest) + `tests/feature_engineering/test_regression_S02_filter_to_numeric_no_copy.py` |
| A2#2 / S03 | P0 | `_apply_pysr_fe` temp `_pysr_y_` column injection leak window | DONE | a411d50 (w1a) + `tests/feature_engineering/test_regression_S03_pysr_temp_target_restored.py` |
| A2#3 / S04 | P0 | `_phase_composite_discovery copy(deep=False)+setitem` BlockManager leak | DONE | a27b6895 (w1a) + `tests/training/test_regression_S04_composite_discovery_no_leak.py` |
| A2#4 / S32 | P1 | `_DEFAULT_DATE_METHODS` mutable module-level default | DEFERRED | Not in w5a/w5b commit. bbff3590 touched FE defaults but for other items (timeseries mutable default = `A2_FE_timeseries_mutable_default=DONE`) - basic.py default still by-reference |
| A2#5 / S33 | P1 | Missing year/week/quarter/is_weekend; no tz handling | DEFERRED | A2_FE_create_date_features_log_level=DONE (bbff3590 log demotion) but richness extension deferred |
| A2#6 / S34 | P1 | bruteforce `to_pandas()` + sub-frame fillna broadcast-copy | DEFERRED | Not addressed |
| A2#7 / S35 | P1 | `pd.DataFrame(list-of-lists, dtype=...)` slow path in timeseries | REJECTED | w2c finding #20 rejected with bench (parity vs astype fallback) |
| A2#8 / S36 | P1 | recursive subset `.copy()` in create_aggregated_features | DEFERRED | Not addressed |
| A2#9 / S37 | P1 | `from_fitted_inner` skips clone asymmetry vs fit | DONE | c043f386 (w1c S26) overrides `__sklearn_clone__` on `from_fitted_inner` instances |
| A2#10 / S38 | P1 | `_median_residual_fit` Python loop with `np.median` per bin | DONE | 05688481 (w4b S45) size-aware dispatcher pandas-groupby fallback |
| A2#11 / S39 | P1 | bruteforce rename mutating caller's frame in place | DEFERRED | Not addressed |
| A2#12 / S40 | P1 | `_ratio_fit` empty-array `np.median` warning pitfall | DEFERRED | Not addressed |
| A2#13 | P2 | `make_discovery_cache_key random_state=0` sentinel arg | DONE | 2243fc33 (w5b A5_make_discovery_cache_key_seed_double_fold=DONE; renamed to `_legacy_random_state_sentinel`) |
| A2#14 | P2 | `create_aggregated_features nonlinear_transforms=[np.cbrt]` literal each call | DONE | bbff3590 (A2_FE_timeseries_mutable_default=DONE per w5a) |
| A2#15 | P2 | `_ewma_kernel`/`_frac_diff_inverse_kernel` no fastmath/dispatcher | DONE | 13326bc / bc55aba (w4b S46) - parallel + dispatcher + kernel_tuning_cache integration |
| A2#16 | P2 | `compute_positional_encoding` float64 intermediate | DONE | bbff3590 (A2_FE_random_features_dtype_intermediate=DONE per w5a; fmod in caller dtype) |
| A2#17 | P2 | `CompositeTargetEstimator.feature_names_in_` bare except Exception | DONE | bbff3590 (A2_FE_composite_target_estimator_bare_except=DONE; narrowed to AttributeError per w5a) |
| A2#18 | P2 | `_PYSR_LOCK` global serialisation - user-facing docs | DONE | bbff3590 (A2_FE_pysr_lock_contract_doc=DONE per w5a) |
| A2#19 | P2 | `pysr_operator_preset or "standard"` swallows empty string | DONE | bbff3590 (A2_FE_or_standard_collapses_empty_string=DONE per w5a) |
| A2#20 | P2 | Composite Transform unit-test coverage gap | DEFERRED | Per-transform unit suites not authored; w3a/w3c did not add families. Backlog |
| A2#21 | Low | `create_date_features` per-call INFO log | DONE | bbff3590 (A2_FE_create_date_features_log_level=DONE; demoted to DEBUG per w5a) |
| A2#22 | Low | Legacy `run_pysr_fe` vs `bruteforce.run_pysr_feature_engineering` duplicate | DONE | bbff3590 (A2_FE_run_pysr_fe_duplicate_deprecation=DONE; DeprecationWarning per w5a) |
| A2#23 | Low | sympy `safe_log` predict semantics ≠ Julia train | DONE | de7b5cb6 (A2_FE_sympy_safe_log_predict_semantics=DONE; Piecewise mapping per w5a) + `test_regression_w5_pysr_piecewise_semantics.py` |
| A2#24 | Low | `_median_residual_fit np.unique` collapse warning gap | DONE | bbff3590 (A2_FE_composite_transform_median_collapse=DONE; RuntimeWarning per w5a) |
| A2#25 | Low | misleading try/except/finally comment | DEFERRED | Cosmetic; not addressed |
| - +Low | Low | `_TRANSFORMS_REGISTRY` MappingProxyType wrap | DEFERRED | Not addressed |
| - +Low | Low | `pd.DataFrame(data=features, dtype=)` SettingWithCopyWarning class | REJECTED | w2c #20 rejected (parity) |
| - +Low | Low | `online_refit_*` docstring gap | DEFERRED | Not addressed |
| A2 W1 | weak | `test_biz_val_bruteforce.py` `model.equations_ is not None` x4 | DONE | 7157af4 (w3c B1_W1_pysr_bruteforce_weak_asserts=DONE) |
| A2 W2 | weak | `test_tvt_round5_*.py` bare `is not None` cluster | DEFERRED | Cluster not surveyed by w3c |
| A2 W3 | weak | `test_biz_val_filters_hermite_fe.py` quantitative floor gap | DEFERRED | Directional finding, not addressed |
| A2 W4 | weak | `test_phase_helpers_clone_elimination.py` clone-elision identity check | DEFERRED | Not addressed |
| A2 W5 | weak | `test_mlp_degenerate_init_*` divergence-detector assert | DEFERRED | Not addressed |
| A2 W6 | weak | `test_dataset_cache_fingerprint.py` idempotence/sensitivity check | DONE | 415d3e23 (W10C `test(wave10c): fast marker on bootstrap suite + strengthen dataset_cache_fingerprint assert`) |
| A2 W7 | weak | `test_automl.py` bare `is not None` | DEFERRED | Not addressed (snapshot/restore in d124f168 closed pollution, not the weak asserts) |
| A2 W8 | weak | `test_predict_polars_fastpath.py` `preds is not None` | DONE | 7157af4 (w3c B1_W8_predict_polars_fastpath_weak_assert=DONE) |
| A2 W-other | weak | `compute_countaggs` weak assert (B1_W7) | DONE | 7157af4 (w3c B1_W7_compute_countaggs_weak_assert=DONE) |

### A3 - Ensembling (ensembling-critique.md, 16 findings)

| ID | Sev | Title | Status | Commit / Sensor / Reason |
|---|---|---|---|---|
| A3#1 / S05 | P0 | `_oof_or_train` silent train-fallback in level-1 stacking | DONE | 56ef5e98 (w1a) + `tests/test_regression_S05_oof_silent_train_fallback.py` (visibility WARN, tactical fix per audit) |
| A3#2 / S41 | P1 | `_ENSEMBLE_RANK_METRIC_CANDIDATES` fallback on `val.*`/`test.*` | DEFERRED | Not addressed in any commit. Comment-vs-tuple contradiction remains |
| A3#3 / S42 | P1 | No probability calibration before classifier blend (Isotonic/Platt) | DEFERRED | Overlaps AP12 calibration policy - not landed at ensemble level |
| A3#4 / S43 | P1 | `_rrf_aggregate_probs` K=1 binary raw RRF (no sigmoid/minmax) | DEFERRED | Not addressed |
| A3#5 | P2 | `combine_probs` static equal weighting; NNLS observational-only | DONE | 5505c773 (AP7: NNLS stacking-aware blend - weights now fed back into combine_probs) + `test_regression_w7_ap7_nnls_weights_applied.py` |
| A3#6 | P2 | Diversity auto-drop tag-index not MAE-quality | DONE | c5727d53 (W9E `fix(models/ensembling): score_ensemble drops higher-MAE member on diversity auto-drop`) + 5a93969a regression sensors |
| A3#7 | P2 | K=2 catastrophic-dropout non-deterministic tiebreak | DONE | c5727d53 (W9E; deterministic K=2 alphabetical tiebreak) + 5a93969a sensors |
| A3#8 | P2 | Lazy import `_choose_ensemble_flavour` cycle-breaker per-target cost | DONE | 9432b098 (W9E `refactor(training/core): move _choose_ensemble_flavour to leaf module; ensembling sibling top-level imports it`) |
| A3#9 | P2 | No nbytes dispatcher for streaming vs materialised ensemble | DONE | 5505c773 (AP8: nbytes streaming dispatcher) |
| A3#10 | P2 | No biz_value `score_ensemble beats best single` test | DONE | 5a93969a (W9E `test(ensembling): regression sensors for the A3 P2/Low cluster`) covers compare_ensembles biz floor alongside per-finding sensors |
| A3#11 | P2 | `rrf_k` metadata stamped unconditionally | DONE | 5a93969a (W9E sensor cluster gates rrf_k stamp on actual RRF aggregation path) |
| A3#12 | P2 | `compare_ensembles` full `copy.deepcopy` per call | DONE | 7bb9fee0 (W9E `fix(models/ensembling): per-class avg Pearson for multiclass diversity + shallow inner pop in compare_ensembles`) |
| A3#13 | Low | Weighted-median branch ignores `sample_weight` kw | DONE | 3538f5a5 (W9E `fix(models/ensembling): quality gate group-aware median collapses per group; documented unweighted predict-side anchor`) |
| A3#14 | Low | Group-aware median weight broadcast bug | DONE | 3538f5a5 (W9E; group-aware median collapses per group) |
| A3#15 | Low | Uniformity check misses `oof_probs` | DONE | c5727d53 (W9E; `uniformity inspects oof_probs`) |
| A3#16 | Low | Multiclass diversity flatten over (N,K) Pearson over interleaved | DONE | 7bb9fee0 (W9E; per-class avg Pearson for multiclass diversity) |

### A4 - Perf hotspots (perf-hotspots-critique.md, 24 findings)

| ID | Sev | Title | Status | Commit / Sensor / Reason |
|---|---|---|---|---|
| A4#1 / S06 | P0 | pandas `get_trainset_features_stats` per-col Python loop | DONE | 5626668e (w4a) + `tests/training/test_regression_S06_precompute_pandas_batch.py` |
| A4#2 / S44 | P1 | `apply_polars_categorical_fixes` per-cat-col sync `.collect()` | DONE | 138cdf73 (w4a) + `test_regression_S44_polars_fixes_batched.py` |
| A4#3 / S45 | P1 | `_median_residual_fit` / `_quantile_residual_fit` Python bin loop | DONE | 05688481 (w4b) + `test_regression_S45_median_residual_perf.py` |
| A4#4 / S46 | P1 | `_ewma_kernel` / `_frac_diff_inverse_kernel` no parallel/dispatcher | DONE | 13326bcd + bc55aba (w4b) + `test_regression_S46_ewma_frac_diff_parallel.py` |
| A4#5 / +P1 | P1 | `_cached_init_params` weight-loop rebuild | DONE | 5c6820ea (w4a follow-up `perf(training): hoist per-target invariants out of inner loops` - NGBoost get_params snapshot cached on first use + CB cat/text/embedding filter hoisted out of weight loop) + `tests/training/test_regression_S47_filter_polars_cat_hoisted.py::test_S47_ngb_fallback_snapshot_cached_outside_loop` sensor pins both hoists. Mis-tracked as DEFERRED in initial Wave-9 sweep; verified DONE by w10a-perf-residue audit on 2026-05-25 (commit 27429ed3 docs-only correction) |
| A4#6 / S47 | P1 | `_filter_polars_cat_features_by_dtype` invariant over weight loop | DONE | cea8a266 (w4a) + `test_regression_S47_filter_polars_cat_hoisted.py` |
| A4#7 / S48 | P1 | train-pred cache keyed by `id(comp)` misses shim wrappers | DONE | 09d6899e (w4a) + `test_regression_S48_composite_cache_shim_aware.py` |
| A4#8 / S49 | P1 | `memory_usage(deep=True)` multi-minute hot point | DONE | 4f3e2cec (w4a) + `test_regression_S49_memory_usage_polars_fastpath.py` |
| A4#9 (F9) | P2 | sort by key=str on cat-col union (50k str-key sort) | REJECTED | w4c F9=REJECTED (4% gain, not worth branch) |
| A4#10 (F10) | P2 | `_disc_df = filtered_train_df.copy(deep=False)` | DONE | F10=DONE pre-existing (was a27b6895 / S04) |
| A4#11 (F11) | P2 | `_numeric_only_view` called twice train+val | REJECTED | w4c F11=REJECTED (per-call schema iter is ~us; different frames) |
| A4#12 (F12) | P2 | `_oof_rmses_list` per-column Python loop | DONE | c6558960 (w4c F12=DONE, 1.31x speedup) + `test_regression_w4c_perf_p2low.py::test_vectorised_oof_rmse_*` |
| A4#13 (F13) | P2 | CB-cat schema scan when CB not in suite | DONE | bc55aba (w4c F13=DONE; gated on `_has_cb`) |
| A4#14 (F14) | P2 | 3 nested tqdm contexts per target | REJECTED | w4c F14=REJECTED (lost user-visible progress > perf gain) |
| A4#15 (F15) | P2 | Per-spec `finite_mask` recompute in residual-fit family | DONE | 12b458dc (W9B `perf(composite_transforms): thread precomputed _finite_mask through 9 residual _fit kernels`; closes AP2) |
| A4#16 (F16) | P2 | Pipeline JSON cache uses stdlib json not orjson | DONE | 9b7a3576 (w4c F16=DONE, 3.26x) + `test_orjson_roundtrip_payload_matches_json` |
| A4#17 (F17) | P2 | post-fit columns metadata duplicate `list()` copies | DONE | bc55aba (w4c F17=DONE) |
| A4#18 (F18) | Low | `np.column_stack` of K per-component preds | DONE | c6558960 (w4c F18=DONE) + `test_preallocated_pred_matrix_*` |
| A4#19 (F19) | Low | `group_ids` per-target re-asarray | DONE | 5c6820e (w4c F19=DONE) |
| A4#20 (F20) | Low | Post-cast `null_count` diagnostic non-verbose | DONE | 138cdf7 (w4c F20=DONE; sibling absorbed) |
| A4#21 (F21) | Low | `_rolling_median` pandas O(n·k log k) | DONE | bc55aba (w4c F21=DONE, 10x via bottleneck.move_median) + `test_rolling_median_matches_pandas_*` |
| A4#22 (F22) | Low | `id(model)` cache key id-recycle risk | DONE | c655896 (w4c F22=DONE; promoted to (id, id(frame), shape)) + `test_frame_content_key_distinguishes_id_collision` |
| A4#23 (F23) | Low | pandas `get_trainset_features_stats` no polars fallback | DONE | bc55aba (w4c F23=DONE) |
| A4#24 (F24) | Low | Cross-target ensemble fits sequential | DEFERRED | ARCH-DEFER → architectural_proposals/F24_ktarget_ensemble_joblib.md. User dropped per Wave-7 plan (AP3 dropped) |

### A5 - Pipeline caching (pipeline-cache-critique.md, 16 findings)

| ID | Sev | Title | Status | Commit / Sensor / Reason |
|---|---|---|---|---|
| A5#1 / S07 | P0 | `precompute_composite_target_specs` + `precompute_dummy_baselines` NotImplementedError stubs | DONE | 55125d61 (W8A AP1 SuiteArtefactCache `src/mlframe/training/suite_artefact_cache.py` covers composite_target_specs + dummy_baselines via SuiteKeyBuilder digest) + `tests/training/test_suite_artefact_cache.py`; W10B b8083ef6 hardens eviction (sidecar .sha256 accounting fix) |
| A5#2 / S08 | P0 | `PipelineCache` plain dict, no size gate | DONE | 7eab5ab5 (w1b) + `tests/training/test_regression_S08_pipeline_cache_size_gate.py` |
| A5#3 / S09 | P0 | Heavyweight pipeline + extensions recomputed every run | DONE | 55125d61 (W8A AP1 SuiteArtefactCache covers `fit_and_transform_pipeline` + `apply_preprocessing_extensions` + `trainset_features_stats`) + `tests/training/test_suite_artefact_cache.py` |
| A5#4 / S51 | P1 | Per-target select_target template rebuild (TODO at :774) | DEFERRED | w5a A3-side related items DEFERRED; A5 P1 not commit-landed |
| A5#5 / S52 | P1 | `DiscoveryCache` vs `FeatureCache` parallel disk-cache divergence | DEFERRED | Architectural unification deferred to user OK (c70c9317 W10B proposal `docs(audit): architectural proposal for DiscoveryCache + SuiteArtefactCache joint-stash`); W10B 65b3dded landed `_discovery_cache_bytes_total` helper for observability ahead of unification |
| A5#6 / S53 | P1 | `MRMR._FIT_CACHE` class attribute, no byte-size cap | DONE | d2fd00d4 (w5b A5_mrmr_fit_cache_no_byte_cap=DONE) + `test_regression_w5_mrmr_lru_byte_cap.py` |
| A5#7 / S54 | P1 | `_PRE_PIPELINE_CACHE_MAX=8` hardcoded, no byte budget | DEFERRED | Count cap exists; byte-budget+ktc not added |
| A5#8 (P2) | P2 | `_pandas_view_cache` size-cap 4 popitem FIFO | DONE | 56e78eb1 (w5b A5_pandas_view_cache_unbounded=DONE; OrderedDict + byte gate) + `test_regression_A5_p2_8_pandas_view_cache_lru.py` |
| A5#9 (P2) | P2 | No `polars.LazyFrame.cache()` in training/core | DEFERRED | ARCH-DEFER → architectural_proposals/A5-P2-9-polars-lazyframe-cache.md + c70c9317 (W10B joint-stash proposal). Not landed pending user OK |
| A5#10 (P2) | P2 | `_loaded_models_cache` per-call dict (no LRU on disk model loader) | DONE | cb5acc82 (w5b A5_load_mlframe_model_no_warm_cache=DONE; LRU keyed by (path, mtime_ns)) + `test_regression_A5_p2_10_load_model_lru.py` |
| A5#11 (P2) | P2 | `_pre_pipeline_cache_key` missing random_seed / lib_versions | DONE | a96077da (w5b A5_pipeline_cache_attribute_only_seeds_missing_from_sig=DONE) + W10B 54d456ae extends with full-X blake2b content digest + `test_regression_A5_p2_11_pre_pipeline_seed_in_key.py` |
| A5#12 (P2) | P2 | `PipelineCache.cache_size_bytes` `sys.getsizeof` "best-effort" | DONE | Subsumed by 7eab5ab5 S08 fix (per-entry nbytes accounting via `_estimate_slot_nbytes`) |
| A5#13 (Low) | Low | `DiscoveryCache(None, None)` warn-only | DONE | 5440c65d (w5b A5_discovery_cache_construction_silent_unbounded=DONE; hard ValueError) + `test_regression_A5_low_13_discovery_cache_hard_error.py` |
| A5#14 (Low) | Low | `_FP_CACHE_MAX=128` no env override | DONE | 38453aca (w5b A5_fp_cache_max_no_env_override=DONE) + `test_regression_A5_low_14_fp_cache_env_override.py` |
| A5#15 (Low) | Low | WeakKeyDictionary pattern doc gap | DEFERRED | Documentation only; not landed |
| A5#16 (Low) | Low | `select_target` no cross-target memo | DEFERRED | Same root cause as S51; not addressed |

### A6 - Polars zero-copy (polars-zerocopy-critique.md, 40 findings)

| ID | Sev | Title | Status | Commit / Sensor / Reason |
|---|---|---|---|---|
| A6#1 / S10 / F1 | P0 | `_main_train_suite.py:814,816 pd.DataFrame(<polars>)` bypasses bridge | DONE | bc73331 (w2a F1=DONE) + `test_regression_w2a_bridge_zerocopy.py` |
| A6#2 (NEC) | P0 | `_predict_pre_pipeline.py:149 pd.DataFrame(_arr,...)` sklearn output | DONE | (NECESSARY shape, documented per w2a) |
| A6#3 / S11 / F3 | P0 | `_pipeline_helpers.py:401-402` bare `held.to_pandas()` | DONE | bc73331 (w2a F3=DONE) |
| A6#4 / F4 | P0 | `_predict_main_from_models.py:246 pl.from_pandas` no rechunk | DONE | 0d2012f (w2a F4=DONE, 15.2x speedup) |
| A6#5 / F5 | P0 | `_phase_helpers_fit_pipeline.py:521 pl.from_pandas` per split | DONE | 0d2012f (w2a F5=DONE) |
| A6#6 / F6 | P0 | `_phase_helpers_fit_split.py:436` fallback bridge | DONE | (NECESSARY fallback documented per w2a; existing primary uses bridge) |
| A6#7 / S12 / F7 | P0 | `_pipeline_extensions.py:497` bare `to_pandas()` | DONE | bc73331 (w2a F7=DONE) |
| A6#8 / S13 / F15 | P0 | `extractors.py:249,371` bare `to_pandas()` head/tail | DONE | bc73331 (w2a F15=DONE) |
| A6#8b (#8 percol) | P1 | `composite_cache.py:411` per-column `to_numpy()` loop | REJECTED | w2b #8=REJECTED (batch breaks per-column str(dtype) signature distinction) |
| A6#9 (#9 percol) | P1 | `_dummy_baseline_regression.py:92-103` per-col extractions | DONE | d31cb35 (w2b #9=DONE; `_extract_col_1d` helper) + `test_regression_w2b_percol_zerocopy.py` |
| A6#10 (#10) | P1 | `composite_auto_detect.py:114-349` per-col scan | REJECTED | w2b #10=REJECTED (per-column np.unique cost dominates) |
| A6#11 (#11) | P1 | `_main_train_suite_target_distribution.py:113,116` full materialise+slice | DONE | d31cb35 (w2b #11=DONE; polars `get_column().gather()`) |
| A6#12 (#12) | P1 | `_LagPredictDeployableModel.predict select().reshape` | DONE | d31cb35 (w2b #12=DONE) |
| A6#13 (#13) | P1 | `_phase_recurrent.py:46,79` ndarray ascontiguousarray cache key | DONE | e0fa554c (w2c #13=DONE; cache key adds cols_signature) + `test_regression_w2c_dtype_gate.py` |
| A6#14 (NEC) | P1 | `_phase_temporal_audit.py:130` pandas fallback | DONE | (NECESSARY; already polars-first per w2a) |
| A6#15 (= F15) | P1 | `extractors.py:249,371` head/tail bare to_pandas | DONE | Same as A6#8; bc73331 |
| A6#16 (#16) | P1 | `_pipeline_extensions.py:728 pd.DataFrame(arr,..)` sklearn return | REJECTED | w2c #16=REJECTED (0.06ms dtype-agnostic; no speedup) |
| A6#17 (#17) | P1 | `_predict_guards.py:354+481+531` `X.to_numpy(dtype=np.float64)` | DONE | 152d65d (w2b #17=DONE; allow_copy=False fastpath) |
| A6#18 (#18) | P1 | `bruteforce.py:151 sampled.to_pandas()` for PySR | REJECTED | w2c #18=REJECTED (3x SLOWER via pyarrow extension array on PySR check_array path) |
| A6#19 (#19) | P1 | `bruteforce.py:58 pd.DataFrame(...dtype=float)` per-fold buffer | DONE | e0fa554c (w2c #19=DONE, 1.46x) + sensor names in w2c manifest |
| A6#20 (#20) | P1 | `timeseries.py:622,627 pd.DataFrame(features,...)` list-of-lists | REJECTED | w2c #20=REJECTED (parity; dtype= kwarg removal doesn't help) |
| A6#21 (#21) | P1 | `random_features.py:150,229 to_numpy()` then astype | REJECTED | w2c #21=REJECTED (polars allow_copy=False raises on multi-col) |
| A6#22 (#22) | P1 | `per_column_rff.py:56 X.to_numpy()` polars | REJECTED | w2c #22=REJECTED (same root) |
| A6#23 (#23) | P1 | `stacked_attention.py:92,103` astype per layer | DONE | e0fa554c (w2c #23=DONE; dtype passthrough) |
| A6#24 (#24) | P1 | `stacked_qnn.py:64,73` `Q1.to_numpy().astype(np.float32)` | REJECTED | w2c #24=REJECTED (float32 intentional intermediate design) |
| A6#25 (#25) | P1 | `boosted_attention.py:93,104` astype | DONE | e0fa554c (w2c #25=DONE) |
| A6#26 (NEC) | P2 | `_ensembling_score.py:259-262 .to_numpy() if pd.Series` | DONE | (NECESSARY for joblib pickle-safety per audit) |
| A6#27 (NEC) | P2 | `_log_loss_and_separation.py:137,139` Series to_numpy | DONE | (NECESSARY for numba) |
| A6#28 (#28) | P2 | `_ice_metric.py:92-249` `.values` mixed with `.to_numpy()` | DONE | 92e6de8 (w2b #28=DONE) + `test_regression_w2b_values_to_numpy.py` |
| A6#29 (#29) | P2 | `_fairness_metrics.py:241,245` redundant np.asarray wrap | DONE | 92e6de8 (w2b #29=DONE) |
| A6#30 (NEC) | P2 | `metrics/core.py:545,547,818,820` to_numpy on Series | DONE | (NECESSARY) |
| A6#31 (NEC) | P2 | `utils.py:608 tbl_fixed.to_pandas()` bridge body | DONE | (NECESSARY - is the bridge) |
| A6#32 (NEC) | P2 | `_dummy_baseline_compute.py:195,297-299` to_numpy after map | DONE | (NECESSARY for sklearn metric) |
| A6#33 (#33) | P2 | `baseline_diagnostics.py:722,732,735,767` per-feat ablation | REJECTED | w2b #33=REJECTED (722/767 are single-feature, not loop; misread) |
| A6#34 (#34) | P2 | `_rfecv.py:716` bare to_pandas vs `_rfecv_fit.py:102` aligned | DONE | 152d65d (w2b #34=DONE; Arrow bridge aligned) + `test_regression_w2b_rfecv_arrow_bridge.py` |
| A6#35 (NEC) | P2 | `_boruta_shap_fit_explain.py:131-133 X to_pandas + y to_pandas` | DONE | (NECESSARY per BorutaShap pandas-only) |
| A6#36 (NEC) | P2 | `bruteforce.py:62 .values` for iloc assignment | DONE | (NECESSARY per audit) |
| A6#37 (#37) | Low | `inference/predict.py:100,159 features.values.tolist()` | DONE | 92e6de8 (w2b #37=DONE; `.to_list()`) + `test_regression_w2b_predict_tolist.py` |
| A6#38 (#38) | Low | `preprocessing/cleaning.py:195,630,639,717 .values` | DONE | 92e6de8 (w2b #38=DONE) + `test_regression_w2b_cleaning_to_numpy.py` |
| A6#39 (Low) | Low | `estimators/custom.py:197-597 .values` cosmetic | DEFERRED | Not in any commit |
| A6#40 (#40) | Low | `votenrank/leaderboard/Leaderboard.py:78,191,207 .values` | DONE | 92e6de8 (w2b #40=DONE) + `tests/votenrank/test_regression_w2b_values_to_numpy.py` |

### B1 - Tests expand (tests-expand.md - 22 module gaps + W1-W8 weak + F1-F8 fuzz + N1-N12 numba + 4 biz_value gaps + sklearn-matrix)

| ID | Sev | Title | Status | Commit / Sensor / Reason |
|---|---|---|---|---|
| B1 U1 / S18 | P0 | `crash_reporting.py` zero tests | DONE | 9e5192a3 (w3a U1=DONE) + `tests/training/test_crash_reporting.py` (9 tests) |
| B1 U2 / S19 | P0 | `metrics_registry.py` zero direct tests | DONE | c4109ec9 (w3a U2=DONE) + `tests/training/test_metrics_registry.py` (24 tests) |
| B1 U3 | P1 | `composite_auto_detect.py` no biz_value | DONE | c9ef9ace (w3a U3=DONE) + `tests/training/test_composite_auto_detect_biz_value.py` (23 tests) - one bug surfaced (FU-w3a-1, fixed in b7d3e4e9) |
| B1 U4 | P1 | `drift_report.py` gap | DEFERRED | Not addressed |
| B1 U5 | P1 | `feature_drift_report.py` gap (translators + drift) | DEFERRED | Not addressed |
| B1 U6 | P1 | `models.py` predicate functions | DEFERRED | Not addressed |
| B1 U7 | P1 | `io.py` `clean_mlframe_model` + `validate_load_meta_sidecar` | DEFERRED | Not addressed (S72 fixed verify, not validate helpers) |
| B1 U8 | P2 | `composite_bayesian.py` bootstrap path | DEFERRED | Not addressed |
| B1 U9 | P2 | `phases.py` registry helpers | DEFERRED | Not addressed |
| B1 U10 | P2 | `lgb_shim.py` capability | DEFERRED | Not addressed |
| B1 U11 | P2 | `xgb_shim.py` capability | DEFERRED | Not addressed |
| B1 U12 | P2 | `ranking.py` qid_to_group_sizes / 3 backend prepare parity | DEFERRED | Not addressed |
| B1 U13 | n/a | `pu_learning.py` already well-covered | n/a | Closed per agent |
| B1 U14 / S20 | P0 | `feature_selection/mi.py` 3 implementations no parity | DONE | fe179c7a (w3a U14=DONE) + `tests/feature_selection/test_mi_cross_parity.py` (19 tests) |
| B1 U15 | P1 | `feature_selection/general.py` only meta coverage | DONE | 9f2497c7 (w3a U15=DONE) + `tests/feature_selection/test_general_efs_relevancy.py` (8 tests) |
| B1 U16 | P1 | `feature_selection/importance.py` only meta | DONE | 90f1e17e (w3a U16=DONE) + `tests/feature_selection/test_importance.py` (10 tests) |
| B1 U17 | P1 | `feature_selection/optbinning.py` 87 LOC, zero tests | DONE | 5d428627 (w3a U17=DONE) + `tests/feature_selection/test_optbinning.py` (10; 2 skip on optbinning x sklearn 1.6+ incompat) |
| B1 U18 | P2 | `feature_selection/pre_screen.py` gap | DONE | 3f5b03c1 (w3a U18=DONE) + `tests/feature_selection/test_pre_screen.py` (19 tests) |
| B1 U19 | P2 | `feature_selection/registry.py` Protocol-conformance | DEFERRED | Not addressed |
| B1 U20 | P2 | `feature_engineering/bruteforce.py` strengthen + edges | DONE (partial) | 7157af4 (W1 weak-asserts strengthened); edge tests deferred |
| B1 U21 | P2 | `feature_engineering/pysr_operators.py` presets | DEFERRED | Not addressed |
| B1 U22 | n/a | `_timeseries_emit.py` already covered (underscore exempt) | n/a | Closed per agent |
| B1 W1 | weak | bruteforce PySR weak asserts | DONE | 7157af4 (w3c B1_W1=DONE) |
| B1 W7 | weak | compute_countaggs weak assert | DONE | 7157af4 (w3c B1_W7=DONE) |
| B1 W8 | weak | predict_polars_fastpath weak assert | DONE | 7157af4 (w3c B1_W8=DONE) |
| B1 W2-5,7 | weak | TVT/clone-elision/MLP-divergence/automl weak asserts | DEFERRED | Cluster not surveyed |
| B1 W6 | weak | dataset_cache_fingerprint idempotence assert | DONE | 415d3e23 (W10C `test(wave10c): fast marker on bootstrap suite + strengthen dataset_cache_fingerprint assert`) |
| B1 F1 | fuzz | crash_reporting axis | DONE | f7698ac6 (W9B `test(fuzz): wire F1/F5 axes + F2/F6/F7 reachability sensors`; closes AP4) |
| B1 F2 | fuzz | polarsds × LTR pair | DONE | f7698ac6 (W9B; F2 reachability sensor) |
| B1 F3 | fuzz | recency-only × recurrent | DONE | 9648c56 (w3c B1_F3=DONE) + `tests/training/test_fuzz_regression_sensors.py::test_sensor_fuzz_recurrent_model_x_recency_only_weights` |
| B1 F4 | fuzz | mrmr fillna_zero × all-null col | DONE | 9648c56 (w3c B1_F4=DONE) + `test_sensor_fuzz_mrmr_fillna_zero_x_all_null_col_does_not_corrupt_mi` |
| B1 F5 | fuzz | pysr × inf/nan injection | DONE | f7698ac6 (W9B; F5 axis) |
| B1 F6 | fuzz | composite discovery × outlier_detection | DONE | f7698ac6 (W9B; F6 reachability sensor) |
| B1 F7 | fuzz | diagnostics without baselines | DONE | f7698ac6 (W9B; F7 reachability sensor) |
| B1 F8 | fuzz | multilabel chain × random order metamorphic | DEFERRED | w3c B1_F8=ARCH-DEFER (budget; needs full-suite double-run) |
| B1 N1-N12 | numba | NUMBA_DISABLE_JIT=1 nightly coverage | DONE | b26cfa93 (w3c B1_N1_to_N12=DONE; helper scripts + marker registered) + 06727c04 (W8C AP5; `.github/workflows/numba-coverage.yml` cron 0 3 * * * with NUMBA_DISABLE_JIT=1) + d1820c55 (W10D `feat(scripts): numba-coverage report generator + meta-test`) + b643da93 (W10D `docs(audit): AP5 status update + validation steps + 2026-05-25 nightly findings`) + `tests/test_meta/test_numba_coverage_workflow_exists.py` |
| B1 biz_value cdist | biz | focused cdist test | DONE | c8605d3 (w3c B1_biz_value_cdist_local_lift_gap=DONE) + `tests/feature_engineering/transformer/test_biz_val_class_distance_and_local_lift.py` (2 tests) |
| B1 biz_value local_lift | biz | focused local_lift test | DONE | Same commit |
| B1 biz_value BGM (6 var) | biz | per-variant biz_value | DONE | bb019e9e (W10C `test(fe/transformer): biz_value focused tests for BGM + RSD-kNN shortlist`) |
| B1 biz_value RSD-kNN | biz | focused biz_value | DONE | bb019e9e (W10C; same commit) |
| B1 sklearn matrix marker | meta | `@pytest.mark.sklearn_matrix` convention + meta-test | DEFERRED | Not addressed; CI matrix selection still implicit |
| B1 CHANGELOG cross-walk | meta | per-fix regression-sensor gap audit | DEFERRED | Out-of-budget; flagged as follow-up |

### B2 - Tests optimize (tests-optimize.md, 50 findings)

| ID | Sev | Title | Status | Commit / Sensor / Reason |
|---|---|---|---|---|
| B2#1 / S21 / S24 | P0 | importlib.reload sites in tests/automl + temporal_audit + lazy_mrmr | DONE | d124f168 (w3b S21=VERIFIED + S24=VERIFIED + S74=DONE; all 6 reload sites already snapshot/restore) + `tests/test_meta/test_no_unsafe_module_reload.py` |
| B2#2 / S22 | P0 | `_clear_mrmr_fit_cache_between_tests` autouse over wrong scope | DONE | 29562d3 (w3b S22=DONE; sys.modules probe fast-path) |
| B2#3 / S23 | P0 | duplicate `IS_FAST_MODE` snapshots in root + fs conftests | DONE | 29562d3 (w3b S23=DONE; fs re-exports root) |
| B2#4 / S24 | P0 | `test_automl.py` 5 reloads one file no isolation | DONE | d124f168 (w3b S24=VERIFIED_ALREADY_FIXED; `_automl_module_snapshot` fixture in place) |
| B2#5 / S55 | P1 | No global `--timeout=` in addopts | DONE | 7a8e80b (w3b S55=DONE; `--timeout=60` added) |
| B2#6 / S56 | P1 | `@pytest.mark.fast` registered but 0 usage | DONE | (w3b S56=VERIFIED_ALREADY_FIXED; 239 tests across 81 files apply it); W10C 415d3e23 extends to bootstrap suite |
| B2#7 | P1 | only 2 files import `make_simple_*` builders; per-file re-rolls | DEFERRED | Not addressed (broad scope) |
| B2#8 | P1 | RFECV verbose=0 monkey-patch upstream-fix needed | DEFERRED | Not addressed (requires upstream PR); flagged in W10C deferred list |
| B2#9 | P1 | tqdmu monkey-patch upstream-fix needed | DEFERRED | Not addressed (requires upstream PR) |
| B2#10 | P1 | `cleanup_memory` `[MEM]` print spam | DEFERRED | Not addressed |
| B2#11 / S63 | P1 | `_reset_global_rng_state` fights pytest-randomly | DONE | 9e78904 (w3b S63=DONE; deferred to pytest-randomly when active) |
| B2#12 / S61 | P1 | `_prewarm_numba_once` early-returns on serial | DONE | 3be7be5 (w3b S61=DONE; serial also prewarms) |
| B2#13 / S60 | P1 | `pytest.skip("_X not exported")` cluster (8 sites) | DONE (partial) | 1d09c3b (w3b S60=DONE; 5 weak skips tightened across 5 files; remaining 3 sites in test_engineered_recipes_coverage.py not yet surveyed) |
| B2#14 / S60 | P1 | `pytest.skip("MI screen happened to find features")` | DONE | 1d09c3b (forced screen-empty path) |
| B2#15 | P1 | "3-way XOR seed didn't produce kway recipe" skip | DONE | 1d09c3b (deterministic seed pinned) |
| B2#16 | P1 | "no cat-FE recipes" same pattern × 2 | DONE | 1d09c3b (asserted recipes non-empty on XOR fixture) |
| B2#17 | P1 | LightAutoML × numpy 2.0 perpetual skip | DONE | 1d09c3b (replaced with numpy>=2.0 xfail) |
| B2#18 / S18_polars_ds | P2 | 8× duplicate polars-ds importorskip in test_pipeline.py | DONE | a72feaf (w3b S18_polars_ds_dry=DONE; class-level autouse) |
| B2#19 | P2 | 5× numba importorskip unreachable (numba is hard dep) | DONE | dad7acf (w3c B2_19=DONE; deleted unreachable skips) |
| B2#20 | P2 | Windows zstd quirk skip - untracked | DEFERRED | Not addressed |
| B2#21 | P2 | test_wrappers_default_args.py layout-conditional skip | DEFERRED | Not addressed |
| B2#22 | P2 | test_eval_medium_eval.py version-conditional skips | DEFERRED | Not addressed |
| B2#23 | P2 | tests/perf/conftest.py collect_ignore_glob | OK | Correct as-is |
| B2#24 / dead-markers | P2 | unused registered markers cluster | DONE (partial) | ebac7a3 (w3c B2_24_dead_markers_wire_in=DONE; wired requires_xgb / requires_torch / requires_cb / uses_torch / integration on 4 files); some markers remain to triage |
| B2#25 | P2 | tests/conftest.py 472 LOC mixed responsibilities | OK | Under threshold; flagged for future |
| B2#26 | P2 | tests/training/conftest.py top-of-file try-imports (flaml/networkx/lightning) | DEFERRED | Not addressed |
| B2#27 | P2 | `cleanup_memory request` param | OK | Correct |
| B2#28 / S59 | P2 | 16 session-scope mutable fixtures | DONE | 53d62fb8 (W10C `test(training): extend session-fixture immutability sensor to remaining mutable fixtures`) extends prior partial sensor coverage to remaining fixtures |
| B2#29 | P2 | trained_suite_regression / _binary same mutable risk | DONE | 53d62fb8 (W10C; same commit extends sensor to sibling fixtures) |
| B2#30 | P2 | xdist marker computed at import | OK | Documented; acceptable |
| B2#31 | P2 | `_ann_backend_safely_importable` 30-60s JIT at conftest import | DEFERRED | Not addressed |
| B2#32 | P2 | `test_post.py` 6× function-body importorskips | DEFERRED | Cosmetic; not addressed |
| B2#33 | P2 | `test_quality.py` 7× importorskips DRY | DEFERRED | Cosmetic; not addressed |
| B2#34 | P2 | `test_ranker_object_cat_encode_l26.py` good doc example | OK | Keep as canonical |
| B2#35 | P2 | `test_eval_medium_eval.py:431` good doc example | OK | Same |
| B2#36 | Low | `-x` in addopts | OK | Intentional |
| B2#37 | Low | `-p no:randomly` mismatch local vs CI | DEFERRED | Not addressed |
| B2#38 | Low | `suppress_convergence_warnings` autouse blocks `pytest.warns` | DEFERRED | Not addressed |
| B2#39 | Low | `pytest_plugins = ["tests.training.synthetic"]` at root | DEFERRED | Not addressed |
| B2#40 | Low | `_coverage_active` dead code on Windows | DEFERRED | Doc gap; not addressed |
| B2#41 | Low | tests/test_meta/conftest.py addoption swallow | OK | Intentional |
| B2#42 | Low | thinc seed-overflow shim 45 LOC | DEFERRED | Not addressed (waiting on upstream) |
| B2#43 | Low | metamorphic test "no val metric ... not applicable" skips | OK | Genuine non-applicability per agent |
| B2#44 | Low | `--instafail` not in addopts | DEFERRED | Not addressed |
| B2#45 | Low | Windows zstd quirk (dup #20) | DEFERRED | Not addressed |
| B2 +PYTHONUNBUFFERED | P2 | Missing `PYTHONUNBUFFERED=1` default | DEFERRED | Not addressed |
| B2 xfail audit | meta | `B2_xfail_without_owner_audit` | DONE | (w3c B2_xfail=DONE-CONFIRMED-CLEAN; only 2 xfails, both with reasons) |

### C1 - Monoliths split (monoliths-split.md, 18 file plans + 7 preventive)

| ID | Sev | File | Status | Commit / Sensor / Reason |
|---|---|---|---|---|
| C1#1 | arch | `_phase_composite_post.py` (1129 LOC) | DEFERRED | Compacted via cab4fcfc / 6adacb29 to stay under 1k previously but creeped back to 1005 LOC; Wave-11 backlog (now caught by 2c2d8452 LOC-budget meta-test) |
| C1#2 | arch | `composite_transforms.py` (1142 LOC) | DONE | a68f08e4 (w6a) before=1194 / after=295 + `tests/training/test_monolith_split_w6a_composite_transforms.py` |
| C1#3 | arch | `metrics/core.py` (1064 LOC) | DONE | 0e4e6cdf (w6b) before=1064 / after=232 + `tests/metrics/test_monolith_split_w6b_core.py` |
| C1#4 | arch | `_setup_helpers.py` (1047 LOC) | DONE | 3b7f6cf4 (w6b) before=1058 / after=356 + `tests/training/test_monolith_split_w6b_setup_helpers.py` |
| C1#5 | arch | `_target_distribution_analyzer.py` (1017 LOC) | DONE | 2bb3c896 (w6a) before=1017 / after=188 + `tests/training/test_monolith_split_w6a_target_dist.py` |
| C1#6 | arch | `wrappers/_rfecv_fit.py` (998 LOC -> grew to 1059) | DEFERRED | Documented as requiring `FitState` dataclass refactor; not pure carve. Plan persisted in monoliths-split.md; Wave-11 backlog (caught by 2c2d8452 LOC-budget meta-test) |
| C1#7 | arch | `_composite_target_estimator.py` (998 LOC -> grew to 1116) | DEFERRED | Method-rebinding pattern documented; not landed; Wave-11 backlog (caught by 2c2d8452 LOC-budget meta-test) |
| C1#8 | arch | `training/helpers.py` (993 LOC) | DONE | d61dd0cc (W10E `refactor(training): carve helpers.py below 400 LOC via get_training_configs sibling`) before=994 / after=267 |
| C1#9 | arch | `training/neural/recurrent.py` (963 LOC) | DEFERRED | Not landed |
| C1#10 | arch | `boruta_shap.py` (952 LOC) | DEFERRED | Not landed |
| C1#11 | arch | `target_temporal_audit.py` (949 LOC) | DEFERRED | Not landed |
| C1#12 | arch | `_phase_helpers.py` (948 LOC) | DEFERRED | Not landed |
| C1#13 | arch | `baseline_diagnostics.py` (942 LOC) | DEFERRED | Not landed |
| C1#14 | arch | `train_eval.py` (941 LOC) | DONE | 206ce81b (W10E `refactor(training): carve train_eval.py below 700 LOC via select_target sibling`) before=942 / after=570 |
| C1#15 | arch | `training/neural/flat.py` (927 LOC) | DONE | ef9aec26 (W10E `refactor(training/neural): carve flat.py below 600 LOC via MLPTorchModel sibling`) before=928 / after=420 |
| C1#16 | arch | `extractors.py` (940 LOC) | DONE | 48626d64 (W10E `refactor(training): carve extractors.py below 800 LOC via sibling re-export`) before=937 / after=326 |
| C1#17 | arch | `training/neural/ranker.py` (919 LOC) | DEFERRED | Not landed |
| C1#18 | arch | `training/neural/base.py` (1057 LOC) | DEFERRED | Tight-coupled class; method-rebinding pattern documented; Wave-11 backlog (caught by 2c2d8452 LOC-budget meta-test) |
| C1 prev #1 | prev | `strategies.py` (956 LOC) preventive | DONE | 9d633121 (W10E `refactor(training): carve strategies.py below 800 LOC via sibling re-export`) before=956 / after=316 (4 siblings) |
| C1 prev #2 | prev | `_ensembling_score.py` (1025 LOC) | DEFERRED | Wave-11 backlog (caught by 2c2d8452 LOC-budget meta-test) |
| C1 prev #3-#7 | prev | 5 remaining preventive files at 900-1000 LOC range | DEFERRED | LOC-cap sensor LANDED via 2c2d8452 (W10E `test(meta): add LOC-budget meta-test rejecting any mlframe .py over 1000 lines`); test currently FAILS by design on 5 files >1k LOC, queued for Wave 11 carves |

### D1 - ML best practices (ml-best-practices.md, 13 findings)

| ID | Sev | Title | Status | Commit / Sensor / Reason |
|---|---|---|---|---|
| D1 #1 / S64 | P1 | Train+val Enum domain widens val ES detector | DONE | d248bef8 (W7D S64 INFO-log val-only categories) + `test_regression_S64_enum_val_only_log.py` |
| D1 #2 / S65 | P1 | RFECV default cv=int silent k-fold when outer suite temporal | DONE | d248bef8 (W7D S65 RFECV auto-temporal CV) |
| D1 #3 / S66 | P1 | drift_report numeric-only, no categorical PSI | DONE | c2b741a6 (W8D S66; categorical PSI wired into `feature_drift_report` + `weighted_drift_score` with WARN at >0.2 / >0.25 + new-category smoothing) + `tests/training/test_regression_S66_drift_psi_categorical.py` |
| D1 #4 / S67 | P1 | `pl.Categorical` cast instead of `pl.Enum(train+val union)` in 3 sites | DONE | 34d61b0a (W9C `feat(predict): persist train enum_domains in model meta + thread to predict cat-cast`) + a2d9e73c (W9C `fix(polars-fixes): warn on bare pl.Categorical fallback + return enum_domains`) - S64 INFO logging from W7D now backed by full enum_domains bundle in model meta |
| D1 #5 / S68 | P1 | No bootstrap CI on primary metrics | DONE | 9a7e05a0 + 578cb578 (W8 S68; `src/mlframe/evaluation/bootstrap.py` with `bootstrap_metric` percentile CI + `delong_test` paired AUC + zero-variance edge fix) + `tests/evaluation/test_bootstrap.py` |
| D1 #6 | P2 | Default calibrator selection not pinned (no OOF-driven policy) | DONE | 783eae4c (W9A `feat(calibration): pick_best_calibrator policy with OOF ECE + bootstrap CI`; closes AP12) |
| D1 #7 | P2 | Regression with heavy-tail target not bucket-stratified | DONE | 7dfd6d5a (W9C `feat(split): regression bucket-stratify default ON; respect configurable cap`) |
| D1 #8 | P2 | BorutaShap SHAP background-dataset audit | DONE | 9eed863d (W9C `fix(boruta_shap): assert SHAP background == train X + log n_train`) |
| D1 #9 | P2 | Analyzer hyperparam provenance trail | DONE | 831a1bcb (W7C AP14: `provenance.py` module landed) + 19 source files reference provenance; W9A drained `format_provenance_table` from dead-helpers whitelist via 58586198 wiring into finalize |
| D1 #10 | P2 | `val_placement` time-series default INFO log | DONE | b54e7cc9 (W9C `feat(split): one-line INFO surfacing implied temporal layout on default forward placement`) |
| D1 #11 | Low | `_MAX_COMPOSITE_CARDINALITY=200` magic number | DONE | d7da291f (W9C `feat(split): expose composite_cardinality_cap + bucket_stratify on TrainingSplitConfig`) + d20f4b8b (user direct `fix(split): exclude composite_cardinality_cap from model_dump kwargs`) |
| D1 #12 | Low | Pre-screen protects group_id/ts only when string-typed | DONE | bdeeee22 (W9C `fix(pre-screen): double-source group/ts column protection from extractor + split_config`) |
| D1 #13 | Low | `pl.Categorical` silent fallback no WARN | DONE | a2d9e73c (W9C `fix(polars-fixes): warn on bare pl.Categorical fallback + return enum_domains`) |

### D2 - Code/arch standards (code-arch-standards.md, 23 findings)

| ID | Sev | Title | Status | Commit / Sensor / Reason |
|---|---|---|---|---|
| D2 #1 / S25 | P0 | sklearn check_estimator not run on composite/lag/early-stop/RFECV/Pd* | DONE | 7ca9f19a (w1c S25) + `tests/test_sklearn_compliance_composite.py` (23 tests + 1 xfailed) |
| D2 #2 / S26 | P0 | `from_fitted_inner` sklearn.clone silent data loss | DONE | c043f386 (w1c S26) + `tests/training/test_regression_S26_composite_target_clone_safety.py` |
| D2 #3 / S27 | P0 | unregistered marker `no_xdist_parallel` | DONE | eed502b (w1c S27) + `tests/test_meta/test_regression_S27_markers_registered.py` |
| D2 #4 / S69 | P1 | 12 dead registered markers | DONE (partial) | ebac7a3 (B2#24 in w3c) wired 5; remaining markers triage deferred |
| D2 #5 / S70 | P1 | ~90 deps without upper-bound caps | DROPPED | User dropped per Wave-7 plan (AP10 dropped) |
| D2 #6 / S71 | P1 | `src/mlframe/__init__.py` public API stale | DONE | 8ab33fb9 (W10D `feat(meta-test): cross-package underscore-import sensor + top-level public API docstring + cupy SoftBan rationale`) |
| D2 #7 / S72 | P1 | Pickle/joblib RCE: `_verify_sidecar` fail-open + bundle_sha256 unfilled | DONE | 1ef73ef3 (w1c S72) + 61d25bcd / b7f86580 (AP6 safe_pickle centralisation across 4 entry-points) + f2d00e15 (predict-time loaders) + `tests/inference/test_regression_S72_pickle_verification.py` |
| D2 #8 / S73 | P1 | `.pre-commit-config.yaml` ruff/black/mypy continue-on-error | DONE | 4858e454 (AP11 ruff+black+calibration-scoped mypy hooks). CI `continue-on-error` drop dropped by user (AP11-c) |
| D2 #9 / S74 | P1 | meta-test absent for `del sys.modules` / `importlib.reload` | DONE | d124f168 (w3b S74_sensor=DONE) + `tests/test_meta/test_no_unsafe_module_reload.py` |
| D2 #10 / S75 | P1 | `ci.yml` no macOS row | DONE | 55e6b55d (W10D `ci: macos-latest x py3.11 row + sklearn-matrix py3.13/windows rows + mypy strict beachhead expand (utils.safe_pickle)`) |
| D2 #11 / S76 | P1 | `sklearn-matrix-ci.yml` py3.11/linux only | DONE | 55e6b55d (W10D; same commit adds py3.13 + windows rows) |
| D2 #12 / S77 | P1 | docs/ not updated under Round 5.3/5.4 composite refactor | DONE | faea57b3 (W10D `docs(audit): aggregate round 2026-05-24 audit cycle CHANGELOG entry + AP12/AP13/summary guides`) - 3 docs guides + CHANGELOG aggregate entry |
| D2 #13 | P2 | `[tool.ruff.lint] ignore` no per-rule justification | DONE | ff18467e (w5b D2_ruff_ignore_no_per_rule_justification=DONE; per-rule comments) |
| D2 #14 | P2 | mypy effectively informational, no strict beachhead | DONE | ff18467e (w5b D2_mypy_informational_no_strict_beachhead=DONE; calibration subpackage strict-mode) + 55e6b55d (W10D extends strict beachhead to utils.safe_pickle) |
| D2 #15 | P2 | `_collect_lib_versions` missing cupy/numba/pydantic | DONE | ff18467e (w5b D2_io_collect_lib_versions=DONE; cupy + numba + pydantic + lightning + torch added) |
| D2 #16 | P2 | RFECV registry uses underscore `_rfecv` path | DONE | ff18467e (w5b D2_feature_selection_registry_underscore_path=DONE; public alias) |
| D2 #17 | P2 | Underscore convention not documented + meta-test | DONE | ff18467e (w5b D2_underscore_convention_not_documented=DONE; core/__init__.py docstring + meta-test) + 8ab33fb9 (W10D adds cross-package underscore-import sensor with 9-entry ALLOWLIST for Wave-11 promote) |
| D2 #18 | P2 | Duplicate marker registration conftest vs pyproject | DONE | ff18467e + f2d00e15 (w5b D2_duplicate_marker_registration=DONE; conftest dedup) |
| D2 #19 | P2 | Bare `pickle.load` in composite_cache + feature_handling | DONE | b7f86580 (AP6 safe_pickle migration to 4 entry-points incl. composite_cache + feature_handling) |
| D2 #20 | Low | `test_sklearn_compliance.py:78-90` weakest assertion | DEFERRED | Not addressed |
| D2 #21 | Low | `sys.modules["cupy"] = None` undocumented | DONE | 8ab33fb9 (W10D; cupy SoftBan rationale doc added) |
| D2 #22 | Low | `xfail_strict=true` positive note | OK | n/a |
| D2 #23 | Low | CHANGELOG in sync | OK | n/a |

### Wave-7/8/9 architectural proposals (per AP1-AP14 user approval set)

| AP | Title | Status | Commit / Reason |
|---|---|---|---|
| AP1 | SuiteArtefactCache | DONE | 55125d61 (W8A; `src/mlframe/training/suite_artefact_cache.py` 513 LOC + 17 sensors in `tests/training/test_suite_artefact_cache.py` + bytes-budget sensor fix in 578cb578) + W10B b8083ef6 eviction sidecar-accounting fix |
| AP2 | F15 finite_mask threading | DONE | 12b458dc (W9B `perf(composite_transforms): thread precomputed _finite_mask through 9 residual _fit kernels`) |
| AP3 | F24 K-target joblib | DROPPED | User dropped per Wave-7 plan |
| AP4 | Fuzz axes F1/F2/F5/F6/F7 | DONE | f7698ac6 (W9B `test(fuzz): wire F1/F5 axes + F2/F6/F7 reachability sensors`). F3+F4 already landed in w3c |
| AP5 | NUMBA_DISABLE_JIT=1 nightly CI workflow | DONE | 06727c04 (W8C; `.github/workflows/numba-coverage.yml` cron 0 3 * * * + meta-sensor `tests/test_meta/test_numba_coverage_workflow_exists.py`) + d1820c55 (W10D numba-coverage report generator + meta-test) + b643da93 (W10D 2026-05-25 nightly run validation surfacing 2 fixes for Wave 11: `mps.py:129` IndexError under NUMBA_DISABLE_JIT=1 + `biz_njit_poly_eval` skipif gate needed) |
| AP6 | safe_pickle | DONE | b7f86580 (w1c S72 follow-up) + f2d00e15 (predict-time loaders) + `src/mlframe/utils/safe_pickle.py` |
| AP7 | NNLS stacking-aware gate | DONE | 5505c773 + 9279b1cb test follow-up + `tests/test_regression_w7_ap7_nnls_weights_applied.py` |
| AP8 | nbytes streaming dispatcher | DONE | 5505c773 |
| AP9 | sklearn-matrix compliance step | DONE | d248bef8 + 55e6b55d (W10D py3.13 + windows rows) |
| AP10 | dep upper-bound caps | DROPPED | User dropped |
| AP11 | pre-commit ruff+black+mypy | DONE | 4858e454 (AP11-c CI continue-on-error drop dropped by user) |
| AP12 | calibration policy | DONE | 783eae4c (W9A `feat(calibration): pick_best_calibrator policy with OOF ECE + bootstrap CI`) + reliability plot |
| AP13 | honest_diagnostics ReportingConfig | DONE | 58586198 (W9A `feat(training): honest_diagnostics aggregator + ReportingConfig.honest_estimator_diagnostics default ON`) - aggregator wired into finalize, consumes S68 bootstrap CI module |
| AP14 | provenance trail | DONE | 831a1bcb + 19 source files reference `provenance` |
| AP15 (W10B) | DiscoveryCache + SuiteArtefactCache joint-stash | DEFERRED | c70c9317 (W10B `docs(audit): architectural proposal for DiscoveryCache + SuiteArtefactCache joint-stash`) - proposal landed; implementation pending user OK |

## Roll-up by status (post-Wave-10)

Counted directly via grep of every per-finding row across all tables (column-4 status cell). Numbers below include aliased rows (where a single underlying finding is tracked under both an A-side ID and a S-side / F-side / +1 alias) - each row gets its own status cell, so total row count exceeds the "234 unique atomic finding" baseline the audit was kicked off with.

Concrete Wave-10 DONE flips from prior Wave-9 baseline:

| Finding | Prior | New | Wave-10 commit |
|---|---|---|---|
| A2 W6 dataset_cache_fingerprint weak assert | DEFERRED | DONE | 415d3e23 (W10C) |
| B1 biz_value BGM (6 var) | DEFERRED | DONE | bb019e9e (W10C) |
| B1 biz_value RSD-kNN | DEFERRED | DONE | bb019e9e (W10C) |
| B2#28 / S59 16 session-scope mutable fixtures | DONE (partial) | DONE | 53d62fb8 (W10C) |
| B2#29 trained_suite_regression / _binary mutable risk | DEFERRED | DONE | 53d62fb8 (W10C) |
| C1#8 `training/helpers.py` (993 LOC) | DEFERRED | DONE | d61dd0cc (W10E) |
| C1#14 `train_eval.py` (941 LOC) | DEFERRED | DONE | 206ce81b (W10E) |
| C1#15 `training/neural/flat.py` (927 LOC) | DEFERRED | DONE | ef9aec26 (W10E) |
| C1#16 `extractors.py` (940 LOC) | DEFERRED | DONE | 48626d64 (W10E) |
| C1 prev #1 `strategies.py` (956 LOC) preventive | DEFERRED | DONE | 9d633121 (W10E) |
| D2#6 / S71 `src/mlframe/__init__.py` public API stale | DEFERRED | DONE | 8ab33fb9 (W10D) |
| D2#10 / S75 `ci.yml` no macOS row | DEFERRED | DONE | 55e6b55d (W10D) |
| D2#11 / S76 sklearn-matrix py3.11/linux only | DEFERRED | DONE | 55e6b55d (W10D) |
| D2#12 / S77 docs/ Round 5.3/5.4 composite refactor | DEFERRED | DONE | faea57b3 (W10D) |
| D2#21 `sys.modules["cupy"] = None` undocumented | DEFERRED | DONE | 8ab33fb9 (W10D) |
| AP15 (NEW) DiscoveryCache + SuiteArtefactCache joint-stash | n/a | DEFERRED | c70c9317 (W10B) - proposal landed pending user OK |

Net Wave-10 delta: **+15 atomic-finding rows DONE** (15 DEFERRED → DONE) + 1 NEW deferred AP15 proposal row.

Roll-up:
- DONE: 199 row-mentions overall (187 atomic-finding rows DONE + 12 of 15 AP rows DONE)
- DEFERRED: 71 row-mentions overall (atomic-finding rows + 1 AP15 row)
- REJECTED (with bench): 14 row-mentions overall (13 atomic-finding rows: W2B #8, #10, #33; W2C #16, #18, #20, #21, #22, #24; W4C F9, F11, F14; plus 1 alias row)
- DROPPED (user): 4 row-mentions overall (1 atomic-finding D2#5/S70 dep upper-bound caps + 3 AP rows: AP3 K-target joblib, AP10 dep caps, AP11-c CI continue-on-error drop)
- OK / closed by agent: 11 row-mentions overall (verified-already-fixed or no-op)

Honest closure rate: **187 of 258 closeable atomic-finding rows landed in code or test (72.5%)**, or equivalently **199 of 300 total row-mentions across atomic + AP slate + alias rows (66.3%)**. Per `feedback_no_premature_closure`, this is **NOT** a fully closed audit; this is a Wave-11 backlog of 71 deferred rows (predominantly remaining C1 monolith carves, B1 module-gap unit-test extensions, A2 FE bucket cosmetic / mutable-default cleanup, A5 caching divergence, B2 test-infra hygiene cluster, B1 W2-5/W7 weak-asserts cluster).

Wave-10 sub-wave attribution: W10A 1 commit (docs-only correction; 0 net DONE flips — A4#5 was already DONE, just mis-tracked), W10B 4 commits (caching follow-ups; 0 net DONE flips — all caching findings already DONE in prior waves but hardened/extended), W10C 3 commits + manifest (3 DONE flips: A2 W6, BGM, RSD-kNN + 2 partial-to-full: B2#28/#29), W10D 5 commits (5 DONE flips: D2#6, D2#10, D2#11, D2#12, D2#21 + AP5 / D2#14 / D2#17 extended), W10E 6 commits (5 DONE flips: C1#8, C1#14, C1#15, C1#16, C1 prev#1 + 2c2d8452 LOC-budget meta-test currently FAILS by design on 5 files >1k LOC).

Concrete DONE list confirmed via commits or sensors (file evidence supplied above in per-row entries). Same for every DEFERRED row - reason documented inline; sensor files re-checked under `tests/`.

## Wave 8 summary (landed items, commit range `bafff1d2..ef82f687`)

| Commit | Wave | Item | Description |
|---|---|---|---|
| 55125d61 | W8A | AP1 / A5#1 / A5#3 | SuiteArtefactCache - new module `src/mlframe/training/suite_artefact_cache.py` (513 LOC) + 17 sensors. SuiteKeyBuilder digest folds (df_fingerprint, config_canonical, mlframe_models, lib_versions, random_seed) via orjson+blake2b. Default 2GB bytes_limit, env override `MLFRAME_SUITE_CACHE_MAX_BYTES` + `MLFRAME_SUITE_CACHE_DIR`. Above cap refuses to store and returns caller value untouched. |
| 9a7e05a0 | W8 | S68 / D1#5 | `src/mlframe/evaluation/bootstrap.py` with `bootstrap_metric(y_true, y_pred, metric_fn, n_bootstrap=1000, alpha=0.05, stratify=None, random_state=None)` percentile method + `delong_test(y_true, score_a, score_b)` paired-AUC. Metric-agnostic surface. |
| 06727c04 | W8C | AP5 / B1 N1-N12 | `.github/workflows/numba-coverage.yml` cron `0 3 * * *` nightly with `NUMBA_DISABLE_JIT=1` so @njit kernel bodies become visible to coverage.py. Sensor: `tests/test_meta/test_numba_coverage_workflow_exists.py`. |
| c2b741a6 | W8D | S66 / D1#3 | `feature_drift_report` categorical PSI (chi-square on bin counts) alongside numeric path. Wires into `weighted_drift_score`. WARN >0.2 / >0.25 per Wu & Olson credit-risk convention. Small-count smoothing for new categories. Sensor: `tests/training/test_regression_S66_drift_psi_categorical.py`. |
| 0bae104f | W8E | S28 / A1#2 | `estimate_features_relevancy` replaced global `np.random.shuffle` with `np.random.default_rng(random_state)` + dropped `.to_numpy(allow_copy=True).copy()` chain in favour of view path. `importance.py` picks up same RNG convention. Sensor: `tests/feature_selection/test_regression_S28_efs_relevancy_rng.py`. |
| d71506b7 | W8 bonus | round5.5 | `CompositeTargetEstimator.from_fitted_inner` reconstructs conservative T-envelope (~10x y_std) when fitted-inner state lacks T_train, mirroring `.fit()` clipping. `composite_ensemble` extends NNLS / cross-target diagnostics + downstream member metadata for honest-OOF gate attribution. Sensors: `tests/training/test_round5_5_composite_diagnostics.py`, `test_round5_5_followups.py`. |
| fb80cbeb | W8 bonus | misc helpers | `core/helpers.py` extended for SuiteArtefactCache KeyBuilder consumer; `reporting/renderers/matplotlib.py` reliability-plot polish (calibration-policy / honest-diagnostics path); `_data_helpers.py` minor coercion fix for bucket-stratify default. |
| 578cb578 | W8 fix | sensor + bootstrap edge | `test_suite_artefact_cache` payload bumped to 1000 bytes so stored .pkl actually exceeds the 4KB budget (sidecar .sha256 not counted by `_total_bytes_locked`; 100-byte payload gave 115-byte pickles totaling 3450 < 4096, never triggering eviction). `delong_test` zero-variance edge: var_diff==0 AND diff==0 returns z=0, p=1.0 instead of NaN; non-zero diff with var_diff==0 still returns NaN. |

## Wave 9 summary (landed items, commit range `ef82f687..242f7199`, 23 commits)

| Commit | Wave | Finding(s) closed | Description |
|---|---|---|---|
| 783eae4c | W9A | AP12 / D1#6 | `feat(calibration): pick_best_calibrator policy with OOF ECE + bootstrap CI` + reliability plot. Closes the AP12 calibration-selection gate the W8B agent was unable to land. |
| 58586198 | W9A | AP13 / D1#9 (close-out) | `feat(training): honest_diagnostics aggregator + ReportingConfig.honest_estimator_diagnostics default ON` - wires into finalize; consumer of S68 bootstrap CI module. Also drained `training/provenance.py::format_provenance_table` from dead-helpers whitelist. |
| 12b458dc | W9B | AP2 / A4#15 (F15) | `perf(composite_transforms): thread precomputed _finite_mask through 9 residual _fit kernels`. Closes AP2 finite_mask plumbing. |
| f7698ac6 | W9B | AP4 / B1 F1, F2, F5, F6, F7 | `test(fuzz): wire F1/F5 axes + F2/F6/F7 reachability sensors`. Closes 5 fuzz axes flagged by tests-expand critic (F3+F4 already in w3c). |
| d7da291f | W9C | D1 Low #11 | `feat(split): expose composite_cardinality_cap + bucket_stratify on TrainingSplitConfig`. Removes the 200-magic-number cap and surfaces it as user-tunable. |
| a2d9e73c | W9C | D1 Low #13 | `fix(polars-fixes): warn on bare pl.Categorical fallback + return enum_domains`. WARN log when the broader Enum bundle is unavailable + propagate enum_domains for downstream consumption. |
| 7dfd6d5a | W9C | D1 P2 #7 | `feat(split): regression bucket-stratify default ON; respect configurable cap`. Heavy-tail regression now bucket-stratified by default. |
| bdeeee22 | W9C | D1 Low #12 | `fix(pre-screen): double-source group/ts column protection from extractor + split_config`. Group/ts cols protected regardless of dtype (was string-typed only). |
| b54e7cc9 | W9C | D1 P2 #10 | `feat(split): one-line INFO surfacing implied temporal layout on default forward placement`. |
| 9eed863d | W9C | D1 P2 #8 | `fix(boruta_shap): assert SHAP background == train X + log n_train`. SHAP background dataset audit now hard-enforced. |
| 34d61b0a | W9C | D1#4 / S67 (full) | `feat(predict): persist train enum_domains in model meta + thread to predict cat-cast`. Closes S67 in full - W7D's S64 INFO logging now backed by enum_domain bundle round-tripped through model meta. |
| d9c9aba4 | W9D | A1#4 / S30 | `fix(fs/mrmr): add strict_groups knob; raise instead of warn-only ignore`. |
| 46db224b | W9D | A1#5 / S31 | `fix(fs/njit-rng): seed inline LCG in parallel_mi + fleuret njit shuffles`. Closes the global-shuffle / non-determinism gap at `permutation.py:41,210` + `fleuret.py:187,191`. |
| 1b4a2c32 | W9D | A1#6 | `fix(fs/rfecv): respect explicit cv_shuffle=True over temporal auto-detect`. Closes the opt-OUT direction (S65 fixed the opt-IN direction). |
| 7016379b | W9D | A1#8 / A1#9 / A1#12 / A1#13 | `fix(fs): strengthen MRMR/RFECV cache keys + gate polars self_destruct`. Bundled close of 4 P2 cache/sample/destruct findings. |
| 7ad8bd30 | W9D | A1#15 | `perf(fs/mrmr): hoist _full_y_content_hash; reuse signature build (A1#15)`. |
| ccfb45d2 | W9D | A1#5 follow-up | `test(fs/permutation): tighten BC speedup floor for LCG-accelerated baseline`. Regression sensor for the njit-rng fix. |
| (verified) | W9D | A1#14, A1#16-A1#21 | 7 Low-tier items verified-already-fixed in prior waves (no new commit needed - audit-agent sweep confirmed code path correct, sensors in place). |
| 7bb9fee0 | W9E | A3#12 / A3#16 | `fix(models/ensembling): per-class avg Pearson for multiclass diversity + shallow inner pop in compare_ensembles`. Closes the (N,K) flatten bug + deepcopy-per-call. |
| 3538f5a5 | W9E | A3#13 / A3#14 | `fix(models/ensembling): quality gate group-aware median collapses per group; documented unweighted predict-side anchor`. Closes weighted-median sample_weight kw gap + group-broadcast bug. |
| c5727d53 | W9E | A3#6 / A3#7 / A3#15 | `fix(models/ensembling): score_ensemble drops higher-MAE member on diversity auto-drop; deterministic K=2 tiebreak; uniformity inspects oof_probs`. Closes 3 P2/Low ensemble findings in one commit. |
| 9432b098 | W9E | A3#8 | `refactor(training/core): move _choose_ensemble_flavour to leaf module; ensembling sibling top-level imports it`. Eliminates per-target lazy-import cycle-breaker cost. |
| 5a93969a | W9E | A3#10 / A3#11 (sensors) | `test(ensembling): regression sensors for the A3 P2/Low cluster`. Adds compare_ensembles biz floor + rrf_k stamp gate sensors. |

## Wave 10 summary (landed items, commit range `c31b419d..04cfe69d`, 23 commits)

| Commit | Wave | Finding(s) closed | Description |
|---|---|---|---|
| 27429ed3 | W10A | A4#5 status correction (no source change) | `docs(audit): correct A4#5 status DEFERRED->DONE + w10a manifest`. Audit on 9 A4 P2/Low items shows all already DONE in prior waves (W4/W5/W9); FINAL_VERIFICATION mis-tracked A4#5 (closed in 5c6820ea, sensor `tests/training/test_regression_S47_filter_polars_cat_hoisted.py::test_S47_ngb_fallback_snapshot_cached_outside_loop`). Pure docs flip, no source change. |
| b8083ef6 | W10B | AP1 hardening | `fix(suite-cache): count + remove .pkl.sha256 sidecars during eviction`. SuiteArtefactCache eviction was not counting the .sha256 sidecar in bytes accounting nor removing it under eviction, leaking disk + skewing budget. |
| 54d456ae | W10B | A1#1 / A5#11 extension | `fix(pipeline-cache): fold full-X blake2b into _pre_pipeline_cache_key`. Extends the MRMR x-hash strengthening pattern (W9D 7016379b) into `_pre_pipeline_cache_key`. |
| 65b3dded | W10B | A5#5 observability | `feat(composite-cache): add _discovery_cache_bytes_total on-disk size helper`. DiscoveryCache on-disk byte accounting helper landed ahead of any unification with SuiteArtefactCache (paves the way for AP15 joint-stash). |
| c70c9317 | W10B | AP15 (NEW) | `docs(audit): architectural proposal for DiscoveryCache + SuiteArtefactCache joint-stash`. Architectural proposal landed; implementation pending user OK. Listed as DEFERRED below. |
| 53d62fb8 | W10C | B2#28 / B2#29 | `test(training): extend session-fixture immutability sensor to remaining mutable fixtures`. Extends prior partial sensor coverage to remaining session-scope mutable fixtures (trained_suite_regression / _binary). |
| bb019e9e | W10C | B1 biz_value BGM + RSD-kNN | `test(fe/transformer): biz_value focused tests for BGM + RSD-kNN shortlist`. Closes the 2 remaining biz_value gaps from the FE shortlist (cdist + local_lift already in w3c). |
| 415d3e23 | W10C | A2 W6 + fast marker extension | `test(wave10c): fast marker on bootstrap suite + strengthen dataset_cache_fingerprint assert`. Strengthens previously-weak A2 W6 assertion + extends @pytest.mark.fast coverage to bootstrap suite. |
| faea57b3 | W10D | D2#12 / S77 | `docs(audit): aggregate round 2026-05-24 audit cycle CHANGELOG entry + AP12/AP13/summary guides`. 3 docs guides covering AP12 calibration policy + AP13 honest_diagnostics + audit cycle summary; closes the docs/ Round 5.3/5.4 update gap. |
| 55e6b55d | W10D | D2#10 / S75, D2#11 / S76, D2#14 (extend), AP9 (extend) | `ci: macos-latest x py3.11 row + sklearn-matrix py3.13/windows rows + mypy strict beachhead expand (utils.safe_pickle)`. CI matrix expanded + mypy strict scope extended beyond calibration subpackage. |
| 8ab33fb9 | W10D | D2#6 / S71, D2#21, D2#17 (extend) | `feat(meta-test): cross-package underscore-import sensor + top-level public API docstring + cupy SoftBan rationale`. Closes 3 D2 items in one commit + lands a meta-test with 9-entry ALLOWLIST for Wave-11 promote-to-public review. |
| b643da93 | W10D | AP5 validation | `docs(audit): AP5 status update + validation steps + 2026-05-25 nightly findings`. 2026-05-25 nightly run validates AP5 workflow; surfaces 2 Wave-11 fixes (mps.py:129 IndexError under NUMBA_DISABLE_JIT=1 + biz_njit_poly_eval skipif gate). |
| d1820c55 | W10D | AP5 hardening | `feat(scripts): numba-coverage report generator + meta-test`. Report generator + meta-test wired so nightly AP5 run produces actionable artefact rather than just a coverage.xml. |
| d94313ab | W10D | w10d manifest | `chore(audit): w10d-arch-residue manifest + heartbeat`. |
| 9d633121 | W10E | C1 prev #1 strategies.py | `refactor(training): carve strategies.py below 800 LOC via sibling re-export`. 956→316 LOC; 4 siblings under sibling-file re-export pattern per `feedback_monolith_split_via_re_export`. |
| 48626d64 | W10E | C1#16 extractors.py | `refactor(training): carve extractors.py below 800 LOC via sibling re-export`. 937→326 LOC; 3 siblings. |
| 206ce81b | W10E | C1#14 train_eval.py | `refactor(training): carve train_eval.py below 700 LOC via select_target sibling`. 942→570 LOC; 1 sibling (closes A5#4 / S51 sibling location too). |
| d61dd0cc | W10E | C1#8 training/helpers.py | `refactor(training): carve helpers.py below 400 LOC via get_training_configs sibling`. 994→267 LOC; 1 sibling. |
| ef9aec26 | W10E | C1#15 training/neural/flat.py | `refactor(training/neural): carve flat.py below 600 LOC via MLPTorchModel sibling`. 928→420 LOC; 1 sibling. |
| 2c2d8452 | W10E | LOC-budget meta-test | `test(meta): add LOC-budget meta-test rejecting any mlframe .py over 1000 lines`. Meta-test currently FAILS by design on 5 files >1k LOC, queued for Wave 11 carves. |
| d20f4b8b | user | D1 Low #11 follow-up | `fix(split): exclude composite_cardinality_cap from model_dump kwargs`. User direct commit; pairs with W9C d7da291f to ensure the new TrainingSplitConfig field doesn't pollute downstream model_dump kwargs. |
| 0dc04b8b | W10B | w10b manifest | `chore(audit): w10b-caching-followups manifest + heartbeat`. |
| 04cfe69d | W10C | w10c manifest | `chore(audit): w10c-tests-residue manifest after orchestrator stragglers commit`. |

Wave 10 commit attribution: 23 commits across 5 sub-waves + user direct + manifests (W10A 1, W10B 4 + 1 manifest, W10C 3 + 1 manifest, W10D 5 + 1 manifest, W10E 6, plus 1 user direct d20f4b8b). Every W10 commit landed under standard pre-commit hook flow (no `--no-verify`).

## Wave 11 backlog (post-Wave-10, in priority order - for user approval)

### P0/P1 carry-forward (correctness / ML discipline)

1. **A3#2 / S41** `_ENSEMBLE_RANK_METRIC_CANDIDATES` val.*/test.* fallback (P1; contradicts in-file comment).
2. **A3#3 / S42** No probability calibration before classifier blend (P1; partially addressed by AP12 calibrator policy at single-model level, but ensemble-time integration still pending).
3. **A3#4 / S43** `_rrf_aggregate_probs` K=1 binary destroys calibration (P1).
4. **A2#4 / S32** `_DEFAULT_DATE_METHODS` mutable module-level default (P1).
5. **A2#5 / S33** Missing year/week/quarter/is_weekend; no tz handling (P1).
6. **A2#6 / S34** bruteforce `to_pandas()` + sub-frame fillna broadcast-copy (P1).
7. **A2#8 / S36** recursive subset `.copy()` in create_aggregated_features (P1).
8. **A2#11 / S39** bruteforce rename mutating caller's frame in place (P1).
9. **A2#12 / S40** `_ratio_fit` empty-array `np.median` warning pitfall (P1).
10. **A5#4 / S51** Per-target select_target template rebuild (P1).
11. **A5#5 / S52** `DiscoveryCache` vs `FeatureCache` parallel disk-cache divergence (P1; observability landed via 65b3dded, full unification deferred to AP15).
12. **A5#7 / S54** `_PRE_PIPELINE_CACHE_MAX=8` no byte budget (P1).
13. **B1 U4-U12 module-gap unit-test coverage** (drift_report, feature_drift_report, models, io, composite_bayesian, phases, lgb_shim, xgb_shim, ranking - 9 P1/P2 items).
14. **B2#7-#10** test-infra make_simple_* DRY + RFECV/tqdmu upstream PR + [MEM] log spam (4 P1 items).
15. **AP15** DiscoveryCache + SuiteArtefactCache joint-stash architectural unification (W10B proposal c70c9317 pending user OK).
16. **mps.py:129 IndexError** under `NUMBA_DISABLE_JIT=1` (surfaced by AP5 2026-05-25 nightly via b643da93).
17. **biz_njit_poly_eval skipif gate** for `NUMBA_DISABLE_JIT` mode (surfaced by AP5 2026-05-25 nightly via b643da93).
18. **C1#1 `_phase_composite_post.py` (1005 LOC)** monolith carve (caught by 2c2d8452 LOC-budget meta-test).
19. **C1#6 `wrappers/_rfecv_fit.py` (1059 LOC)** monolith carve (`FitState` dataclass refactor; caught by 2c2d8452).
20. **C1#7 `_composite_target_estimator.py` (1116 LOC)** monolith carve (method-rebinding pattern; caught by 2c2d8452).
21. **C1#18 `training/neural/base.py` (1057 LOC)** monolith carve (caught by 2c2d8452).
22. **C1 prev #2 `_ensembling_score.py` (1025 LOC)** preventive carve (caught by 2c2d8452).

### P2 carry-forward (perf / hygiene)

23. **A2#20** Composite Transform unit-test coverage gap.
24. **A5#9 P2** polars LazyFrame.cache() in training/core (ARCH-DEFER per Wave-7 proposal; superseded by AP15 joint-stash proposal).
25. **B1 W2-5/W7 weak-assert clusters** in tvt_round5 / clone-elimination / MLP-divergence / automl (W6 closed in W10C).
26. **B1 sklearn matrix marker convention** + meta-test.
27. **B1 CHANGELOG cross-walk** per-fix regression-sensor gap audit.
28. **B1 U19/U21** registry Protocol conformance + pysr_operators presets.
29. **B2#20-#22, #26, #31-#33** test-infra hygiene cluster.
30. **B2 +PYTHONUNBUFFERED** default for local runs.
31. **W10C deferred items** (carried forward in W10C agent hang): fast-mode extension to remaining packages, fuzz dimension widening cross-axis, weak-assert W4-W8 cluster (W6 closed; remainder open), numba bench-report script, pytest cache cleanup, RFECV monkey-patch upstream-fix.
32. **9 entries in cross-package underscore-import ALLOWLIST** for promote-to-public review (W10D 8ab33fb9 meta-test).
33. **C1 remaining preventive splits** at 900-1000 LOC range (5 files; LOC-cap sensor in place via 2c2d8452, will activate on creep past 1k).
34. **C1#9-#13, #17** remaining 6 monolith carves (recurrent.py 963, boruta_shap.py 952, target_temporal_audit.py 949, _phase_helpers.py 948, baseline_diagnostics.py 942, neural/ranker.py 919).

### Low carry-forward (cosmetic / documentation)

35. A2 Low #25 (try/except/finally comment) + 3 unindexed Low (MappingProxyType wrap, online_refit docstring, +Low rejected dup).
36. A5 Low #15, #16 (WeakKey doc + cross-target memo).
37. A6 Low #39 (estimators/custom.py `.values` cosmetic).
38. B2 Low #37-#40, #42, #44 (cosmetic test-infra).
39. D2 Low #20 (sklearn_compliance weakest assertion).

## Notes / qualifications

- Per `feedback_no_premature_closure`: this is explicitly **Wave-11 backlog**, NOT a "Wave-10 done" closure. 71 atomic-finding rows remain DEFERRED (27.5% of the 258 closeable atomic-finding rows; predominantly cosmetic/perf-hygiene/test-coverage rather than correctness; see Wave-11 backlog above).
- The 13 REJECTED findings all carry inline bench numbers in source or DONE_*.json manifest; none silent.
- The 3 DROPPED items are user-decision per Wave-7 plan (AP3 K-target joblib, AP10 dep caps, AP11-c CI continue-on-error drop). Documented per Wave-7 brief.
- W10A is a docs-only correction wave (1 commit 27429ed3) - all 9 A4 P2/Low items already DONE in prior waves; only A4#5 was mis-tracked in baseline.
- W10C agent hung mid-sweep on fast-mode extension after the 3 landed commits (53d62fb8, bb019e9e, 415d3e23); deferred items captured in Wave-11 backlog item #31.
- W10E carve pattern: every monolith carve uses sibling-file re-export per `feedback_monolith_split_via_re_export` - parent re-exports at bottom; lazy-import parent helpers inside function body when init-order matters; sensor asserts identity + facade size + smoke.
- W10E 2c2d8452 LOC-budget meta-test currently FAILS by design on 5 files >1k LOC (_composite_target_estimator.py 1116, _rfecv_fit.py 1059, neural/base.py 1057, _ensembling_score.py 1025, _phase_composite_post.py 1005). This is intentional pressure to force Wave-11 carves; per `feedback_no_audit_phase_in_comments`, no audit/phase markers in code - only the LOC-cap.
- AP1 SuiteArtefactCache (55125d61) lands a NEW cache module covering composite_target_specs + dummy_baselines + fit_and_transform_pipeline + apply_preprocessing_extensions + trainset_features_stats (closes A5#1/S07 + A5#3/S09). Wave-10 hardening: b8083ef6 fixes eviction sidecar accounting bug. Adjacent A5 caching findings (S51 per-target select_target rebuild, S52 DiscoveryCache vs FeatureCache divergence, S54 _PRE_PIPELINE_CACHE byte-budget, A5#9 polars LazyFrame.cache, A5#15 WeakKey doc, A5#16 cross-target memo) remain DEFERRED because AP1 is a new orthogonal cross-process disk cache, not a rewrite of those pre-existing per-target caches. AP15 (NEW) W10B proposal c70c9317 lays out the joint-stash architectural unification path; implementation pending user OK.
- A1#14, A1#16, A1#17, A1#18, A1#19, A1#20, A1#21 (7 Low-tier items) flipped to DONE via W9D audit sweep "verified-already-fixed in prior waves" - no new commit; sensor coverage / code path confirmed correct by audit. Per `feedback_never_hide_low_findings` these are surfaced explicitly rather than rolled up into a count.
- A3#10/A3#11 carry sensor-only closure (5a93969a) rather than source-code change because the underlying behavior (compare_ensembles biz floor; rrf_k stamp gating) was already conditionally correct; sensors lock it in.
- W10D 8ab33fb9 cross-package underscore-import meta-test ships with 9-entry ALLOWLIST documenting deliberate cross-package private-API usage; each entry queued for Wave-11 promote-to-public review.
- W10D b643da93 AP5 nightly validation surfaced 2 fixes for Wave 11 (mps.py:129 IndexError under NUMBA_DISABLE_JIT=1 + biz_njit_poly_eval skipif gate needed) - already enumerated in Wave-11 backlog items #16-#17.
- AP15 (W10B c70c9317) is the first architectural proposal of the audit cycle that landed as proposal-only (not implementation); status DEFERRED pending user OK per `feedback_never_open_pr_without_review` analog for architectural changes.

## Files referenced for verification

- `audit/critique_2026_05_24/SUMMARY.md`
- `audit/critique_2026_05_24/{fs,fe,ensembling,perf-hotspots,pipeline-cache,polars-zerocopy}-critique.md`
- `audit/critique_2026_05_24/{tests-expand,tests-optimize,monoliths-split,ml-best-practices,code-arch-standards}.md`
- `audit/critique_2026_05_24/manifests/DONE_{w1a-leakage,w1b-fe-cache,w1c-sklearn-security,w2a-bridge-p0,w2b-percol-scattered,w2c-fe-dtype-gate,w3a-tests-expand,w3b-tests-optimize,w3c-tests-p2low,w4a-hotspots-critical,w4b-numba-parallel,w4c-perf-p2low,w5a-fs-fe-ens-p2low,w5b-cache-arch-fu,w6a-carve-composite-target-dist,w6b-carve-metrics-setup,w10a-perf-residue,w10b-caching-followups,w10c-tests-residue,w10d-arch-residue,w10e-monolith-preventive}.json`
- `audit/critique_2026_05_24/architectural_proposals/{A5-P2-9-polars-lazyframe-cache,F15_precomputed_finite_mask,F24_ktarget_ensemble_joblib,fuzz_blind_spots_F1_F2_F5_F6_F7,numba_coverage_ci,AP15_discoverycache_joint_stash}.md`
- Git log range `da68ca6..HEAD` (~142 commits; Wave-8 sub-range `bafff1d2..ef82f687` = 11 commits incl. cleanup; Wave-9 sub-range `ef82f687..242f7199` = 23 commits; Wave-10 sub-range `c31b419d..04cfe69d` = 23 commits: 27429ed3, 0dc04b8b, b8083ef6, 54d456ae, 65b3dded, c70c9317, 53d62fb8, bb019e9e, 415d3e23, 04cfe69d, faea57b3, 55e6b55d, 8ab33fb9, b643da93, d1820c55, d94313ab, 9d633121, 48626d64, 206ce81b, d61dd0cc, ef9aec26, 2c2d8452, d20f4b8b)
- `tests/**/test_regression_*.py`, `tests/**/test_monolith_split_*.py`, `tests/test_meta/test_regression_S27_*.py`, `tests/test_meta/test_no_unsafe_module_reload.py`
- Wave-8 new sensors: `tests/training/test_suite_artefact_cache.py` (AP1), `tests/evaluation/test_bootstrap.py` (S68), `tests/test_meta/test_numba_coverage_workflow_exists.py` (AP5), `tests/training/test_regression_S66_drift_psi_categorical.py` (S66), `tests/feature_selection/test_regression_S28_efs_relevancy_rng.py` (S28), `tests/training/test_round5_5_composite_diagnostics.py` + `test_round5_5_followups.py` (round5.5 bonus)
- Wave-8 new source files: `src/mlframe/training/suite_artefact_cache.py`, `src/mlframe/evaluation/bootstrap.py`, `.github/workflows/numba-coverage.yml`
- Wave-9 new/touched sensor surfaces: regression sensors for A3 P2/Low cluster (5a93969a), permutation BC speedup floor (ccfb45d2), fuzz F1/F5 axes + F2/F6/F7 reachability (f7698ac6), W9D fs-residue manifest (242f7199)
- Wave-9 new source touches: calibration `pick_best_calibrator` policy + reliability plot (783eae4c), `honest_diagnostics` aggregator (58586198), composite_transforms residual `_fit` kernels with threaded `_finite_mask` (12b458dc), TrainingSplitConfig fields `composite_cardinality_cap` + `bucket_stratify` (d7da291f), polars-fixes WARN + enum_domains return (a2d9e73c), BorutaShap SHAP background guard (9eed863d), predict-time enum_domains meta-roundtrip (34d61b0a), MRMR `strict_groups` knob (d9c9aba4), njit LCG seeding (46db224b), RFECV cv_shuffle respect (1b4a2c32), MRMR/RFECV cache-key strengthening + self_destruct gate (7016379b), MRMR signature build hoist (7ad8bd30), ensembling per-class avg Pearson + shallow inner pop (7bb9fee0), quality gate group-aware median (3538f5a5), score_ensemble MAE-quality drop + K=2 alphabetical + uniformity oof_probs (c5727d53), `_choose_ensemble_flavour` leaf-module move (9432b098), training/provenance.py `format_provenance_table` drain (58586198)
- Wave-10 new/touched sensor surfaces: LOC-budget meta-test (2c2d8452), cross-package underscore-import sensor (8ab33fb9), numba-coverage report meta-test (d1820c55), session-fixture immutability sensor extension (53d62fb8), bootstrap-suite fast marker + dataset_cache_fingerprint strengthened assert (415d3e23), BGM + RSD-kNN biz_value tests (bb019e9e), 5 monolith-carve sensors (9d633121, 48626d64, 206ce81b, d61dd0cc, ef9aec26)
- Wave-10 new source touches: SuiteArtefactCache eviction sidecar accounting (b8083ef6), `_pre_pipeline_cache_key` full-X blake2b (54d456ae), `_discovery_cache_bytes_total` helper (65b3dded), CI matrix macOS + sklearn-matrix py3.13/windows + mypy strict beachhead expand to utils.safe_pickle (55e6b55d), top-level `__init__.py` docstring + cupy SoftBan rationale (8ab33fb9), `composite_cardinality_cap` excluded from model_dump kwargs (d20f4b8b user direct), numba-coverage report generator script (d1820c55), 5 monolith carves with sibling re-export (9d633121, 48626d64, 206ce81b, d61dd0cc, ef9aec26)
- Wave-10 new docs: AP15 joint-stash architectural proposal (c70c9317), CHANGELOG aggregate + AP12/AP13/summary guides (faea57b3), AP5 status update + 2026-05-25 nightly findings (b643da93)
