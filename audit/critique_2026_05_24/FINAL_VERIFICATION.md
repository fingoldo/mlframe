# Final verification audit (2026-05-25)

Pass-through of every finding from the 11 Wave-0 critic agents + the Wave-7 architectural-proposal slate. Status derived from (a) per-wave DONE_*.json manifest, (b) git log `da68ca6..HEAD` (88 commits), (c) regression-sensor file existence under `tests/`, (d) targeted source greps for items not in any manifest.

Per `feedback_show_all_agent_findings` and `feedback_use_all_agent_findings`: every atomic finding gets a row. No round-number aggregation. No silent filtering. Low-tier surfaced alongside P0/P1/P2.

## Per-finding status table

### A1 - Feature selection (fs-critique.md, 21 findings)

| ID | Sev | Title | Status | Commit / Sensor / Reason |
|---|---|---|---|---|
| A1#1 / S01 | P0 | `_pre_pipeline_cache_key` 4-cell target fingerprint collision | DONE | f1d4212 + `tests/training/test_regression_S01_pipeline_cache_fp_collision.py` (w1a manifest) |
| A1#2 / S28 | P1 | `estimate_features_relevancy` `.copy()` + global `np.random.shuffle` | DEFERRED | w5a marked DEFERRED ("touches general.py hot path; w5b owns general.py edits") - no commit landed |
| A1#3 / S29 | P1 | `_rfecv_fit.cv_n` silent fallback to 3 for splitters w/o `n_splits` | DONE | 73aaac4d w5a "strengthen FS cache/sigs + add group-n_groups floor" (overlaps S65 RFECV gate) |
| A1#4 / S30 | P1 | MRMR `groups=` accepted but silently ignored (warn-only) | DEFERRED | No commit; w5a/w5b did not land a strict-mode toggle. Backlog candidate |
| A1#5 / S31 | P1 | `np.random.shuffle` global RNG in fleuret / permutation njit kernels | DEFERRED | No commit. LCG migration referenced in source but legacy sites at `permutation.py:41,210` + `fleuret.py:187,191` untouched |
| A1#6 / +1 | P1 | RFECV `cv_shuffle=True` silently swapped to TimeSeriesSplit on polars datetime hint | DEFERRED | Overlap with S65. S65 fixed the inverse direction (int→TSS when temporal); the opt-OUT path remains as flagged |
| A1#7 | P2 | `_mlframe_use_sample_weights_in_fs_` attr-marker not in pipeline signature | DONE | a96077da `fix(pipeline_cache): fold attribute-only seeds into signature` (w5b manifest A5_pipeline_cache_attribute_only_seeds_missing_from_sig=DONE) |
| A1#8 | P2 | `_mrmr_fingerprints._content_array_signature` 10-cell X sample | DEFERRED | w5a folded full-y hash strengthening but X-side 10-cell sample preserved; no commit replaced it |
| A1#9 | P2 | `_rfecv_fit` sample_weight length not checked vs y | DEFERRED | Not in any manifest |
| A1#10 | P2 | `_mrmr_compute_y_fingerprint_sample` 6-decimal rounding | DONE | 73aaac4d (folded into A1_FS_mrmr_y_fingerprint_6decimal_rounding=DONE per w5a manifest) |
| A1#11 | P2 | `_rfecv_fit n_samples < 2*cv_n` floor ignores n_groups for GroupKFold | DONE | 73aaac4d (A1_FS_groupkfold_n_groups_vs_n_samples=DONE per w5a manifest) |
| A1#12 | P2 | RFECV polars `to_pandas(self_destruct=True)` risk if caller-owned | DEFERRED | Not addressed; w5a marked as ARCH-DEFER A1_FS_rfecv_self_destruct_pl_pd |
| A1#13 | P2 | RFECV `_x_hash` 10-row strided sample collision risk | DEFERRED | Not in any manifest |
| A1#14 | Low | `benchmark_mi_algos` warmup arrays use global RNG, no seed | DEFERRED | Not addressed |
| A1#15 | Low | MRMR cache lookup signature build paid twice | DEFERRED | Not addressed (low leverage per agent ranking) |
| A1#16 | Low | `_rfecv_cv_setup` shallow-copy of splitter for early_stopping_val_nsplits | DEFERRED | Not addressed |
| A1#17 | Low | MRMR `max_nbytes` dead under `backend="threading"` | DEFERRED | Not addressed (cosmetic) |
| A1#18 | Low | RFECV per-fold `_eval_fold` non-deterministic log ordering | DEFERRED | Not addressed (score result is order-independent per agent) |
| A1#19 | Low | MRMR `groups`/`sample_weight` asymmetric wrapper vs `_fit_impl` | DEFERRED | Documentation gap; not addressed |
| A1#20 | Low | RFECV `X.select_dtypes("number").to_numpy()` full-frame copy for Inf check | DEFERRED | Not addressed |
| A1#21 | Low | RFECV verbose `X.isna().to_numpy()` materialises bool N×M frame | DEFERRED | Not addressed |

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
| A2 W6 | weak | `test_dataset_cache_fingerprint.py` idempotence/sensitivity check | DEFERRED | Not addressed |
| A2 W7 | weak | `test_automl.py` bare `is not None` | DEFERRED | Not addressed (snapshot/restore in d124f168 closed pollution, not the weak asserts) |
| A2 W8 | weak | `test_predict_polars_fastpath.py` `preds is not None` | DONE | 7157af4 (w3c B1_W8_predict_polars_fastpath_weak_assert=DONE) |
| A2 W-other | weak | `compute_countaggs` weak assert (B1_W7) | DONE | 7157af4 (w3c B1_W7_compute_countaggs_weak_assert=DONE) |

### A3 - Ensembling (ensembling-critique.md, 16 findings)

| ID | Sev | Title | Status | Commit / Sensor / Reason |
|---|---|---|---|---|
| A3#1 / S05 | P0 | `_oof_or_train` silent train-fallback in level-1 stacking | DONE | 56ef5e98 (w1a) + `tests/test_regression_S05_oof_silent_train_fallback.py` (visibility WARN, tactical fix per audit) |
| A3#2 / S41 | P1 | `_ENSEMBLE_RANK_METRIC_CANDIDATES` fallback on `val.*`/`test.*` | DEFERRED | Not addressed in any commit. Comment-vs-tuple contradiction remains |
| A3#3 / S42 | P1 | No probability calibration before classifier blend (Isotonic/Platt) | DEFERRED | Overlaps AP12 calibration policy - not landed |
| A3#4 / S43 | P1 | `_rrf_aggregate_probs` K=1 binary raw RRF (no sigmoid/minmax) | DEFERRED | Not addressed |
| A3#5 | P2 | `combine_probs` static equal weighting; NNLS observational-only | DONE | 5505c773 (AP7: NNLS stacking-aware blend - weights now fed back into combine_probs) + `test_regression_w7_ap7_nnls_weights_applied.py` |
| A3#6 | P2 | Diversity auto-drop tag-index not MAE-quality | DEFERRED | w5a A3_Ens_diversity_auto_drop_tag_index_not_mae=DEFERRED |
| A3#7 | P2 | K=2 catastrophic-dropout non-deterministic tiebreak | DEFERRED | w5a A3_Ens_K2_tiebreak_nondeterministic=DEFERRED |
| A3#8 | P2 | Lazy import `_choose_ensemble_flavour` cycle-breaker per-target cost | DEFERRED | w5a A3_Ens_choose_ensemble_flavour_lazy_import=DEFERRED |
| A3#9 | P2 | No nbytes dispatcher for streaming vs materialised ensemble | DONE | 5505c773 (AP8: nbytes streaming dispatcher) |
| A3#10 | P2 | No biz_value `score_ensemble beats best single` test | DEFERRED | w5a A3_Ens_compare_ensembles_deepcopy_per_call=DEFERRED (related); biz_value suite not added |
| A3#11 | P2 | `rrf_k` metadata stamped unconditionally | DEFERRED | w5a A3_Ens_rrf_k_metadata_stamped_unconditionally=DEFERRED |
| A3#12 | P2 | `compare_ensembles` full `copy.deepcopy` per call | DEFERRED | w5a A3_Ens_compare_ensembles_deepcopy_per_call=DEFERRED |
| A3#13 | Low | Weighted-median branch ignores `sample_weight` kw | DEFERRED | w5a A3_Ens_weighted_median_no_weights_kw=DEFERRED |
| A3#14 | Low | Group-aware median weight broadcast bug | DEFERRED | w5a A3_Ens_group_aware_median_weight_broadcast=DEFERRED |
| A3#15 | Low | Uniformity check misses `oof_probs` | DEFERRED | w5a A3_Ens_uniformity_mix_missing_oof_probs=DEFERRED |
| A3#16 | Low | Multiclass diversity flatten over (N,K) Pearson over interleaved | DEFERRED | w5a A3_Ens_multiclass_diversity_flatten_NK=DEFERRED |

### A4 - Perf hotspots (perf-hotspots-critique.md, 24 findings)

| ID | Sev | Title | Status | Commit / Sensor / Reason |
|---|---|---|---|---|
| A4#1 / S06 | P0 | pandas `get_trainset_features_stats` per-col Python loop | DONE | 5626668e (w4a) + `tests/training/test_regression_S06_precompute_pandas_batch.py` |
| A4#2 / S44 | P1 | `apply_polars_categorical_fixes` per-cat-col sync `.collect()` | DONE | 138cdf73 (w4a) + `test_regression_S44_polars_fixes_batched.py` |
| A4#3 / S45 | P1 | `_median_residual_fit` / `_quantile_residual_fit` Python bin loop | DONE | 05688481 (w4b) + `test_regression_S45_median_residual_perf.py` |
| A4#4 / S46 | P1 | `_ewma_kernel` / `_frac_diff_inverse_kernel` no parallel/dispatcher | DONE | 13326bcd + bc55aba (w4b) + `test_regression_S46_ewma_frac_diff_parallel.py` |
| A4#5 / +P1 | P1 | `_cached_init_params` weight-loop rebuild | DEFERRED | Not addressed by w4a/w4c |
| A4#6 / S47 | P1 | `_filter_polars_cat_features_by_dtype` invariant over weight loop | DONE | cea8a266 (w4a) + `test_regression_S47_filter_polars_cat_hoisted.py` |
| A4#7 / S48 | P1 | train-pred cache keyed by `id(comp)` misses shim wrappers | DONE | 09d6899e (w4a) + `test_regression_S48_composite_cache_shim_aware.py` |
| A4#8 / S49 | P1 | `memory_usage(deep=True)` multi-minute hot point | DONE | 4f3e2cec (w4a) + `test_regression_S49_memory_usage_polars_fastpath.py` |
| A4#9 (F9) | P2 | sort by key=str on cat-col union (50k str-key sort) | REJECTED | w4c F9=REJECTED (4% gain, not worth branch) |
| A4#10 (F10) | P2 | `_disc_df = filtered_train_df.copy(deep=False)` | DONE | F10=DONE pre-existing (was a27b6895 / S04) |
| A4#11 (F11) | P2 | `_numeric_only_view` called twice train+val | REJECTED | w4c F11=REJECTED (per-call schema iter is ~us; different frames) |
| A4#12 (F12) | P2 | `_oof_rmses_list` per-column Python loop | DONE | c6558960 (w4c F12=DONE, 1.31x speedup) + `test_regression_w4c_perf_p2low.py::test_vectorised_oof_rmse_*` |
| A4#13 (F13) | P2 | CB-cat schema scan when CB not in suite | DONE | bc55aba (w4c F13=DONE; gated on `_has_cb`) |
| A4#14 (F14) | P2 | 3 nested tqdm contexts per target | REJECTED | w4c F14=REJECTED (lost user-visible progress > perf gain) |
| A4#15 (F15) | P2 | Per-spec `finite_mask` recompute in residual-fit family | DEFERRED | ARCH-DEFER → architectural_proposals/F15_precomputed_finite_mask.md; AP2 in W7 plan but W7E did not land |
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
| A5#1 / S07 | P0 | `precompute_composite_target_specs` + `precompute_dummy_baselines` NotImplementedError stubs | DEFERRED | Not addressed. Architectural; overlaps AP1 SuiteArtefactCache (W7A did NOT land - only AP6 safe_pickle landed) |
| A5#2 / S08 | P0 | `PipelineCache` plain dict, no size gate | DONE | 7eab5ab5 (w1b) + `tests/training/test_regression_S08_pipeline_cache_size_gate.py` |
| A5#3 / S09 | P0 | Heavyweight pipeline + extensions recomputed every run | DEFERRED | Architectural (SuiteArtefactCache). AP1 not landed in W7 |
| A5#4 / S51 | P1 | Per-target select_target template rebuild (TODO at :774) | DEFERRED | w5a A3-side related items DEFERRED; A5 P1 not commit-landed |
| A5#5 / S52 | P1 | `DiscoveryCache` vs `FeatureCache` parallel disk-cache divergence | DEFERRED | Not addressed in commit |
| A5#6 / S53 | P1 | `MRMR._FIT_CACHE` class attribute, no byte-size cap | DONE | d2fd00d4 (w5b A5_mrmr_fit_cache_no_byte_cap=DONE) + `test_regression_w5_mrmr_lru_byte_cap.py` |
| A5#7 / S54 | P1 | `_PRE_PIPELINE_CACHE_MAX=8` hardcoded, no byte budget | DEFERRED | Count cap exists; byte-budget+ktc not added |
| A5#8 (P2) | P2 | `_pandas_view_cache` size-cap 4 popitem FIFO | DONE | 56e78eb1 (w5b A5_pandas_view_cache_unbounded=DONE; OrderedDict + byte gate) + `test_regression_A5_p2_8_pandas_view_cache_lru.py` |
| A5#9 (P2) | P2 | No `polars.LazyFrame.cache()` in training/core | DEFERRED | ARCH-DEFER → architectural_proposals/A5-P2-9-polars-lazyframe-cache.md. Not landed in W7 |
| A5#10 (P2) | P2 | `_loaded_models_cache` per-call dict (no LRU on disk model loader) | DONE | cb5acc82 (w5b A5_load_mlframe_model_no_warm_cache=DONE; LRU keyed by (path, mtime_ns)) + `test_regression_A5_p2_10_load_model_lru.py` |
| A5#11 (P2) | P2 | `_pre_pipeline_cache_key` missing random_seed / lib_versions | DONE | a96077da (w5b A5_pipeline_cache_attribute_only_seeds_missing_from_sig=DONE) + `test_regression_A5_p2_11_pre_pipeline_seed_in_key.py` |
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
| B1 W2-6 | weak | TVT/clone-elision/MLP-divergence/fingerprint/automl weak asserts | DEFERRED | Cluster not surveyed |
| B1 F1 | fuzz | crash_reporting axis | DEFERRED | w3c B1_F1=ARCH-DEFER → architectural_proposals/fuzz_blind_spots_F1_F2_F5_F6_F7.md |
| B1 F2 | fuzz | polarsds × LTR pair | DEFERRED | w3c B1_F2=ARCH-DEFER |
| B1 F3 | fuzz | recency-only × recurrent | DONE | 9648c56 (w3c B1_F3=DONE) + `tests/training/test_fuzz_regression_sensors.py::test_sensor_fuzz_recurrent_model_x_recency_only_weights` |
| B1 F4 | fuzz | mrmr fillna_zero × all-null col | DONE | 9648c56 (w3c B1_F4=DONE) + `test_sensor_fuzz_mrmr_fillna_zero_x_all_null_col_does_not_corrupt_mi` |
| B1 F5 | fuzz | pysr × inf/nan injection | DEFERRED | w3c B1_F5=ARCH-DEFER |
| B1 F6 | fuzz | composite discovery × outlier_detection | DEFERRED | w3c B1_F6=ARCH-DEFER |
| B1 F7 | fuzz | diagnostics without baselines | DEFERRED | w3c B1_F7=ARCH-DEFER |
| B1 F8 | fuzz | multilabel chain × random order metamorphic | DEFERRED | w3c B1_F8=ARCH-DEFER (budget; needs full-suite double-run) |
| B1 N1-N12 | numba | NUMBA_DISABLE_JIT=1 nightly coverage | DONE (partial) | b26cfa93 (w3c B1_N1_to_N12=DONE; helper scripts + marker registered). GH Actions CI workflow ARCH-DEFER per numba_coverage_ci.md proposal (AP5) - not landed |
| B1 biz_value cdist | biz | focused cdist test | DONE | c8605d3 (w3c B1_biz_value_cdist_local_lift_gap=DONE) + `tests/feature_engineering/transformer/test_biz_val_class_distance_and_local_lift.py` (2 tests) |
| B1 biz_value local_lift | biz | focused local_lift test | DONE | Same commit |
| B1 biz_value BGM (6 var) | biz | per-variant biz_value | DEFERRED | w3c deferred (budget; 6 tests × 2-3s) |
| B1 biz_value RSD-kNN | biz | focused biz_value | DEFERRED | w3c deferred (budget) |
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
| B2#6 / S56 | P1 | `@pytest.mark.fast` registered but 0 usage | DONE | (w3b S56=VERIFIED_ALREADY_FIXED; 239 tests across 81 files apply it) |
| B2#7 | P1 | only 2 files import `make_simple_*` builders; per-file re-rolls | DEFERRED | Not addressed (broad scope) |
| B2#8 | P1 | RFECV verbose=0 monkey-patch upstream-fix needed | DEFERRED | Not addressed (requires upstream PR) |
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
| B2#28 / S59 | P2 | 16 session-scope mutable fixtures | DONE (partial) | w3c B2_28_29=PARTIAL (immutability sensor + DO NOT MUTATE docstrings on regression+classification fixtures); other 4 fixtures uncovered |
| B2#29 | P2 | trained_suite_regression / _binary same mutable risk | DEFERRED | Not addressed (sibling fixtures uncovered) |
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
| C1#1 | arch | `_phase_composite_post.py` (1129 LOC) | DEFERRED | Not yet carved. Top-1 priority remains. Compacted via cab4fcfc / 6adacb29 to stay under 1k but no carve |
| C1#2 | arch | `composite_transforms.py` (1142 LOC) | DONE | a68f08e4 (w6a) before=1194 / after=295 + `tests/training/test_monolith_split_w6a_composite_transforms.py` |
| C1#3 | arch | `metrics/core.py` (1064 LOC) | DONE | 0e4e6cdf (w6b) before=1064 / after=232 + `tests/metrics/test_monolith_split_w6b_core.py` |
| C1#4 | arch | `_setup_helpers.py` (1047 LOC) | DONE | 3b7f6cf4 (w6b) before=1058 / after=356 + `tests/training/test_monolith_split_w6b_setup_helpers.py` |
| C1#5 | arch | `_target_distribution_analyzer.py` (1017 LOC) | DONE | 2bb3c896 (w6a) before=1017 / after=188 + `tests/training/test_monolith_split_w6a_target_dist.py` |
| C1#6 | arch | `wrappers/_rfecv_fit.py` (998 LOC) | DEFERRED | Documented as requiring `FitState` dataclass refactor; not pure carve. Plan persisted in monoliths-split.md |
| C1#7 | arch | `_composite_target_estimator.py` (998 LOC) | DEFERRED | Method-rebinding pattern documented; not landed |
| C1#8 | arch | `training/helpers.py` (993 LOC) | DEFERRED | Not landed |
| C1#9 | arch | `training/neural/recurrent.py` (963 LOC) | DEFERRED | Not landed |
| C1#10 | arch | `boruta_shap.py` (952 LOC) | DEFERRED | Not landed |
| C1#11 | arch | `target_temporal_audit.py` (949 LOC) | DEFERRED | Not landed |
| C1#12 | arch | `_phase_helpers.py` (948 LOC) | DEFERRED | Not landed |
| C1#13 | arch | `baseline_diagnostics.py` (942 LOC) | DEFERRED | Not landed |
| C1#14 | arch | `train_eval.py` (941 LOC) | DEFERRED | Not landed |
| C1#15 | arch | `training/neural/flat.py` (927 LOC) | DEFERRED | Not landed |
| C1#16 | arch | `extractors.py` (940 LOC) | DEFERRED | Not landed |
| C1#17 | arch | `training/neural/ranker.py` (919 LOC) | DEFERRED | Not landed |
| C1#18 | arch | `training/neural/base.py` (1057 LOC) | DEFERRED | Tight-coupled class; method-rebinding pattern documented |
| C1 prev #1-#7 | prev | 7 preventive files at 900-1000 LOC range | DEFERRED | LOC-cap sensors not added |

### D1 - ML best practices (ml-best-practices.md, 13 findings)

| ID | Sev | Title | Status | Commit / Sensor / Reason |
|---|---|---|---|---|
| D1 #1 / S64 | P1 | Train+val Enum domain widens val ES detector | DONE | d248bef8 (W7D S64 INFO-log val-only categories) + `test_regression_S64_enum_val_only_log.py` |
| D1 #2 / S65 | P1 | RFECV default cv=int silent k-fold when outer suite temporal | DONE | d248bef8 (W7D S65 RFECV auto-temporal CV) |
| D1 #3 / S66 | P1 | drift_report numeric-only, no categorical PSI | DEFERRED | W7D plan did not land. Source grep finds 0 `categorical_psi` / `cat_psi` in `src/`. Backlog |
| D1 #4 / S67 | P1 | `pl.Categorical` cast instead of `pl.Enum(train+val union)` in 3 sites | DEFERRED (partial) | W7D plan; partial via S64 INFO logging. No `pl.Enum` migration commit found |
| D1 #5 / S68 | P1 | No bootstrap CI on primary metrics | DEFERRED | W7C plan; not landed. Source grep finds no `bootstrap_metric`/`bootstrap_ci_metric` in training/evaluation.py |
| D1 #6 | P2 | Default calibrator selection not pinned (no OOF-driven policy) | DEFERRED | W7C plan AP12 calibration policy; not landed |
| D1 #7 | P2 | Regression with heavy-tail target not bucket-stratified | DEFERRED | Not addressed |
| D1 #8 | P2 | BorutaShap SHAP background-dataset audit | DEFERRED | Not addressed |
| D1 #9 | P2 | Analyzer hyperparam provenance trail | DONE | 831a1bcb (W7C AP14: `provenance.py` module landed) + 19 source files reference provenance |
| D1 #10 | P2 | `val_placement` time-series default INFO log | DEFERRED | Not addressed |
| D1 #11 | Low | `_MAX_COMPOSITE_CARDINALITY=200` magic number | DEFERRED | Not addressed |
| D1 #12 | Low | Pre-screen protects group_id/ts only when string-typed | DEFERRED | Not addressed |
| D1 #13 | Low | `pl.Categorical` silent fallback no WARN | DONE (partial) | d248bef8 S64 added INFO log on val-only categories; the broader Enum WARN gap (#4) remains |

### D2 - Code/arch standards (code-arch-standards.md, 23 findings)

| ID | Sev | Title | Status | Commit / Sensor / Reason |
|---|---|---|---|---|
| D2 #1 / S25 | P0 | sklearn check_estimator not run on composite/lag/early-stop/RFECV/Pd* | DONE | 7ca9f19a (w1c S25) + `tests/test_sklearn_compliance_composite.py` (23 tests + 1 xfailed) |
| D2 #2 / S26 | P0 | `from_fitted_inner` sklearn.clone silent data loss | DONE | c043f386 (w1c S26) + `tests/training/test_regression_S26_composite_target_clone_safety.py` |
| D2 #3 / S27 | P0 | unregistered marker `no_xdist_parallel` | DONE | eed502b (w1c S27) + `tests/test_meta/test_regression_S27_markers_registered.py` |
| D2 #4 / S69 | P1 | 12 dead registered markers | DONE (partial) | ebac7a3 (B2#24 in w3c) wired 5; remaining markers triage deferred |
| D2 #5 / S70 | P1 | ~90 deps without upper-bound caps | DROPPED | User dropped per Wave-7 plan (AP10 dropped) |
| D2 #6 / S71 | P1 | `src/mlframe/__init__.py` public API stale | DEFERRED | Not addressed |
| D2 #7 / S72 | P1 | Pickle/joblib RCE: `_verify_sidecar` fail-open + bundle_sha256 unfilled | DONE | 1ef73ef3 (w1c S72) + 61d25bcd / b7f86580 (AP6 safe_pickle centralisation across 4 entry-points) + f2d00e15 (predict-time loaders) + `tests/inference/test_regression_S72_pickle_verification.py` |
| D2 #8 / S73 | P1 | `.pre-commit-config.yaml` ruff/black/mypy continue-on-error | DONE | 4858e454 (AP11 ruff+black+calibration-scoped mypy hooks). CI `continue-on-error` drop dropped by user (AP11-c) |
| D2 #9 / S74 | P1 | meta-test absent for `del sys.modules` / `importlib.reload` | DONE | d124f168 (w3b S74_sensor=DONE) + `tests/test_meta/test_no_unsafe_module_reload.py` |
| D2 #10 / S75 | P1 | `ci.yml` no macOS row | DEFERRED | Not addressed |
| D2 #11 / S76 | P1 | `sklearn-matrix-ci.yml` py3.11/linux only | DEFERRED | Not addressed |
| D2 #12 / S77 | P1 | docs/ not updated under Round 5.3/5.4 composite refactor | DEFERRED | Not addressed |
| D2 #13 | P2 | `[tool.ruff.lint] ignore` no per-rule justification | DONE | ff18467e (w5b D2_ruff_ignore_no_per_rule_justification=DONE; per-rule comments) |
| D2 #14 | P2 | mypy effectively informational, no strict beachhead | DONE | ff18467e (w5b D2_mypy_informational_no_strict_beachhead=DONE; calibration subpackage strict-mode) |
| D2 #15 | P2 | `_collect_lib_versions` missing cupy/numba/pydantic | DONE | ff18467e (w5b D2_io_collect_lib_versions=DONE; cupy + numba + pydantic + lightning + torch added) |
| D2 #16 | P2 | RFECV registry uses underscore `_rfecv` path | DONE | ff18467e (w5b D2_feature_selection_registry_underscore_path=DONE; public alias) |
| D2 #17 | P2 | Underscore convention not documented + meta-test | DONE | ff18467e (w5b D2_underscore_convention_not_documented=DONE; core/__init__.py docstring + meta-test) |
| D2 #18 | P2 | Duplicate marker registration conftest vs pyproject | DONE | ff18467e + f2d00e15 (w5b D2_duplicate_marker_registration=DONE; conftest dedup) |
| D2 #19 | P2 | Bare `pickle.load` in composite_cache + feature_handling | DONE | b7f86580 (AP6 safe_pickle migration to 4 entry-points incl. composite_cache + feature_handling) |
| D2 #20 | Low | `test_sklearn_compliance.py:78-90` weakest assertion | DEFERRED | Not addressed |
| D2 #21 | Low | `sys.modules["cupy"] = None` undocumented | DEFERRED | Cosmetic; not addressed |
| D2 #22 | Low | `xfail_strict=true` positive note | OK | n/a |
| D2 #23 | Low | CHANGELOG in sync | OK | n/a |

### Wave-7 architectural proposals (per AP1-AP14 user approval set)

| AP | Title | Status | Commit / Reason |
|---|---|---|---|
| AP1 | SuiteArtefactCache | DEFERRED | W7A only landed AP6 portion; cache module not created. Per `feedback_no_premature_closure`: W7A=partial, not closed |
| AP2 | F15 finite_mask threading | DEFERRED | W7E hung mid-sweep; no commit. Proposal persisted at architectural_proposals/F15_precomputed_finite_mask.md |
| AP3 | F24 K-target joblib | DROPPED | User dropped per Wave-7 plan |
| AP4 | Fuzz axes F1/F2/F5/F6/F7 | DEFERRED | W7E hung; proposal persisted at fuzz_blind_spots_F1_F2_F5_F6_F7.md. F3+F4 already landed in w3c |
| AP5 | NUMBA_DISABLE_JIT=1 nightly CI workflow | DEFERRED | W7E hung; helper scripts landed in b26cfa93. GH Actions YAML not added |
| AP6 | safe_pickle | DONE | b7f86580 (w1c S72 follow-up) + f2d00e15 (predict-time loaders) + `src/mlframe/utils/safe_pickle.py` |
| AP7 | NNLS stacking-aware gate | DONE | 5505c773 + 9279b1cb test follow-up + `tests/test_regression_w7_ap7_nnls_weights_applied.py` |
| AP8 | nbytes streaming dispatcher | DONE | 5505c773 |
| AP9 | sklearn-matrix compliance step | DONE | d248bef8 |
| AP10 | dep upper-bound caps | DROPPED | User dropped |
| AP11 | pre-commit ruff+black+mypy | DONE | 4858e454 (AP11-c CI continue-on-error drop dropped by user) |
| AP12 | calibration policy | DEFERRED | W7C plan; not landed |
| AP13 | honest_diagnostics ReportingConfig | DEFERRED | W7C plan; not landed |
| AP14 | provenance trail | DONE | 831a1bcb + 19 source files reference `provenance` |

## Roll-up by status

- DONE: 116 findings (incl. NECESSARY shape no-ops documented per agent)
- DEFERRED: 102 findings
- REJECTED (with bench): 13 findings (W2B #8, #10, #33; W2C #16, #18, #20, #21, #22, #24; W4C F9, F11, F14)
- DROPPED (user): 3 items (AP3 K-target joblib; AP10 dep caps; AP11-c CI continue-on-error drop)
- OK / closed by agent: 9 findings (verified-already-fixed or no-op)

Concrete DONE list confirmed via commits or sensors (file evidence supplied above in per-row entries). Same for every DEFERRED row - reason documented inline; sensor files re-checked under `tests/`.

Honest closure rate: **116 of 234 atomic findings landed in code or test** (49.6%). Per `feedback_no_premature_closure`, this is **NOT** a closed wave-7; this is a Wave-8 backlog of 102 deferred items.

## Wave 8 backlog (in priority order — for user approval)

### P0/P1 carry-forward (correctness / ML discipline)

1. **AP1 SuiteArtefactCache** (A5 S07 + S09 + S51 + S52 + S54) - heaviest leverage; PySR / fit_pipeline / extensions recomputed every run; W7A delivered only AP6 safe_pickle portion. Five linked A5 findings remain.
2. **A1#2 / S28** `estimate_features_relevancy` full-frame .copy + global RNG shuffle (P1; w5a marked DEFERRED with reason "w5b owns general.py edits" - but w5b also did not land it).
3. **A1#5 / S31** `np.random.shuffle` global RNG in fleuret + permutation njit kernels (P1; LCG migration referenced in source but legacy sites untouched).
4. **A1#4 / S30** MRMR `groups=` silently ignored (P1; warn-only; backlog).
5. **A1#6** RFECV opt-out from TimeSeriesSplit silent swap (P1; opposite direction of S65).
6. **A3#2 / S41** `_ENSEMBLE_RANK_METRIC_CANDIDATES` val.*/test.* fallback (P1; contradicts in-file comment).
7. **A3#3 / S42** No probability calibration before classifier blend (P1; overlaps AP12).
8. **A3#4 / S43** `_rrf_aggregate_probs` K=1 binary destroys calibration (P1).
9. **AP2 F15 finite_mask** Threading precomputed mask through ~20 residual-fit kernels (W7E unfinished).
10. **AP4 Fuzz axes** F1/F2/F5/F6/F7 - 30-80 LOC per axis; W7E unfinished.
11. **AP5 NUMBA_DISABLE_JIT nightly workflow** - GH Actions YAML (W7E unfinished).
12. **AP12 calibration policy** + **AP13 honest_diagnostics** - W7C plans not landed.
13. **D1 #3 / S66** drift_report categorical PSI - 0 hits in src/; not landed.
14. **D1 #4 / S67** `pl.Enum(train+val union)` migration in 3 sites (W7D landed S64 logging only).
15. **D1 #5 / S68** Bootstrap CI on primary metrics (W7C not landed).
16. **D2 #6 / S71** `src/mlframe/__init__.py` public API stale vs CHANGELOG.
17. **D2 #10/#11 / S75+S76** CI macOS row + sklearn-matrix py3.13+windows row.
18. **D2 #12 / S77** docs/ Round 5.3/5.4 composite refactor update.

### P2 carry-forward (perf / hygiene)

19. **A1#8 / #9 / #12 / #13** MRMR + RFECV sample/x-hash / self_destruct / 10-row collision (4 deferred P2).
20. **A2#4-#12** FE bucket: mutable defaults, date richness, bruteforce broadcast-copy, subset .copy, in-place rename, ratio_fit empty-median (8 deferred).
21. **A3#6/7/8/10/11/12** Ensemble bucket: diversity drop policy, K=2 tiebreak, lazy import, biz_value test suite, rrf_k metadata, deepcopy per call (6 deferred).
22. **A4#5 / +P1** `_cached_init_params` weight-loop rebuild.
23. **A5#4 / S51, A5#5 / S52, A5#7 / S54, A5#9 P2, A5#15, A5#16** caching architectural items (overlap with AP1).
24. **B1 module-gap U4/U5/U6/U7/U8/U9/U10/U11/U12/U19/U21** 11 module unit-test gaps.
25. **B1 BGM (6 variants) + RSD-kNN biz_value tests** - shortlist transformer coverage.
26. **B1 W2-6 weak-assert clusters** in tvt_round5 / clone-elimination / MLP-divergence / dataset-cache-fingerprint / automl.
27. **B1 sklearn matrix marker convention** + meta-test.
28. **B1 CHANGELOG cross-walk** per-fix regression-sensor gap audit.
29. **B2#7, #10, #20-#22, #26, #28b/#29, #31-#33, #37-#40, #42, #44** test-infra hygiene cluster (~14 items).
30. **B2 +PYTHONUNBUFFERED** default for local runs.
31. **C1 monolith carves #1, #6-#18** (17 file plans remaining) + 7 preventive LOC-cap sensors.
32. **D1 #6/#7/#8/#10** calibration-policy + bucket-stratify + SHAP-bg audit + val_placement INFO log.

### Low carry-forward (cosmetic / documentation)

33. A1 Low #14-#21 (8 items)
34. A2 Low (4 items + comment cleanup)
35. A3 Low #13-#16 (4 items)
36. A5 Low #15, #16 (2 items)
37. A6 Low #39 (estimators/custom.py `.values` cosmetic)
38. D1 Low #11, #12 (2 items)
39. D2 Low #20, #21 (2 items)

## Notes / qualifications

- Per `feedback_no_premature_closure`: this is explicitly **Wave-8 backlog**, NOT a "Wave-7 done" closure. 102 of 234 findings remain unaddressed (43.6%).
- The 13 REJECTED findings all carry inline bench numbers in source or DONE_*.json manifest; none silent.
- The 3 DROPPED items are user-decision per Wave-7 plan (AP3 K-target joblib, AP10 dep caps, AP11-c CI continue-on-error drop). Documented per Wave-7 brief.
- W5A and W5B hung mid-sweep on numba.cuda conftest warmup; both finalized partial commits. Several A3 P2/Low items DEFERRED by w5a are documented in manifest with rationale (need bench, need sensor, follow-up wave).
- W7E hung after W7A/B/C/D landed. AP2/AP4/AP5 entries deferred not from rejection but from incomplete wave dispatch.
- Pre-commit hook AP11 commit (4858e454) created baseline thrash leading to follow-up CI fix commits (39f20034, 2c1804e2, 6a4578f7, c23f4fde, cdf1ec1c, 4cb901d1, f050594e). All accounted for as commits in `da68ca6..HEAD`.
- 88 commits in the audit window. 16 wave manifests on disk. 30+ regression-sensor files cross-referenced.
- Some manifest entries reference commits with sibling-agent attribution mixing (e.g. w4b S46 "kernels in _composite_transforms_nonlinear.py shipped on master in bc55aba (sibling-agent inadvertently swept my unstaged changes into their commit)") - documented inline.
- DEFERRED items without sensor file under tests/ are NOT_FOUND-equivalent for the purpose of regression-gate enforcement; user should treat them as Wave-8 backlog rather than silently-closed.

## Files referenced for verification

- `audit/critique_2026_05_24/SUMMARY.md`
- `audit/critique_2026_05_24/{fs,fe,ensembling,perf-hotspots,pipeline-cache,polars-zerocopy}-critique.md`
- `audit/critique_2026_05_24/{tests-expand,tests-optimize,monoliths-split,ml-best-practices,code-arch-standards}.md`
- `audit/critique_2026_05_24/manifests/DONE_{w1a-leakage,w1b-fe-cache,w1c-sklearn-security,w2a-bridge-p0,w2b-percol-scattered,w2c-fe-dtype-gate,w3a-tests-expand,w3b-tests-optimize,w3c-tests-p2low,w4a-hotspots-critical,w4b-numba-parallel,w4c-perf-p2low,w5a-fs-fe-ens-p2low,w5b-cache-arch-fu,w6a-carve-composite-target-dist,w6b-carve-metrics-setup}.json`
- `audit/critique_2026_05_24/architectural_proposals/{A5-P2-9-polars-lazyframe-cache,F15_precomputed_finite_mask,F24_ktarget_ensemble_joblib,fuzz_blind_spots_F1_F2_F5_F6_F7,numba_coverage_ci}.md`
- Git log range `da68ca6..HEAD` (88 commits)
- `tests/**/test_regression_*.py`, `tests/**/test_monolith_split_*.py`, `tests/test_meta/test_regression_S27_*.py`, `tests/test_meta/test_no_unsafe_module_reload.py`
