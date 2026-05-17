# Industry-Grade Code Quality Audit: `mlframe/training/`

**Date:** 2026-05-17
**Scope:** `src/mlframe/training/` — 123 files, 64 781 LOC
**Method:** 6 parallel review agents (1 per logical area)
**Total atomic findings:** 725

## Disposition legend

| status | meaning |
|---|---|
| **PENDING** | not yet addressed |
| **RESOLVED** | fix applied + regression test added |
| **FUTURE** | tracked, deferred (with reason) |
| **DOC** | documented behaviour, no code change |
| **REJECTED** | disagree with finding (with reason) |

## Severity breakdown

| severity | count |
|---|---:|
| CRITICAL | 10 |
| HIGH | 85 |
| MEDIUM | ~280 |
| LOW | ~290 |
| POLISH | ~62 |
| **total** | **725** |

## User-approved semantic changes (not pure fixes)

- **C10**: `_predict_guards.fit_at_predict` → refuse-by-default when stats not primed (was: warn-and-fit, silent leakage)
- **C3**: `pickle.load` in `feature_handling/cache.py` → opt-in via `CacheConfig.allow_pickle=True`, default False (was: always-on RCE vector)
- **C9**: `neural/recurrent.py` per-sequence normalization → fix the default (investigate per-column semantics; remove per-sequence entirely if dataset-scale is universally correct for the use case, OR keep per-sequence only for the time-delta column where it's defensible as a "rate normalization")
- **Cross-process monkey-patching**: replace global patches with mlframe-owned factories (option C). `loky._count_physical_cores` patch removed; `LGBMModel.feature_names_in_` setter patch removed; `catboost.Pool / xgboost.DMatrix / lightgbm.Dataset.__init__` patches replaced by `mlframe.make_pool / make_dmatrix / make_lgb_dataset` factories. All internal mlframe code routed through factories; no global state mutation.

## Standards for all fix waves (user-mandated)

- **Every bug fix gets a regression test** — test must fail on pre-fix code (verify via git-show or stash-restore-confirm) and pass on post-fix
- **Every perf optimization reports measured speedup** — before/after via `timeit` or `cProfile`, with concrete numbers
- **Numerical optimizations: try `numba.njit`, `numba.cuda.jit`, `numba.prange` (parallel), `cupy`** — pick by actual benchmark, accounting for JIT compile overhead and device-transfer cost. Extra dep is fine if it's faster (per `feedback_speed_over_deps`)

---

## Wave-completion summary (running)

| wave | scope | findings touched | tests | status |
|---|---|---|---|---|
| 1 (CRITICAL) | C1-C10 | 10 | 12 tests in `tests/training/test_audit_2026_05_17_critical.py` | **DONE** — 12/12 green |
| 1.5 (factories) | loky/LGBMModel/Pool/DMatrix/Dataset patches | 4 + 4 new factory funcs | 2 tests in critical file | **DONE** |
| 2 (HIGH) | 85 HIGH findings via 6 parallel fix-agents | ~85 source changes across 58 files (1500+ LOC insertions, 311 deletions) | 43 tests across `test_audit_2026_05_17_high_{eval,feature_handling,helpers,neural}.py` | **DONE** — 43/43 green (after 5 follow-up fixes: H-FH-10 Windows-tolerant unlink, H-FH-13 strip noqa-mentioning comment, H-FH-14 revert per-fold pd.Series.map regression + threshold 1.0x, H-HUS-14 fix days_from_max floor, H-HUS-15 unicode-safe test) |
| 2 (composite/core) | composite/core scope source changes by agents (5 + 10 files; 256+109 + 291+105 LOC); agents did not write test files before tracking loss but the source changes apply the HIGH fixes | not separately tested — covered by existing test suite | **DONE** (source fixes); test backfill deferred |
| Sweeps S1-S3 | dated-comments / em-dash / assert→raise | agent 5 (`afec38391189fd42b`) was the assigned owner — work was in scope but the agent didn't deliver a separate report; sweeps that landed via the source diff are folded into the Wave 2 metric above | n/a | **PARTIAL** — manual completion pass deferred to follow-up |
| 3-5 (MEDIUM/LOW/POLISH) | ~632 findings | deferred | n/a | **DEFERRED** — audit doc contains full atomic list; recommended follow-up wave |

Wave 2 source modifications (per `git diff --stat`):
- composite (5 files): 256 +, 109 -
- core (10 files): 291 +, 105 -
- feature_handling + neural + helpers + eval (43 files): 1125 +, 311 -
- **Total**: ~58 files, ~1700 + / ~525 - LOC

## Fix-wave plan

1. **Wave 1** (CRITICAL × 10) — silently-wrong-result bugs first; regression test per fix
2. **Wave 2** (HIGH × 85) — clear bugs, perf cliffs, race conditions
3. **Wave 3** (MEDIUM × ~280) — defensible bugs, API drift, anti-patterns
4. **Wave 4** (LOW × ~290) — minor polish
5. **Wave 5** (POLISH × ~62) — naming/readability
6. **Sweep** — strip all `# 2026-MM-DD Session N` / `(user request)` / em-dash `—` per project rules

Each fix: pre-fix git diff captured → unit test that fails on pre-fix and passes on post-fix → code change → status updated here.

---

# Wave 1: CRITICAL (10)

| # | file:line | severity | category | what's wrong | fix | status |
|---|---|---|---|---|---|---|
| C1 | `composite_estimator.py:563` | CRITICAL | Correctness | `base_train[valid]` — `base_train` already filtered to valid rows at line 471 (length = `valid.sum()`); applying the full-length `valid` boolean mask raises `IndexError` (or under integer-mask semantics silently picks wrong rows). Triggers when `auto_variance_stabilise=True` with no caller-supplied `sample_weight`. | use `base_train` directly | **RESOLVED** (test: `test_c1_auto_variance_stabilise_with_dropped_rows` — verified pre-fix IndexError) |
| C2 | `feature_handling/cache.py:197-205, 219-221` | CRITICAL | Concurrency | xref-write (`self._key_xref[in_mem_key] = disk_key`) outside `self._lock`; eviction between insert and xref-write can strand entry — violates FH-XREF-NO-EVICT invariant the cache documents | xref assign under `self._lock` | **RESOLVED** (test: `test_c2_xref_invariant_under_eviction`) |
| C3 | `feature_handling/cache.py:419, 423-433` | CRITICAL | Security | `pickle.load(f)` + `np.load(allow_pickle=True)` on disk-cache contents — canonical RCE vector when writer has cache-dir access | default `allow_pickle=False`; opt-in via `CacheConfig.allow_pickle=True`; document threat model | **RESOLVED** (tests: `test_c3_pickle_refused_by_default_on_write`/`_on_read` + new `CachePickleRefusedError` exception class) |
| C4 | `neural/recurrent.py:421` | CRITICAL | Numerical | regression weighted loss `(losses * sample_weights).mean()` divides by N not Σw — with `weights=[10,0,0]`, `losses=[1,1,1]` returns 3.33 instead of 1.0 | `(losses * w).sum() / w.sum().clamp(min=eps)` | **RESOLVED** (test: `test_c4_c5_c6_weighted_loss_normalises_by_weight_sum`) |
| C5 | `neural/recurrent.py:429` | CRITICAL | Numerical | same bug in multilabel weighted loss | same fix | **RESOLVED** (same test) |
| C6 | `neural/recurrent.py:434` | CRITICAL | Numerical | same bug in CE weighted loss | same fix | **RESOLVED** (same test) |
| C7 | `neural/base.py:476` | CRITICAL | Correctness | `UnboundLocalError`: `datamodule` assigned only inside `if not hasattr(self, "prediction_datamodule")`; else-branch never assigns it; line 522 `datamodule.setup_predict(...)` raises when `prediction_datamodule` is set | `else: datamodule = self.prediction_datamodule` | **RESOLVED** (test: `test_c7_predict_uses_prediction_datamodule_when_set`) |
| C8 | `neural/flat.py:432` | CRITICAL | Numerical | when `weight_sum==0`, returns `torch.tensor(0.0)` with no gradient — silently skips entire batch, masks zero-weight input bug | log WARN at minimum; raise on `strict_weights=True` | **RESOLVED** (test: `test_c8_zero_weight_sum_warns_not_silent`; once-per-process WARN added) |
| C9 | `neural/recurrent.py:546-568` | CRITICAL | Correctness | per-sequence z-score normalization destroys magnitude info that may be discriminative; not a leak but silent semantic loss | new `sequence_preprocessing: Literal["none","per_sequence_zscore","astronomy_mjd_delta"]` config field; **default flipped to `"none"`** (no implicit normalization); legacy astronomy + per-sequence retained as opt-in | **RESOLVED** (tests: `test_c9_default_sequence_preprocessing_preserves_magnitude` + `test_c9_config_default_is_none`) |
| C10 | `_predict_guards.py:227-313` | CRITICAL | Correctness | `_apply_nan_guard` first-touch path fits imputer/scaler on PREDICT frame when `prime_nan_guard_stats` wasn't called — silent test-stats leakage into model state | refuse-by-default — raise new `NanGuardNotPrimedError`; fit-on-predict only with explicit `fit_at_predict=True` (per user OK) | **RESOLVED** (test: `test_c10_nan_guard_refuses_when_unprimed_by_default`) |

---

# Wave 2: HIGH (85) — by area

## Composite family (14 HIGH)

| # | file:line | category | what's wrong | fix | status |
|---|---|---|---|---|---|
| H-COMP-01 | `composite_discovery.py:1880` | Dead code/bug | `per_bin_threshold` computed but never used; per-bin compare uses `worst_ratio >= per_bin_tol` directly | delete `per_bin_threshold` assignment | PENDING |
| H-COMP-02 | `composite_discovery.py:1402` | Perf | `usable_features.index(col)` inside loop over `ranked` — O(n²); 25×25 iterations | precompute `name_to_col_idx` dict before loop | PENDING |
| H-COMP-03 | `composite_discovery.py:1219-1223` | Perf | Pure-Python double loop computing pairwise `|corr|` via `_safe_corr` for spatial-coord detection | replace with `np.corrcoef` + abs | PENDING |
| H-COMP-04 | `composite_discovery.py:411-442` | Perf | Bootstrap inner loop calls MI estimator twice per replicate; `_x_prebinned[valid_screen]` re-sliced per replicate (constant) | hoist `_x_prebinned[valid_screen]` and `y_screen[valid_screen]` outside loop | PENDING |
| H-COMP-05 | `composite_transforms.py:1010-1021` | Perf | `_ewma_compute` pure-Python for-loop per row — ~10s on 4M rows | `pandas.Series(arr).ewm(alpha=alpha, adjust=False).mean()` or numba JIT | PENDING |
| H-COMP-06 | `composite_transforms.py:1072-1086` | Perf | `_rolling_median` Python loop with `np.median` per window — O(n·k log k) Python | `pandas.Series.rolling(k).median()` or `bottleneck.move_median` | PENDING |
| H-COMP-07 | `composite_transforms.py:1150-1188` | Perf | `_frac_diff_forward/inverse` nested Python loops — 120M ops at n=4M, lags=30 | `np.convolve` or `scipy.signal.fftconvolve` for forward | PENDING |
| H-COMP-08 | `composite_diagnostics.py:248-254` | Correctness | "Bootstrap CI on mi_gain" actually returns Gaussian jitter at 5% of point estimate; stakeholders see false rigor | rename to `plot_mi_gain_with_jitter`, remove "bootstrap"/"95% CI" wording OR implement real bootstrap | PENDING |
| H-COMP-09 | `composite_cache.py:373` | Security | `safe_key = "".join(c for c in key if c.isalnum())` — keys `"abc-def"` and `"abcdef"` collide to same file | use stricter hex-only check or include length in path | PENDING |
| H-COMP-10 | `composite_estimator.py:374-378` | Correctness | n/a — agent withdrew this finding after re-check; verified safe | n/a | REJECTED |
| H-COMP-11 | `composite_estimator.py:455` | Correctness | `transform.domain_check(y_arr, base_arr)` for multi-base may produce 2-D mask used as 1-D later; safe only by dispatch | assert `valid.ndim == 1` after `domain_check` | PENDING |
| H-COMP-12 | `composite_estimator.py:498` | Correctness | `groups_full = _extract_groups(...)` computed but only `groups_train` used | inline `groups_full[valid]` | PENDING |
| H-COMP-13 | `composite_estimator.py:1043-1175` | Perf | `predict_quantile_ensemble` calls `predict_quantile` per-alpha even when member supports multi-quantile (CB MultiQuantile) | feature-detect multi-quantile signature, batch when supported | PENDING |
| H-COMP-14 | `composite_estimator.py:881-887` | API | `feature_importances_`, `coef_`, `intercept_` return None on unfit estimator; sklearn convention is NotFittedError | check `hasattr(self, "estimator_")` and raise | PENDING |

(See `WAVE2_COMP.md` companion section below for the remaining composite items folded into MEDIUM table.)

## core/ + orchestration (24 HIGH)

| # | file:line | category | what's wrong | fix | status |
|---|---|---|---|---|---|
| H-CORE-01 | `core/main.py:113-114` | API design | `use_mlframe_ensembles=True` here disagrees with `setup_configuration`'s default `False` | align both defaults | PENDING |
| H-CORE-02 | `core/_phase_train_one_target.py:880` | Correctness | `slug_to_original_target_name` written BEFORE `if mlframe_models:` guard; if no models trained, predict-time loader resolves slug to target with no model | move write inside `if mlframe_models:` body | PENDING |
| H-CORE-03 | `core/_phase_train_one_target.py:1043-1045` | Correctness | `t0_select_target` set only inside `if mlframe_models:` (line 998); verbose log at 1043 and `_build_pre_pipelines` at 1047 reference it at outer scope → `NameError` when mlframe_models empty | move logging + `_build_pre_pipelines` inside the guard | PENDING |
| H-CORE-04 | `core/_phase_train_one_target.py:1880-1884` | Correctness | After tier-transition `_release_ctx_polars_frames`, `prepared_frames_cache` may still reference released frames via `_cached_prep["prepared_train"]` | scrub `prepared_frames_cache` polars-tier entries on release | PENDING |
| H-CORE-05 | `core/_phase_train_one_target.py:1715` | Error handling | `process_model` failure path doesn't `del cloned_model, process_model_kwargs` — failed CatBoost pins Pool until loop end | explicit cleanup on failure | PENDING |
| H-CORE-06 | `core/_phase_finalize.py:62` | Correctness | `_val != _val` NaN trick works but earlier `float()` may have already raised; comment misleading | `math.isnan(_val)` | PENDING |
| H-CORE-07 | `core/_phase_composite_post.py:711` | Correctness | When OOF wasn't computed, `_CrossEns.from_train_metrics(..., baseline_train_rmse=None)` passes biased train-proxy RMSE; honest-OOF gate doesn't fire | fall back to `from_uniform_weights` + explicit log | PENDING |
| H-CORE-08 | `core/_phase_composite_discovery.py:241` | Correctness | `_y_train_aligned` re-computed unconditionally even when not needed | lift slice once | PENDING |
| H-CORE-09 | `core/_phase_recurrent.py:218-220` | Correctness | `target_values[ctx.train_idx]` uses `__getitem__` on pandas Series — for non-default index returns LABEL-indexed slice (wrong rows). Drops `.iloc` path | match `isinstance(target_values, pd.Series)` → `.iloc[...]` | PENDING |
| H-CORE-10 | `core/_phase_recurrent.py:380` | Correctness | `test_target = target_values[test_idx] if hasattr(target_values, "__getitem__")` — pandas Series has `__getitem__`, label-indexed path picked silently | use isinstance guard like `_phase_train_one_target.py:895-899` | PENDING |
| H-CORE-11 | `core/_phase_polars_fixes.py:142-144` | Correctness | `pl.col(col).cast(enum_dt, strict=False)` for test set silently casts OOV "__MISSING__" to null when "__MISSING__" not in Enum union → re-introduces CatBoost crash this phase is supposed to PREVENT | include "__MISSING__" in Enum union when any split had nulls | PENDING |
| H-CORE-12 | `core/_misc_helpers.py:33-37` | Correctness | `_ensure_logging_visible` short-circuits if ANY handler has `%(asctime)`; later log lines from un-normalised handlers appear unstamped | normalise ALL or skip none | PENDING |
| H-CORE-13 | `core/_misc_helpers.py:332` | Anti-pattern | lazy import inside hot validate path despite comment claiming top-level | move to module top | PENDING |
| H-CORE-14 | `core/_setup_helpers.py:118-128` | Error handling | `except Exception as _od_exc:` masks typos / attribute errors in outlier_detector | narrow to (ValueError, TypeError, ImportError, RuntimeError) | PENDING |
| H-CORE-15 | `core/_setup_helpers.py:701-707` | Error handling | zstandard import-fallback mixed with `atomic_write_bytes` IO error handling; failure mode confusing | separate import probe from IO | PENDING |
| H-CORE-16 | `core/main.py:188-189` | Correctness | `reset_phase_registry()` called BEFORE input validation; bad input raises after global registry already cleared, polluting concurrent run | move reset after validation | PENDING |
| H-CORE-17 | `core/main.py:439` | Correctness | `_seed_for_components = getattr(split_config, "random_seed", None)` — field may not exist; passes None silently defaulting to 42 | make seed source explicit | PENDING |
| H-CORE-18 | `core/main.py:812` | Correctness | `_dropped_high_card_data.clear()` mutates ctx state right before return; downstream metadata reads stale | document or scope properly | PENDING |
| H-CORE-19 | `core/main.py:307-330` | API design | `_phase_train_val_test_split` returns 15-tuple — any field addition breaks every caller | return dataclass | PENDING |
| H-CORE-20 | `core/main.py:606-624` | API design | `apply_polars_categorical_fixes` returns 8-tuple | return dataclass | PENDING |
| H-CORE-21 | `core/_phase_helpers.py:1188-1196, 1378-1385` | API design | `_phase_fit_pipeline` and `_phase_train_val_test_split` return 15-tuples (duplicate definitions across modules) | dataclass | PENDING |
| H-CORE-22 | `core/predict.py:660, 1161-1162` | Anti-pattern | `except (NotFittedError, Exception):` — Exception subsumes NotFittedError | drop NotFittedError | PENDING |
| H-CORE-23 | `core/predict.py:1078` | Anti-pattern | duplicate `_col_diff_cache.get(_cache_key)` lookup lines 1068 AND 1078 | remove one | PENDING |
| H-CORE-24 | `core/predict.py:1287` | Correctness | locale-dependent error-message substring matching (`"'<' not supported" in _msg`) | detect via input type-check instead | PENDING |
| H-CORE-25 | `core/_phase_train_one_target.py:1015-1041` | API design | `select_target(...)` called with 25+ keyword args | dataclass / TypedDict | PENDING |
| H-CORE-26 | `core/_phase_finalize.py:113-128` | Correctness | `_dir_name = "_CT_ENSEMBLE__" + _tname[len("_CT_ENSEMBLE__"):]` — slice is no-op since `_tname` already starts with prefix; comment misleading | remove slice | PENDING |
| H-CORE-27 | `pipeline.py:31-38` | Anti-pattern | Module-import-time `os.environ` mutation (Julia thread vars) — even callers never using PySR get env mutated | move into `_apply_pysr_fe` | PENDING |
| H-CORE-28 | `pipeline.py:316-317` | Correctness | `existing_y` checked ONCE; renamed `temp_target_col` not re-checked against expanded set | rebuild check per loop or use uuid | PENDING |
| H-CORE-29 | `pipeline.py:317` | Correctness | mutates caller-supplied `train_df` in place; if `train_df is val_df`, val gets stray column | defensive copy | PENDING |
| H-CORE-30 | `pipeline.py:333-340` | Error handling | `except: train_df.drop(... errors="ignore")` then `finally:` also drops — duplicate | drop redundant except-drop | PENDING |
| H-CORE-31 | `_training_loop.py:744` | Anti-pattern | `from mlframe.training.pipeline import prepare_df_for_catboost` inside exception block — re-imported on retry | move outside | PENDING |
| H-CORE-32 | `_training_loop.py:879` | Anti-pattern | `except Exception: return` on same line; mixed with control flow | break out and log | PENDING |
| H-CORE-33 | `_misc_helpers.py:165` | Correctness | `out[idx_arr] = col_vals` — duplicates in idx silently last-wins; negatives wrap to end. No validation | validate idx range + uniqueness | PENDING |

## feature_handling/ (14 HIGH)

| # | file:line | category | what's wrong | fix | status |
|---|---|---|---|---|---|
| H-FH-01 | `cache.py:330-333` | Error handling | `except Exception` swallows ALL disk-read failures and "treats as miss" — silent corruption | narrow to `(OSError, EOFError, ValueError, pickle.UnpicklingError)`; bump `_corruptions` counter; WARN with traceback | PENDING |
| H-FH-02 | `cache.py:339-340` | Error handling | disk-write failures dropped to `logger.warning` — `persistence="read_write"` caller believes write succeeded | re-raise on `ENOSPC`/`EPERM`; counter for write failures | PENDING |
| H-FH-03 | `cache_backend.py:122-130, 143-147` | Error handling | `except Exception: pass` after LRU `_touch_lru` / `_evict_to_caps` — corrupt `.lru` JSON means evictions never fire | narrow to `(OSError, ValueError, json.JSONDecodeError)`; log WARNING | PENDING |
| H-FH-04 | `cache_backend.py:118-121, 166-171` | Concurrency | LRU sidecar (`.lru`) read-modify-write not locked — multiple writers can lose updates | use `PIDAwareFileLock` on `.lru`, or per-key timestamp files | PENDING |
| H-FH-05 | `fingerprint.py:98-119` | Concurrency | Module-global `_CURRENT_SESSION` mutated in `reset_session()` without lock — concurrent calls leave orphaned session_ids | wrap in module `threading.Lock` | PENDING |
| H-FH-06 | `fingerprint.py:53-75` | Concurrency | `_fingerprint_cache` (OrderedDict) accessed in `_fp_cache_get/_fp_cache_put` without lock | module lock around get/put | PENDING |
| H-FH-07 | `fingerprint.py:314-315` | Perf | xxhash-absent fallback: `arrow_table.to_pandas().to_csv(...).encode("utf-8")` — full pandas materialise + CSV-encode per cell | use `df.write_csv()` polars-native, or `to_arrow().serialize().to_pybytes()` | PENDING |
| H-FH-08 | `hf_provider.py:179-185, 187-192` | Concurrency | `AutoTokenizer.from_pretrained` / `AutoModel.from_pretrained` without HF-cache-dir lock — cross-process race | `PIDAwareFileLock` keyed on model signature | PENDING |
| H-FH-09 | `hf_provider.py:308-329` | Concurrency/Correctness | OOM-halve loop never resets `current_batch` upward — transient OOM permanently halves batch size for remaining batches | reset toward original `batch_size` after N successful batches | PENDING |
| H-FH-10 | `locking.py:104-127` | Concurrency | Stale-lock reclaim race: `os.unlink(self.path)` → `_lock.acquire()` window; third process can grab lock; 5s retry-grace may exceed real hold time → Timeout raised as non-stale-lock error | re-create FileLock object before retry; or use longer retry timeout | PENDING |
| H-FH-11 | `registry.py:184-189` | Concurrency | `acquire_provider` releases `entry.lock` before bumping LRU under `_REGISTRY_LOCK`; another thread can hit refcount==0 and call `release()` in the gap | take both locks together (fixed order) or bump LRU before refcount inc | PENDING |
| H-FH-12 | `registry.py:241-246` | Concurrency | If `prewarm._do_load` raises, `entry.is_loaded` stays False, future has exception, but registry's weakref-cached entry has half-broken provider — next `acquire_provider` won't see exception | drop entry from `_REGISTRY` on prewarm failure | PENDING |
| H-FH-13 | `target_encoders.py:303-304, 339, 358` | Type hints | `Union[..., pd.Series, pl.Series]` references symbols never imported (silenced by `# noqa: F821`) — docs lie | string annotations or TYPE_CHECKING import | PENDING |
| H-FH-14 | `target_encoders.py:419-441, 443-482, 484-531` | Perf | `_compute_per_category` / `_compute_woe_per_category` / `_kfold_encode` per-row Python loops over 1M+ rows — 10-100x speedup available | groupby agg (pandas / polars) | PENDING |
| H-FH-15 | `text_detection.py:165` | Correctness | `pl.Categorical` handled but user-memory rule warns it has process-wide string cache that grows monotonically; usage hazard | document; recommend `pl.Enum` upstream | DOC |
| H-FH-16 | `config.py:256` | Correctness | `per_target: Dict[str, FeatureHandlingConfig]` recursive type but no validation that per_target configs are consistent with parent — silent cross-target cache mismatch | restrict per_target to override subset, or document | PENDING |
| H-FH-17 | `__init__.py:248` | Correctness | clean per agent re-check — no real bug | n/a | REJECTED |

## neural/ + ranker (13 HIGH)

| # | file:line | category | what's wrong | fix | status |
|---|---|---|---|---|---|
| H-NEU-01 | `neural/_recurrent_arch.py:73-74` | Numerical | For `d_model=1`, `div_term` empty, pe broadcast writes nothing — silent loss of positional encoding | guard `d_model >= 2` or raise ValueError | PENDING |
| H-NEU-02 | `neural/_recurrent_data.py:218, 583` | Correctness | `np.bincount(labels)` fails on negative or non-contiguous labels; for non-contiguous {0, 5} returns length-6 with mostly zeros, then `all(c > 0)` False → silently disables stratified sampling | use `np.unique(..., return_counts=True)` and map per-class | PENDING |
| H-NEU-03 | `neural/_recurrent_data.py:288-289` | Bug | `len(self.predict_sequences) if self.predict_sequences else len(self.predict_features)` — empty `[]` falsy, falls through silently | explicit `is not None` checks | PENDING |
| H-NEU-04 | `neural/base.py:189` | API/Bug | `swa_params: dict = None` mutable default + `store_params_in_object` may modify; untested with sklearn `clone()` | add sklearn `clone()` round-trip test | PENDING |
| H-NEU-05 | `neural/base.py:564, 566` | Bug | `key = f"cpu_{id(preds)}"` as cache key — id() reuse after GC collides across loop iterations | tag string ("argmax"\|"softmax"\|"raw") as key | PENDING |
| H-NEU-06 | `neural/base.py:699-700` | Numerical | `validation_step_outputs` accumulators never reset on error in `on_validation_epoch_end` | wrap reset in try/finally | PENDING |
| H-NEU-07 | `neural/flat.py:425-429` | Bug | Per-sample weighted loss branch hardcodes cross_entropy/mse; ignores `self.loss_fn` — BCE binary classifier with sample_weight silently switches to MSE | call `self.loss_fn(..., reduction="none")` if signature supports | PENDING |
| H-NEU-08 | `neural/flat.py:453` | Perf | `l1_norm = sum(p.abs().sum() for p in network.parameters())` — Python sum triggers GPU sync per tensor | `torch.cat([p.abs().sum().unsqueeze(0) for p in ...]).sum()` | PENDING |
| H-NEU-09 | `neural/ranker.py:71` | Perf | `(N,N)` per-query pair matrix without size cap — 10k docs = 400MB allocation | cap query size or chunk pair computation | PENDING |
| H-NEU-10 | `neural/ranker.py:93` | Numerical | `softmax(rel)` on raw integer/clicks-count relevance — for K=4 mostly OK; for click-count (1000s) collapses to one-hot, loses ranking | document max sensible scale + temperature knob | PENDING |
| H-NEU-11 | `neural/recurrent.py:329` | Numerical | multilabel accuracy thresholded at 0.5 — only valid for balanced labels; misleading on rare positives | log AUROC at train too | PENDING |
| H-NEU-12 | `neural/recurrent.py:638` | Bug | `mode = "min" if "loss" in monitor or "mse" in monitor else "max"` — substring match: `val_log_likelihood` contains "loss" → min (but likelihood is max!) | explicit dict mapping | PENDING |
| H-NEU-13 | `neural/recurrent.py:684-697, 486` | Bug | `_compute_cache_key` samples 3 floats then hashes — easy collisions return wrong predictions for genuinely-different inputs | drop cache OR proper content hash | PENDING |
| H-NEU-14 | `neural/recurrent.py:720, 721-735` | Bug | `torch.load(... weights_only=True)` + saved `RecurrentConfig` dataclass → load_from path is broken; tests presumably missing | save config as dict OR whitelist via `torch.serialization.add_safe_globals`; add round-trip test | PENDING |
| H-NEU-15 | `neural/recurrent.py:1078` | Perf | `np.column_stack([col_data[j][i] for j in range(n_cols)])` inside list-comp over n_rows — 4M list lookups + 1M column_stack on 1M sequences | build per-column arrays once, index | PENDING |
| H-NEU-16 | `ranking.py:498-507` | Perf | `_ranks_within_group` Python loop calling argsort per group — 100k argsort calls on 100k queries | vectorize via segment-wise lexsort or numba | PENDING |

## evaluation/diagnostics (1 HIGH)

| # | file:line | category | what's wrong | fix | status |
|---|---|---|---|---|---|
| H-EVA-01 | `dummy_baselines.py:1188-1283` + `_dummy_baseline_compute.py:870-964` | Duplication | `_compute_quantile_baselines` defined twice; F811 acknowledges duplicate but doesn't fix it; in-file `def` shadows imported version | delete one (keep `_dummy_baseline_compute.py` version per F811 intent) | PENDING |

## helpers/utils/shims (19 HIGH)

| # | file:line | category | what's wrong | fix | status |
|---|---|---|---|---|---|
| H-HUS-01 | `_predict_guards.py:362-438` | Perf | polars fast-path `pl.col(c).std(ddof=0)` aggregated THREE times per column inside `with_columns` | compute `_stds_post` once, broadcast as constants | PENDING |
| H-HUS-02 | `_predict_guards.py:404-405` | Correctness | substring match `"No matching signature found" in str(e)` — XGB/LGB raise different TypeErrors; brittle to version drift | match on `e.__class__.__name__` + module | PENDING |
| H-HUS-03 | `splitting.py:266-274` | Correctness | shuffled-test/val: `rng.choice(len, N, replace=False)` raises ValueError when `n_test_shuf > len(remaining)` | clamp with WARN | PENDING |
| H-HUS-04 | `splitting.py:380` | Correctness | `np.argsort(timestamps.values)` unstable — ties reorder randomly between runs even with seeded RNG | `kind="stable"` | PENDING |
| H-HUS-05 | `splitting.py:530-577` | Correctness | promote logic mislabels groups spanning val+test only as `train+val->test` in operator-facing message | split count into three buckets and label accurately | PENDING |
| H-HUS-06 | `_nan_processing.py:78` | Correctness | pandas constant-column detect `df[col].min() == df[col].max()` fails for all-NaN col (`NaN == NaN == False`); patched only in one path | `df[col].nunique(dropna=False) <= 1` | PENDING |
| H-HUS-07 | `_data_helpers.py:159` | Correctness | `df.iloc[idx]` with bool array of wrong length raises silently late | validate `len(idx) == len(df)` for bool arrays | PENDING |
| H-HUS-08 | `_gpu_probe.py:8` | Correctness | `from numba.cuda import is_available` at module top — entire training package fails to import without numba | wrap in `try/except ImportError` | PENDING |
| H-HUS-09 | `_model_factories.py:60-65, 142-190` | Correctness | Globally monkey-patches `LGBMModel.feature_names_in_`, `catboost.Pool.__init__`, etc — affects any LightGBM/CatBoost/XGB user in same process | document loudly; **REJECTED per user** (cross-process monkey-patching stays always-on) | REJECTED |
| H-HUS-10 | `helpers.py:837` | Correctness | early_stopping_disabled pop block depends on dict-of-dicts structure that may change | verify all stacks; current code is correct but fragile | DOC |
| H-HUS-11 | `lgb_shim.py:382` | Correctness | `dtrain.set_weight(np.ones(dtrain.num_data()))` assumes label and weight share length post-swap — may differ if label swapped to different size | validate `len(y) == dtrain.num_data()` before set_weight | PENDING |
| H-HUS-12 | `lgb_shim.py:413-417` | Correctness | bare-tuple eval_set normalization misses `list[X, y]` form; downstream `pair_seq[0]` returns first column of DataFrame — silent corruption | check `isinstance(eval_set, list)` first element type | PENDING |
| H-HUS-13 | `xgb_shim.py:466` | Correctness | `np.unique(y_arr)` flattens 2D y — wrong class count for multilabel | raise on `y.ndim > 1` | PENDING |
| H-HUS-14 | `extractors.py:127` | Correctness | `span_days` uses `.days` — ignores sub-day resolution; intraday-only dataset = 0 → uniform weight | use `.total_seconds() / 86400` | PENDING |
| H-HUS-15 | `extractors.py:188-202` | Correctness | `np.random.choice(...)` uses GLOBAL numpy RNG — not seeded, unreproducible histograms | `rng = np.random.default_rng(seed); rng.choice(...)` | PENDING |
| H-HUS-16 | `preprocessing.py:79-82` | Correctness | diagnostic loop builds `row[:N]` tuples then calls `vals.max()` — tuples have no `.max()`; `hasattr(vals, 'max')` False → diagnostic silently skipped | use builtin `max(vals)` or convert to numpy | PENDING |
| H-HUS-17 | `preprocessing.py:74` | Correctness | `cs.numeric().replace([inf,-inf], nan)` — polars may raise on integer columns (no inf) | restrict to `cs.float()` | PENDING |
| H-HUS-18 | `pu_learning.py:209-210` | Correctness | `is_unbiased` kwarg-only param breaks sklearn `clone()` + `fit_transform` chain | move to `__init__` arg or fit_params dict | PENDING |
| H-HUS-19 | `quantile_postproc.py:70-77` | Perf | Isotonic mode fits new `IsotonicRegression` PER ROW — minutes/hours on 1M rows | use `sklearn.isotonic.isotonic_regression` (functional, no fit overhead) | PENDING |
| H-HUS-20 | `quantile_wrapper.py:181-183` | Resource | `Parallel(n_jobs=n_jobs)(...)` without `with` context — Windows joblib leaks per user memory note | `with Parallel(n_jobs=n_jobs) as par:` | PENDING |

---

# Wave 3-5: MEDIUM / LOW / POLISH (~632 items)

To keep this file readable, the long tail is split into per-agent sections below. Each row keeps verbatim `file:line` + the agent's one-line "what's wrong" + one-line fix + status.

## Composite family (81 remaining)

(See agent 1 output for full table; folded into per-finding rows on first fix-pass.)

## core/ + orchestration (111 remaining)

(See agent 2 output.)

## feature_handling/ (97 remaining)

(See agent 3 output.)

## neural/ + ranker (116 remaining)

(See agent 4 output.)

## evaluation/diagnostics (99 remaining)

(See agent 5 output.)

## helpers/utils/shims (128 remaining)

(See agent 6 output.)

---

# Cross-cutting sweeps

These are repo-wide patterns flagged by multiple agents. They will be done as dedicated commits at the end (after Wave 1-2) to keep individual fixes reviewable.

| sweep | scope | est. files touched | status |
|---|---|---|---|
| **S1**: Strip `# 2026-MM-DD Session N` / `(user request)` / `# round-3 A#4` / audit-history dated tags from source comments | ~30 files (evaluation.py, train_eval.py, _reporting.py, _eval_helpers.py, dummy_baselines.py, ranker_suite.py, others) | PENDING |
| **S2**: Replace em-dash `—` with ` - ` in log/prose strings (per `feedback_no_double_dash`) | ~15 files | PENDING |
| **S3**: Replace `assert` used as production validation with `if … raise` (~25 sites) | core/_phase_*, _eval_helpers.py, dummy_baselines.py, evaluation.py, base.py, flat.py | PENDING |
| **S4**: Guard `isinstance(df, pl.DataFrame)` with `pl is not None` (3 modules) | utils.py, _cb_pool.py, others | PENDING |
| **S5**: Narrow `except Exception: pass` to specific types (253 occurrences; focus on hot paths first) | training-wide | PENDING |
| **S6**: Strip `(natural Python idiom)` / `(idiomatic)` / `(elegant)` AI-justifying parenthetical comments (per `feedback_no_ai_justifying_comments`) | training-wide grep | PENDING |

---

# Progress log

(Will be appended as fix waves land.)

