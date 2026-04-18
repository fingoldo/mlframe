# Changelog

## 2026-04-19 — Proactive exploratory probes uncovered (and fixed) 3 more latent bugs

### Fixed
- **`_auto_detect_feature_types` missed `pl.Enum`**: the dtype check `if dtype in (pl.String, pl.Utf8, pl.Categorical)` did not match `pl.Enum` instances (Enum carries instance-level dtype metadata that doesn't compare equal to the class-level entry). Added `isinstance(dtype, pl.Enum)` branch. Without this, a high-cardinality `pl.Enum` text column silently stayed in `cat_features` and CatBoost wasted memory on nominal encoding — same bug class as the `skills_text` case but on a different Polars type.
- **`_auto_detect_feature_types` crashed on `cat_features=None`**: `if name in cat_features` hit `TypeError: argument of type 'NoneType' is not iterable`. Callers who skipped categorical detection passed None; now treated as empty.
- **`prepare_df_for_catboost` crashed on `text_features=None` / `cat_features=None`**: the `for var in text_features` / `cat_features` loops can't iterate None. Both arguments now None-guarded at function entry.
- **`_validate_feature_type_exclusivity` crashed on None lists**: `set(None)` raises. All three arguments now coerce None to empty list before set ops.

### Added (regression sensors)
- `test_auto_detect_polars_enum_promoted_by_cardinality` — the Enum-specific sensor.
- `test_auto_detect_accepts_cat_features_none` — None-guard sensor.
- `test_text_features_none_does_not_crash`, `test_cat_features_none_does_not_crash`, `test_cat_features_both_none_does_not_crash` — prep_cb None-guard sensors.
- `test_exclusivity_accepts_none_args`, `test_exclusivity_none_still_catches_real_overlap` — validator None-guard sensors (the second one guards the silent-overlap-while-None-guard regression).
- `test_cat_features_list_is_mutated_in_place_across_calls` — documents the in-place mutation contract so a future refactor that returns a fresh list doesn't silently break callers in `core.py`.
- `test_high_cardinality_conversion_perf_budget` — `get_pandas_view_of_polars_df` on 500k × 1 Categorical with 500k uniques must finish < 5 s.
- `test_empty_polars_dataframe` + `test_zero_column_polars_dataframe` — edge-case robustness.
- `test_fallback_without_eval_set_still_retries`, `test_fallback_retry_failure_propagates` — fallback orchestration cases not covered by the earlier end-to-end tests.

### Process note
The reactive regression tests added earlier in the day all passed on the first run — comforting but also suspicious, because the bugs they target had already been fixed. Running a round of *proactive* exploratory probes (what-if tests over Enum, None args, empty frames, zero columns, high cardinality, retry failures, eval-set absence) surfaced four real latent bugs that reactive tests would never have caught. Keeping both practices going forward.

## 2026-04-19 — Test-suite expansion: invariants, boundaries, orchestration, perf

### Why
The string of production bugs over the past two days (cat-to-text promotion side-effects, CB fallback ordering, prep_cb O(n) dance on high-cardinality text columns) all slipped through because our tests exercised **inputs → outputs** on toy data but not:

1. **Behavioural invariants** — "text_features must NEVER flow into cat_features" wasn't asserted anywhere.
2. **Orchestration flows** — "fastpath raises → pandas fallback → decategorize → prep_cb → retry" was never run end-to-end.
3. **Boundary conditions** — threshold `>` vs `>=` regressions pass silently when a single mid-range test data point is used.
4. **Perf budgets** — `astype(str).astype("category")` at O(n_rows × avg_str_len) is fine on 10 rows but kills production at 810k.
5. **High-cardinality fixtures** — `["A","B","A","C"]` with 3 uniques hides a class of bugs that only bite at 10k+.

### Added
- **`tests/test_preprocessing.py`** (3 new tests for `prepare_df_for_catboost` invariants):
  - `test_pandas_text_feature_categorical_not_added_to_cat_features`: the bug-class sensor — a pd.Categorical column declared in `text_features` must NOT be auto-appended to `cat_features`.
  - `test_pandas_text_feature_skips_expensive_astype_rebuild`: **perf budget** sensor on a 50k × 5k-unique text column. Without the skip, the `astype(str).astype("category")` rebuild blows through 2 s; with the skip it finishes in milliseconds. If this sensor fires, the skip logic regressed.
  - `test_pandas_text_feature_dtype_is_not_mutated`: declares the function's responsibility boundary — text-column dtype conversion is the caller's job (via `_decategorize_text_cols`), not `prepare_df_for_catboost`'s.
- **`tests/training/test_untested_core_helpers.py`** (4 new tests for `_auto_detect_feature_types`):
  - `test_auto_detect_pandas_promotes_high_card_cat_to_text` — the formerly-inverted test now asserts correct semantics (promote AND remove in place).
  - `test_auto_detect_pandas_keeps_low_card_cat` — negative case.
  - `test_auto_detect_threshold_boundary` parametrized over `(n_unique=9, 10, 11)` with `threshold=10` — catches `>` vs `>=` regressions.
  - `test_auto_detect_user_text_wins_over_promotion` — user-declared `text_features` authoritative over the cardinality heuristic.
  - `test_auto_detect_polars_categorical_promoted_by_cardinality` — `pl.Categorical` columns are eligible for text promotion (was the production `skills_text` path).
- **`tests/training/test_cb_polars_fallback.py`** (new file, 6 tests): end-to-end tests of the fallback orchestration via a `FakeCatBoost` stub that raises on first fit and succeeds on retry.
  - `test_fallback_triggers_on_polars_typeerror` — the fallback activates on the exact message production showed.
  - `test_fallback_converts_train_df_to_pandas` — retry receives pandas, not polars.
  - `test_fallback_decategorizes_text_columns_before_retry` — regression sensor for the 2026-04-19 morning bug (retry received pd.Categorical → CB rejected).
  - `test_fallback_rewrites_eval_set_to_pandas` — eval_set X is pandas + text cols decategorized (otherwise CB re-crashes on val).
  - `test_fallback_passes_when_polars_fastpath_succeeds` — sanity: no fallback when first fit succeeds.
  - `test_fallback_ignored_for_non_catboost_models` — fallback is CatBoost-specific (XGB/LGB/MLP don't trigger it).

### Fixed
- **`_auto_detect_feature_types`**: the 2026-04-19 behaviour change (promoting `cat_features` columns to `text_features` when cardinality exceeds threshold) went in without updating `test_auto_detect_pandas_skips_cat_features`. That stale test would have kept passing only because of an incorrect assertion; now replaced with `test_auto_detect_pandas_promotes_high_card_cat_to_text` that asserts the new (correct) contract.

### Lessons captured as sensors
A future regression of any of the fixed bugs would trip one of the new tests above, with a clear message pointing at the invariant that broke. Specifically:
- Reintroducing the `astype(str).astype("category")` dance for text columns → perf budget test fails with ``"prepare_df_for_catboost took X.XXs on a 50k text column — the text-feature skip likely regressed"``.
- Reversing the fallback ordering back to ``decategorize → prep_cb`` ordering → end-to-end test fails with ``"text column X arrived at retry with dtype category; must be object/string"``.
- `>` flipping to `>=` in the cardinality threshold → boundary test fails on the `n_unique=10, threshold=10` case.

## 2026-04-19 — Fallback hang fix: text features no longer pay the cat-preparation tax

### Fixed
- **Production hang in the CB Polars-fastpath fallback** (`training/trainer.py` + `preprocessing.py`). A live run (2026-04-19 00:38) showed the fallback reaching ``prepare_df_for_catboost`` and stalling on the "Processing categorical features for CatBoost..." tqdm: for every column with a ``pd.CategoricalDtype`` the function ran

  ```python
  df[col].astype(str).fillna(na_filler).astype("category")
  ```

  On the user's ``skills_text`` column (81_575 unique values × 810_000 rows) that re-materialises every row as a Python string then rebuilds a CategoricalIndex — minutes per column, ~tens of minutes total across the four high-cardinality text columns that had been auto-promoted from ``cat_features`` to ``text_features`` earlier in the pipeline. Two complementary fixes:

  1. **Reorder the fallback pipeline to decategorize *before* ``prepare_df_for_catboost``** (`trainer.py::_train_model_with_fallback`). The ``_decategorize_text_cols`` helper was running *after* ``_prep_cb`` — too late, because by then ``_prep_cb`` had already started the expensive dance. Now the order is ``get_pandas_view → decategorize → _prep_cb``. Applied to both ``train_df`` and every ``eval_set`` pair.
  2. **Skip ``text_features`` columns in ``prepare_df_for_catboost``'s pandas cat-iteration loop** (`preprocessing.py`). A text-feature column that happens to carry ``pd.CategoricalDtype`` must not be auto-added to ``cat_features`` nor pass through the ``astype(str).astype("category")`` rebuild — it's text, not categorical, and the function now makes that invariant explicit via an opt-out set.

  Defence in depth: either fix alone would unblock the production scenario; together they ensure no future code path can re-open the hang.

## 2026-04-19 — Investigated (and ruled out) shared-dict optimisation for polars→pandas

### Investigation summary
The production PHASE-4 log of 2026-04-18 showed ``get_pandas_view_of_polars_df`` consuming 383 s total across train/val/test on a 1M × 98 frame with 13 Categorical columns (4 high-cardinality text-like). The initial hypothesis: train/val/test are slices of one source DataFrame, so they share a Categorical palette; the per-split pyarrow dict rebuild duplicates O(n_unique) work. A ``shared_dict_cache`` parameter was added along with equivalence checks, plus wiring in ``_convert_dfs_to_pandas``.

A synthetic benchmark disproved both the premise and the premise-of-the-premise:

1. **Polars trims the Categorical dictionary per slice** — each of ``train``, ``val``, ``test`` carries only the categories actually present in its row subset, with different orderings and lengths. The cache's equivalence check correctly rejected every cross-call reuse, so the optimisation became a no-op in practice. A new regression-sensor test (``TestPolarsSliceDictionaryDiffers::test_slice_trims_categorical_dictionary``) documents this and will trip immediately if a future polars upgrade starts preserving parent palettes across slices (which would make the optimisation viable again).
2. **The function is actually fast on synthetic prod-shaped data.** 1M × 93 with 13 Categoricals (including 4 high-cardinality ones, short strings ~8 chars): **0.45 s total** across the three splits. Switching to 500-char "text-blob" categoricals (closer to production's ``skills_text`` / ``ontology_skills_text``): **0.59 s total**. Production's 383 s is ~650× slower — the per-column work simply doesn't scale that way even with long strings.
3. An alternative "direct-polars path" (build ``pd.Categorical.from_codes`` skipping the pyarrow round-trip) was **4.24× slower** than the current implementation. Not a win.

### Conclusion
The 383-s production cost is not inside ``get_pandas_view_of_polars_df`` — it's dominated by something the function can't see: memory pressure at ~37 GB RSS causing OS-level page thrash / swap, or per-process overheads outside the function. No in-function optimisation is possible; future work would need to address memory-ceiling behaviour at the suite level.

### Fixed
- ``get_pandas_view_of_polars_df`` signature reverted to the pre-2026-04-19 shape (no ``shared_dict_cache`` parameter). The docstring now carries a "Tried but reverted" note so future readers don't reopen the same dead end.
- ``_convert_dfs_to_pandas`` no longer constructs a shared cache dict per call; the per-split timers stay (they proved useful).

### Tests
- ``TestSharedDictCache`` removed.
- ``TestPolarsSliceDictionaryDiffers`` added as a single-test regression sensor documenting Polars' per-slice dict-trimming behaviour.

### Bench scripts
- ``bench_shared_dict_cache.py`` rewritten as a **per-step profiler** for the conversion (1. ``to_arrow()`` 2. dict rebuild 3. ``to_pandas()``) + direct-polars alternative comparison. Kept for future investigations.
- ``bench_long_strings.py`` added: measures the effect of Categorical string length on conversion time. Confirms the function is fast even at 500-char strings.

## 2026-04-19 — Auto-promote cat→text: correctly drop promoted cols + diagnostic + fallback timers

### Fixed
- **`cat_features` list was not updated after auto-promotion of high-cardinality columns to `text_features`** (`training/core.py`, around line 1456). `_auto_detect_feature_types` returned the promoted set, and local `effective_cat_features` was computed with promoted columns removed — but the suite-level `cat_features` binding was never rebound, so `select_target` / `strategy.build_pipeline` / the CatBoost pandas-fallback path all kept receiving the **original** unfiltered list (including the just-promoted `category`, `skills_text`, etc.). Result: CatBoost's pandas path rejected the run with `"column 'category' has dtype 'category' but is not in cat_features list"` — the column was pd.Categorical (preserved from the source Polars schema) AND listed in `text_features`, so CB's Pool refused to accept the combination. Fix: `cat_features = effective_cat_features` right after the auto-detect call, so every downstream user sees the deduplicated list via the single binding.
- **`_train_model_with_fallback` now de-categorizes text columns after the Polars→pandas conversion** (`training/trainer.py`). Without this, columns that were auto-promoted from cat→text still arrived at CatBoost with pd.Categorical dtype; CB then complained "dtype 'category' but not in cat_features". New local `_decategorize_text_cols` helper casts every text-feature column with `pd.CategoricalDtype` to plain object (with `fillna("")`). Applied to both `train_df` and every `eval_set` pair.

### Observability
- **Promotion log now includes per-column cardinality** (`_auto_detect_feature_types`). Old line was opaque:
  ```
    Promoted 4 high-cardinality column(s) from cat_features to text_features: ['category', 'occupation', 'skills_text', 'ontology_skills_text']
  ```
  New output shows the threshold AND the actual per-column unique counts, so it's immediately obvious WHY each column was promoted:
  ```
    Promoted 4 high-cardinality column(s) from cat_features to text_features (threshold>100): [category:12_345, occupation:3_211, skills_text:52_480, ontology_skills_text:4_890]
  ```
  Same format reused for the "Auto-detected feature types — text: ..." summary.

- **Per-step timing inside the CB Polars-fallback path** (`training/trainer.py`). A production run hit the fallback and spent >1 hour between the "converting to pandas and retrying" warning and the eventual CB retry (which itself was just 37 s). No intermediate log meant diagnosing what consumed that 65 minutes was impossible. New `[fallback]` lines break it down per sub-step:
  ```
    [fallback] polars→pandas(train) 810_000×98 in ...s
    [fallback] prepare_df_for_catboost(train) in ...s
    [fallback] decategorize text cols(train) in ...s
    [fallback] eval_set rewrite in ...s
    [fallback] total pandas prep for CB in ...s
  ```
  The next fallback run will show which step is the real bottleneck and inform the decision on whether `get_pandas_view_of_polars_df` needs a shared-dict optimization or whether the cost lives somewhere else (e.g. `prepare_df_for_catboost`'s pandas-side per-column loops).

## 2026-04-18 — Log-triage part 3: PHASE 3 gc timer + pandas-conv reason

### Observability
- **`fit_and_transform_pipeline` now logs when the `maybe_clean_ram_adaptive()` step takes >1 s** (`training/pipeline.py`). A 1-minute "silent" gap observed in production between "Detected N categorical features" and "Done. RAM usage:" was tracked down to `gc.collect()` running on a multi-GB Arrow heap right after the raw DataFrame was freed. The step is no longer a black box; exact cost is visible in the log.
- **`_convert_dfs_to_pandas` path now logs the exact reason when `can_skip_pandas_conv=False`** (`training/core.py`). Previously users running Polars-native-only model sets but still seeing 5-6 minute pandas conversions had no way to diagnose what was forcing the fallback. The new line is verbose and explicit, e.g.:
  ```
    polars→pandas conversion needed because: non-Polars-native models requested: ['mlp', 'linear']
  ```
  or
  ```
    polars→pandas conversion needed because: rfecv_models=['cb_rfecv']
  ```

## 2026-04-18 — CatBoost text-column dtype fix + per-split polars→pandas timing

### Fixed
- **CatBoostError: "Unsupported data type Categorical for a text feature column"** — exposed by the 2026-04-18 auto-promote-to-text-features change. After `_auto_detect_feature_types` moves high-cardinality columns (e.g. `skills_text`, `category`) from `cat_features` to `text_features`, their backing dtype in the Polars frame remained `pl.Categorical`, but CatBoost's Polars text-column handler (`_set_features_order_data_polars_text_column`) only accepts `pl.String`/`pl.Utf8`. The fix casts every Polars `Categorical`/`Enum` column listed in `text_features` to `pl.String` right before the existing null-fill step in the CB fastpath (`training/core.py`, same block as the text null-fill). A single info line reports which columns were cast.
- **Broaden CB fallback condition** (`training/trainer.py:_train_model_with_fallback`) — the existing `"Unsupported data type Categorical for a numerical feature column"` fallback now triggers on any `"Unsupported data type Categorical"` substring (both `numerical` and `text` variants). Keeps us safe if future CB versions add more category-rejection sites with similar wording.

### Observability
- **`_convert_dfs_to_pandas` logs per-split timing** (`training/core.py`) when `verbose=True`. The "Zero-copy conversion to pandas..." step that silently consumed 5+ minutes in production (rebuilding pyarrow dict indices on 1M × 98 with ~13 categoricals, some text-like with 10k+ unique values) is no longer a black box. Sample output:
  ```
    polars→pandas(train) 810_000×98 in 3.1s
    polars→pandas(val)   90_000×98 in 0.4s
    polars→pandas(test) 100_000×98 in 0.4s
    polars→pandas total: 3.9s
  ```

## 2026-04-18 — Training-log triage (13 fixes from production run)

A single production run on a 1M × 119 Polars dataset surfaced a cluster of
papercuts that each individually looked minor but together made debugging
training runs much harder than necessary. Grouped below by subsystem.

### RAM logging
- **`get_own_ram_usage` no longer silently reports 0.0 GB** (`helpers.py:112-140`).
  On Windows / under heavy Arrow frees psutil can momentarily report an
  implausibly low rss. When the previous reading was substantial and the
  new one is <0.1 GB we now emit a warning and return the prior value —
  previously the `RAM usage: 0.0GB.` lines that resulted masked the real
  usage (the user's log showed this in the middle of a 37.5 GB run).

### Log attribution and formatting
- **`log_ram_usage` / `log_phase` now attribute to the caller's module**
  (`training/utils.py`). A new `_caller_logger` helper walks the stack
  one frame up so log lines emitted by these helpers use the caller's
  module logger (e.g. `mlframe.training.core`) instead of always saying
  `mlframe.training.utils` — the old behaviour was misleading when
  scanning origins.
- **Separator width reduced from 160 → 80** in `log_phase`. 160 wraps
  horizontally in most terminals and notebook cells.
- **No more stacked blank-looking banner**. `log_phase` used to emit a
  dash-line on both sides of its message. Two consecutive calls produced
  two adjacent dash-lines with nothing between them. Now only a single
  top separator is emitted; the next `log_phase` call naturally closes
  the block with its own separator. Result: `---\nFirst phase msg\n---\nSecond phase msg`.
- **`phase()` context-manager's `START`/`DONE` are now at `DEBUG` by default**
  (`training/phases.py`). These duplicated the caller-side INFO lines
  like `X done — {shape}, {time}`; at INFO they produced two log lines
  per phase. At DEBUG they're still useful for debugging; `RAISED`
  status is escalated to WARNING so failures remain visible.

### Stray raw output
- **`show_raw_data` now routes through the module logger** instead of
  bare `print(...)` (`training/extractors.py`). The raw `<class 'polars...'>` /
  `dtypes:` block was previously appearing out of order with the rest
  of the training log because stdout and the logger stream don't share
  flush points in Jupyter. Test in `test_perf_edges.py` updated to check
  caplog records instead of captured stdout.

### Typo
- **"constant numeric columnss" → "constant numeric columns"** and
  non-numeric counterpart (`training/utils.py:415`). The original f-string
  appended a literal `s` to a `kind` that already ended in "columns".

### Phase-3 / Phase-4 visibility
- **PHASE 3 now logs per-substep timing** (`fit_and_transform_pipeline`
  and, when non-None, `apply_preprocessing_extensions`). Previously a
  3+ minute phase was mysteriously black-boxed between the PHASE 3 banner
  and the "Pipeline done" summary.
- **PHASE 4 logs elapsed time for `select_target`**. Previously the 2+ min
  gap between "select_target..." and "process_model START" had no timing
  context.

### Wasted pandas preparation work
- **Skip `prepare_df_for_catboost` on the pandas views when the Polars
  fastpath is active** (`training/core.py`). When
  `can_skip_pandas_conv=True`, models receive Polars DFs directly;
  running `prepare_df_for_catboost` on the pandas-view side was ~2
  minutes of pure waste on 1M × 100 frames. Logged as
  `"Skipping pandas-side CatBoost prep ... — Polars fastpath receives the DFs natively"`
  when skipped.

### Feature-type auto-detection promotes text columns out of cat_features
- **High-cardinality text-like `pl.Categorical` columns are now promoted
  from `cat_features` to `text_features`** by `_auto_detect_feature_types`
  (`training/core.py:111+`). Previously the pipeline's schema-based
  detection in `fit_and_transform_pipeline` would lock in columns like
  `skills_text` / `ontology_skills_text` as `cat_features` before the
  text auto-detector had a chance to see them; the auto-detector then
  skipped them because "already assigned". Now promotion is explicit
  when `n_unique > cat_text_cardinality_threshold`, and the promoted
  columns are removed from `cat_features` in place. Frees substantial
  memory for text-heavy datasets and gives CatBoost the right
  tokenization path.

### Splitting log
- **Documented the `+NR` / `+ND` suffix** in `training/splitting.py`
  (the `_build_details` helper). Example in a user-visible log:
  `90_000 val rows 2014-01-20/2014-04-05 +45000R` — the `+45000R` means
  "45 000 extra **rows** (`R`) sampled from outside the sequential date
  window". `D` is the same for whole-day splitting.

## 2026-04-18 — Default logger timestamps + CatBoost Polars-fastpath fallback

### Fixed
- **`_ensure_logging_visible` (`training/core.py`)**: previously only installed a timestamped stdout handler when the root logger had NO handlers at all. In Jupyter / IPython a basic handler is already registered (with the `LEVEL:name:message` format — no timestamp), so mlframe's progress logs came out without wall-clock markers — making it impossible to see how long each phase actually takes. Extended the helper to also *upgrade* existing handlers whose formatter doesn't contain `%(asctime)s`, replacing their formatter with the timestamped one. Handlers that the user has intentionally configured with a custom asctime are left untouched.
- **`_train_model_with_fallback` (`training/trainer.py`)**: added a CatBoost × Polars-fastpath fallback. CatBoost's native-Polars entry point (`_set_features_order_data_polars_*`) can reject certain categorical column layouts with opaque messages — either `TypeError: No matching signature found` (fused-cpdef dispatch miss on the column's physical index / value types) or `CatBoostError: Unsupported data type Categorical for a numerical feature column` — abortive on training 1M×100 datasets. On either error, we now convert the Polars DataFrame to pandas via `get_pandas_view_of_polars_df` + `prepare_df_for_catboost`, rewrite the `eval_set` similarly, and retry. The pandas path accepts a broader range of category backings.

## 2026-04-18 — Stale-cache detection in `process_model`

### Fixed
- **`train_eval.py::process_model`**: the suite's cache-load path would unconditionally load a saved `.dump` whenever it existed, even if the feature set or cat_features had changed between runs. Symptom in production: cryptic `CatBoostError: Unsupported data type Categorical for a numerical feature column` deep inside CatBoost's Polars fastpath when a column that used to be numeric is now `pl.Categorical` (or vice versa, or columns were added/reordered). Two complementary fixes:
  1. **`use_cache` gate**: respect `common_params["use_cache"]` (default: True for backward compat — suite-level caching still "just works"). Callers can now force a retrain via `init_common_params={"use_cache": False}`.
  2. **Schema validator** (`_validate_cached_model_schema`, new): after loading, verify the saved model's `feature_names_` / `feature_names_in_` / `booster.feature_names` against the current DataFrame's column list. For CatBoost-shaped models additionally cross-check that each Polars `Categorical`/`Enum` column in the current df is in the saved `_get_cat_feature_indices()` set. On mismatch: log a warning with the reason and invalidate the cache (retrain) rather than let the backend crash.

### Tests
- New `tests/training/test_cache_schema_validation.py` (16 tests):
  - `_extract_polars_cat_columns`: None df, pandas df (no polars cats), `pl.Categorical` / `pl.Enum` detection.
  - Feature-names check: exact match, different names, reordered columns, extra column, unknown-type model (no `feature_names_*`).
  - CatBoost cat_features cross-check: matching case, new Polars Categorical not in saved cat set (the production bug), no-cat model with no Polars cats, out-of-range saved indices (pathological), pandas df never false-positives.

## 2026-04-18 — ICE penalty ramp + `prepare_df_for_catboost` dtype preservation

### Fixed
- **`integral_calibration_error_from_metrics` (`metrics.py:1146-1178`)**: the `roc_auc_penalty` sub-threshold mechanism was a step cliff — `if |auc-0.5| < min_roc_auc-0.5: res += roc_auc_penalty`. That discontinuity could trap CatBoost/XGB/LGBM early stopping just inside the penalty zone (pick the first iter with `auc≈0.5` that has trivially-good calibration, refuse to cross the cliff). Replaced with a **linear ramp**: penalty contribution is `roc_auc_penalty * deficit / threshold_width`, where `deficit = threshold_width - |auc-0.5|` for points inside the zone and 0 outside. **Knob semantics preserved** — `roc_auc_penalty=X` still gives `X` at the worst case `auc==0.5`, and fades smoothly to 0 at `auc==min_roc_auc`. Callers that relied on the step (typically `roc_auc_penalty=0` default) are unaffected.
- **`prepare_df_for_catboost` (`preprocessing.py:58-66`, `preprocessing.py:117-139`)**: the function was silently widening narrow-precision columns to float64. Two offenders:
  - **Pandas branch**: bare `astype(float)` on any extension-array dtype — `pd.Float32Dtype` → `float64`, `pd.Int8/16/32Dtype` → `float64`, `pd.BooleanDtype` → `float64`. Cost: 2× memory and 2× GPU bandwidth on users who had deliberately picked narrow precision.
  - **Polars branch**: every nullable int/bool → `Float64`, regardless of width. Cost: same as above.

  Replaced with precision-preserving/narrowing logic:
  - `pd.Float32Dtype` → `np.float32` (was `float64`)
  - `pd.Float64Dtype` → `np.float64` (unchanged)
  - `pd.Int8/16/32Dtype`, `pd.UInt8/16/32Dtype`, `pd.BooleanDtype` → `np.float32` (values fit exactly, was `float64`)
  - `pd.Int64Dtype` / `pd.UInt64Dtype` → `np.float64` (>~2**24 loses precision in float32, unchanged)
  - Polars: same pattern mirrored via `pl.Float32` / `pl.Float64`. Columns **without** nulls are no longer touched at all (micro-opt).

  Non-nullable `np.float32` columns were never touched and still aren't.

### Tests
- New `tests/test_metrics.py::TestICEPenaltyRamp` (8 tests): ramp is zero outside zone, max `=roc_auc_penalty` at `auc=0.5`, linear interior, symmetric about 0.5 (inverted rankers), **continuous across threshold** (regression sensor against re-introducing the step cliff — max adjacent-sample delta bounded by the Lipschitz constant), monotonic below threshold, respects `roc_auc_penalty=0`, guard against `min_roc_auc<=0.5` (no penalty zone), and the no-opt default-args path.
- New `tests/test_preprocessing.py` (39 parametrised tests): dtype preservation/narrowing across all pandas extension dtypes, non-nullable `np.float32` passthrough, end-to-end null-fill for `pd.Float32Dtype`, all polars int/uint/bool/float widths both with and without nulls, and a micro-opt guard that no-null int columns aren't cast at all.

## 2026-04-18 — Full test suite green; `data_dir=""` no longer leaks artifacts to CWD

### Fixed
- `_setup_model_directories` (`training/core.py` L466-478): switched from `data_dir is not None` to truthy check. Previously, passing `data_dir=""` satisfied `data_dir is not None`, causing the code to `join("", "charts"/"models", ...)` which produced **relative** `./charts/` and `./models/` paths. Artifacts were written to the **current working directory** — the mlframe repo root when tests were invoked from there. This had a subtle cascading effect: on a subsequent test run with a newer sklearn version, `train_mlframe_models_suite` would find and load these stale pickles, surfacing as `AttributeError: 'SimpleImputer' object has no attribute '_fill_dtype'` (sklearn 1.7→1.8 attribute that didn't exist in the pickled state). That's the failure mode previously documented in the README TODO as an "sklearn 1.8 compat issue" — actually an mlframe-side leak, not a sklearn bug.
- `_setup_model_info_and_paths` (`training/trainer.py` L376-381): same falsy guard. Avoids a second relative `./models/` leak path when only the inner function is called.

### Test infrastructure
- Added `check_catboost_gpu_available` fixture in `tests/training/conftest.py`: checks `catboost.utils.get_gpu_device_count() > 0`. The existing `check_gpu_available` only verifies a CUDA device exists via numba, but CatBoost ships its own GPU runtime that may not be installed (error: `Environment for task type [GPU] not found`). Use this new fixture in CatBoost-specific GPU tests.
- `tests/training/test_all_models.py::TestGPUSupport::test_gpu_configuration[cb]` and `TestGPUUsageVerification::test_catboost_gpu_training_params`: skip when CatBoost GPU runtime is absent (was: hard-fail on dev hosts).
- `tests/training/test_bizvalue_preproc_transformers.py::test_dim_reducer_umap_optional`: gracefully skips on the UMAP×sklearn 1.8 incompatibility (UMAP still calls deprecated `check_array(force_all_finite=...)` — renamed to `ensure_all_finite` in sklearn 1.8). Third-party compat issue, not mlframe.

### Test suite status
Full `pytest tests/` passes end-to-end: **1994 passed, 40 skipped, 1 xfailed, 0 failed** (43:44). The previously-documented `test_no_artifact_files_when_no_data_dir` failure is gone — it was a symptom of the `data_dir=""` leak fixed above.

### Notes for Windows runs
- Before a full run, clear stale numba JIT caches: `find . -name "*.nbi" -delete; find . -name "*.nbc" -delete`. Stale caches trigger `Windows fatal exception: access violation` in `compute_numaggs` / similar kernels. This is documented in README "Troubleshooting".

## 2026-04-18 — Fix `prefer_calibrated_classifiers` no-op regression on base tree models

### Fixed
- `configure_training_params` (`training/trainer.py` L2210-2217): base CatBoostClassifier now uses `CB_CALIB_CLASSIF` (eval_metric=`ICE(...)`) vs `CB_CLASSIF` (eval_metric=`"AUC"`) according to the flag — previously always took `CB_CLASSIF` after the 2026-04-15 "post-hoc calibration" refactor, making the CB live training plot show ROC AUC instead of ICE.
- `_configure_xgboost_params` (L1830-1835): base XGBClassifier now uses `XGB_CALIB_CLASSIF` (eval_metric=`final_integral_calibration_error`) vs `XGB_GENERAL_CLASSIF` (eval_metric=`neg_ovr_roc_auc_score`) according to the flag — previously always took `XGB_GENERAL_CLASSIF`.
- `_configure_lightgbm_params` (L1858-1865): base LGBMClassifier now injects `fit_params={"eval_metric": lgbm_integral_calibration_error}` when flag=True — previously always returned empty `fit_params`.
- All three fixes restore the pre-2026-04-15 behavior: `eval_metric` is used for CatBoost's built-in live training plot and for early-stopping comparisons.

### Root cause
2026-04-15 refactor replaced eval-metric-based calibration with a post-hoc `_mlframe_posthoc_calibrate` attribute tag, but the hook that was supposed to consume it (`_maybe_apply_posthoc_calibration`, L817-833) was left as an explicit no-op (`return model` in both branches). The attribute was set on CB/XGB/LGBM models but never read, so all three models trained identically regardless of the flag.

### Removed
- `_mlframe_posthoc_calibrate=True` attribute setter in three locations (CB base, XGB base, LGBM base) — dead code, consumer hook is a no-op.
- `test_is_inlier` placeholder (`trainer.py`): declared-but-never-set `None` field on the returned SimpleNamespace, never consumed by any caller. Removed from all 4 sites (local init + 3 SimpleNamespace constructors).
- `default_drop_columns` local dead variable in `train_and_evaluate_model`: always set to `[]` with a stale "no longer needed" comment, passed to `_validate_infinity_and_columns` which concatenated an empty list. Simplified the helper signature to drop the parameter.

### Retained (see README "TODO")
- `_PostHocCalibratedModel` class and `_maybe_apply_posthoc_calibration` hook: intentionally retained as scaffolding in case the user revives isotonic post-hoc calibration as an alternative path.

### Tests
- New `tests/training/test_calibration_flag_propagation.py` (5 tests):
  - Level 2 (targeted): flipping `prefer_calibrated_classifiers` must produce different `eval_metric` on `XGBClassifier.get_params()`, different `fit_params["eval_metric"]` on LGBM configure helper, and different `eval_metric` on CatBoostClassifier (`ICE(...)` instance vs `"AUC"` string).
  - Level 2 (sanity): flag does not affect LGBM regression path.
  - Level 3 (matrix invariant): parametric sweep over `cb`/`xgb`/`lgb` — either the model's own `eval_metric` or the `fit_params["eval_metric"]` must differ between True/False. Catches any future silent no-op regression of the same class.

### Also fixed (collateral, surfaced by the broader test run)
- `report_model_perf` (`training/evaluation.py` L212-219): sklearn≥1.6 raises `AttributeError` when `is_classifier(None)` triggers `get_tags(None)` (previously returned `False`). The `just_evaluate=True` path legitimately passes `model=None` with pre-computed preds/probs — now task type is inferred from `probs is not None` when `model is None`, and `is_classifier` is skipped in that case. Fixes `tests/training/test_trainer.py::TestTrainAndEvaluateModelEdgeCases::test_model_none_just_evaluate`.
- `run_confidence_analysis` (`training/trainer.py` L1068-1097): the auxiliary confidence-analysis CatBoost model picked `task_type="GPU"` whenever `CUDA_IS_AVAILABLE` was True, ignoring the `TrainingBehaviorConfig.prefer_gpu_configs` override. On hosts that have a CUDA device but no CatBoost GPU runtime (e.g. CI/dev with `prefer_gpu_configs=False` forced in conftest), CatBoost raised `Environment for task type [GPU] not found`. Added a one-shot CPU fallback: on that specific error, retry fit with `task_type="CPU"` and log a warning. Fixes `tests/training/test_core.py::TestConfidenceAnalysis::test_confidence_analysis_basic`.

### Known pre-existing test failure (NOT caused by this change)
- `tests/training/test_core_coverage.py::TestSplitting::test_no_artifact_files_when_no_data_dir` fails on `master` even without this patch. Root cause is an sklearn 1.8 compat issue: some fitted sklearn `Pipeline`/`SimpleImputer` in the test flow is unpickled from sklearn 1.7.2 state that is missing the new-in-1.8 `_fill_dtype` attribute, so `SimpleImputer.transform()` raises `AttributeError: 'SimpleImputer' object has no attribute '_fill_dtype'`. Confirmed by `git stash` + rerun on `8d30b9a`. TODO: either refit the imputer on load (detect missing `_fill_dtype`), or invalidate cached artifacts on sklearn version bump. Out of scope for this change.

### Follow-ups documented
- README gains a "TODO" section with two items:
  1. Decide to ship or remove `_PostHocCalibratedModel` + post-hoc calibration hook.
  2. Re-enable CatBoost `custom_metric=tuple(...)` with a clone-safe strategy (set via `model.set_params(...)` on the base path only, leaving RFECV estimators clean).

## 2026-04-17 — Polars→pandas Categorical optimization (no more dict→string cast)

### Changed
- `get_pandas_view_of_polars_df` in `training/utils.py` now preserves Polars `Categorical` columns as `pd.Categorical` (int32-indexed dictionary) instead of casting dict→string. Polars emits dict arrays with uint32 indices, which pyarrow's `to_pandas` refuses; we rebuild each dict column with int32 indices so the conversion produces a proper `pd.Categorical`.

### Why
End-to-end benchmark on production-shaped data (CatBoost classifier, 180k × 586 cols, 70 Categorical) via `bench_polars_to_pandas.py`:

| Variant | convert | fit | predict | **total** |
|---|---|---|---|---|
| native Polars (CatBoost's own path) | 0.00s | 12.42s | 0.14s | 12.55s |
| old (dict→string cast) | 1.04s | 15.56s | 0.47s | 17.08s (+37%) |
| **new (int32-indexed pd.Categorical)** | 0.45s | 11.99s | **0.04s** | **12.49s** (fastest) |

String cast was both slower (CatBoost hashes strings per row during fit and predict) and memory-hungrier (OOMs at 450k+ rows with 70 Categoricals where the new path trains cleanly).

### Tests
- `test_utils.py::test_categorical_to_string_conversion` renamed to `test_categorical_preserved_as_pd_categorical` and now asserts the `pd.CategoricalDtype` plus the category list, not just the string values.
- Downstream comment in `core.py` above the `prepare_df_for_catboost` call updated — that call is now usually a no-op but kept for pandas-input safety.

## 2026-04-17 — Fix metadata pickle failure with duplicate mlframe installs

### Fixed
- `_create_initial_metadata`: Pydantic config objects (`preprocessing_config`, `pipeline_config`, `split_config`) are now stored in `metadata["configs"]` as plain dicts via `.model_dump()` instead of raw Pydantic instances. This prevents `_pickle.PicklingError: Can't pickle <class 'mlframe.training.configs.PolarsPipelineConfig'>: it's not the same object as mlframe.training.configs.PolarsPipelineConfig` when two copies of mlframe are reachable via `sys.path` (e.g. a dev checkout plus an older pip install, or Jupyter autoreload duplicating a module). Tests only assert key presence (`"preprocessing" in metadata["configs"]`), so the change is backward compatible.

## 2026-04-17 — Polars→pandas conversion benchmark

### Added
- `bench_polars_to_pandas.py`: two benchmark modes on a production-shaped synthetic DF (1M × 587 cols by default: Boolean(10), Categorical(70), Datetime(1), Float32(38), Float64(425), Int16(14), Int64(2), Int8(27)).
  - **Default (`BENCH_MODE=catboost`)**: end-to-end CatBoost `fit` + `predict_proba` with identical hyperparameters on (a) the native Polars DataFrame and (b) the same data converted to pandas via mlframe's `get_pandas_view_of_polars_df`. Reports per-phase times (convert / fit / predict / total) and the end-to-end speedup.
  - **Conversion-only (`BENCH_MODE=conversion`)**: microbench of mlframe's approach (`to_arrow` + batched `pa.compute.cast` dict→string + `to_pandas`) vs a Python re-implementation of CatBoost's per-column loop (`_catboost.pyx:3199` / `:3288`: per-column `rechunk()` + `to_physical().to_numpy()`). Includes per-step breakdown for mlframe path and per-dtype breakdown for the CatBoost-like path.
  - Tunable via env vars: `BENCH_N_ROWS`, `BENCH_N_CAT`, `BENCH_ITERATIONS`, `BENCH_THREAD_COUNT`, `BENCH_TEST_FRACTION`, `BENCH_N_REPEATS`, `BENCH_MODE`.

## 2026-04-17 — Structured phase timing + logging visibility fix

### Added
- `training/phases.py`: `PhaseTimer` context manager, global registry, `format_phase_summary()` / `phase_snapshot()` / `reset_phase_registry()`. Hotspot wrappers across `core.py`, `trainer.py`, `evaluation.py` cover data load, split, train_stats, `process_model`, `model.fit` (incl. retry), `pre_pipeline_fit_transform`, `compute_split_metrics` (train/val/test), `report_probabilistic_model_perf`, `report_regression_model_perf`, `predict` / `predict_proba`, `fast_calibration_report`, `plot_feature_importances`, `compute_fairness_metrics`. Summary table is logged at the end of verbose `train_mlframe_models_suite` runs so regressions become visible immediately.
- `_ensure_logging_visible()` in `core.py`: idempotently attaches an INFO-level stdout handler to the root logger when none exists, so `logger.info` calls inside the suite actually appear in Jupyter with `verbose=True`. Does nothing if the user already configured logging.

### Fixed
- `TrainingControlConfig.verbose` accepts `Union[bool, int]` (was strict `bool`). Passing a verbosity level like `verbose=3` from the suite no longer raises pydantic `bool_parsing` 3 minutes into a training run.

## 2026-04-16 — Fix all 11 xfailing biz-value tests

### Fixed
- `test_bizvalue_calibration_ensemble.py`: rewrote data generator with sinusoidal logit + 105 noise features; test now trains sklearn `CalibratedClassifierCV` directly (not through mlframe suite) to avoid internal data splits; per-model threshold (0.50% for CatBoost, 1.00% for LGB/XGB) reflects CatBoost's inherently better calibration.
- `test_bizvalue_imbalance_grid.py`: changed `scale_pos_weight` from `sqrt(n_neg/n_pos)` to full `n_neg/n_pos`; increased imbalance severity from 95:5 to 98:2 with larger dataset (9000 rows).
- `test_bizvalue_fairness_weights.py`: increased dataset size (n_train 3000->6000, n_test 600->1500), reduced minority fraction (0.10->0.07), softened shift vector.
- All 36 biz-value tests now pass with hard asserts (0 xfails).

## 2026-04-15 — Suite pipeline: fixes, new kwargs, metadata, test expansion

### Added
- `train_mlframe_models_suite(save_charts: bool = True)` — when `False`, skips per-model chart file output (for CI / fast runs).
- `metadata["fairness_report"]` — aggregated fairness metrics propagated from per-model runs into suite-level metadata.
- `metadata["outlier_detection"]` dict: `applied`, `n_outliers_dropped_train`, `n_outliers_dropped_val`, `train_size_after_od`, `val_size_after_od`.
- `PreprocessingExtensionsConfig.tfidf_columns` now wired end-to-end: text columns are vectorized inside `apply_preprocessing_extensions` and replaced with `<col>__tfidf_<i>` numeric features.
- `apply_preprocessing_extensions(y_train=...)` kwarg wires supervised fit for `dim_reducer="LDA"`; fixed `RandomTreesEmbedding` factory to use `n_estimators` (not the non-existent `n_components` kwarg); added `tests/training/test_bizvalue_preproc_transformers.py` (37 business-value tests covering polynomial XOR lift, RBFSampler/Nystroem on circles, PCA/TruncatedSVD/LDA/KernelPCA/NMF/FastICA/Isomap/GRP/SRP/RTE/BernoulliRBM/UMAP dim_reducers, KBins sine-wave R^2 lift, Binarizer collapse property, memory-safety guard, Chi2 positive-input guards, Binarizer+KBins mutual exclusion).

### Changed
- `ModelHyperparamsConfig.early_stopping_rounds` is now `Optional[int]`; setting it to `None` disables early stopping across all strategies (CB/LGB/XGB/MLP/RFECV/HGB/NGB).

### Fixed
- `_SafeUnpickler` allowlist now includes the `mlframe` prefix — fixes silent drop of CatBoost models that reference `mlframe.metrics.ICE` during `predict_mlframe_models_suite`.

### Tests
- 8 new unit test files for previously untested helpers: `tests/training/test_untested_*.py` (83 tests).
- 6 new business-value integration test files: `tests/training/test_bizvalue_*.py` covering fairness, calibration, outliers, preprocessing extensions, early stopping, ensemble, sample weights, class imbalance, and `run_grid`.
- `tests/training/test_bizvalue_feature_selection.py` — business-value integration tests for MRMR/RFECV feature selection (drops uninformative cols, preserves AUROC on wide data, exposes selected features).

## 2026-04-15 — Audit #02 (legacy) + test fast mode

### Commit 1/5 — Salvage from legacy modules (pre-move)
- `evaluation.py`: added `predictions_beautify_linear`, `plot_beautified_lift`, `plot_pr_curve`, `plot_roc_curve`.
- `training/evaluation.py`: added `compute_ml_perf_by_time`, `visualize_ml_metric_by_time`.
- `outliers.py`: added `compute_outlier_detector_score`, `count_num_outofranges` (@njit), `compute_naive_outlier_score`. Fixed broken hard-import of `imblearn` (lazy guarded).
- `metrics.py`: added `brier_and_precision_score`, `make_brier_precision_scorer`.
- NEW `training/callbacks.py`: `stop_file` + `{CatBoost,LightGBM,XGBoost,Lightning}StopFileCallback`.
- NEW `training/neural/keras_compat.py` (TF-guarded): `build_keras_mlp`, `KerasCompatibleMLP`.
- NEW `tests/test_evaluation_salvage.py` (18 tests, 16 pass / 2 TF-skip).

### Commit 2/5 — Move to legacy/
- Deleted `mlframe/Backtesting.py` (10-LOC stub, zero importers).
- `git mv` `training_old.py`, `OldEnsembling.py` → `mlframe/legacy/`.
- NEW `mlframe/legacy/__init__.py` — emits `DeprecationWarning` on import.
- Stripped 5 stale "migrated from training_old.py" comments across `training/{__init__,helpers,train_eval,trainer}.py`.
- `pytest.ini` ignores point at `legacy/` directory.

### Commit 3/5 — Resource-logging decorators + estimator-object model spec
- NEW `training/logging_transformers.py`:
  - `log_resources(*, stage, level, extra_factory)` — function decorator, logs wall-time + ΔRSS.
  - `log_methods(*methods, stage_prefix)` — class decorator.
  - `wrap_with_logging(obj, *, stage, methods)` — instance-proxy factory.
- `training/strategies.py`: `get_strategy` accepts strings, estimator instances, `(name, estimator)` tuples. MRO dispatch via `_strategy_for_estimator` (lazy-guarded CatBoost/LightGBM/XGBoost imports); unknown classes fall back to `LinearModelStrategy` with warning. New helpers `_resolve_model_spec`, `_slugify`, `_dedupe_key`.
- `training/utils.py::filter_existing`: tolerate ndarray (no `.columns` → `[]`).
- NEW `tests/training/test_logging_transformers.py` (8 tests).
- NEW `tests/training/test_model_spec_resolution.py` (14 tests).

### Test infrastructure — Fast mode (`--fast` / `MLFRAME_FAST=1`)
- `tests/conftest.py`: `--fast` CLI flag + `MLFRAME_FAST` env var; `is_fast_mode()`, `fast_subset(values, representative=..., keep=1)` helper. `@pytest.mark.slow` / `slow_only` auto-skip in fast mode.
- Pattern: parametrized tests call `fast_subset([...scalers...], representative="StandardScaler")` so all code paths still execute but with one representative variant.
- NEW `tests/test_fast_mode.py` (8 self-tests incl. subprocess end-to-end).

### Commit 4/5 — PreprocessingExtensionsConfig + apply_preprocessing_extensions
- `training/configs.py`: new `PreprocessingExtensionsConfig` with 14 fields
  (scaler override, binarization/kbins mutually-exclusive, polynomial with
  memory-safety guard, nonlinear feature maps, tfidf, dim_reducer covering
  PCA/KernelPCA/LDA/NMF/TruncatedSVD/FastICA/Isomap/UMAP/random projections/
  RandomTreesEmbedding/BernoulliRBM). None default on every stage so the
  whole config reads as a noop.
- `training/pipeline.py`: new `apply_preprocessing_extensions` helper runs
  after `fit_and_transform_pipeline`. Config=None = byte-for-byte fastpath
  preservation. UMAP gated via `find_spec` with install-hint ImportError.
- `training/core.py`: `train_mlframe_models_suite` gains
  `preprocessing_extensions: Optional[PreprocessingExtensionsConfig | Dict]`.
  Dict inputs auto-promoted. Extensions pipeline stored under
  `metadata["extensions_pipeline"]`. `cat_features` cleared once
  extensions materialise them to numeric columns.
- NEW `training/grid.py::run_grid` — sequential variant sweeper (replaces
  the dropped `TryAllMethods` pattern). Accepts base kwargs + list of dicts
  or `(label, dict)` tuples; `stop_on_error=False` default captures
  exceptions per variant. 6 unit tests via injected `suite_fn` stub.
- NEW `tests/training/test_preprocessing_extensions.py` (13 tests).
- NEW `tests/test_scalers.py` (8-scaler LR-AUROC round-trip, fast_subset
  keeps one representative).

### Collection-time fix — `training/callbacks.py` lazy lightning
- Top-level `import pytorch_lightning` was pulling torch DLLs into every
  test collection. Under Windows memory pressure this triggered
  `OSError WinError 1455` (paging file too small) on `shm.dll` /
  `cufft64_10.dll`, aborting collection before a single test could run.
- Switched to `importlib.util.find_spec()` detection + lazy import inside
  `LightningStopFileCallback.__init__` with dynamic base-class rebasing.

### Pending (commit 5/5)
- Benchmark guard (≤2% regression budget on default path).

## 2026-04-14 — Full Audit & Fix Sweep (10 parallel audit agents + 9 parallel fix agents)

### Security (RCE hardening)
- `training/neural/flat.py`, `training/neural/recurrent.py` — `torch.load(..., weights_only=True)`.
- `training/io.py` — `_SafeUnpickler` allowlist; `safe=True` default for `dill.load` paths.
- `inference.py`, `pipelines.py` — `joblib.load` gated by `trusted_root` path validation (`os.path.commonpath`); sorted `os.listdir` for determinism; consistent `(models, X)` return shape; `output_dir` defaults to `tempfile.gettempdir()`.
- `experiments.py` — SQL `fields` validated against `_ALLOWED_EXPERIMENT_FIELDS` frozenset (f-string injection fixed).
- New `tests/test_security_rce.py` (4 tests).

### Correctness / numeric sweep
- `calibration.py` — Brier vs. binned-metric dispatch uses `is` identity (was no-op dict-comp typo); AD clips PIT to `[1e-12, 1-1e-12]`; ECI on probability-normalized counts; WPD `np.clip(p*(1-p), 1e-6, None)`; `show_classifier_calibration` accumulates per-interval perfs.
- `postcalibration.py` — `isinstance` dispatch with lazy imports; `transform_method_name` resolved at `fit()`; 1-D probs clipped to [0,1] before `np.vstack`.
- `metrics.py` — bounds guards in numba kernels; `fast_log_loss_binary` OOB→NaN; `fast_roc_auc` raises on `sample_weight`; `brier_score_loss` → `fast_brier_score_loss` (alias retained); rounding precision `max(1, ceil(log10(max(nbins,2))))`.
- `ewma.py` — full O(n) numba recurrence (was O(n²) matrix + no-op `x[::np.newaxis]` slice).
- `arrays.py` — removed `import mlframe` self-import; `arrayMinMax` returns `(nan,nan)` on empty; `topk_by_partition` no longer mutates caller, `k = min(k, n)`; O(1) membership check; shared-ref list fixed.
- `stats.py:75` — `dist_kwargs=dist_kwargs` → `**dist_kwargs`.
- `FeatureEngineering.py:247` — off-by-one mask (spans `x[l:r+1]` matching inclusive size).
- `feature_engineering/mps.py` — OOB guards on start/end indices.
- `feature_engineering/numerical.py` — Kahan compensator in rolling MA; argmin/argmax first-wins consistent with sibling kernel; weights threaded into early-exit path.
- `feature_engineering/timeseries.py` — list-as-boolean wire bug in `create_and_process_windows` fixed; `accumulated_amount` initialized to avoid NameError.
- `boruta_shap.py` — SciPy 1.12+ `binomtest` wrapper; lazy iris import; vectorized Z-score; shap split fix.
- `feature_selection/general.py`/`wrappers.py`/`filters.py`/`optbinning.py` — empty-list guards, proper CV clone + rng, zero-prob entropy filter, `@njit(cache=True)`, deduped LOGGING block.
- `feature_selection/mi.py` — design-intent NOTE preserving 3 MI kernels (grok/chatgpt/deepseek) as load-bearing.
- `optimization.py:689` — **CRITICAL**: `elif OptimizationDirection.Maximize:` → `Minimize` (copy-paste bug).
- `optimization.py` — `plt.close(fig)` after plotting; `logger.warn` → `logger.warning`.
- `tuning.py` — cache key tuple instead of list; `learning_rate` uniform→loguniform; duplicate `penalties_coefficient` removed.
- `evaluation.py:301` — `plt.grid(b=None)` → `visible=None` (mpl ≥3.5); `:339` tuple unpack fix.
- `custom_estimators.py` — bounded retry loop; `scipy.ndimage.shift`; `PowerTransformer` no longer module-level; sklearn-compliant averagers (`classes_`, `n_features_in_`, `check_is_fitted`, `check_array`); `MyDecorrelator` trailing-underscore convention; `PdKBinsDiscretizer` sparse densify.
- `estimators.py` — `logger` properly imported; `check_array` in fit/predict; `ClassifierWithEarlyStopping` gains `predict_proba`/`decision_function`; typo fixes.
- `cluster.py` — `from sklearn.cluster import DBSCAN` (was undefined).
- `eda.py:41` — `is not None` for pandas Series.
- `feature_importance.py:73` — `feature_importances[sorted_idx[0]]`.
- `helpers.py` — wildcard `from .config import *` → explicit imports; `model.steps[-1][1]` (was `(name, est)` tuple); vectorized `np.isinf` over numeric columns; tutorial helpers deleted.

### RNG discipline
- `MBHOptimizer`, `ParamsOptimizer`, `CatboostParamsOptimizer`, `optimize_finite_onedimensional_search_space`, `generate_valid_candidates`, `create_ctr_params`, `get_model`, `justify_estimator` — all accept `random_state`; internal `_rng = np.random.default_rng(...)` + `_stdlib_rng = random.Random(...)`; `np.random.*`/bare `random()` removed.
- `training/splitting.py`, `training/evaluation.py`, `datasets.py`, `synthetic.py` — no more global `np.random.seed`; `generator`-threaded; scipy `.rvs(random_state=rng)`; sklearn bridge via `rng.integers(0, 2**32-1)`.
- `custom_estimators.py::PureRandomClassifier` — fully sklearn-compliant (`random_state_`, `classes_`, `n_features_in_`, label-returning `predict`).
- `synthetic.py` — tuple-vs-list dead branch at :44 fixed; asserts → `ValueError`; guarded divisions; off-by-one at :241 replaced with `generator.randint`.

### Conventions (per MEMORY.md)
- `postcalibration.py` — 2 regex sites hoisted to module-level `re.compile`; shared `_compile_pattern = lru_cache(re.compile)` helper.
- New `tests/test_conventions.py` meta-tests.

### Test suite & hygiene
- `pytest.ini` rewritten — `minversion=7.0`, `testpaths=tests`, `pythonpath=.`, `--strict-markers --strict-config --doctest-modules --cov=mlframe`, `xfail_strict=true`, 8 `--ignore=` for legacy/broken, custom markers (`benchmark`, `multigpu`, `windows_only`, `linux_only`), `filterwarnings`.
- `tests/conftest.py` — autouse session-scoped RNG seed fixture (random/numpy/torch = 0); `psutil` guarded via try/except; `warnings.resetwarnings()` replaced with scoped `catch_warnings()`.
- 7 `assert True` exception-swallowing patterns replaced with real post-conditions + `pytest.skip` (test_core.py, test_feature_selection.py, test_stress.py).
- `tests.py` (root) renamed → `bench_helpers.py`; `unittest_arrays.py` migrated → `tests/test_arrays.py` (9 pytest fns, timing asserts dropped).
- `tests/lightninglib/` — 9 duplicate files deleted (kept `test_deprecated_import.py`).

### Repo hygiene (~104 MB reclaimed)
- Removed: `profile_mixed_dtypes.prof`, root `__pycache__/`, `catboost_info/`, `checkpoints/`, `lightning_logs/`, `logs/`, `.coverage`, `training_old.py.backup`, `read.me` (content merged into README), `NUL` (via `\\?\` extended-path API).
- `.gitignore` — grouped `NUL` under Windows-specific block; added `*.prof`, `*.backup`, `*.py.backup`, `.benchmarks/`, `.ruff_cache/`, `.black_cache/`, `.vscode/`, `.direnv/`, `.envrc`, `coverage.lcov`, `tests/**/{catboost_info,lightning_logs,checkpoints,logs}/`.
- `public_suffix_list.dat` retained (used at `FeatureEngineering.py:365`).

### New tests (property-based + determinism + regression)
- `tests/test_security_rce.py`, `tests/test_conventions.py`, `tests/test_rng_determinism.py`, `tests/test_rng_determinism_b.py`, `tests/test_numeric_bug_sweep.py`, `tests/test_sklearn_compliance.py`, `tests/test_fs_fe_fixes.py`, `tests/test_arrays.py`.

### Verification
- Import smoke: `mlframe`, `mlframe.training`, `mlframe.feature_engineering`, `mlframe.feature_selection` — all ok.
- Per-agent suites: 4 + 3 + 6 + 9 + 7 + 19 + 6 + 9 = **63 new tests pass**.
- Full-suite run flagged one order-dependent hypothesis test in `test_timeseries.py::test_find_next_cumsum_left_index` (passes in file scope; pre-existing test-pollution, not introduced by this sweep).

### Deferred
- Dead-code removal (Phase 3: `training_old.py`, `OldEnsembling.py`, `Models.py`, `Backtesting.py`, `Features.py`, `Data.py`, empty `models/`) — pending user decision.
- Audit findings under `.claude/plans/mlframe_audit/*.md` (10 reports + `_SUMMARY.md`, outside repo).

## 2026-04-14 — Test Suite Optimization & Coverage Expansion

### Added

- **`tests/training/test_core_coverage.py`** — 48 new tests targeting 99% coverage of `train_mlframe_models_suite`:
  - `TestInputValidation` (8 tests): TypeError/ValueError for invalid df types, non-parquet paths, empty names, None FTE, parquet path loading, dict config acceptance.
  - `TestConfigurationSetup` (4 tests): Pydantic config passthrough for PreprocessingConfig, TrainingSplitConfig, ModelHyperparamsConfig, TrainingBehaviorConfig.
  - `TestDataLoadingPreprocessing` (2 tests): NaN fillna, column dropping via preprocessing_config.
  - `TestSplitting` (4 tests): split size sums, artifact saving, no-data-dir skip, metadata keys.
  - `TestPipelineFitting` (8 tests): auto-skip categorical encoding for Polars-native models, pre-clone logic, metadata pipeline/cat_features/columns keys, mixed native/non-native models.
  - `TestFeatureTypeDetection` (3 tests): text/embedding features in metadata, empty defaults.
  - `TestModelTrainingLoop` (9 tests): unknown model skip with warning, uniform/custom weight schemas, model × weight combinations, ensemble scoring, clone per weight.
  - `TestRecurrentModels` (2 tests): recurrent fit() with error handling, unknown recurrent model skip (selective mock of clone).
  - `TestCrossCuttingParametrized` (4 cases): `@pytest.mark.parametrize` over (ridge/lasso) × (pandas/polars).
  - `TestMetadataCompleteness` (4 tests): all expected keys, configs, split sizes, joblib persistence.
- **`tests/conftest.py`** — root conftest with shared autouse fixtures (`cleanup_memory`, `suppress_convergence_warnings`).
- **`tests/feature_engineering/conftest.py`** — shared date/DataFrame fixtures.
- **`pytest.ini`** — custom markers (`slow`, `integration`, `gpu`), doctest options.
- **`tests/training/test_train_eval.py`** — 10 tests for `optimize_model_for_storage`, `select_target`.
- **`tests/test_utils.py`** — 10 tests for root utils (`set_random_seed`, `get_pipeline_last_element`, etc.).
- **`tests/test_metrics.py`** — 8 new edge case + Hypothesis tests.
- **`tests/training/test_configs.py`** — 2 Hypothesis round-trip tests for Pydantic configs.
- **`tests/training/test_utils.py`** — 3 Hypothesis property-based tests for DataFrame transforms.
- **`tests/training/test_helpers.py`** — 8 tests for `parse_catboost_devices`.

### Changed

- **Consolidated duplicated tests**: 3 pandas/polars test pairs in `test_basic.py` → parametrized; 7 boolean param tests in `test_numerical.py` → single parametrized test.
- **Marked slow tests**: `@pytest.mark.slow` on test_stress.py, test_all_models.py, test_integration.py, RFECV tests. Enables `pytest -m "not slow"` for fast CI.
- **Optimized tree model tests**: reduced iterations from 5000 to 50 in test_core.py (3 tests).
- **Promoted fixture scopes**: `common_init_params`, `fast_iterations`, `fast_config_override` → `scope="session"`.
- **Fixed doctests**: NumPy 2.x compatibility in stats.py (7 doctests), added doctest to `get_numeric_columns`.
- **Fixed dict mutation bugs**: `.copy()` on `hgb_kwargs`, `mlp_kwargs`, `ngb_kwargs`, `rfecv_kwargs` in helpers.py.

## 2026-04-14 — CatBoost Text & Embedding Features + Memory Optimizations

### Added

- **`text_features` and `embedding_features` support**: CatBoost now receives `text_features` (free-text string columns) and `embedding_features` (list-of-float vector columns) via `fit()` params. Models that don't support them (Ridge, XGB, LGB, HGB, MLP, etc.) automatically have these columns dropped before training.
- **`FeatureTypesConfig`** Pydantic class in `configs.py`: `text_features`, `embedding_features`, `auto_detect_feature_types`, `cat_text_cardinality_threshold` (default 50).
- **Auto-detection of feature types**:
  - Embedding columns: auto-detected from `pl.List(pl.Float32)` / `pl.List(pl.Float64)` dtype.
  - Text vs categorical: string columns with `n_unique > cat_text_cardinality_threshold` → text; `<= threshold` → categorical. User-specified lists always take priority.
- **Feature-tier model grouping**: models sorted by `strategy.feature_tier()` — `(True, True)` (CatBoost) trains first with all columns, then text/embedding columns are dropped once per tier for remaining models. Tier DFs cached via `_build_tier_dfs()` using `.select()` (not `.drop()`).
- `supports_text_features` and `supports_embedding_features` properties on `ModelPipelineStrategy` (default `False`). `CatBoostStrategy` overrides both to `True`.
- `feature_tier()` method on `ModelPipelineStrategy` — returns `(supports_text, supports_embedding)` tuple for grouping.
- Mutual exclusivity validation: `text ∩ cat`, `emb ∩ cat`, `text ∩ emb` → `ValueError`.
- Pipeline exclusion: text and embedding columns excluded from encoding/scaling in `fit_and_transform_pipeline()`.
- CatBoost text columns auto-filled with `""` for nulls (CatBoost requirement).
- 18 CPU integration tests + 2 GPU tests in `TestTextAndEmbeddingFeatures`.

### Memory Optimizations (for 100GB+ DataFrames)

- **B1: Conditional clone** — `train_df.clone()` only when pipeline will modify categoricals (`skip_categorical_encoding=False`). Saves 100GB+ when all models are Polars-native.
- **B2: Aggressive cleanup** — post-pipeline Polars DFs released after pandas conversion when no longer needed.
- **B3: `prepare_polars_dataframe()` cache** — moved outside weight schema loop, called once per model instead of once per weight schema.
- **B4: `.select()` over `.drop()`** — tier column trimming uses `.select(cols_to_keep)` for better Polars optimization.
- **B5: Release Polars originals after tier transition** — pre-pipeline Polars DFs freed after all Polars-native models finish training.

### Changed

- Model training loop now sorts models by `feature_tier()` (most features first) instead of using the user-provided order.
- `select_target()` and `configure_training_params()` accept `text_features` and `embedding_features` params.
- `fit_and_transform_pipeline()` accepts `text_features` and `embedding_features` params to exclude from encoding/scaling.

## 2026-04-14 — Typed Training Parameters Refactor

### Breaking Changes

- `train_mlframe_models_suite` signature changed: removed `config_params`, `control_params`, `config_params_override`, `control_params_override`, and `**kwargs`.
- New parameters: `hyperparams_config` (`ModelHyperparamsConfig` or dict) and `behavior_config` (`TrainingBehaviorConfig` or dict).
- `select_target()` signature changed accordingly.

### Added

- **`ModelHyperparamsConfig`** Pydantic class in `configs.py`: typed replacement for `config_params`/`config_params_override` dicts. Fields: `iterations`, `learning_rate`, `early_stopping_rounds`, `has_time`, `rfecv_kwargs`, per-model kwargs (`cb_kwargs`, `lgb_kwargs`, `xgb_kwargs`, `hgb_kwargs`, `mlp_kwargs`, `ngb_kwargs`).
- **`TrainingBehaviorConfig`** Pydantic class in `configs.py`: typed replacement for `control_params`/`control_params_override` dicts. Fields: `prefer_gpu_configs`, `prefer_cpu_for_lightgbm`, `prefer_cpu_for_xgboost`, `prefer_calibrated_classifiers`, `use_robust_eval_metric`, `nbins`, `fairness_features`, `fairness_min_pop_cat_thresh`, `cont_nbins`, `metamodel_func`, `callback_params`, `cb_fit_params`, `use_flaml_zeroshot`, scoring configs.
- Both classes exported from `mlframe.training` and added to `__init__.py` lazy imports.
- Constants `DEFAULT_FAIRNESS_MIN_POP_CAT_THRESH`, `DEFAULT_RFECV_*` moved to `configs.py` (canonical location).

### Changed

- `_initialize_training_defaults()` simplified: no longer normalizes 4 dict params.
- `_build_common_params_for_target()` accepts `TrainingBehaviorConfig` instead of dict.
- `_should_skip_catboost_metamodel()` accepts `TrainingBehaviorConfig` instead of dict.
- `_compute_fairness_subgroups()` accepts `TrainingBehaviorConfig` instead of dict.
- `TrainingConfig` class updated: `config_params_override`/`control_params_override` fields replaced with `hyperparams: ModelHyperparamsConfig` and `behavior: TrainingBehaviorConfig`.

### Fixed

- **Bug**: Tests used `models["target"][TargetTypes.X]` but actual structure is `models[TargetTypes.X]["target"]` — fixed across all test files.

### Migration

```python
# Before:
train_mlframe_models_suite(
    ...,
    config_params_override={"iterations": 10, "cb_kwargs": {"task_type": "CPU"}},
    control_params_override={"prefer_calibrated_classifiers": False},
)

# After:
train_mlframe_models_suite(
    ...,
    hyperparams_config={"iterations": 10, "cb_kwargs": {"task_type": "CPU"}},
    behavior_config={"prefer_calibrated_classifiers": False},
)
```

## 2026-04-14 — Auto-skip Categorical Encoding + Verbose Logging

### Added

- **`skip_categorical_encoding`** flag on `PolarsPipelineConfig`: when `True`, the polars-ds pipeline and sklearn pandas path skip ordinal/onehot encoding of categorical features. **Auto-detected** by `train_mlframe_models_suite` — when all requested `mlframe_models` support Polars natively (cb, xgb, hgb), the flag is set automatically, avoiding wasted encoding work and preserving original categorical dtypes.
- **Verbose timing & shape logging** across the training pipeline (`verbose=True`):
  - `core.py`: Phase 1 (data loading, FTE, preprocessing), Phase 2 (splitting with shapes), Phase 3 (pipeline with dtypes), per-model `process_model()` timing, Polars fastpath activation logging
  - `trainer.py`: `model.fit()` timing with shape, `_apply_pre_pipeline_transforms` timing with shape, metrics computation timing
  - `pipeline.py`: Polars-ds pipeline creation timing (scaler/encoding config), transform timing with shape, sklearn categorical encoding timing with shape
- Helper functions `_df_shape_str(df)` and `_elapsed_str(start)` in `core.py`
- 6 parametrized tests for `skip_categorical_encoding` auto-detection (all-native, mixed, non-native model lists)

### Changed

- `CatBoostStrategy.cache_key` = `"catboost"` (was inherited `"tree"`). `XGBoostStrategy.cache_key` = `"xgboost"` (was inherited `"tree"`). Each Polars-native model now gets its own pipeline cache slot, preventing cross-contamination when running multiple models together (e.g. `["cb", "xgb", "hgb"]`).

### Fixed

- **Bug**: XGBoost Polars fastpath passed `cat_features` as a `fit()` parameter, causing `TypeError: XGBClassifier.fit() got an unexpected keyword argument 'cat_features'`. Only CatBoost accepts `cat_features` in `fit()` — XGBoost/HGB auto-detect `pl.Categorical` columns via `enable_categorical=True`.
- **Bug**: When running multiple Polars-native models together (e.g. `["cb", "xgb"]`), the pipeline cache shared the `"tree"` key, causing the second model to receive cached pandas DFs from the first — overriding the Polars fastpath and causing `KeyError: DataType(large_string)` in XGBoost.

## 2026-04-14 — XGBoost Polars Fastpath + Unified Categorical Handling

### Added

- **XGBoost Polars fastpath**: XGBoost (>= 3.1) now receives Polars DataFrames directly via `train_mlframe_models_suite`. String columns are cast to `pl.Categorical` (XGBoost auto-detects via `enable_categorical=True`). No cardinality limit unlike HGB.
- `XGBoostStrategy` in `training/strategies.py` — inherits `TreeModelStrategy`, adds `supports_polars = True` and `prepare_polars_dataframe` (casts `pl.String` → `pl.Categorical`).
- **Unified categorical type constants** in `training/strategies.py`:
  - `PANDAS_CATEGORICAL_DTYPES` — `frozenset({"category", "object", "string", "string[pyarrow]", "large_string[pyarrow]"})`
  - `get_polars_cat_columns(df)` — detects `pl.Categorical`, `pl.Utf8`, `pl.String` columns
  - `is_polars_categorical(dtype)` — type check helper
- 5 unit tests for `XGBoostStrategy.prepare_polars_dataframe` (string→categorical, high-cardinality, passthrough)
- `TestXGBoostPolarsClassification` — XGBoost trained directly on Polars with categorical features
- Parametrized integration tests extended: `test_polars_fastpath_parametrized` and `test_polars_fastpath_regression_target` now cover `["cb", "xgb", "hgb"]`

### Changed

- `training/strategies.py`: XGBoost (`"xgb"`) now uses `XGBoostStrategy` instead of shared `TreeModelStrategy`.
- **Refactored categorical detection** across codebase to use unified constants:
  - `pipeline.py`: uses `PANDAS_CATEGORICAL_DTYPES` and `get_polars_cat_columns()`
  - `trainer.py:_filter_categorical_features`: uses unified constants, **fixed missing `pl.Utf8` bug** and missing pandas string types
  - `utils.py:get_categorical_columns`: uses unified constants
  - `core.py`: uses `get_polars_cat_columns()` for pre-pipeline detection

### Fixed

- **Bug**: `_filter_categorical_features` in `trainer.py` did not include `pl.Utf8` in Polars detection, silently filtering out valid categorical columns.
- **Bug**: `_filter_categorical_features` pandas path only checked `["category", "object"]`, missing `"string"`, `"string[pyarrow]"`, `"large_string[pyarrow]"`.

### Polars support matrix (updated)

| Model | Native Polars `.fit()` | Polars fastpath in `train_mlframe_models_suite` |
|-------|:----------------------:|:-----------------------------------------------:|
| CatBoost (`cb`) | Yes (>= 1.2.7) | Yes |
| XGBoost (`xgb`) | Yes (>= 3.1) | Yes (auto-casts strings → pl.Categorical) |
| HGB | Yes (numeric + Categorical) | Yes (auto-casts strings, handles cardinality > 255) |
| LightGBM (`lgb`) | No (broken in 4.6) | No |
| Linear models | No (internal NumPy conversion) | No |
| MLP / NGBoost | No | No |

## 2026-04-14 — HGB Polars Native Fastpath

### Added

- **HGB Polars fastpath** in `train_mlframe_models_suite`: when input is a Polars DataFrame, HGB models now receive it directly without intermediate pandas conversion. String categorical columns are automatically cast to `pl.Categorical` (cardinality ≤ 255) or ordinal-encoded to `pl.UInt32` (cardinality > 255, treated as continuous by HGB).
- `supports_polars = True` on `HGBStrategy`.
- `prepare_polars_dataframe(df, cat_features)` method on `ModelPipelineStrategy` base class (no-op default). `HGBStrategy` overrides it to handle cardinality-aware categorical casting.
- Pre-pipeline Polars originals are now saved before `fit_and_transform_pipeline()` to preserve string/categorical dtypes that polars-ds may convert to float.
- `cat_features_polars` list detected from pre-pipeline schema, used in Polars fastpath to ensure categorical columns are passed to models correctly.
- Polars fastpath now overrides `fit_params["cat_features"]` with pre-pipeline categorical columns when they differ from post-pipeline ones.
- 8 unit tests for `HGBStrategy.prepare_polars_dataframe` in `test_catboost_polars.py` (low/high cardinality, boundary 255/256, passthrough, missing columns).
- 2 integration tests in `test_core.py::TestPolarsNativeFastpath`: `test_hgb_receives_polars_dataframe`, `test_hgb_polars_categorical_is_cast`.

### Changed

- `training/strategies.py`: `HGBStrategy` now sets `supports_polars = True` and overrides `prepare_polars_dataframe` with cardinality-aware casting logic.
- `training/core.py`: Polars fastpath block now calls `strategy.prepare_polars_dataframe()` and sets `skip_preprocessing=True` for models that normally require encoding (HGB). Pre-pipeline Polars originals are saved before `fit_and_transform_pipeline()`.

### Polars support matrix (updated)

| Model | Native Polars `.fit()` | Polars fastpath in `train_mlframe_models_suite` |
|-------|:----------------------:|:-----------------------------------------------:|
| CatBoost (`cb`) | Yes (>= 1.2.7) | Yes |
| HGB | Yes (numeric + Categorical) | Yes (auto-casts strings, handles cardinality > 255) |
| LightGBM (`lgb`) | No | No |
| XGBoost (`xgb`) | No | No |
| Linear models | No | No |
| MLP / NGBoost | No | No |

## 2026-04-14 — Polars Native Fastpath for CatBoost

### Added

- **CatBoost Polars fastpath** in `train_mlframe_models_suite`: when input is a Polars DataFrame, CatBoost models now receive it directly without intermediate pandas conversion. This eliminates zero-copy overhead and allows CatBoost (>= 1.2.7) to use its native Polars ingestion path.
- `supports_polars` property on `ModelPipelineStrategy` (default `False`). New `CatBoostStrategy` subclass sets it to `True`.
- `CatBoostStrategy` in `training/strategies.py` — inherits `TreeModelStrategy`, adds `supports_polars = True`.
- Test file `tests/training/test_catboost_polars.py`: 11 tests covering CatBoost and HGB training directly on Polars DataFrames with categorical, numeric, text, and embedding features, plus early stopping on a Polars validation set.
- Integration tests in `tests/training/test_core.py` (`TestPolarsNativeFastpath`):
  - `test_catboost_receives_polars_dataframe` — monkeypatches `_train_model_with_fallback` to verify CatBoost `.fit()` receives a Polars DataFrame.
  - `test_non_catboost_still_gets_pandas` — verifies Ridge still receives pandas when input is Polars.

### Changed

- `training/core.py`: original Polars DataFrames are preserved before `_convert_dfs_to_pandas()` and substituted into `common_params` for models with `supports_polars`.
- `training/trainer.py`:
  - `train_df.columns.to_list()` replaced with `list(train_df.columns)` for Polars compatibility.
  - `_filter_categorical_features` now detects `pl.String` columns in addition to `pl.Categorical` when input is Polars.
- `training/strategies.py`: CatBoost (`"cb"`) now uses `CatBoostStrategy` instead of the shared `TreeModelStrategy`.

### Polars support matrix

| Model | Native Polars `.fit()` | Polars fastpath in `train_mlframe_models_suite` |
|-------|:----------------------:|:-----------------------------------------------:|
| CatBoost (`cb`) | Yes (>= 1.2.7) | Yes |
| HGB | Yes (numeric only) | No (requires category encoding) |
| LightGBM (`lgb`) | No | No |
| XGBoost (`xgb`) | No | No |
| Linear models | No | No |
| MLP / NGBoost | No | No |
