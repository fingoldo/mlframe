# training misc small subsystems (ranking, strategies, callbacks, cb, extractors, slicing, diagnostics) -- mlframe audit

## Scope

Every file under the assigned cluster was opened and read in full (no partial reads, nothing skipped):

- `src/mlframe/training/callbacks/__init__.py`
- `src/mlframe/training/callbacks/_benchmarks/bench_worsening_vs_monotonic_stop.py`
- `src/mlframe/training/callbacks/_callbacks.py`
- `src/mlframe/training/callbacks/iteration_metrics.py`
- `src/mlframe/training/callbacks/monotonic_decline.py`
- `src/mlframe/training/callbacks/stop_file.py`
- `src/mlframe/training/cb/__init__.py`
- `src/mlframe/training/cb/_cb_pool.py`
- `src/mlframe/training/cb/_cb_pool_build.py`
- `src/mlframe/training/diagnostics/__init__.py`
- `src/mlframe/training/diagnostics/learning_curve.py`
- `src/mlframe/training/extractors/__init__.py`
- `src/mlframe/training/extractors/_benchmarks/__init__.py` (0 LOC, empty file, confirmed via `wc -l`)
- `src/mlframe/training/extractors/_benchmarks/bench_recency_weights_fused.py`
- `src/mlframe/training/extractors/_extractors_dtype_helpers.py`
- `src/mlframe/training/extractors/_extractors_showcase.py`
- `src/mlframe/training/extractors/_extractors_simple.py`
- `src/mlframe/training/ranking/__init__.py`
- `src/mlframe/training/ranking/_benchmarks/__init__.py` (0 LOC, empty file, confirmed via `wc -l`)
- `src/mlframe/training/ranking/_benchmarks/bench_binned_mi_group_relevance.py`
- `src/mlframe/training/ranking/_ranker_fs.py`
- `src/mlframe/training/ranking/_ranker_suite_train.py`
- `src/mlframe/training/ranking/ranker_suite.py`
- `src/mlframe/training/ranking/ranking.py`
- `src/mlframe/training/slicing/__init__.py`
- `src/mlframe/training/slicing/_slice_helpers.py`
- `src/mlframe/training/slicing/_slice_pareto_plot.py`
- `src/mlframe/training/strategies/__init__.py`
- `src/mlframe/training/strategies/base.py`
- `src/mlframe/training/strategies/hgb.py`
- `src/mlframe/training/strategies/neural.py`
- `src/mlframe/training/strategies/pipeline_cache.py`
- `src/mlframe/training/strategies/tree_cb.py`
- `src/mlframe/training/strategies/xgboost.py`

Total files reviewed: 34. Total LOC reviewed: 9434 (sum of `wc -l` across every file above, including the two empty `__init__.py` stubs).

`src/mlframe/training/ranking/_ranker_fs.py` imports `_su_redundancy_matrix` from `mlframe.feature_selection.filters.group_aware` and `select_ltr_features` imports `mlframe.feature_selection.registry.get("MRMR")` -- per the excluded-scope instructions I only read the call signature/docstring of these excluded-package symbols for context, and did not analyze or report on their internals.

Overall impression: this cluster is unusually well-hardened already -- most files carry visible scars from prior audit "waves" (explicit fix-history comments, defensive guards, regression tests referencing specific fuzz-combo IDs). The bugs found below are the residue that survived that hardening, mostly in less-traveled combinations of independently-added knobs.

## Findings

| ID | Severity | Category | File:Line | Summary |
|----|----------|----------|-----------|---------|
| F1 | P0 | correctness | `slicing/_slice_helpers.py:215,228-230` | `build_slice_eval_sets(source="both", group_ids=...)` silently uses plain/Stratified KFold instead of GroupKFold for the random-shard half, breaking ranking query-group boundaries with no warning |
| F2 | P1 | reproducibility/silent-failure | `callbacks/_callbacks.py:274,357-364` | `slice_min_delta_in_se` silently no-ops (falls back to the plain absolute `min_delta`) unless the unrelated `slice_persist_history` flag is also set to `True` |
| F3 | P1 | silent-failure (id-reuse cache bug class) | `cb/_cb_pool_build.py:210-225,305` | Cached CatBoost train-Pool label-swap decision keys off `id(train_target)`; CPython id-reuse after GC can make a fresh, different target array appear "unchanged", silently leaving a stale label on the reused Pool |
| F4 | P1 | silent-failure (id-reuse cache bug class) | `cb/_cb_pool.py:694-702,745` | Same id()-identity bug as F3, in the val-Pool cache (`_maybe_rewrite_eval_set_as_cb_pool`) |
| F5 | P1 | memory/perf discipline | `diagnostics/learning_curve.py:421-424` | `compute_learning_curve` defaults to `joblib.Parallel(prefer="processes")`, which pickles/copies the full `X_pool` to every worker process -- contradicts the file's own "RAM-safe on 100GB-class frames ... never a frame copy" design claim |
| F6 | P2 | edge-case gap | `ranking/ranking.py:266-280` | `_fit_cb_ranker`'s null-fill guard only handles object-dtype cat columns; pandas `CategoricalDtype` columns with nulls reach `CatBoostRanker.fit`/`Pool` unfilled, unlike the parity guard in the main CB training path |
| F7 | P2 | docs/comment accuracy | `extractors/_extractors_dtype_helpers.py:211-214` | Comment claims the code switched the log baseline to "seconds" for sub-day spans; the code still computes `np.log(span_days)` -- comment describes a fix that was never made (math happens to still be correct by cancellation, but the comment is actively misleading) |
| F8 | P2 | architecture / duplication | `strategies/hgb.py:12-30` vs `strategies/__init__.py:37-66` | `HGBStrategy`'s module-level `_polars_categorical_dtypes`/`_is_polars_categorical`/`_get_polars_cat_columns` duplicate `strategies/__init__.py`'s `is_polars_categorical`/`get_polars_cat_columns` verbatim instead of importing them -- a future dtype-detection fix (e.g. the "str"/StringDtype fix already applied to the pandas-side constant) risks landing in only one copy |

### F1 -- `build_slice_eval_sets(source="both")` bypasses GroupKFold for ranking data (P0)

`build_slice_eval_sets` explicitly guards against unsafe shard partitioning for ranking (LTR) data: when `group_ids` is supplied and `source == "random"` it switches to `GroupKFold` and logs a WARNING explaining why (`"NDCG on partial queries is meaningless"`, line ~217-221). But the gate is a strict `source == "random"` equality check. When `source == "both"` (random shards unioned with fairness shards -- a documented, valid enum value of `_SliceSource`) and `group_ids` is supplied, the `if source in ("random", "both")` branch at line 228 calls `_random_shards` (plain/Stratified `KFold`) directly, completely skipping the group-aware branch and its warning. The result: for a ranker's slice-stable early stopping configured with `source="both"`, the random half of the shards silently splits query groups across shards, so each shard's NDCG/MAP metric is computed on broken/partial queries -- a genuinely meaningless number that then feeds the ES aggregate and the Pareto-plot diagnostic, with zero log signal that anything went wrong (contrast with the `source="random"` path, which warns loudly for the identical underlying risk). Confirmed no existing test exercises `source="both"` together with `group_ids` (`tests/training/slicing/test_slice_helpers.py` only covers `source="random"` + `group_ids`, per its own docstring "GroupKFold auto-switch when group_ids supplied with source='random'"). Suggested fix: widen the group-aware gate to `source in ("random", "both")` (mirroring the block above it), or explicitly document/reject `source="both"` + `group_ids` if group-fairness combos were never intended to be supported.

### F2 -- `slice_min_delta_in_se` is a silent no-op unless `slice_persist_history=True` (P1)

`_effective_min_delta` (the SE-scaled min_delta feature, gated by the `slice_min_delta_in_se` constructor kwarg) is only ever invoked with a non-empty `shard_values` when `slice_shards` is non-`None` in `should_stop()`. `slice_shards` is populated exclusively from `self.slice_shard_score_history[-1]`, which is itself only appended to inside `_compute_slice_aggregate` when `self.slice_persist_history` is `True` (see `_callbacks.py:274`). `slice_persist_history` defaults to `False` and its own docstring frames it purely as a Pareto-plot memory optimization ("only kept when `slice_persist_history=True` to keep the no-slice path's memory profile unchanged"), with no mention that it also gates `slice_min_delta_in_se`. So a caller who sets `slice_min_delta_in_se=1.0` (wanting the ES threshold to scale with the K-shard standard error) but leaves `slice_persist_history` at its default `False` silently gets the *plain* absolute `min_delta` every iteration -- the feature they configured never activates, with no warning. This is not a hypothetical: production wiring in `_trainer_train_and_evaluate.py` (outside this cluster's scope, but visible via grep) happens to tie `slice_persist_history` to `pareto_plot_enabled` (defaulting `True`), which is why the coupling doesn't bite by default in the trainer -- but any direct `UniversalCallback`/`LGBMCallback` caller (as the callback classes are designed to be used) who sets `slice_persist_history=False` explicitly (as `tests/training/slicing/test_slice_es_biz_value.py` does) while wanting SE-scaled min_delta would hit this silently. Suggested fix: decouple the shard-history-for-min_delta-scaling need from `slice_persist_history` (e.g. always keep the *last* shard's values around cheaply for the min_delta computation, independent of whether the full Pareto-plot history is retained), or raise/log a warning when `slice_min_delta_in_se` is set with `slice_persist_history=False`.

### F3 / F4 -- `id(target)` reused as a "did the label change" check in the CB Pool caches (P1 x2)

Both `_maybe_get_or_build_cb_pool` (train-Pool cache, `cb/_cb_pool_build.py:210-225`) and `_maybe_rewrite_eval_set_as_cb_pool` (val-Pool cache, `cb/_cb_pool.py:694-702`) decide whether to re-call `Pool.set_label` by comparing `id(train_target)` / `id(val_target)` against a stashed `_mlframe_last_target_id`. This is the *exact* bug class the surrounding code's own comments say was already found and fixed for the cache **key** (`id(train_df)` -> content-fingerprint via `compute_signature`, explicitly called out at `_cb_pool_build.py:161-166` and `_cb_pool.py:682-686`: *"pre-2026-05-23 used id(train_df) which broke across sklearn.clone() and train_df.iloc[...] slicing"*) -- but the fix was never extended to this sibling identity check on the **label array**. CPython can and does reuse a just-freed object's memory address for a new allocation of matching size, which is routine in exactly the loop pattern this file's own comments describe as the motivating scenario (RFECV inner CV folds slicing a fresh `train_target = y[fold_mask]` each iteration, the prior array's refcount dropping to 0 as the next slice replaces it). If a fold-N target array's id happens to be reused by fold-N+1's target array, `id(train_target) == last_target_id` is (accidentally) `True`, `set_label` is skipped, and the cached Pool keeps fold-N's stale label while training proceeds with fold-N+1's data -- a silently-wrong training signal that no exception surfaces. No test in the repo references `_mlframe_last_target_id` or exercises id-reuse directly. Suggested fix: use the same content-fingerprint approach already adopted for the cache key (or at minimum a `weakref` + generation counter, or a cheap content hash of the label array) instead of `id()` for this check too.

### F5 -- `compute_learning_curve` defaults to process-based joblib parallelism over full-frame closures (P1)

`diagnostics/learning_curve.py`'s module docstring and inline comments explicitly claim RAM-safety on 100GB-class frames via "column views ... never a frame copy" (lines 27, 34-36). But `compute_learning_curve`'s default parallel path (`n_jobs=-1` is the default parameter value) calls `joblib.Parallel(n_jobs=n_jobs, prefer="processes")(delayed(_job)(c) for c in run_counts)` at line 424, where `_job` closes over `X_pool` (and `X_hold`, `y_pool`, `estimator_factory`) from the enclosing scope. `prefer="processes"` means joblib spawns separate OS processes with no shared memory; every process receives its own pickled copy of the closure, including the full `X_pool`/`X_hold` views. On the large frames this project explicitly designs around (100+ GB), the "cheap column view" claim is defeated the moment this parallel path is taken -- each of `n_jobs` worker processes gets its own full serialized copy, multiplying peak memory by worker count (and paying large IPC serialization cost) instead of the claimed zero-copy behavior. The existing test suite only exercises this at small-data scale (`tests/training/test_learning_curve.py` uses small synthetic X/y with `n_jobs=-1` purely for a correctness A/B against `n_jobs=1`, never a memory check). Suggested fix: default to `prefer="threads"` (numpy/pandas/sklearn `.fit` calls release the GIL for their heavy lifting) or gate the processes-backend on a byte-size threshold consistent with this project's other eager/lazy conversion gates (documented ~2GB rule elsewhere in the codebase).

### F6 -- CatBoostRanker null-fill guard misses category-dtype cat columns (P2)

`_fit_cb_ranker._fill_obj_cat_nones` (`ranking/ranking.py:268-278`) only fills `None` for cat_feature columns whose pandas dtype is `object`. The main CB training path (`cb/_cb_pool.py`'s `_polars_nullable_categorical_cols` / `_polars_fill_null_in_categorical`) treats nullable Categorical dtype as an equally-crashing case and fills it too (documented root cause: CatBoost's Polars/pandas fastpath dispatch has no signature for nullable-categorical). The ranker path has no equivalent guard for a pandas `CategoricalDtype` column with nulls, so a caller whose `cat_features` includes a `pd.Categorical` column with `NaN` cells can still hit the same class of CatBoost crash the main path was hardened against. Suggested fix: extend `_fill_obj_cat_nones` to also handle `pd.CategoricalDtype` columns (add the sentinel as a category, then fillna), mirroring the main path's `_polars_fill_null_in_categorical` treatment.

### F7 -- Stale/misleading comment in `get_sample_weights_by_recency` (P2)

`extractors/_extractors_dtype_helpers.py:211-214` comments: *"log(span_days) for span<1 day is negative -> max_drop negative. Use log(span_in_seconds) baseline so the gradient stays positive for sub-day spans too."* The code on the very next line still computes `max_drop = (np.log(span_days) - _log_min_age) * weight_drop_per_year` -- it was never changed to use seconds. The formula happens to still be numerically correct (the `_log_min_age` floor subtraction makes the expression scale-invariant, i.e. `log(span_days) - log(min_age_days) == log(span_days/min_age_days)`, so the "seconds vs days" unit choice is actually irrelevant to correctness), but the comment describes a fix that doesn't exist in the code, which will mislead the next person auditing this arithmetic into thinking a units bug was already addressed when it wasn't (or into "fixing" something that isn't broken). Suggested fix: correct the comment to explain the actual reason sub-day spans stay positive (the ratio-cancellation via `_log_min_age`), not a units switch that was never implemented.

### F8 -- Duplicated polars-categorical-detection helpers between `strategies/__init__.py` and `strategies/hgb.py` (P2)

`strategies/__init__.py` defines `_polars_categorical_dtypes()` / `is_polars_categorical()` / `get_polars_cat_columns()` as "the single source of truth" per its own comment (line 29), specifically because a prior bug (Round-9, referenced inline) came from `pl.Enum` not comparing equal to `pl.Categorical` and a per-strategy detector missing it. `strategies/hgb.py` (lines 12-30) re-implements an identical `_polars_categorical_dtypes()` / `_is_polars_categorical()` / `_get_polars_cat_columns()` triplet instead of importing the package-level versions -- currently in sync, but the exact kind of drift risk the "single source of truth" comment was written to prevent (e.g. a future new polars dtype variant fixed in one copy and not the other). Suggested fix: have `hgb.py` import `is_polars_categorical` / `get_polars_cat_columns` from `. import` (the package `__init__`) instead of re-defining local copies.

## Proposals

| ID | Category | File | Summary |
|----|----------|------|---------|
| PR1 | test-coverage | `tests/training/slicing/test_slice_helpers.py` | Add a case combining `source="both"` with `group_ids` (would have caught F1) |
| PR2 | test-coverage | `tests/training/slicing/test_slice_callback.py` or similar | Add a case combining `slice_min_delta_in_se` with `slice_persist_history=False` and assert the SE-scaling actually takes effect (would have caught F2) |
| PR3 | test-coverage | `tests/training/cb/` | Add a regression test that forces `id()` collision between two distinct target arrays across two `_maybe_get_or_build_cb_pool` calls (e.g. via `del` + immediate re-allocation of a same-size array) and asserts the cached Pool's label reflects the SECOND target, not the first (would have caught F3/F4) |
| PR4 | test-coverage | `tests/training/ranking/` | Add a case for `_fit_cb_ranker` with a pandas `CategoricalDtype` cat_feature column containing nulls (would have caught F6) |
| PR5 | perf/memory | `diagnostics/learning_curve.py` | Consider `prefer="threads"` as the joblib default, with a `parallel_backend=` escape hatch for callers whose `estimator_factory` genuinely needs process isolation (non-GIL-releasing pure-Python estimators) |
| PR6 | ML-best-practice | `diagnostics/learning_curve.py` | `_fit_warm_curve`'s `per_step = max(1, base_n // len(counts)) if base_n else 50` heuristic for incremental booster growth is a fixed guess; consider deriving it from `time_budget_s` when set, so warm-start sizing adapts to the caller's actual budget rather than a hardcoded `50` |
| PR7 | refactor | `strategies/hgb.py` | Import the shared `is_polars_categorical` / `get_polars_cat_columns` from the package `__init__` instead of the local duplicate (ties to F8) |
| PR8 | ML-best-practice | `ranking/ranker_suite.py` (ensemble gates, `_ranker_suite_train.py:674-763`) | The quality/diversity ensemble gates are good practice already present; consider surfacing the gate decisions (which members were dropped and why) in the returned `metadata` dict as well as the log line, so downstream automated reporting doesn't need to scrape logs |

## Coverage notes

- No files were skipped or partially read; every file in the assigned scope was read to completion (verified against the `wc -l` file list before starting).
- `_ranker_fs.py`'s two call-outs into the excluded MRMR engine (`mlframe.feature_selection.filters.group_aware._su_redundancy_matrix`, `mlframe.feature_selection.registry.get("MRMR")`) were read only at the call-site signature level, per the excluded-scope instructions; their internals were not analyzed and no findings were sought there.
- No dynamic/runtime testing was performed (read-only audit per task constraints) -- all findings above are static-analysis-derived and cross-checked against the existing test suite (via `grep`) to confirm the gap is real rather than already covered by a test I simply didn't see.
- The `_benchmarks/` scripts in this cluster (`bench_recency_weights_fused.py`, `bench_binned_mi_group_relevance.py`, `bench_worsening_vs_monotonic_stop.py`) are dev-only harnesses, not shipped/imported production code; they were read for correctness of the underlying claims (e.g. rejected-optimization notes) but are not scored for the same bug classes as production modules.
