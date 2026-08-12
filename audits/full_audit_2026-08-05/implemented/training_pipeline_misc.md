# Audit report: training_pipeline_misc

**Scope**: `src/mlframe/training/baselines/`, `src/mlframe/training/pipeline/`, `src/mlframe/training/reporting/`,
`src/mlframe/training/callbacks/`, `src/mlframe/training/cb/`, `src/mlframe/training/extractors/`,
`src/mlframe/training/diagnostics/`, `src/mlframe/training/slicing/`, `src/mlframe/training/_benchmarks/`
(`feature_selection/filters/**` and `feature_selection/shap_proxied_fs/**` explicitly excluded — separately audited
2026-07-25).

**Files reviewed in full**: 50 production files (~16,090 LOC read line-by-line). This covers every non-benchmark
`.py` file in `baselines/`, `pipeline/`, `callbacks/`, `cb/`, `extractors/`, `diagnostics/`, `slicing/`, and the
`reporting/` facade + its regression/probabilistic/diagnostics siblings. The `_benchmarks/` subdirectories under each
package (~90 files, mostly `bench_*.py` perf-microbenchmark harnesses and a handful of `profile_*.py` cProfile
scripts) received a lighter sampling pass (1 file read in full, several others grepped for bug-class patterns:
mutable default args, bare `except:`, non-ASCII text) rather than a full line-by-line read — these are developer
benchmarking scripts, not code on the production training path, and no correctness issues were found in the sample.
Package `__init__.py` re-export shims (`pipeline/__init__.py`, `reporting/__init__.py`, `cb/__init__.py`,
`callbacks/__init__.py`) were not read in full; a `Grep` sweep across the whole cluster for bare-except and
mutable-default-argument patterns found none outside what's already discussed below. `baselines/_dummy_report_type.py`
and the `_profile_*.py` / `_smoke_*.py` scripts in `baselines/` were not read (dataclass/schema-version stub and
manual profiling/smoke-test scripts respectively, not part of the reviewed correctness surface).

**LOC reviewed**: ~16,090 (line-by-line); cluster nominal size is ~20k LOC per the assignment brief (the larger
~29k figure from a raw `wc -l` over every file includes benchmarks/pycache).

## Findings table

| ID | Severity | File:Line | Summary | Suggested fix | Meta-test idea |
|----|----------|-----------|---------|----------------|-----------------|
| TRAINING_PIPELINE_MISC-1 | P0 | `pipeline/_composite_fe_shared.py:26-32` (root cause); reached from `pipeline/_entity_time_composite_fe.py:103,133`, `pipeline/_ma_crossover_composite_fe.py:106,118`, `pipeline/_cross_sectional_composite_fe.py:98-100`, `pipeline/_event_proximity_decay_composite_fe.py:91`, `pipeline/_latent_interaction_svd_composite_fe.py:48-49`, `pipeline/_target_encoding_composite_fe.py:106-107,117,157` | `attach_new_columns()` joins new feature columns onto `train_df`/`val_df`/`test_df` by pandas **index label** (`df.join(new_cols)`), but every one of these 6 sibling composite-FE modules builds `new_cols` with a **fresh `RangeIndex(0..n-1)`** while the caller's `df` carries the **original, non-contiguous row-index labels** left over from the upstream `df.iloc[train_idx]` split (train/val/test indices are a shuffled/stratified subset of the source frame's index, not `0..n-1`). The join silently produces NaN or, worse, cross-row-misattributed feature values for a large fraction of rows on any pandas-backed run. | Fix the root cause once in `attach_new_columns()`: re-index `new_cols` onto `df.index` positionally before the join/concat (e.g. `new_cols = new_cols.set_axis(df.index, axis=0)` when `len(new_cols) == len(df)`), so every call site is fixed without touching the 6 downstream modules. Alternatively require every caller to build `new_cols` with `index=df.index` (the pattern `_categorical_composite_fe.py` already uses correctly) and add an assertion inside `attach_new_columns` that raises if `new_cols.index` isn't index-comparable to `df.index`. | A property test: for a pandas `df` with a non-default (shuffled) index, call `attach_new_columns(df, new_cols_built_with_range_index)` and assert the returned frame's new column values equal the row-order-based expectation (not index-label-based). Generalize as an AST/grep scanner flagging any `pd.DataFrame(..., index=range(...))` or `.reset_index(drop=True)` result that is later passed to a helper doing `.join()`/`pd.concat(axis=1)` against a differently-indexed frame. |
| TRAINING_PIPELINE_MISC-2 | P3 | `callbacks/iteration_metrics.py:162-177` (`CBIterationMetricsCallback._predict`) | `is_reg = "regression" in self.target_type` is a substring check used to decide whether to call `model.predict()` (regression path) or `model.predict_proba()` (classification path). It only produces correct behaviour for `learning_to_rank` today because the sole production wiring site (`_data_helpers.py::_build_cb_iteration_metrics_callback`, out of this cluster's scope) derives `target_type` from `sklearn.base.is_classifier(model_obj)` rather than the real target type, so a CatBoost ranker (non-classifier) happens to get `target_type="regression"` and thus the correct `.predict()` path — by coincidence, not by contract. If this public class is ever wired directly with the genuine `target_type="learning_to_rank"` string (a plausible refactor, since the class's own docstring says it dispatches on "the model's `target_type`"), `"regression" in "learning_to_rank"` is `False`, so it calls `model.predict_proba()` on a ranker, which raises `AttributeError`, is swallowed by the broad `except Exception` in `_predict`, logged only at `debug`, and `iteration_metrics_` silently stays empty for every round. | Replace the substring check with an explicit allow-list (`target_type in ("regression", "quantile_regression", "multi_target_regression", "learning_to_rank")`) or, better, dispatch on `hasattr(model, "predict_proba")` instead of a string. | A small unit test constructing `CBIterationMetricsCallback` directly with `target_type="learning_to_rank"` against a mock CatBoost ranker (`predict` present, `predict_proba` absent) and asserting `iteration_metrics_` gets populated (not silently empty) after a few `after_iteration` calls. Generalizable as a grep-based scanner for `"<word>" in some_string_variable` used as a type/kind dispatch key, cross-checked against the variable's declared `Literal[...]`/enum domain to catch any value the substring check doesn't actually cover. |
| TRAINING_PIPELINE_MISC-3 | P3 | `baselines/_dummy_baseline_regression.py:201-202` | The `seasonal_naive_pP (ts)` dummy baseline builds its val/test predictions via a pure-Python list comprehension (`np.array([train_y[-P + (k % P)] for k in range(n_val)])`) instead of vectorized modulo indexing. Negligible at typical val/test sizes but scales linearly with a Python-loop constant on multi-million-row val/test splits (this exact code path is documented elsewhere in this codebase's own perf-fix history as the pattern to avoid: "fuse into one vectorized/njit call"). | Replace with `train_y[-P:][np.arange(n_val) % P]` (equivalent result, one array-index call, no per-row Python overhead). | A microbenchmark (`_benchmarks/bench_seasonal_naive_vectorized.py`) comparing the list-comprehension form against the vectorized form at n=10k/1M and asserting bit-identical output plus a documented speedup; generalizable as a grep scanner for `np.array([... for ... in range(...)])` patterns feeding a downstream numeric array (a strong signal of an un-vectorized hot loop). |
| TRAINING_PIPELINE_MISC-4 | P3 | `baselines/_dummy_metrics_pick_plot.py:517-518` | A source comment contains an unintentional Cyrillic phrase embedded in otherwise-English prose: `# cell auto-flush to re-render the figure (the "толпа графиков" double-render seen 2026-05-26).` This is mojibake/foreign-language leakage into a comment, plus a date-stamp reference the project's own comment-style convention (CLAUDE.md "Comment style") disallows for new code ("no ... date stamps"). | Replace with the equivalent English phrase, e.g. `# ... (the "figure pile-up" double-render bug)`, and drop the date stamp. | A CI lint step (regex `[^\x00-\x7F]` scoped to `#`-comment text, excluding docstring examples/data literals) flags any non-ASCII character in a Python comment outside string literals — catches accidental foreign-language/encoding leaks before they ship. |

**Counts**: P0=1, P1=0, P2=0, P3=3.

## Narrative

### TRAINING_PIPELINE_MISC-1 (P0) — composite-FE pandas index-misalignment silently corrupts feature values

**How found**: While reading `pipeline/_categorical_composite_fe.py` I noticed it builds its new-column frame with
`pd.DataFrame(index=_pd_view.index)` — i.e. it reuses the caller's own row index — and only *then* calls the shared
`attach_new_columns()` helper. Reading the shared helper (`pipeline/_composite_fe_shared.py`) showed its pandas
branch is `df.join(new_cols)` (falling back to `pd.concat([df, new_cols], axis=1)` for non-pandas/non-join types),
both of which are **index-label** joins in pandas, not positional concatenation. I then read the other seven
composite-FE sibling modules and found six of them build `new_cols` with a **fresh `RangeIndex`** instead of reusing
`df.index`:
- `_entity_time_composite_fe.py:103`: `new_cols = pd.DataFrame(index=range(row_count(df)))`
- `_ma_crossover_composite_fe.py:106`: `new_cols = pd.DataFrame(index=range(n_rows))`
- `_cross_sectional_composite_fe.py:99`: `new_cols = new_cols.reset_index(drop=True)`
- `_event_proximity_decay_composite_fe.py:91`: `attach_new_columns(df, result.reset_index(drop=True))`
- `_latent_interaction_svd_composite_fe.py:48`: `new_cols = pd.DataFrame(vecs, columns=cols, index=range(n))`
- `_target_encoding_composite_fe.py:106,117,157`: `pd.DataFrame({out_col: ...}, index=range(len(...)))`

**Why this is reachable in production, not just theoretical**: I traced where `train_df`/`val_df`/`test_df` — the
`df` argument these composite-FE functions receive — come from. `preprocessing.py::create_split_dataframes` builds
them via plain `df.iloc[train_idx]` / `.iloc[val_idx]` / `.iloc[test_idx]` with **no `reset_index()`** afterward.
`train_idx`/`val_idx`/`test_idx` themselves come from `_split_helpers.py`'s stratified/shuffled splitters
(`sklearn.model_selection.StratifiedShuffleSplit` / `ShuffleSplit`/`GroupKFold`-style splitters) — i.e. random,
non-contiguous subsets of the original row positions for any non-time-ordered split (the common case). So by the
time any of these six composite-FE steps runs (they're invoked from
`core/_phase_helpers_fit_pipeline.py:414-493`, immediately after the split and before categorical encoding),
`train_df.index` is a shuffled subset like `[47, 3, 1912, ...]`, not `0..n_train-1`.

**Concrete failure mode**: `df.join(new_cols)` (or `pd.concat([df, new_cols], axis=1)`, which has the same
index-alignment semantics) aligns row-by-row on index *label*. Since `new_cols` was built with labels `0..n-1`:
- Any row of `df` whose original label is `>= n` gets **NaN** for every new composite-FE column (a large majority
  of rows for any reasonably-sized source frame).
- Any row of `df` whose original label happens to fall inside `[0, n)` gets the **wrong** `new_cols` row — silently
  attaching another row's engineered feature values to it. This is not merely a NaN/robustness gap; it's silent
  feature-value corruption that trains the downstream model on wrong data, matching the rubric's P0 bar ("would
  corrupt a trained model or its predictions").

**Contrast with the safe pattern already in this codebase**: `_categorical_composite_fe.py` gets this right —
`_new_cols = pd.DataFrame(index=_pd_view.index)` reuses the caller's actual row-index object, so the later
`attach_new_columns(train_df, _pending_new_cols["train"])` join is a correct identity-index alignment. This proves
the fix is a one-line, already-precedented pattern; the other six modules simply didn't follow it. The polars branch
of `attach_new_columns` (`df.with_columns([pl.Series(c, new_cols[c].to_numpy()) for c in new_cols.columns])`) is
**not** affected — polars has no index concept, so that branch is purely positional and correct; the bug is pandas-
only, but pandas is a fully supported first-class input format throughout this codebase (extensive dedicated pandas
handling exists in nearly every file reviewed in this cluster).

**Verification performed**: this was traced statically end-to-end (call graph + index-construction sites +
split-index generation) rather than reproduced by running the suite (I am read-only for this audit), but every link
in the chain is directly quoted above with file:line evidence; I did not find any `reset_index()` call between the
split (`preprocessing.py::create_split_dataframes`) and the six composite-FE call sites that would neutralize the
bug.

### TRAINING_PIPELINE_MISC-2 (P3) — CB iteration-metrics callback's target_type dispatch is fragile-by-coincidence

Found while reading `callbacks/iteration_metrics.py`'s three per-booster callback classes for the "capture full
metric suite every N rounds" opt-in diagnostic. `CBIterationMetricsCallback._predict` decides `.predict()` vs
`.predict_proba()` via `"regression" in self.target_type`. Grepping the sole production constructor
(`_data_helpers.py::_build_cb_iteration_metrics_callback`, outside this cluster) showed it never actually passes the
literal string `"learning_to_rank"` — it infers `target_type` from `sklearn.base.is_classifier(model_obj)`, so a
ranker (not a classifier) coincidentally lands on the correct `"regression"` branch today. This means there is no
live production bug right now, but the class's own contract (a `target_type` string parameter, matching every other
target-type-driven mlframe API) invites a future caller or refactor to pass the real value, at which point the
substring check silently breaks (caught by a broad `except Exception`, logged only at `debug`, capture becomes a
silent no-op for the rest of that fit). Downgraded from what I initially suspected was a live P2 bug once I traced
the actual wiring site and confirmed today's only caller side-steps the fragile branch by accident.

### TRAINING_PIPELINE_MISC-3 (P3) — un-vectorized seasonal_naive baseline loop

Minor efficiency nit spotted while reading the regression dummy-baseline dispatcher: the two `seasonal_naive_pP (ts)`
prediction arrays are built with a Python-level list comprehension indexing `train_y` once per output row, instead
of a single vectorized `train_y[-P:][idx % P]` gather. This is a well-known un-vectorized-loop pattern this project's
own CLAUDE.md explicitly calls out to fix elsewhere (the many documented `njit`/vectorization perf-win entries), so
flagging it here for consistency even though its absolute cost is small (dummy baselines run once per target, not in
a hot per-iteration loop).

### TRAINING_PIPELINE_MISC-4 (P3) — mojibake in a source comment

A grep for Cyrillic characters across the whole cluster found exactly one hit: a stray Russian phrase embedded
mid-sentence in an English comment in `_dummy_metrics_pick_plot.py`, alongside a date stamp the project's own
"Comment style" convention says new code should not carry. Purely cosmetic but explicitly in-scope per the audit
brief's hygiene dimension ("comment cruft (stale audit-wave markers, mojibake, dashes in prose)").

## Dimension coverage notes

- **Correctness bugs**: 1 finding (P0, above). No mutable-default-argument bugs, no bare `except:` clauses, and no
  other index/off-by-one bugs were found outside the one documented above; the reviewed code is otherwise unusually
  well-defended (extensive narrow `except` clauses with debug/warning logging, explicit contract-violation `raise`s,
  and many in-line regression-test-worthy comments citing prior fuzz-caught incidents).
- **ML correctness (leakage / reproducibility / calibration / sample_weight)**: no leakage issues found. Sample-weight
  threading through the pre-pipeline cache key, the CB Pool cache, and the composite-FE target-encoding path is
  explicitly and correctly guarded (weight-aware markers folded into cache keys; train-only fit for two-step target
  encoding; honest train-only "prior"/"most_frequent" dummy baselines vs. explicitly-labeled `oracle_prior` for the
  eval-peeking reference). Reproducibility: `_per_target_seed` deliberately uses `blake2b` instead of Python's salted
  `hash()`; RNGs throughout use `np.random.default_rng(seed)` with explicit seeds, not global state.
- **Computational efficiency**: 1 finding (P3, above). No unnecessary `.copy()` on large frames was found in the
  reviewed files — the codebase is unusually disciplined about avoiding full-frame copies (extensive comments citing
  the project's own "no `.copy()` on 100GB frames" rule, shallow-copy-only mutation patterns, Arrow split-blocks
  bridges instead of `.to_pandas()`, LRU caches with content-fingerprint keys rather than full-frame pickling).
- **Edge cases and robustness**: no new findings. Empty-input, all-NaN, single-class, and single-row guards are
  pervasive and mostly already regression-tested per in-line comments referencing specific fuzz-caught incidents.
- **Test coverage gaps**: not independently assessed against the actual `tests/` tree in this pass (out of the
  read-only file set reviewed); the in-line comments throughout reference many specific regression tests by name,
  suggesting good coverage discipline, but I did not cross-check test-file existence for every function.
- **Code quality/architecture**: no findings beyond TRAINING_PIPELINE_MISC-2/4 above. Module-split discipline (the
  "carved out of X to stay under 1k LOC" pattern) is followed consistently across every reviewed file.
- **OSS/hygiene**: 1 finding (P3, above — mojibake). The pervasive `Wave NN (YYYY-MM-DD): ...` narrative-comment
  style used throughout this cluster technically conflicts with the project's own "no phase/wave markers, no date
  stamps" CLAUDE.md convention, but it is applied so uniformly and deliberately (hundreds of instances, each citing
  a specific fuzz-caught incident with reproduction detail) that it reads as an established, intentional
  documentation style for this specific codebase rather than a new hygiene defect introduced by any one file — not
  flagged as a standalone finding to avoid manufacturing a bulk low-value P3 list against what is evidently a
  deliberate team convention already in wide, consistent use.
