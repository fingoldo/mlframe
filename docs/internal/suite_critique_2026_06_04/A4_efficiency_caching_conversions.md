# A4 — Efficiency / Caching / Polars→Pandas Conversions critique

Scope: READ-ONLY critique of `train_mlframe_models_suite` and its per-target / per-model /
per-fold hot path. Every claim below was verified by reading the cited source. Findings are
split into **confirmed-by-reading** (reported) vs measurement-gated perf claims (flagged).

Overall verdict: the suite is **already heavily hardened** against the headline failure modes in
the project conventions — there are no hot-path `df.copy()` / `df.clone()` / `pd.DataFrame(df)`
reference-copies, the polars→pandas conversion is genuinely deferred / size-gated / content-cached,
the booster-dataset and pipeline caches are content-keyed (not `id()`-keyed), pickle is protected via
`_PolarsDsPipelineJsonProxy.__reduce__`, and the polars fastpath stores references only. The findings
below are mostly **second-order micro-overheads in the inner loop** and a few correctness-adjacent
caching nits, not gross violations. This matches the user-memory note "mlframe perf is mature
(2026-06)".

---

## A4-01 — Pipeline cache key recomputes suite-invariant feature/dtype digests per (pre_pipeline × model)
- **Severity:** P2
- **File:** `src/mlframe/training/core/_phase_train_one_target_body.py:446-455` →
  `src/mlframe/training/core/_phase_train_one_target.py:393-447` (`_compute_pipeline_cache_key`)
- **What's wrong:** `_compute_pipeline_cache_key` is called once per (pre_pipeline × model)
  iteration of the inner loop. Inside it (lines 421-426) it rebuilds
  `repr((sorted(cat_features), sorted(text_features), sorted(embedding_features)))` and a blake2b
  digest of that repr on EVERY call, even though `cat_features` / `text_features` /
  `embedding_features` are invariant across the entire per-target loop (they're fixed after
  auto-detect, lines 463-464 of `_main_train_suite.py`). When the strategy supports polars it also
  rebuilds `_canonical_dtype_pairs(train_df)` (line 438-442 + 450-494) which walks every column and
  blake2b-hashes the result — recomputed per (model, pre_pipeline) although the polars frame's
  schema is invariant across all models that share `supports_polars=True`.
- **Why it matters:** O(n_cols) string-build + sort + two blake2b hashes per inner iteration. On a
  suite of 5 models × N pre_pipelines × M targets this is paid 5·N·M times for a value that changes
  only when the feature lists / schema change. Cheap per call (~µs–sub-ms) but pure waste — exactly
  the "dispatch / config parsing not hoisted out of hot loop" pattern the audit targets.
- **Recommendation:** Hoist the `_feats_suffix` (feature-list digest) to once per target (or once
  per suite — the lists don't change per target) and cache the `_dtype_suffix` keyed by
  `id(train_df_polars)`/schema once per (pre_pipeline) sweep. Compose the final key from the three
  invariant suffixes + the per-model `_content_key`/`tier`/`kind` parts.
- **Measurement needed first:** YES — measure `_compute_pipeline_cache_key` cumtime via cProfile on
  a production-shape frame (wide cat/text lists, many columns) before acting; per the "measure
  before optimize" rule the absolute saving may be sub-1% on narrow frames.
- **Confidence:** High (waste confirmed by reading; magnitude unmeasured).

## A4-02 — `_pre_pipeline_cache_key` computed twice per fit; relies entirely on a single-slot memo
- **Severity:** P2
- **File:** `src/mlframe/training/_pipeline_helpers.py:590-599` (compute `_cache_key_entry`, then
  immediately `_pre_pipeline_cache_get` recomputes the same key) and
  `src/mlframe/training/_pipeline_cache.py:509,550-551,579-581` (`_LAST_KEY_CACHE` single-entry memo).
- **What's wrong:** `_build_and_transform_pre_pipeline` computes the full content key once into
  `_cache_key_entry` (line 590) for the later populate path (line 865), then calls
  `_pre_pipeline_cache_get` (line 595) which calls `_pre_pipeline_cache_key` a SECOND time with the
  same args. The key build is expensive — it folds `_full_x_content_hash(train_df)` +
  `_full_x_content_hash(val_df)` + `_full_target_content_hash(train_target)`, each an O(rows×cols)
  blake2b (the docstring at `_pipeline_cache.py:494-499` measures ~1.17 s/call at 100k×16). The only
  thing preventing the double-pay is the single-entry `_LAST_KEY_CACHE` memo, which hits on the
  immediate repeat. This is fragile: any interleaving call to `_pre_pipeline_cache_key` with
  different inputs between line 590 and line 595 evicts the single slot and forces a full recompute.
- **Why it matters:** Correct today (the get immediately follows the compute) but a latent ~1 s
  landmine if a future edit inserts any keying call between them. The `_full_x_content_hash` memo
  (`_PIPELINE_X_HASH_CACHE`, 16 slots) backstops the X/val hashes, but `_full_target_content_hash`
  has NO memo — so a memo miss recomputes the full target hash twice.
- **Recommendation:** Pass the already-computed `_cache_key_entry` into a key-accepting variant of
  `_pre_pipeline_cache_get(key=...)` instead of re-deriving it; this removes the dependency on
  `_LAST_KEY_CACHE` for correctness of the double-call and is bit-identical.
- **Measurement needed first:** NO for the refactor (it's a pure dedup of a known-expensive call);
  the existing memo measurements already quantify the saving.
- **Confidence:** High.

## A4-03 — `_full_target_content_hash` has no memo while `_full_x_content_hash` does
- **Severity:** Low
- **File:** `src/mlframe/training/_pipeline_cache.py:413-447` (`_full_target_content_hash`, no cache)
  vs `:254-410` (`_full_x_content_hash`, has `_PIPELINE_X_HASH_CACHE` 16-slot LRU).
- **What's wrong:** The X-side full hash is memoised by `(id, shape)`; the target-side full hash is
  not. In the suite, `train_target` for a given target is identity-stable across the get/set pair and
  across weight schemas, so the same O(rows) `to_numpy()` + blake2b is repaid on every
  `_pre_pipeline_cache_key` call that misses `_LAST_KEY_CACHE`.
- **Why it matters:** Targets are 1-D so the absolute cost is lower than the X hash, but the
  asymmetry is gratuitous — the X memo exists precisely because the suite calls the keyer repeatedly
  with identity-stable frames, and the same argument applies to the target.
- **Recommendation:** Add the same `(id, shape)`-keyed small-LRU memo to `_full_target_content_hash`
  (mirror the X helper). Note the standard `id()`-recycle caveat already documented for the X memo
  applies identically and is acceptable here.
- **Measurement needed first:** YES — target hash is O(rows), so confirm it's a measurable share of
  keyer cumtime before adding the memo.
- **Confidence:** High (asymmetry confirmed; magnitude unmeasured).

## A4-04 — `_canonical_dtype_pairs` stringifies polars schema per call on a path that could reuse the cached suffix
- **Severity:** Low
- **File:** `src/mlframe/training/core/_phase_train_one_target.py:450-494`
- **What's wrong:** For a polars `train_df` the function iterates `train_df.columns`, reads
  `train_df.schema[c]` per column, and `str(dt)`-stringifies non-categorical dtypes (the Enum/Cat
  isinstance fast-path is good and avoids the multi-KB Enum repr, but the numeric/temporal columns
  still get `str(dt)`-built + `_canonicalise_dtype`-normalised every call). This is folded into A4-01
  (same call site) but is the inner cost driver of it.
- **Why it matters:** Same redundancy as A4-01; called per (model × pre_pipeline) for an invariant
  schema.
- **Recommendation:** Memoise `_canonical_dtype_pairs` result keyed by `(id(train_df), width)` —
  the polars frames are identity-stable across the model loop (they live on `ctx.*_df_polars`).
- **Measurement needed first:** YES (sub-ms per call on typical widths).
- **Confidence:** High (recompute confirmed).

## A4-05 — Model-input fingerprint cache keyed on `id(train_df)` is correct only because frames are pinned; brittle by construction
- **Severity:** Low (correctness-adjacent, not a live bug)
- **File:** `src/mlframe/training/core/_phase_train_one_target_body.py:495-531`
- **What's wrong:** `_fp_cache_key` folds `id(_fp_train_df_pre)` (line 502) plus column-count. The
  code comment (lines 488-501) acknowledges this is an `id()`-keyed cache and justifies it by the
  frame being strong-ref-pinned at that point. Unlike the booster-dataset caches (which were
  migrated OFF `id()` to content fingerprints in `_dataset_cache_fingerprint.py` precisely because
  `id()` recycles), this fingerprint cache is still `id()`-keyed. It's safe TODAY because the
  prepared frames are cached in `prepared_frames_cache` and stay alive, but it's the same class of
  hazard the rest of the codebase deliberately moved away from.
- **Why it matters:** If a future change drops the strong ref earlier (e.g. a more aggressive polars
  release between prep and fingerprint), a GC-recycled `id()` collision would serve a stale schema
  hash → wrong `model_file_name` suffix / wrong recorded schema. The blast radius is metadata /
  filenames, not predictions, so severity is Low.
- **Recommendation:** Either document the pin invariant with an assertion, or key on
  `compute_signature(_fp_train_df_pre)` (the existing content fingerprint helper) which is O(n_cols)
  and already used by the booster caches — consistent with the rest of the codebase.
- **Measurement needed first:** NO (correctness hardening; `compute_signature` is already cheap).
- **Confidence:** Medium (no live bug found; risk is latent).

## A4-06 — Verbose-only `get_strategy(m)` re-walk in the conversion phase
- **Severity:** Low
- **File:** `src/mlframe/training/core/_phase_helpers.py:525,541` (`[get_strategy(m) for m in ...]`)
- **What's wrong:** Inside `_phase_pandas_conversion_and_cat_prep`, the verbose-log branches rebuild
  the per-model strategy list via `get_strategy(m)` for every model. The comment at lines 466-470
  explicitly notes this is intentional (avoids the build when verbose=False) — but `ctx.strategy_by_model`
  already holds the resolved strategies (built once in `setup_configuration`, per
  `_phase_train_one_target_body.py:205`). The phase doesn't receive ctx here, so it can't reuse it.
- **Why it matters:** Verbose-only and once-per-suite (not per-target/fold), so negligible runtime
  cost. Flagged only for completeness as a redundant-recompute the suite already solved elsewhere.
- **Recommendation:** Thread the resolved `strategy_by_model` (or the precomputed `non_native` list)
  into the phase so the verbose branch reads it instead of re-resolving. Low priority.
- **Measurement needed first:** NO (cost is trivially once-per-suite, verbose-gated).
- **Confidence:** High.

## A4-07 — `get_pandas_view_of_polars_df` single-slot last-result memo can thrash on train/val/test alternation
- **Severity:** Low
- **File:** `src/mlframe/training/utils.py:420,467-495` (`_PD_VIEW_LAST_CACHE`, single slot) +
  consumer `src/mlframe/training/core/_phase_train_one_target_polars_fastpath.py:211-263`
  (`ctx._pandas_view_cache`, 4-slot OrderedDict with byte budget).
- **What's wrong:** There are TWO layers of pandas-view caching: the module-level single-slot
  `_PD_VIEW_LAST_CACHE` in `utils.py` and the ctx-scoped 4-slot `_pandas_view_cache`. The ctx-scoped
  one (the real cache) is keyed by `id(polars_df)` and correctly LRU-bounded. The module-level
  single-slot memo only helps the immediate-repeat case; in the lazy-pandas branch the loop converts
  train→val→test in sequence (lines 212-263), so the single-slot memo is overwritten on each of the
  three frames and never hits within one strategy. It's purely a backstop for adjacent identical
  calls. Not wrong, just mostly inert given the better ctx cache sits in front of it.
- **Why it matters:** No correctness issue; the ctx cache carries the real reuse. The single-slot
  memo adds an `id()`-recycle surface (documented at lines 476-487) for marginal benefit.
- **Recommendation:** Leave as-is unless profiling shows it never hits — in which case remove it to
  shed the `id()` surface. Do not expand it (the ctx cache is the correct place and is byte-gated).
- **Measurement needed first:** YES (confirm hit-rate before removing).
- **Confidence:** Medium.

---

## Areas explicitly checked and found CLEAN (no finding)

These are reported so the negative results are on record (per "no hand-wave" / "objectivity" rules):

- **No hot-path frame copies.** Grepped `trainer.py`, `_trainer_train_and_evaluate.py`,
  `_phase_train_one_target*`, `_phase_helpers.py`, `_misc_helpers.py`. The only `.copy()` calls are
  shallow dict copies (`common_params.copy()` at `_phase_train_one_target_body.py:571,686`,
  `process_model`'s `effective_common_params = common_params.copy()` at `train_eval.py:416`) and
  MLP param-dict copies (`trainer.py:716-760`) — all dict-level, none copy a DataFrame. The
  `_misc_helpers._augment_with_dropped_high_card_cols._attach` uses `pd.concat`/`with_columns` to
  build a fresh frame (line 130-140) but documents it produces a fresh frame from extras, not a copy
  of the caller's frame, and runs once per dummy-baseline re-attach, not per fold.

- **Polars→pandas conversion is properly deferred + size-gated + content-cached.**
  `_phase_pandas_conversion_and_cat_prep` (`_phase_helpers.py:478-483`) computes
  `defer_pandas_conv` and skips the upfront conversion when all blockers are polars-native; the
  lazy per-strategy conversion (`_phase_train_one_target_polars_fastpath.py:201-263`) is
  `id()`-cached on ctx with a 4-entry + byte-budget cap (env `MLFRAME_PANDAS_VIEW_CACHE_MAX_MB`).
  `get_pandas_view_of_polars_df` (`utils.py:497-740`) uses `split_blocks=True` zero-copy Arrow views
  and a size-aware dispatcher (helper for >50 MB, raw `to_pandas` only for small Cat-heavy frames).
  This honours the "frame-type conversions are caller responsibility, gate eager conversion on byte
  size" convention.

- **In-wrapper conversions removed.** `xgb_shim._build_quantile_dmatrix` (`xgb_shim.py:184-216`)
  passes polars straight to `QuantileDMatrix` (no `to_pandas`). The only target-side coercions in
  `_trainer_train_and_evaluate.py:453,620` are 1-D `val_target.to_numpy()` / `train_target.to_numpy()`
  on `pl.Series` — narrow single-column pulls, which the convention explicitly permits.

- **Booster-dataset caches are content-keyed, not `id()`-keyed.** `_dataset_cache_fingerprint.compute_signature`
  (O(n_cols), 3-row sample hash) replaced the old `id(X)` keys across xgb/lgb/cb caches — verified
  in `xgb_shim.py:112-120`. The XGB DMatrix cache is module-level LRU (cap 8) surviving `clone()`.

- **Cross-target dataset reuse + polars-frame release are sound.** The dataset-reuse cache is keyed
  by `(model_name, pp_name)` (`_phase_train_one_target_dataset_cache.py:101-106`), avoiding the
  prior bare-name cross-PP collision; `_release_ctx_polars_frames` scrubs the pandas-view cache,
  recurrent-numpy cache, FH cache, and dataset-reuse cache by released `id()` token to prevent
  recycled-id stale hits (lines 293-320).

- **PipelineCache (`_strategies_pipeline_cache.py`) is byte-budgeted + RAM-fraction-aware**, never
  pickled, re-checks available RAM on each insert (lines 180-213), and pins the active key. It caches
  transformed frames keyed by the content key from A4-01; it does NOT cache live Trainer / CUDA /
  compiled objects. Suite-scoped via `ctx._pipeline_cache` so hits carry across targets.

- **Pickle safety for the polars-ds pipeline.** `_PolarsDsPipelineJsonProxy.__reduce__`
  (`_setup_helpers_pipeline_cache.py:221-228`) serialises via `to_json()` and reconstructs lazily —
  no descent through every `pl.Expr`, no live-framework-object pickling. The JSON-roundtrip
  validation result is cached in-memory + on-disk, content-keyed by `hash(_js)` + polars/polars-ds
  version tag (correct invalidation on wheel upgrade).

- **The pipeline cache IS hit, not dead.** HIT/MISS counters are merged into
  `ctx._cache_stats["pipeline_cache"]` (`_phase_train_one_target_body.py:996-1001`); the polars
  fastpath feature-side cache and fingerprint cache likewise track hits/misses
  (`_phase_train_one_target_polars_fastpath.py:101-110`, `_phase_train_one_target_body.py:519-531`).

- **No O(n²) algorithmic issues on the suite path.** `_non_neural_train_times` uses
  `np.percentile` (O(n log n)) once per neural fit; auto-detect uses a single lazy `collect()` for
  all text-like columns (`_misc_helpers.py:631-653`) rather than per-column kernel launches; the
  pandas branch uses one `df[cols].agg(["nunique","count"])` (`_misc_helpers.py:750`). Slugify is
  LRU-memoised (`_phase_train_one_target.py:58-63`); `inspect.signature` is WeakKeyDict-memoised
  per class (`:33-48`). No full sorts where partial would do on the hot path.

- **Per-call overhead largely hoisted.** `strategy_by_model` / `sorted_mlframe_models` precomputed
  on ctx; `_cb_extra_fit_invariant` and `_ngb_fallback_snapshot` hoisted outside the weight loop
  (`_phase_train_one_target_body.py:537-556`); `psutil` and `TargetTypes` imports hoisted to module
  scope (annotated `PSUTIL-IMPORT-HOT` / `TARGETTYPES-IMPORT-LOOP`); `Path` import hoisted
  (`_main_train_suite.py:43`). The remaining lazy in-body imports (`_train_one_target` re-imports its
  siblings at call entry) are documented as cycle-avoidance and are sub-µs dict lookups.

---

## Notes on method
- All file:line references verified by reading the cited source on 2026-06-04.
- Per project "measure before optimize" rule, A4-01/03/04/07 are flagged as needing a cProfile pass
  on a production-shape frame before any change; the wins are likely small (the suite is mature) and
  could be sub-1% on narrow frames. A4-02 and A4-05 are correctness-hardening / latent-landmine
  refactors that are bit-identical and don't need a perf measurement to justify.
