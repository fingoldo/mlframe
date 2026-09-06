# Cross-cutting audit: memory & resource contracts at 100+ GB scale

**Date:** 2026-09-05
**Scope:** `src/mlframe` (2466 .py files, `_benchmarks/` excluded)
**Method:** AST scan for the six target shapes (copy/reconstruct calls with loop-nesting context;
`__getstate__`-less classes assigning cache/device/handle attributes to `self`; unmanaged
`open`/`Pool`/`Executor`/`memmap`/`mkstemp` construction; `n_boot * n` style products;
eager `.collect()`/`.to_pandas()`), then hand-reading candidates in
`feature_selection/` (189k LOC), `training/` (166k LOC), `feature_engineering/` (44k LOC),
`evaluation/`, `utils/`.
**Read-only.** No source file was modified.

The repository is unusually disciplined here: it already has `fe_polars_exceeds()` (~2 GB eager gate),
`_hash_array_chunked()` (chunked hashing "instead of a whole-frame `.tobytes()` copy, which on a 100+ GB
frame..."), `_fit_constant_key()`'s explicit `h.update(a.data)` memoryview idiom ("O(n) time but O(1)
additional memory, same discipline as the rest of the package's never-copy-a-large-frame rule"),
`_build_resample_indices()`'s int32 + `MemoryError` ceiling, and weakref-guarded bounded LRUs on every
`id()`-keyed cache. **Every finding below is a site that departs from an idiom the repo itself
establishes elsewhere** - which is what makes them actionable and low-risk to fix.

---

# CONFIRMED

### XMC-01 [P0] bootstrap-resample-index-matrix-int64-unguarded
**File:** `src/mlframe/evaluation/_bootstrap_fused_binary_bundle.py:181`
**Summary:** `_generate_resample_idxs` materialises the entire `(n_bootstrap, n)` resample-index
matrix as **int64** in one allocation before any metric is computed:
`idxs = np.empty((n_bootstrap, n), dtype=np.int64)`.
It is called unconditionally at `:244` from `bootstrap_auc_brier_ll_ece_batch`, which
`training/honest_diagnostics.py:236` invokes with a hardcoded `n_bootstrap=1000` on the full
prediction vector of a test/OOS split.

**Failure scenario:**
- n = 200k -> 1000 x 200 000 x 8 B = **1.60 GB**
- n = 1M -> 1000 x 1 000 000 x 8 B = **8.00 GB**
- n = 7M (the frame size `_predict_guards.py`'s own docstring cites as production: "a 50-70s rebuild
  on 7M-row frames") -> 1000 x 7 000 000 x 8 B = **56.0 GB** in a single `np.empty`.

The sibling implementation of the *same* matrix, `calibration/policy.py:_build_resample_indices`,
uses int32 **and** raises `MemoryError` before allocating once `4 * n_bootstrap * n` exceeds ~1 GiB
(`MLFRAME_CALIBRATION_RESAMPLE_MAX_BYTES`). This path has neither guard and uses the wider dtype, so
it OOMs silently at half the row count that the guarded sibling refuses outright.

**Evidence:**
- `_bootstrap_fused_binary_bundle.py:181` int64 alloc, no size check anywhere in the function or in
  `bootstrap_auc_brier_ll_ece_batch` (read `:172-250`).
- The consumer at `:250-265` **already iterates in `chunk_size=200` row blocks**
  (`for lo in range(0, n_bootstrap, chunk_size): ... idxs[lo:hi] ...`), so the full matrix is never
  needed at once - only 200 rows are live per kernel call.
- `calibration/policy.py:383-408`: "RAM ceiling: the matrix is n_bootstrap x resample_len **int32** ...
  A proactive size guard computes the projected `4 * n_bootstrap * n` bytes BEFORE any allocation and
  raises MemoryError". The same docstring records that int32 draws are **bit-identical** to the former
  int64 draws because `Generator.integers` keys entropy off the range, not the output dtype.

**Suggested fix:** **Chunk** - generate `idxs` one `chunk_size` block at a time inside the existing
`for lo in range(0, n_bootstrap, chunk_size)` loop. For `stratify is None` the current code draws
row-by-row (`for i in range(n_bootstrap): idxs[i] = rng.integers(...)`), and the stratified branch
likewise draws per `(i, c)`, so moving the draw inside the chunk loop preserves the RNG call order
exactly - the documented bit-identity contract holds. Additionally switch to **int32** (policy.py
already proves this is bit-identical), and port policy.py's projected-bytes `MemoryError` guard.
Combined: 56 GB -> 200 x 7M x 4 = **5.6 GB** peak, a 10x reduction, with no behaviour change.

---

### XMC-02 [P0] ungated-whole-frame-to_pandas-in-mrmr-synergy-screen
**File:** `src/mlframe/feature_selection/filters/_mrmr_fit_impl/_friend_graph_and_redundancy/_group1.py:176`
**Summary:** `_X_pd_syn = fe_to_pandas(X)` converts the **entire** fit frame polars->pandas inside the
MRMR greedy friend-graph / redundancy stage, with no byte-size gate. `fe_to_pandas` is
`X.to_pandas()` with no `use_pyarrow_extension_array`, so numeric columns are densified to numpy -
a genuine full materialisation, not the zero-copy Arrow bridge that `MRMR.fit` uses at its own boundary.

**Failure scenario:** X is the MRMR fit frame, the frame CLAUDE.md sizes at 100+ GB. A 100 GB polars
frame becomes a second ~100 GB pandas frame held simultaneously -> **~200 GB peak**. Concretely at a
modest 20M rows x 200 float64 columns: 20e6 x 200 x 8 = 32.0 GB polars, +32.0 GB pandas = **64.0 GB
peak** on a box sized for the 32 GB frame. The only guard on the call is
`if 2 <= len(_cand_syn) <= 60` - a *candidate-count* bound with no relation to frame bytes, so a
100 GB frame with 3 candidate operands takes the path.

**Evidence:**
- `_group1.py:176`, inside the `interactions_max_order >= 2` synergy-combo block (read `:165-190`).
- `grep -n fe_polars_exceeds _group1.py` returns **nothing** - the gate is neither imported nor called
  in that module.
- The sibling FE stage `_fe_stage_cascade_early_b.py` guards **every** `fe_to_pandas(X)` call with
  `if fe_polars_exceeds(X):` first (lines 52, 138, 193, 367, 559 gate the calls at 91, 151, 235, 275,
  314, 418, 427, 429, 463, 493, 595, 628, 661, 695).
- `_fe_frame_ops.py:102-106`: "Eager polars->pandas materialisation is bounded to frames under this
  size (CLAUDE.md eager-conversion rule); a larger polars frame must not be whole-frame-copied to
  pandas". `FE_EAGER_MATERIALIZE_MAX_BYTES = 2 * 1024**3`.

**Suggested fix:** **Move to the boundary / gate.** Wrap the block in `if not fe_polars_exceeds(X):`
exactly as `_fe_stage_cascade_early_b.py` does, so an oversized frame skips the synergy screen rather
than duplicating itself. Better still, `detect_synergy_combos` only needs the `_cand_syn` columns
(2..60 of them) - pass an `X.select(cand_cols).to_pandas()` projection (or `fe_subsample_to_pandas`)
instead of the whole frame, which makes the gate unnecessary and costs `n x 60 x 8`
(9.6 GB at 20M rows, still worth subsampling) rather than `n x n_all_cols x 8`.

---

### XMC-03 [P1] key-bank-fingerprint-full-tobytes-copy
**File:** `src/mlframe/feature_engineering/transformer/_key_bank.py:123`
**Summary:** `h.update(X_train.tobytes())` in `_key_bank_fingerprint` allocates a full contiguous
**bytes copy of the entire training matrix** purely to feed sha256, then frees it.

**Failure scenario:** `X_train` is `(n_train, d_input)`. The docstring's own worked example is
n=10M, d=64: 10e6 x 64 x 4 B (float32) = 2.56 GB - `.tobytes()` makes a **second 2.56 GB
allocation live simultaneously with the array**, so peak is 5.12 GB where 2.56 GB is required. At
float64 it is 5.12 GB + 5.12 GB = **10.24 GB peak**. The docstring explicitly budgets the *CPU* cost
("~3 GB/s sha256 ... 1-3 s for 10M, d=64") but says nothing about the doubled RAM.

**Evidence:**
- `_key_bank.py:110` (docstring) and `:123` (the call).
- The repo's own contrary idiom, `feature_selection/filters/_joblib_safe.py:263-274`:
  "Hashing the full buffer via a memoryview (no `.tobytes()` copy - `hashlib.update` consumes the
  buffer-protocol object directly) is O(n) time but **O(1) additional memory**, same discipline as
  the rest of the package's 'never copy a large frame' rule", implemented as `h.update(a.data)`.
- Same idiom again at `feature_selection/wrappers/rfecv/_fit_init.py:32-43` (`_hash_array_chunked`),
  whose docstring reads "never materialises the whole buffer as one extra bytes copy ... Peak extra
  RAM is ~one chunk, not the whole frame."

**Suggested fix:** **View, no copy** - `h.update(np.ascontiguousarray(X_train).data)`
(or reuse `_hash_array_chunked` from `_fit_init.py`). Byte-for-byte identical digest for a
C-contiguous array, O(1) extra memory. The same one-line change applies to XMC-04/05/06 below.

---

### XMC-04 [P1] fe-accuracy-gate-baseline-key-double-full-copy-per-candidate
**File:** `src/mlframe/feature_selection/filters/_fe_accuracy_gate.py:56-57`
**Summary:** `_baseline_cv_key` builds its cache key as
`hash(X_base.tobytes()), ... hash(y.tobytes())` - a full bytes copy of the FE base matrix **plus**
a full bytes copy of y, on **every** `measure_feature_uplift` call.

**Failure scenario:** The whole point of the memo (per its own docstring at `:36-40`) is that
"several engineered SIBLINGS derived from the SAME raw source (x__He2, x__He3, x__T2, x__L2, ...)
call `measure_feature_uplift` with an IDENTICAL X_base/y/seed" - i.e. the key is computed once per
*candidate*, and the memo only pays off across many candidates, so the copy recurs per candidate.
At a 1M-row FE subsample with 30 base columns, float64: 1e6 x 30 x 8 = **240 MB copied and freed per
candidate**; over a 500-candidate FE round that is ~120 GB of allocator churn and a 240 MB transient
peak sitting on top of the array itself. `hash()` on a 240 MB bytes object also walks it with
siphash, so nothing is saved by the copy.

**Evidence:** `_fe_accuracy_gate.py:51-62`; called at `:123` from `measure_feature_uplift`, which is
called from `_mrmr_fit_impl/_fit_impl_core.py:616` and `_fe_accuracy_gate.py:283`.
Contrast `_fe_resident_operands.py:79-105`, which fixed **exactly this** for a smaller array:
"The old `hash(host.tobytes())` walked the buffer AND allocated a full host copy first (an ~8 MB
tobytes churn for a 1M-row f64 operand). `xxh3_64` walks the array buffer directly via the buffer
protocol - no intermediate bytes copy - at ~8x the throughput".

**Suggested fix:** **View** - reuse `_fe_resident_operands._content_hash(X_base)` (same package,
already copy-free via xxh3 with an njit/tobytes fallback). Since this is a private in-process cache
key, changing the hash function is behaviour-neutral - no persisted artefact depends on the digest.

---

### XMC-05 [P1] collinear-keep-mask-hash-2gb-tobytes-copy
**File:** `src/mlframe/training/composite/discovery/_collinear_numba.py:83`
**Summary:** `h.update(fm.tobytes())` inside `_keep_mask_cache_key`. The function has a byte cap -
but the cap only decides whether to *cache*; below it, the full copy is still made.

**Failure scenario:** `_KEEP_MASK_HASH_MAX_BYTES = 2_000_000_000` (`:65`), so a feature matrix of up
to **2.0 GB** is duplicated as a bytes object on every call - peak 4.0 GB for a 2.0 GB matrix
(e.g. 5M rows x 50 float64 = 2.0 GB). The docstring at `:75-77` claims "Hashes the full contiguous
buffer (**no source-frame copy** - `fm` is already an ascontiguousarray the kernel owns)", which is
true of the *frame* but false of the buffer: `.tobytes()` is precisely a buffer copy.

**Evidence:** `_collinear_numba.py:72-85` read in full; cap at `:65`.

**Suggested fix:** **View** - `h.update(fm.data)`. `fm` is documented as already
`ascontiguousarray`, so the digest is byte-identical and the extra allocation drops to zero. The
2 GB cap then only governs hash *time*, which is what the docstring intends.

---

### XMC-06 [P2] prebin-signature-tobytes-contradicts-its-own-docstring
**File:** `src/mlframe/training/composite/cache.py:586`
**Summary:** `prebin_matrix_signature` does `h.update(arr.tobytes())` immediately after a docstring
asserting "The hash is over the contiguous matrix buffer - O(matrix bytes), **no copy of the source
frame**."

**Failure scenario:** Bounded - this is the discovery *screen sample* matrix, described at `:537` as
"the SMALL screen-sized float feature matrix". At a 200k-row x 300-column screen sample, float64:
200 000 x 300 x 8 = 480 MB, so the transient copy is **480 MB** on top of the matrix (960 MB peak).
Called once per prebin cache probe, not per column, so it does not compound - hence P2, not P1.

**Evidence:** `training/composite/cache.py:568-587`; the surrounding `PrebinCache` (`:590-640`) is
properly byte-gated per entry via `_prebin_cache_max_bytes()` and count-capped, so the cache itself
is sound - only the key computation copies.

**Suggested fix:** **View** - `h.update(arr.data)` (`arr` is `np.ascontiguousarray(feature_matrix)`
one line above), making the docstring's claim true.

---

### XMC-07 [P1] cb-pool-caches-capped-on-entry-count-not-bytes
**File:** `src/mlframe/training/cb/_cb_pool.py:542-543` and `src/mlframe/training/_predict_guards.py:102`
**Summary:** Two module-level dicts, `_CB_POOL_CACHE` (train Pools) and `_CB_VAL_POOL_CACHE` (val
Pools), each retain up to `_CB_POOL_CACHE_MAX_ENTRIES = 16` **CatBoost `Pool` objects**. A `Pool`
owns the full quantised dataset. The only eviction bound is entry count; there is no byte budget and
no per-entry size gate.

**Failure scenario:** A quantised CatBoost `Pool` costs roughly 1 byte per (row, feature) cell plus
border tables. On the frame size this module's own docstring cites - "a 50-70s rebuild on 7M-row
frames" (`_predict_guards.py:129-131`) - with 500 features:
7e6 x 500 x 1 B = **3.5 GB per Pool**. Sixteen entries = **56 GB** retained in a module-level dict
that outlives every fit. Both dicts are capped independently, so the combined worst case is
**32 Pools ~= 112 GB**. Realistically a suite over 4 targets x 2 folds fills 8 val entries = 28 GB
held for the whole process. Keys are content fingerprints (`compute_signature`), so distinct
targets/folds/pre-pipeline outputs each mint a new entry rather than reusing one.

**Evidence:**
- `_cb_pool.py:542-543`; eviction loops at `_cb_pool.py:778-779` and `_cb_pool_build.py:288-289` -
  both `while len(cache) >= MAX_ENTRIES: cache.pop(next(iter(cache)))`, count only.
- Key construction at `_cb_pool.py:736-740` via `compute_signature(val_df, extra=(...))`.
- The suite does clear both at startup (`training/core/_phase_config_setup.py:440-443`) but never
  mid-suite.
- The repo demonstrates the byte-budgeted alternative in three places:
  `training/composite/cache.py:559` `_prebin_cache_max_bytes()`,
  `_collinear_numba.py:65` `_KEEP_MASK_HASH_MAX_BYTES`,
  `calibration/policy.py` `_resample_matrix_max_bytes()`.

**Suggested fix:** Add a **byte budget** alongside the entry cap: record
`n_rows * n_features` (or `Pool.num_row() * Pool.num_col()`) at insert, evict oldest-first until the
running total is under an env-overridable ceiling (mirror `_prebin_cache_max_bytes()`'s shape), and
refuse to cache a single Pool that exceeds it outright. The 16-entry cap can stay as a secondary
bound.

---

### XMC-08 [P2] fe-pair-sweep-threadpool-leaked-on-any-exception
**File:** `src/mlframe/feature_selection/filters/_feature_engineering_pairs/_pairs_core.py:1155`
**Summary:** `_chunk_state["pipeline_ex"] = ThreadPoolExecutor(max_workers=1)` is created at
line 1155 and shut down at line 1354 - ~200 lines and the entire pair sweep later - with **no
`try`/`finally` covering the span**. The second chunk buffer `_chunk_buffer2 = np.empty_like(...)`
(line 1153) is registered into the same dict and is held by any in-flight future.

**Failure scenario:** Any exception raised in the pair loop (a cupy `OutOfMemoryError`, a kernel
launch failure, `KeyboardInterrupt`) skips line 1354. `ThreadPoolExecutor`'s worker is registered
with `threading._register_atexit`, so the thread survives to interpreter exit and, if a submitted
future is still pending, keeps a reference to the double chunk buffer. Sizing: the chunk buffer is
`(chunk_rows, n_operands)` float64 - at a 2M-row chunk x 40 operands that is
2e6 x 40 x 8 = **640 MB per leaked pair-sweep**, x2 for the double buffer = **1.28 GB**, retained
per aborted FE step. An MRMR fit that retries FE across N targets in one process accumulates N
threads and up to N x 1.28 GB. The exception classes that trigger this are exactly the ones this
path is prone to (GPU OOM under `MLFRAME_FE_GPU_MATERIALISE`).

**Evidence:**
- Executor created `:1155`, shut down `:1354` ("...is awaited by shutdown so the worker never
  outlives the shared buffers" - the comment assumes the shutdown always runs).
- AST check: the enclosing function `check_prospective_fe_pairs` spans lines 261-1358 and contains
  **no `Try` node with a `finalbody` that covers line 1155**.
- The near-identical sibling **does** guard it: `_mrmr_fe_step/_step_pairmi.py:382` creates
  `ThreadPoolExecutor(max_workers=1)` and shuts it down inside a `finally:` at `:386-389`.
- Note the *setup-failure* path at `:1161-1164` is correctly handled - only the steady-state span is
  unprotected.

**Suggested fix:** Wrap lines 1155-1354 in `try: ... finally:` that pops and `shutdown(wait=True)`s
`pipeline_ex` and drops `pipeline_buffers`, mirroring `_step_pairmi.py:386-389` verbatim.

---

### XMC-09 [P2] fisher-gradient-stack-materialised-twice
**File:** `src/mlframe/feature_engineering/transformer/fisher_weighted_residual.py:115`
**Summary:** `stack = np.broadcast_to(X, (d, n, d)).reshape(d * n, d).copy()`.
`np.broadcast_to` returns a zero-stride view; `.reshape(d*n, d)` on a non-contiguous broadcast view
**cannot** be a view, so numpy already allocates a full `d*n*d` array - and then `.copy()` allocates
a **second** one. The transient peak is 2x the intended stack.

**Failure scenario:** Gated by `_MAX_STACK_ELEMS = 64_000_000` (`:28`, checked at `:114`), so the
intended allocation is up to 64e6 x 8 B = **512 MB** (float64). The double materialisation makes the
real transient peak **1.02 GB**, plus the `predict_proba` output `(d*n, n_classes)` on top. At the
cap with d=64: n = 64e6/(64*64) = 15 625 rows - so the ceiling is hit on quite ordinary frames once
d grows, and the gate's own budget is silently doubled.

**Evidence:** `fisher_weighted_residual.py:110-127` read in full; the `else` fallback branch at
`:129-136` (`X_plus = X.copy()` per feature) is correct - one copy live at a time, bounded by `n*d`.

**Suggested fix:** Drop the redundant `.copy()` - `.reshape()` on a broadcast view already returns a
fresh writable C-contiguous array, so `stack` is safe to mutate at `:118` without it. Halves the
peak to the 512 MB the gate budgets for, bit-identically. (If defensiveness is wanted, use
`np.ascontiguousarray(...)`, a no-op on the already-fresh reshape result.)

---

# LEADS

*(shape confirmed by reading, but I could not pin the runtime dimensions to a defensible byte estimate)*

### XMC-L1 [P3] keybank-device-buffers-and-ann-indices-not-excluded-from-pickle
**File:** `src/mlframe/feature_engineering/transformer/_key_bank.py:29`
**Summary:** `KeyBank` is a `@dataclass` holding `k_proj_device: list[Any] | None` (a list of cupy
device arrays, populated by `to_device()`) and `y_train_device`, plus `ann_indices: list[Any]`
(hnswlib `Index` objects). It defines **no `__getstate__`**, so a plain pickle of a `KeyBank`
attempts to serialise live device handles and native ANN indices.
**Failure scenario:** LEAD - I found no current call site that pickles a `KeyBank` whole:
`save_key_bank`/`try_load_key_bank` (`:148+`) persist field-by-field to `.npy` + per-head
`ann_h{h}.pkl`, and `free_device()` exists for the device side. The exposure is latent: any future
joblib fan-out over banks (`Parallel(delayed(...))(bank)`) would either raise or, worse, silently
D2H the whole `k_proj` per worker - `(n_heads, n_train, head_dim)` float32; at 8 heads x 10M x 32
that is 8 x 10e6 x 32 x 4 = **10.2 GB per worker**. Marked LEAD because no such call site exists today.
**Evidence:** `_key_bank.py:29-91`; `grep KeyBank` outside the module returns only
`row_attention.py:221/223/270` (the file-based cache).
**Suggested fix:** Add `__getstate__` dropping `k_proj_device`/`y_train_device` (and `__setstate__`
restoring them as `None`), matching `feature_selection/filters/_gpu_strict_fe/_state.py:157` and the
warning `training/_io_save.py:348-352` already emits for exactly this class of attribute.

### XMC-L2 [P3] fe_to_pandas full-n fallbacks in fe_decide_on_subsample are ungated
**File:** `src/mlframe/feature_selection/filters/_mrmr_fit_impl/_helpers.py:398, 424, 444, 458`
**Summary:** Four `return fit_with_recipes_fn(fe_to_pandas(X), y, **kwargs)` fallbacks - no
subsample configured (`:398`), unexpected family return shape (`:424`), partial recipe coverage
(`:444`), full-n replay failure (`:458`) - each materialises the whole polars frame to pandas with no
`fe_polars_exceeds` check (`grep` confirms the gate is not imported in this module).
**Failure scenario:** Same arithmetic as XMC-02 (frame duplicated: a 32 GB polars frame -> 64 GB
peak). Marked LEAD rather than CONFIRMED because these are explicitly documented, *warning-logged*
degradation paths ("costly full-n path - the subsample bypass is lost") that the module treats as
correctness-over-speed, and `_fe_frame_ops.fe_to_pandas`'s own docstring sanctions them: "ONLY for
the rare full-n FE fallback paths ... On the normal subsampled path this is never called". I could
not establish how often the partial-coverage branch actually fires in production.
**Suggested fix:** If kept, add the `fe_polars_exceeds(X)` check and log-and-skip the family above
2 GB instead of duplicating the frame - the same trade `_fe_stage_cascade_early_b.py` already makes.

### XMC-L3 [P3] finite-difference mixed-partial holds four full copies simultaneously
**File:** `src/mlframe/feature_selection/filters/_gradient_interaction_seeder.py:258-261`
**Summary:** Per candidate pair, `Xpp/Xpm/Xmp/Xmm = Xs.copy()` - four whole-matrix copies **alive at
once** (all four are arguments to the single `predict_fn` expression at `:262`).
**Failure scenario:** Bounded by `max_rows: int = 2000` (`:246`, row-subsampled at `:254-255`), so at
d=1000 float64 the peak is 4 x 2000 x 1000 x 8 = **64 MB** - real but small, hence P3. It becomes a
P1 only if a caller passes a larger `max_rows`; I did not find a call site that does, so this is a
LEAD on the parameterisation, not the default.
**Suggested fix:** Mutate-and-restore on a single buffer (`Xs[:, a] += h; ...; Xs[:, a] -= h` in a
`try/finally`), or evaluate the four corners sequentially, reducing peak to 1 copy.

### XMC-L4 [P3] KFold split cache holds full-n index arrays under a byte-blind 256-entry cap
**File:** `src/mlframe/training/composite/discovery/_screening_tiny.py:47`
**Summary:** `_KFOLD_SPLIT_CACHE` keyed `(n_rows, cv_folds, random_state)` stores
`list[(train_idx, val_idx)]`; each entry totals `cv_folds * n_rows` int64 index values. The cap is
`_KFOLD_SPLIT_CACHE_MAX = 256` **entries**, byte-blind (it does a full `.clear()` on overflow).
**Failure scenario:** One entry at 5 folds is `5 * n_rows * 8` B: at n=1M that is **40 MB**, at
n=100M **4.0 GB** - and 256 such entries would be ~1 TB. In practice the docstring argues the key is
*identical* across a sweep ("Across all N_SPECS in one rerank sweep that triple is identical"), so
occupancy should be 1-2 entries, and this module operates on a *screen sample*. Marked LEAD because
I could not establish an upper bound on `n_rows` reaching this cache. The comment "Bounded LRU-ish:
cap entries so a long-lived process can't grow it unboundedly" is accurate about count, not bytes.
**Suggested fix:** Cap on `sum(cv_folds * n_rows)` bytes rather than entry count, and store `int32`
indices when `n_rows < 2**31` (halves it; the arrays are documented read-only downstream).

---

# Summary

| ID | Sev | Shape | File:line | Peak / retained (worked case) |
|---|---|---|---|---|
| XMC-01 | P0 | 3 - `n_boot * n` product, chunked form available | `evaluation/_bootstrap_fused_binary_bundle.py:181` | 8.0 GB @ n=1M; **56 GB @ n=7M** (1000 x n x 8 B); consumer already chunks at 200 |
| XMC-02 | P0 | 6/1 - ungated whole-frame polars->pandas | `.../_friend_graph_and_redundancy/_group1.py:176` | frame duplicated: **64 GB peak** on a 32 GB frame; ~200 GB at the stated 100 GB scale |
| XMC-03 | P1 | 1 - whole-array `.tobytes()` copy | `feature_engineering/transformer/_key_bank.py:123` | **+2.56 GB** transient (n=10M, d=64, f32); 5.12 GB peak |
| XMC-04 | P1 | 1 - full copy per candidate | `feature_selection/filters/_fe_accuracy_gate.py:56-57` | **240 MB/candidate** (1M x 30 f64); ~120 GB churn over 500 candidates |
| XMC-05 | P1 | 1 - full copy up to the cache cap | `training/composite/discovery/_collinear_numba.py:83` | **+2.0 GB** transient (cap is 2 GB); 4.0 GB peak |
| XMC-06 | P2 | 1 - full copy, contradicts docstring | `training/composite/cache.py:586` | **+480 MB** (200k x 300 f64) |
| XMC-07 | P1 | 2 - cache capped on count, not bytes | `training/cb/_cb_pool.py:542`, `training/_predict_guards.py:102` | 3.5 GB/Pool x 16 = **56 GB** per dict; 112 GB across both |
| XMC-08 | P2 | 4 - executor + buffers without `finally` | `.../_feature_engineering_pairs/_pairs_core.py:1155` | 1 thread + **1.28 GB** double buffer leaked per aborted FE step |
| XMC-09 | P2 | 1 - reshape-of-broadcast copied twice | `feature_engineering/transformer/fisher_weighted_residual.py:115` | **1.02 GB** vs the 512 MB the gate budgets |
| XMC-L1 | P3 | 5 - device buffers, no `__getstate__` | `feature_engineering/transformer/_key_bank.py:29` | LEAD - 10.2 GB/worker *if* ever pickled; no such call site today |
| XMC-L2 | P3 | 6 - ungated fallback conversions | `_mrmr_fit_impl/_helpers.py:398,424,444,458` | LEAD - frame duplicated; documented rare paths |
| XMC-L3 | P3 | 1 - 4 simultaneous copies | `_gradient_interaction_seeder.py:258-261` | LEAD - 64 MB at `max_rows=2000` default |
| XMC-L4 | P3 | 2 - byte-blind 256-entry cache | `training/composite/discovery/_screening_tiny.py:47` | LEAD - 40 MB/entry @ n=1M; occupancy argued to be ~1 |

**Counts:** 9 CONFIRMED (2 P0, 4 P1, 3 P2), 4 LEADS (all P3). 13 total.

**Cheapest high-value batch:** XMC-03/04/05/06 are all the same one-line change
(`.tobytes()` -> `.data` / `_content_hash`), bit-identical digests, ~5.3 GB of transient peak removed
across four subsystems. XMC-02 is a two-line gate that mirrors an existing sibling. XMC-01 is the
single largest win and is structurally easy because the consumer already loops in 200-row chunks.

**Shapes that came back clean.** Shape 2 (`id()`-keyed caches) is systematically correct: every
`id()`-keyed table found (`_cmi_cuda._FORDER_CACHE`, `_gpu_resident_fe._RESIDENT_CODES_HANDOFF`,
`_gpu_resident_materialise._OPERAND_TABLE_CACHE`, `_mah._get_y_binning`, `_mi_dispatch`,
`_fe_resident_operands._HASH_MEMO`, `batch_mi_noise_gate_gpu`) pairs the id with a weakref +
shape/dtype co-validation and an `OrderedDict` LRU with an explicit max-entries pop, and several
carry a `clear_*()` teardown hook. Shape 5 is broadly handled - 40+ classes define `__getstate__`,
and `training/_io_save.py:348-352` actively warns when an unpicklable attribute escapes one. Shape 4
turned up only `_pairs_core.py` (XMC-08); `registry.py:75`'s module-level `ThreadPoolExecutor` has
both an explicit `shutdown_prewarm_executor()` and an `atexit` registration, and every `mkstemp`
found is paired with cleanup.
