# MRMR 100000 x 100 all-families fit: host-side structural speed analysis (read-only agent, 2026-11-17)

Scope: host orchestration only (the device / kernel side is the other agent's report). No source file was changed, no fit was run. Evidence comes from the code at `D:\Temp\wt2\src\mlframe\feature_selection\filters` (paths below are relative to it), from `D:/Temp/exp/fit_all.prof` (cProfile, all threads, stage concurrency off), `D:/Temp/exp/spy_gil2.txt` (py-spy `--gil`, 50 Hz) and `_TRACKER.md` MR11-01..40. Both recordings are from earlier today (10:00 and 14:37) and predate some of the day's commits (lane rebalance MR11-27, MR11-29..38), so per-family numbers are a ranking, not today's exact values. Every statement marked INFERENCE was not measured.

## Progress

- [x] Tracker MR11-01..40, stage scheduler, edge pool, variant-group scheduler read.
- [x] Per-lane GIL-held breakdown from `spy_gil2.txt`; own-time ranking and njit census from `fit_all.prof`.
- [x] Serial tail: dependencies of the pair step and of `_assign_support` read in the code.
- [x] Fit-wide memo inventory and repeat counts.
- [x] Scheduler limits.
- [x] Ranked table, rejected list, open questions.

## 0. The two numbers that bound everything

1. **GIL-held time is 29.3 s per fit and it splits 8.4 s main thread / 13.5 s stage lanes / 7.4 s pool threads** (my aggregation of `spy_gil2.txt`; MR11-02 notes py-spy undercounts native calls that keep the lock). The two concurrent sections last about 20.5 s of wall (8.4 + 12.1) and the lanes plus their pools hold the lock for roughly 18-19 s of it: those sections are lock-saturated, which is why a fourth lane (MR11-28) or a narrower/wider pool (MR11-05) changes nothing. The serial stretches (set-up, early_a's serial head and tail, the 17 s tail: about 26 s of wall) hold the lock only about 11 s (8.4 main + roughly 2.5 pool, the pool share being my estimate). **About 15 s of serial wall is spent with no thread holding the lock** (device waits, lock-free kernels, numpy sorts). That is the only place where re-scheduling, as opposed to removing work, can pay.
2. **CPU headroom on this box is small.** INFERENCE from `util_1117c.txt` (CPU 51.8 % of 8 logical = 4.1 logical cores, 17 % = 1.4 cores of it background before the fit) and from the code's own note that the second hardware thread of a core gives 15-25 % (`_adaptive_nbins.py:697`): the fit already uses about 2.8 of roughly 3.8 available core-equivalents. Perfect parallelisation of the unchanged work would give about 1.3-1.7x (47 s -> roughly 30-35 s), consistent with MR11-02(b) (two Python loops side by side slow each other 1.33x). **A 10 s wall therefore needs the work itself to shrink about 3x or move to the card; no host scheduler, process pool or free-threaded build reaches it alone.** The host-side proposals below add up to roughly 7-10 s (47 -> about 38 s) if all land; they overlap partly, so the sum is an upper bound.

## (a) Ranked proposals

| # | What | Where | Mechanism | Est. wall gain | Selection risk | Effort | Verify |
|---|------|-------|-----------|----------------|----------------|--------|--------|
| 1 | Pool bodies as one compiled lock-free kernel per column block | `_orth_auto_scorer_fe.py` (`one_resample`), `_orthogonal_scorer_auto_fe.py` (`_one`), `_oof_warp_fe.py:96` (`_one_column`), `_adaptive_nbins.py` (`_run_chunk`), `_composite_group_agg_fe.py` (`_run`), `_binned_numeric_agg_redundancy.py` (`_run`), `_orth_extra_basis_fe_generate.py` (`prepare`, `_one_column`) | The pools hold the lock 7.4 s; the bodies listed hold 4.6 s of it. Each is 15-30 medium numpy calls per column. One `prange` kernel per family removes the lock from the saturated sections, where a second of lock time is about a second of wall | 2-3 s | low-medium (float order inside reductions; selection-equivalence bar) | L (per family S-M) | per body: old vs new on 30000 x 100, inside an 8-thread pool (MR11-34 rule); then old/new fit alternation |
| 2 | Raw-only parts of `_assign_support` computed on a worker while the pair step runs | `_fe_pure_form_retention.py:89, 350, 770`; `_mrmr_fit_impl/_assign_support_tail.py:180-203` | `_retention_prep`, the form pool + non-separability filter + CV greedy of `retain_usable_pure_forms`, and the CV greedy of `retain_usable_raw_columns` read only raw columns (14 of them, 3000 rows), `y`, the seed and the raw marginal MIs; the estimator state is read only by cheap filters before and after | 1.0-1.5 s | low (same function, same inputs; result discarded when the cheap pre-checks say so) | S-M | assert future result == in-place result on the benchmark and on the retention tests; timeline of `assign_support` |
| 3 | Pair early_a's serial stretches with each other instead of with the saturated lanes | `_mrmr_fit_impl/_fe_stage_cascade_early_a.py:205-223, 364, 476, 509`; `_hybrid_orth_family_variants/_concurrent.py:103-128`; `_fe_stage_cascade_run.py:152-159` | About 6 s of early_a runs on the main thread alone (hybrid main 0.9, extra bases 1.7, ordered folds + three-gate 1.8, CMI-greedy 1.1, dominant check 0.5). All of these stages read the step input. Two-thread pairs in the unsaturated stretches: hybrid main beside extra bases; CMI-greedy and MI-greedy beside the folds / three-gate; the dominant check beside the start of section 2 | 1.0-1.5 s | low (ordered fold already exists for exactly this) | M | stage walls from the existing debug line (`_fe_stage_concurrency.py:211`); same features; `MLFRAME_FE_STAGE_CONCURRENCY=0` parity |
| 4 | Pair loop in three phases: score all pairs, ordered decisions, batched emit | `_feature_engineering_pairs/_pairs_core.py:401-440`, `_pairs_score.py:203-255`, `_pairs_emit_extval.py:223, 315` | The 11 pairs are independent up to the acceptance decision; only the external-validation draw, the record order and the chunk buffer are order-dependent. `_score_one_pair` holds the lock 0.64 s of its 2.7 s | 1.0-1.5 s | medium (generator draw order must be kept) | M-L | per-pair results equal to the serial loop (names, MI, recipes, rejection records in order) |
| 5 | Honest-null and honest-spec work of the raw columns moved under the cascades | `_honest_spec.py:86-150`, `_honest_mdlp_null.py`, `_fit_impl_stages/_binning.py:254` | `_edges_of` reads one column only; the raw half (about 100 of 184 columns) can ride on the existing discretisation warm-up. The null of the raw columns (1.4 s, device time, little lock time) needs the evaluation rows and target codes published at fit start | 0.2-0.3 s (spec) + 1.0-1.4 s (null) | spec: none; null: medium | S (spec), M (null) | spec edges identical; null columns identical to the in-place ones; FDR floor value identical |
| 6 | Ranks of the cross-stage dedup computed beside the accuracy gate | `_fit_impl_stages/_engineered_dedup.py:127`, `_eng_dedup_scan.py:112` | `_precompute_full_ranks` (0.38 s, parallel lock-free kernel) depends on column values only; the gate (0.5 s) only removes columns | 0.2-0.3 s | none (ranks of a dropped column are unused) | S | same dropped set |
| 7 | `_base_trim_select`: one batched standard deviation instead of 107 `np.nanstd` | `_fe_pure_form_retention.py:75` | point fix found on the way: 107 calls, 0.30 s in the profile, on the serial tail (subsumed by #2 if that lands) | 0.2 s | none | S | same `base_names_trimmed` |

## (b) Detail and evidence

### Q1. Where the lock is held, per lane (GIL-held seconds, `spy_gil2.txt`, innermost timed family)

| Thread group | GIL-held | Leaf split (mlframe line / cupy / pandas / numpy) | Largest holders |
|---|---|---|---|
| Main thread (serial parts) | 8.40 | 3.62 / 0.88 / 0.82 / 0.88 | `retain_usable_pure_forms` 0.92; `materialise_and_finalise_fe_candidates` 0.78; `screen_predictors` 0.54 + `compute_fdr_gain_floor` 0.34; `_score_one_pair` 0.42 + pair family gate 0.22; `build_fe_operand_pool` 0.40 (scipy 0.24); greedy CMI 0.38; orth pair family 0.30; `score_prospective_pairs` 0.28 |
| Pool threads | 7.38 (the group-4 variant lane, 1.04 s, was first counted here by my script and is moved to the lanes) | 3.4 / 0.9 / 0.4 / 2.0 | auto-scorer bodies `_one` 1.00 + `one_resample` 0.92 + `<lambda>` 0.36; oof-warp `_one_column` 0.52; `_run_chunk` (edge search) 0.48; composite group agg `_run` 0.40; binned-agg redundancy `_run` 0.40; univariate `_one_column` 0.32; wavelet legs 0.30; extra-basis `_one_column` 0.28 + `prepare` 0.26; honest spec `_edges_of` 0.28 |
| mid_a | 4.50 | 2.52 / 1.06 / 0.44 / 0.18 | pairwise modular 0.90; row argmax 0.72 (cupy 0.46); conditional gate 0.58; cat triple 0.44; composite group agg 0.38 + 0.18; numeric decompose 0.36; grouped agg 0.30; modular 0.20 |
| variant lane "group 3 + tail" | 2.26 | 1.32 / 0.22 / 0.28 / 0.24 | adaptive degree 0.50; routing 0.36; JMIM 0.34; adaptive arity 0.26; dCor 0.24 |
| early_b | 2.18 | 1.36 / 0.20 / 0.42 / 0.12 | binned numeric agg 1.22; k-fold TE 0.26; `_p_added_by_column` 0.22 |
| variant lane "group 1 head" | 1.88 | 1.08 / 0.20 / 0.36 / 0.18 | quadruplet 0.78; triplet 0.48; KSG 0.30 (cupy 0.20) |
| mid_b | 1.70 | 0.78 / 0.26 / 0.16 / 0.34 | untimed cascade body 1.00; conditional quantile rank 0.18; ordinal pattern 0.16 |

Classification of what the lock time is:

- **Compiled kernels are not it.** Of the 121 numba functions that appear in `fit_all.prof`, 119 carry `nogil=True` (11.2 thread-seconds of own time); the two without it (`fast_numba_auc_nonw`, `_fast_r2_score_seq`) total under 0.01 s. Converting more code to numba helps only where it replaces interpreter-level sequences, not where a kernel already exists.
- **It is flat interpreter and call overhead.** The 14.5 s of "mlframe" leaf samples spread over 329 functions and 466 source lines, the largest single line 0.30 s (`_recipe_core.py:85`, `_copy_plain`). By source line: 7.9 s plain Python lines, 3.6 s lines that call numpy, 2.7 s lines that launch or read a device array. Counts in the all-thread profile: `ufunc.reduce` 98638 calls, `np.asarray` 98291, `np.ascontiguousarray` 36366, `ndarray.astype` 24345, `isinstance` 1.4 M.
- **numpy in bulk releases the lock** (sort 2.78 s, partition 3.25 s, argsort 1.04 s of thread time): these are not the problem; MR11-34 showed replacing them inside a pool can lose.
- **Device waits** (`cupy.ndarray.get` 2989 calls, 4.6 s) release the lock; they are the main thread's idle time in the serial stretches.

Consequences:

- *Per-family compiled pipelines*: worth doing for the pool bodies (proposal 1), because a pool body is a per-column Python function run under `map_on_edge_pool` (`_edge_pool.py:33-52`) and the pools' 7.4 s of lock time competes directly with the lanes. `_oof_warp_fe._one_column` (lines 96-125) is typical: `isfinite`, `where`, `quantile`, two `held_out_codes`, `fit_oof_warp1d` (already compiled, `_oof_warp_service._fit_core`), four `plugin_mi_of_codes`, two `paired_gain_se` per column and target; 107 calls, 7.1 thread-seconds cumulative in the profile. The auto-scorer resample bodies launch device work from pool threads (cupy 0.8 s of their 2.3 s), so their fix belongs with the device agent's batching; the host part is the per-resample Python.
- *Worker processes*: see (c).
- *Free-threaded build*: see (d).
- *Coarse loops that dominate a lane*: none. The longest lane (mid_a, 4.5 s of lock) is eight families of 0.2-0.9 s.

### Q2. The serial tail

Chain as coded (`_fit_impl_core_parts.py:740-789` then `:59-183`, then `_fit_impl_core.py:83-141`): cascades -> accuracy gate -> cross-stage dedup -> discretise -> screen 1 -> FE pair step -> screen 2 -> cluster hierarchy -> `_assign_support`. Each stage consumes the previous one's column set, so whole stages cannot be reordered. What can move is the part of a stage that reads only raw columns, and the inside of the pair step.

**`_assign_support` (2.2 s).** Profile (inflated about 1.3x): `retain_usable_pure_forms` 1.23, `run_selection_compaction` 0.55, `retain_usable_raw_columns` 0.47, `_retention_prep` 0.34, `drop_forms_not_beating_their_sides` 0.29, final subsumption 0.16.

- `_retention_prep` (`_fe_pure_form_retention.py:89-135`): numeric raw names, `_base_trim_select` (reads `mrmr.cached_MIs` and the standard deviation of each raw column, lines 72-79), a seeded 3000-row draw. Inputs: raw `X`, `y`, seed, the raw marginal MIs.
- `retain_usable_pure_forms` (`:350-483`): reads the estimator at line 401 (`_existing_form_coverage`, `_engineered_recipes_`), line 409 (`_pure_form_trap_present`, a cheap gate that may return early) and line 477 (`existing_names`). Everything between (lines 423-475: `_build_form_pool`, the non-separability verdicts, `drop_forms_not_beating_their_sides`, `usability_greedy`) takes `X_fit`, `y_fit`, `base_names`, seed and one configuration read (`_retain_usable_pure_relative_screen_under_ranks`, line 227).
- `retain_usable_raw_columns` (`:770-859`): `mrmr.support_` is read only to build the `already` set (lines 823-825); the candidate pool and greedy use `_prep`.
- Decomposition: submit one task after screen 1 (when `cached_MIs` holds the raw marginals) that runs prep, the pure-form block 423-475 and the raw-column greedy on raw data; at the current call site apply the estimator-dependent filters (401, 409, 477, 823-825, and the subsumption guard `_assign_support_tail.py:225-271`) to the returned lists. When the trap pre-check declines, the result is dropped, exactly as today's early return. To verify before building: that screen 2 does not change the raw entries of `cached_MIs` that `_base_trim_select` reads (I did not trace this; if it does, take the snapshot the in-place call would see, i.e. run the task after screen 2 and overlap only cluster hierarchy + compaction, gain about 0.5 s instead).

**Pair step (6.1 s): operand pool 1.3, score pairs 0.9, `check_prospective_fe_pairs` 2.7, materialise 1.5.**

Shared state of the pair loop, read in the code:

| State | Where | Order-dependent? |
|---|---|---|
| running best MI `_sweep_best_mi` | `_pairs_core_helpers.py:383-401` | No. It feeds the progress-bar postfix only, inside `if verbose:`; no decision reads it. MR11-37 listed it as a blocker; it is not one |
| `st._rng_extval` | created `_pairs_core_parts.py:262`; drawn at `_pairs_emit_extval.py:223, 315` | Yes, but only in emit, only for accepted pairs, only when the factor count exceeds the cap |
| rejection records | appended `_pairs_core_helpers.py:355, 376`, `_pairs_score_helpers.py:458, 500` | order of the list only |
| chunk buffers (`_chunk_state`, `_chunk_buffer`, `final_transformed_vals_shared`) | `_pairs_score.py:281-294` | a pair reads its chunk's active buffer; one buffer is live at a time |
| `times_spent` | merged once per pair (`_pairs_score.py:307-348`) | no |
| family gate level `_pair_family_alpha` | computed before the loop (`_pairs_core.py:398`) | no |

Per-pair cost (profile, 11 pairs, 2.61 s): candidate MI sweep 0.59, operand floor 0.18, family gate 0.65 (11 calls), emit 1.16 for the 6 accepted pairs (external-validation codes 0.66 over 24 calls, noise-gate MI 0.24).

Proposed decomposition:

1. Phase A, all pairs: `_score_one_pair` up to and including the family gate (`_pairs_score.py:203-231`). No cross-pair state except the chunk buffer; give each chunk its own buffer or run phase A chunk by chunk as now but for all pairs of the chunk in one scoring call (the device agent's batching).
2. Phase B, serial and cheap: walk the pairs in the original order, append rejection records, and draw the external-validation factors for the accepted pairs from `st._rng_extval` in that order (the draws need only the sorted factor list, lines 223 and 315).
3. Phase C: emit for the accepted pairs with the pre-drawn factors; the 24 `gpu_materialise_extval_codes_host` calls become one per chunk.

Caution from the code's own history: spreading pairs over joblib threads with the device path on was measured at 105 s against 63 s serial (`_step_core_parts.py:825-831`, an older code state, the loss was joblib's sleep-polling). Phases, not threads, are the proposal. The existing multi-chunk route already seeds one generator per chunk call (`_pairs_core_parts.py:262`), so the serial draw order is not an invariant across `n_jobs` today.

**Operand pool (1.3 s)** is the gradient seeder (`rank_gradient_interaction_pairs` 1.27 s, scipy/BLAS). Its operand set comes from screen 1's `selected_vars` (`_step_pool.py:150-157`), so it cannot start earlier; it could run beside `compute_pair_mis_and_floor` if the two are independent (not traced, see (d)).

**Screen FDR floor (2.5 s).** `build_honest_spec`'s per-column work `_edges_of` (`_honest_spec.py:137-145`) reads one column: sort, distinct count, two quantile edge sets. For the raw columns it can be computed during the cascades by the existing warm-up (`start_discretization_warmup`, `_fit_impl_core_parts.py:739`) and stored by column content; `np.sort` releases the lock, so it does not load the saturated sections. The null of the raw columns is MR11-08 / MR11-11's open item: blocked on the evaluation rows and binned target codes being created by the first consumer. Making those two a fit-start service removes the blocker; the device slot question (stage slots cap the pool, MR11-38 hit 456 MB) belongs to the device agent.

**Accuracy gate / dedup.** `scan_engineered_duplicates` spends 0.38 s in `_precompute_full_ranks` before any pair comparison; it can start when the gate starts. The keep-first scan itself must see the gate's survivors (a dropped earlier column must not evict a later twin), so only the ranks move.

### Q3. Redundant work and existing memos

Fit-wide memos that exist and are entered for the whole fit (`_fit_impl_core.py:64`): source-column dedup (`_orth_dedup.py:60`), heavy-tail verdicts (`hermite_fe/_hermite_robust.py:167`), raw-MI baselines and correlation matrices (`_orth_scoring_memo.py:33`). Also present: the univariate basis memo (`generate_univariate_basis_features`, 12 calls), the resident-operand cache (`_content_hash_memoized` 11108 calls for 0.26 s), the edge cache of the discretisation, the honest-null per-column cache across screens (MR11-08).

Repeat counts from the all-thread profile and what a shared service would save:

| Repeated computation | Calls | Thread time | Already shared? | Ceiling |
|---|---|---|---|---|
| `np.quantile` / `nanquantile` | 935 + 214 | 2.9 s | no; callers are distinct uses: `held_out_codes` 220 (oof-warp, per column and target), `quantile_edges` 102, Fourier `core_span` 200, parity 214 | under 0.3 s wall: different rows / quantile sets per caller |
| host `np.sort` | 439 | 2.3 s | no; honest spec 184 (full n), triplet `_quantile_codes` 130 (subsample) | 0.2-0.3 s for the raw half of the honest spec (proposal 5) |
| `np.median` | 1903 | 1.7 s | partly: 984 of them in `_detect_heavy_tail_numpy` / `_robust_scale`, memoised since MR11-20 | small |
| `np.std` | 7115 | 5.2 s | reduced by MR11-31 after this profile | already taken |
| `np.unique` | 3339 | 2.3 s | no; 2939 of them are tiny (quantile internals) | none |
| quantile codes of candidate columns | 2385 columns / 23 calls | 0.79 s | no; 34 % repeated content (MR11-04) | under 0.3 s (MR11-04) |
| device sorts | 447 | n/a | 10 % repeated input (MR11-19) | device agent |
| decision-subsample gather | 36 | 0.39 s | per family | MR11-17 tried and reverted |
| frame-to-matrix `_interleave` | 81 | 1.3 s | no | MR11-37: left with pandas |

Conclusion: the fit-wide shared-service idea has a ceiling of about 0.5 s on the host side. This agrees with MR11-04, MR11-18 and MR11-19; the families compute different statistics on different row sets, and what repeated wholesale has been memoised today.

### Q4. Scheduler

- `run_stages_concurrently` (`_fe_stage_concurrency.py:167-224`): one thread per stage, private estimator copies, merge in stage order. Nothing in it limits occupancy beyond the lock; the per-stage device synchronise (line 202) runs once per stage.
- Pool width: `_resolve_edge_jobs` gives each family pool `cores // active stages` = 2 threads in the concurrent sections and 8 in the serial ones (`_adaptive_nbins.py:714-716`, `_edge_pool.py:16-30`). MR11-05 measured 8 / 4 / 2 as equal.
- `map_on_edge_pool` builds and tears down a `ThreadPoolExecutor` per call (`_edge_pool.py:50`; 308 threads in one profiled fit). INFERENCE: tens of milliseconds in total, not a lever.
- Lanes are static (`_concurrent.py:71-97`, `_fe_stage_cascade_run.py:88-106`); with walls of 11.1 / 11.5 / 12.1 s and 7.2 / 8.4 / 8.2 s there is nothing left to rebalance.
- The structural limits are (1) the lock in the two concurrent sections, (2) **26 s of single-threaded stretches**: early_a's serial head and tail (about 6 s, proposal 3) and the 17 s tail (proposals 2, 4, 5, 6), (3) early_a must finish before section 2 only because of the dominant-candidate early exit (`_fe_stage_cascade_run.py:152-156`); the four-together route (`:29-55`) already shows how to discard later stages when a dominant column appears, so the check itself (0.5 s, mostly lock-free) can run beside the start of section 2.

Proposal 3 detail: every early_a stage takes the step input `X` and is folded with `_fe_merge_new_columns(X_acc, stage(X), X)` (`_fe_stage_cascade_early_a.py:223, 364, 476`); MR11-28 established that CMI-greedy and the extra-basis stage can run on private estimator copies with a by-name fold. That attempt put them beside three saturated lanes and gained nothing. The stretches before and after the lanes are not saturated (main holds the lock about 2.5 s of those 6 s, greedy CMI 0.38 s of its 1.1 s), so pairing serial stages with each other is a different experiment from MR11-28.

## (c) Checked and rejected

- **Worker processes for families.** `import mlframe...mrmr` takes 5.0 s in a fresh process (measured here, warm disk, 3552 modules, cupy loaded), before numba cache loads and a CUDA context. The benchmark runs in STRICT device mode (`cprof_all.py`: `MLFRAME_FE_GPU_STRICT=1`), so there is no CPU-only family to hand to a process, and a second process on the card costs 2.2x on device sorts (MR11-02). The code's own bench found loky 3.9x slower with frame copies (`_step_core_parts.py:803-808`). Would need a persistent pre-warmed pool and CPU-only work units; none of today's units qualifies.
- **Fit-wide binning / sort / rank service**: ceiling about 0.5 s (Q3).
- **Converting kernels to `nogil`**: already done for every kernel that runs in the fit.
- **Second discretisation warm-up beside gate and dedup**: MR11-11, cores contended, reverted. Proposal 6 moves only the dedup's rank pass, which is smaller, but the same contention may apply.
- **Fourth lane / CMI-greedy beside the lanes / per-lane streams / switch interval / pool width**: MR11-02, -05, -19, -28; consistent with the saturation numbers in section 0.
- **Running best MI as a pair-loop blocker**: display only (Q2).
- **Per-call thread-pool construction** as a cost: negligible (INFERENCE from the thread count).

## (d) Open questions

1. **Free-threaded interpreter.** The stack is Python 3.11.5, numpy 1.26.4, numba 0.65.1, cupy 13.6.0, pandas 3.0.3, scipy 1.16.3, scikit-learn 1.8.0 (printed here). A free-threaded build needs Python 3.13t or later and numpy 2.x first, plus free-threaded wheels of numba / llvmlite and cupy; I could not check their availability offline and do not want to state it from memory. Several memos rely on the lock for dict safety. Even if it all works, section 0's core arithmetic caps the gain near 1.3-1.7x on this machine. Cheap test: a venv with the free-threaded interpreter, `import` the package, run the 20000 x 30 warm-up fit.
2. Does screen 2 alter the raw entries of `cached_MIs` read by `_base_trim_select` (proposal 2's start point)?
3. Is the gradient seeder independent of `compute_pair_mis_and_floor` (could save up to 0.2 s more in the pair step)?
4. How much of the 2.7 s of `check_prospective_fe_pairs` is device compute and how much is launch / read latency? The lock is held only 0.64 s; the split decides whether phase batching (proposal 4) gives 1 s or 2 s. Needs the device agent's numbers.
5. The estimates of proposals 2, 3, 5 assume the overlapped work does not slow its neighbour; MR11-11 shows that can fail when both use the pools. Each needs an old/new alternation run (MR11-39 rule).
6. `mid_b`'s 1.0 s of untimed lock time inside the cascade body was not broken down (my script failed on that lane).
7. py-spy and cProfile recordings predate MR11-27..38; a fresh `--gil` recording would re-rank proposal 1's body list.
