# MRMR all-families fit, device side: why the card waits (read-only analysis, 2026-11-17)

Scope: 100000 x 100, all families, GTX 1050 Ti, three lanes, one context. No source file was changed.
Marks: **[read]** = read in the code at the cited line; **[prof]** = read from `D:/Temp/exp/fit_all.prof` (all-thread cProfile, stage concurrency off, thread-seconds, profiler-inflated; taken before MR11-32..38, so line numbers of `_honest_null_device.py` are 20 lower there); **[nvprof]** = `D:/Temp/exp/fit_nvprof.txt` (2026-11-16, `family_profile ALL --n 100000 --p 100 --mode gpu`, one fit under nvprof + cProfile; its counts may include the harness warm-up, so they are an upper bound; the tracker's nsys count is 133000 launches); **[bench]** = microbenchmark of this session (idle card 65-69 C, SM 1721 MHz, one process, synchronised, scripts in the session scratchpad); **[est]** = my estimate, with the arithmetic given.

## Progress

- [x] CUDA.md, tracker MR11-01..40
- [x] profile: device-facing leaves and their callers
- [x] launch/read chains of the hottest callers read in the code
- [x] host-built candidate uploads
- [x] tiny-launch rows that remain
- [x] resident operands, slots, transfers
- [x] kernel-level points, four microbenchmarks
- [x] ranked table

## Answer in short

1. The card is idle because **the host cannot issue work fast enough, and most of what it issues is tiny**. [nvprof] 183941 kernel launches, 12.68 s of kernel time: 126872 of them are cupy / cub / thrust internals with 3.02 s of device time (mean 23.8 us), 57069 are mlframe RawKernels with 9.65 s (mean 169 us); 118272 launches (64 percent) run under 20 us. [bench] issuing one launch costs the host 18.3 us (RawKernel), 31.6 us (cupy elementwise into a given output), 45.9 us (`astype` copy), 55.5 us (elementwise with allocation), 60.5 us (`take`), 70.8 us (a small `cupy.asarray`), 92 to 100 us (a reduction read as a Python scalar), and at 30000 rows the device finishes each of these before the next one is issued (issue 55.5 us, with the queue drained 66.2 us). So a lane that runs cupy chains on subsample-sized columns keeps the card under about 20 percent busy whatever else happens.
2. Device work per fit is about 11 s (23 percent of 49.1 s; [nvprof] 12.7 s of kernels + 1.25 s of copies). The host is blocked on it for about 6 thread-seconds ([prof] `ndarray.get` 4.62 s, cupy sorts 1.39 s); the other half already overlaps host work.
3. **Device-side changes alone are worth about 2.5 to 4 s of wall (5 to 8 percent) and one to three points of occupancy.** With the interpreter lock held about 29 s per fit and 11 s of device work, perfect overlap would still give at most 11 / 29 = 38 percent occupancy; the 10 s target needs the host's operation count to fall, not a faster card (section Q6).
4. The single largest device-bound stretch on the serial tail (the screen's honest null) sorts every continuous column twice (host for the spec, device for the null), sends the sort order device -> host -> device, and recounts a 100 MB label matrix per block. That is proposal 1 and it is worth about 0.8 to 1.1 s.
5. Thermal throttling is not a limiter here (section Q-thermal).

## (a) Ranked proposals

Wall gain: seconds of the 46 to 49 s fit, [est] unless marked. Occupancy: change of the 23 percent mean.

| # | what | where | mechanism | wall gain | occupancy | selection risk | effort | verify |
|---|---|---|---|---|---|---|---|---|
| 1 | One device sort for the honest spec and the honest null; order stays on the card; no label recount; presort floor from a sweep | `_honest_spec.py:137-151`, `_honest_null_presort_gpu.py:23,42-91`, `_honest_null_accept_bits.py:292-298,396-400`, `_honest_null_accept.py:65-74` | the spec's `np.sort` per column and the null's device argsort sort the same filled columns; the null already computes the sorted values and drops them | 0.8 to 1.1 (serial tail, so wall 1:1) | +0.5 pt (same device seconds, shorter wall) | none expected: order and ties measured identical [bench]; edges come from the same sorted values | M | `stage:honest_*` walls, `test_honest_null_device_equivalence.py`, `test_honest_null_accept_bits.py`, old/new alternation |
| 2 | Launch census per family and fusion of the cupy chains and copy kernels it shows | first targets below (2a-2e) | 126872 cupy-internal launches cost about 5.7 s of host issue time for 3.0 s of device time; 46289 are `cupy_copy` | 0.7 to 1.3 | +1 to 2 pt (fewer, larger kernels; wall shorter) | bit-level only where a fused kernel reorders float ops (CUDA.md 19.11 recipe: `--fmad=false`) | M to L, incremental | nvprof summary launch count per family before/after; selection of the seven features |
| 3 | One resident block per bootstrap resample for all seven auto-scorers; KSG scaling and noise on the device; one gather kernel for the sorted values and labels | `_orth_auto_scorer_fe.py:313-335`, `_ksg_cd_fast.py:141-161`, `_ksg_cd_gpu.py:112-137` | the resample is stacked on the host and uploaded up to four times; KSG widens it to float64 on the host (342 MB a fit) | 0.3 to 0.5 | +0.5 pt | selection-equivalent, not bit-identical (device `std` / `mean` round differently; noise is 1e-10) | M | per-scorer value comparison at 1e-9, LCB table equality, family wall |
| 4 | Raw-column marginal MI once per fit | `_extra_fe_families.py:586-610` and its ten callers | ten families rank the same raw columns by the same MI, each with a pandas block copy and an upload; the fit-wide memo exists and is not used here | 0.2 to 0.3 | 0 | none (same values, same tie-break) | S | call counter of `_mi_classif_batch` from `_top_mi_num_cols`; ranking equality |
| 5 | Host cost of the launch path: launch-geometry memo, running byte total of the operand cache, label facts | `pyutilz gpu_hw_launch.block_size_for` (4273 calls), `_fe_resident_operands.py:262-264`, `_fe_resident_operands.py:378-382` | Python executed per launch with the lock held | 0.2 to 0.35 | 0 | none | S | cProfile of the three functions; wall alternation |
| 6 | Reads only where a decision needs them: device-returning scorer entry, one read per loop, scalar gates folded into the next kernel | 6a-6f below | about 1100 of the 2989 explicit reads are per group / per basis / per candidate inside one family call | 0.1 to 0.2 | +0.2 pt | none | M (mechanical, many sites) | counter on `ndarray.get` plus implicit scalar conversions |
| 7 | Fit-wide device column store with row-set handles and per-column facts | new module beside `_fe_resident_operands.py`; consumers of `resident_operand` (106 call lines in 45 files, by grep) | handles replace content hashing; gathers of the subsample / resamples / folds happen on the card; sorted values, quantile edges, moments computed once per column per row set in one batched pass | device side alone 0.5 to 0.8; with host adoption of the facts 3 to 6 | alone +1 pt; with adoption +5 to 10 pt (host numerics become device work) | selection-equivalent where float32 facts replace float64 host quantiles | L | per-family wall, uploads and `np.partition` / `np.sort` / `np.std` call counts |
| 8 | Block sizes from the lane's pool, and counted fallbacks | 36 modules read device-wide free memory, 29 use `device_room` | a block sized from the whole card inside a 456 MB slot runs out of memory and falls back at debug level | unknown; avoids 0.3 to 0.7 s per occurrence (MR11-38) | unknown | none | S (gate) + M (sites) | fallback counter printed at the end of the fit |

Sum of 1 to 6 and the device part of 7: about 2.5 to 4 s. They are not additive with the host agent's proposals where both shorten the same family.

## (b) Detail and evidence

### Q1. Launch / read anatomy of the hottest device paths

Device-facing leaves [prof]: `ndarray.get` 2989 calls / 4.62 s; `cupy.array` 7750 / 2.04 s; cupy `argsort` 333 / 0.96 s; cupy `sort` 262 / 0.43 s; `take_along_axis` 192 / 0.29 s; all cupy functions together 11.8 s; device-named mlframe modules 7.7 s of own time; total profiled work about 116 thread-seconds (148.1 s less 31.9 s of queue / lock waits).

A blocking read on an idle queue costs 0.02 ms more than the launch it follows [bench: 0.108 ms launch + 8-element read against 0.086 ms launch alone]. So the 4.62 s inside `get` (1.5 ms a call on average) is device work finishing, not round-trip latency; removing a read saves wall only when the host has something to do meanwhile or when the read forces a launch chain to be issued in pieces.

Reads that are not in the 2989: `float(cupy_scalar)`, `int(x.max())`, `bool(...)` call `get` from C and appear only as own time of the caller. Example: `_gpu_resident_basis.py:903` `Syy = float((yc * yc).sum())`, [prof] `_gpu_batched_abs_corr` 0.57 s of own time in 100 calls. [bench] such a read costs 92 to 100 us on an idle queue.

Chains (K = kernels launched, R = blocking reads per call):

| chain | calls [prof] | K / R per call | host between | verdict |
|---|---|---|---|---|
| resident plug-in MI `_plugin_mi_classif_batch_cuda_resident` (`_hermite_fe_mi.py:399`) -> `binned_mi_from_values_gpu` (`_fe_batched_mi.py:396-473`) | 642, 1.32 s | cast, transpose, radix select, edge round-up, MI kernel (1 or 2) / 1 (`:466`, `:473`) | none | already one pipeline; the entry always returns a host vector, so callers that loop over groups read once per group (6a) |
| residue grid `residue_grid_mis_resident` (`_pairwise_modular_resident.py:93-150`) | 45 calls -> 360 MI calls, 0.42 s | 1 broadcast `%` + 8 x the chain above / 8 | none | one read per call possible (6a); one launch if the MI kernel takes a bin count per column (d) |
| basis routing `_gpu_route_bases_batched` (`_gpu_resident_basis.py:926-966`) | 72 + 100 abs-corr | per basis: basis block, `mean`, `sum`, abs-corr / 2 (`:903` implicit, `:951`) | Python argmax over `meta` | reads per basis -> one read: max over degrees and first-wins argmax on the device, or stack the per-basis vectors and read once (6b) |
| pair analytic MI `_fused_observed_mi_and_bins` (`_pairs_analytic_gpu.py:262-309`) | 35, 0.63 s | min/max kernel, fused MI (+ finish) / 2 (`:197`, `:308`) | none | the 0.58 s inside the min/max read is the upstream materialise + binning of the pair candidates finishing; the read itself gates only the histogram width. Fold: pass the binner's bin count as `kx` (the `kx` / `ky` pattern of `_fe_batched_mi.py:714-719`) and have the kernel flag a negative code. Saves one launch and one read per pair, not the 0.58 s (6c) |
| KSG `mi_cd_columns_device` (`_ksg_cd_gpu.py:91-141`) | 28, 1.88 s | 4 uploads, transpose, 2-D argsort (thrust, lock held), `take_along_axis` (about 4 launches), label gather, KSG kernel, `sum`, slice write / 1 | host float64 copy, `std`, noise before the call | one pipeline already; the waste is before it (proposal 3) and in the gather (`take_along_axis` 0.31 s / 28 calls) |
| auto-scorer resample `one_resample` (`_orth_auto_scorer_fe.py:313-335`) | 10, 6.46 s | up to 4 uploads of the same block (`:319` -> `:278`, `:324` -> `_orth_mi_backends.py:444`, `:329` -> `_ksg_cd_gpu.py:125`, dCor / HSIC samples) / about 6 | `column_stack` of fancy-indexed columns 0.66 s, KSG prep 1.3 s | proposal 3 |
| honest null `_continuous_accept_block` (`_honest_null_accept.py:77-163`) | 3, 1.65 s | upload, transpose, per-column argsort, `take_along_axis`, tie compare, 2 reads; then 5 uploads, accept kernel / 3 reads | label recount, valid-cut packing, host prep (already overlapped, `:152-154`) | proposal 1 |
| family max-T null `null_max` (`_family_maxt_gate_device.py:227-242`) | 12, 0.22 s | per 256 permutations: bitset upload, kernel parts, `max`, `maximum` / 1 (`:241`) | draws the next block only after the read | read after the loop, or draw block i+1 before reading block i (CUDA.md section 16) (6d); small |
| linear usability `_rows_to_host` (`_binned_agg_linear_usability.py:59-64`, loop at `:233-234`) | 184, 0.43 s | index upload, gather / 1 per candidate | standardise, gain | one stacked gather and one read per pair (46 instead of 184); the row index is uploaded 184 times, cache it (6e) |
| CMI greedy `_cmi_from_binned_cupy` (`_mi_greedy_cmi_fe_gpu.py:174`) | 367 calls, 446 entropy launches, 0.38 s | 1 to 3 / 1 to 3 | none | per-candidate route; `_cmi_gate_scores_batched` (MR11-02) is its batch; left |
| Fleuret confirm `fleuret_confirm_batch` (`_fleuret_confirm_gpu.py:218-320`) | 25, 0.55 s | 3 uploads (440 MB a fit), 1 kernel / 1 | compiled shuffles 0.36 s | fine as it is (see rejected) |

Reads that fetch a scalar or small vector gating the next launch, found by reading: `_pairs_analytic_gpu.py:197` (min/max), `_gpu_resident_basis.py:903` (Syy), `_gpu_resident_basis.py:842,941` (heavy-tail verdict per routing call), `_fe_batched_mi.py:602-603` (`int(y.max())`, `int(C.max())` when the caller gives no bin count), `_orth_mi_backends.py:352` and `_hermite_fe_mi.py` y min/max when `y` is a device array, `_fe_batched_mi.py:665-666` (argmax index then value: two reads where one two-element read would do).

### Proposal 1. Honest screen: one sort, resident order

[read] `build_honest_spec` sorts every numeric column on the host: `_honest_spec.py:140` `v = np.sort(raw[:, j])`, 184 calls on the edge pool; [prof] `_edges_of` 3.05 thread-seconds, of which `np.sort` 1.60 s; the function is 0.53 to 0.60 s of wall on the serial tail (MR11-34).
[read] The null then stacks the same filled columns (`_honest_mdlp_null.py:234`, `spec["raw"][:, c]`) and `presort_and_ties_gpu` uploads them, argsorts each, **computes the sorted values** `sv = cp.take_along_axis(cols, idx, axis=1)` (`_honest_null_presort_gpu.py:63`), uses them for the tie flags only, and copies `idx` and the flags to the host (`:66-67`, 68 MB + 17 MB at 170 columns).
[read] `root_split_accept_gpu` uploads the order again (`_honest_null_accept_bits.py:397`), builds the valid-cut bits on the host from the tie flags (`:398`, `valid_cut_bits` `:280-289`) and calls `accept_fits`, which evaluates `(y_raw_perm == 1).sum(axis=1)` over the whole (999, 100000) matrix (`:297`) for every block, and `:396` once more for row 0. [prof] `accept_fits` 0.26 s in 2 calls. [bench] 31.7 ms at 256 permutations x 100000 rows, i.e. about 124 ms per block at 999.
[bench] accept kernel, 48 columns x 256 permutations x 100000 rows, end to end: 101.9 ms, of it 31.7 ms `accept_fits`, 10.3 ms `valid_cut_bits`, 5.8 ms the order upload; so the kernel is about 50 ms for 12288 pairs (4.1 us a pair, 0.69 s for 170 x 999, which matches the 0.71 s of MR11-35).
[bench] presort, 100000 rows, host (`_presort_columns` + `_tie_flags`, 8 threads) against device (upload and both reads included), identical outputs: 6 columns 22.6 / 8.3 ms, 16 columns 58.0 / 19.8 ms, 40 columns 234.4 / 64.8 ms. [read] the device is used only from `DEVICE_MIN_ELEMENTS = 5_000_000` (`_honest_null_presort_gpu.py:23,39`), a constant that predates the per-vector sort of 2026-11-07; a block under 50 columns goes to the host ([prof] one `_presort_columns` call, 0.27 s).

Change, in steps that each stand alone:
- (1a) `accept_fits` / `accept_fits_global`: every row of `y_raw_perm` is a permutation of the same labels, and the packed bits are already cached (`_pack_bits_cached`, `_honest_null_device.py:257`); take the count from row 0 (or a popcount of the packed rows when the invariant must be checked) and cache it with the packed bits. About 0.25 s.
- (1b) `use_device_presort`: replace the constant by a tuner entry swept in the host-in / host-out placement (CUDA.md section 4); on this host the device wins from 6 columns. Up to 0.2 s when a block has fewer than 50 columns.
- (1c) Presort before the spec: run the device presort over all numeric columns first, hand the sorted values to `_edges_of` (one 74 MB read for 184 float32 columns, or the distinct count and the two edge sets taken on the device by position), keep `order` (int32) and the tie flags on the card, build the valid-cut bits there (one small kernel: `tie == 0 & in_range`, packed 32 to a word), and fetch a host copy of `order[j]` only for the columns of accepted pairs (`_redo_accepted`, `_honest_null_device.py:553-581` reads it only when `items_b.size`). Removes the 184 host sorts (about 0.5 s of wall) and the 85 MB round trip (about 0.1 s).
- Constraint: the presort handles finite float32 / float64 columns (`:45`); others keep today's path. On the serial tail no slot cap applies (default pool): 184 x 100000 x (4 + 4 + 4 + 1) bytes = 239 MB peak for values, order, sorted values and flags, in column blocks as today.

Not gained here: the accept kernel itself (see rejected: the precomputed variant is slower at this row count; the tracker has its counters at 75 percent occupancy).

### Proposal 2. Launch census and fusion

[nvprof] by family of cupy internals: `cupy_copy` 46289 launches / 0.52 s device; cub + thrust 25413 / 1.21 s; `cupy_take` 5358; `cupy_multiply` 5306; `cupy_subtract` 3978; `cupy_add` 3569; `cupy_isfinite` 3394; `cupy_true_divide` 2924; `cupy_cub_all` 2 x 2768; `cupy_prepare_array_indexing` 2147; `cupy_where` 2087. RawKernels with many small launches: `radix_interp` 4170 at 2.4 us, `mlframe_edges_rounded_up_f32` 3837 at 1.8 us, `minmax_i8` 2579 at 14 us, `analytic_mi_from_hist` 2565 at 36 us.
[est] host issue time = 126872 x 45 us + 57069 x 18 us = 5.7 + 1.0 s (4.1 + 0.7 s with the tracker's 133000 launches in the same proportion), nearly all of it with the lock held. Removing a third of the cupy-internal launches returns 1.4 to 1.9 s of lock-held time; inside the cascades lock-held time is wall almost 1:1 (MR11-02), on the tail fully.

First targets, each read in the code:
- (2a) `resident_code_operand` returns `dev.astype(cp.int64)` on every call (`_fe_resident_operands.py:382`): 752 widening copies a fit [prof], and the histogram kernels then read 8-byte codes. Let the consumers take int16 / uint8 codes as they are (CUDA.md 19.8, 19.10; `_batched_cmi_gpu_once` already keeps a byte matrix, `_fe_batched_mi.py:697`), or cache the widened array beside the narrow one.
- (2b) `radix_interp` + `mlframe_edges_rounded_up_f32` after every radix select (4170 + 3837 launches, 2 us of device time each, about 18 us of host each): emit the rounded float32 edges from the select kernel's epilogue or from one interp kernel; about 0.14 s of host time.
- (2c) `take_along_axis` after an argsort (`_ksg_cd_gpu.py:127-128`, `_honest_null_presort_gpu.py:63`): about four launches and two index temporaries where one gather kernel writing the sorted values (and the labels) does it; [prof] 0.31 s / 28 calls in KSG alone.
- (2d) `cp.stack(cols, axis=1)` of resident operands (`_fe_resident_operands.py:333,350`): one copy launch per column; with proposal 7 the block is one gather.
- (2e) the per-call `isfinite().all()` checks (3394 + 2768 launches): the verdict is a fact of the column; carry it with the operand (proposal 7) instead of recomputing it per consumer.
The census itself: run `nvprof --print-gpu-summary` once per family with the profiler started around the family (the `fit_nv.py` pattern) and rank the families by cupy-internal launches; the profile here cannot attribute launches to families.

### Proposal 3. Auto-scorer resamples on the device

[read] `one_resample` builds `block = np.column_stack([c[idx] for c in columns])` (`_orth_auto_scorer_fe.py:316`), then: `_device_block(block)` uploads it and transposes (`:261-282`); `_mi_classif_batch(block, yb)` uploads it again (`_orth_mi_backends.py:444`); `ksg_mi_discrete_target_columns(block, yb)` makes a float64 host copy, `std`, divides, adds `1e-10 * means * one_draw[:, None]` (`_ksg_cd_fast.py:148-158`: **one noise vector of n values is shared by all columns**) and uploads the float64 matrix (`_ksg_cd_gpu.py:125`); dCor and HSIC upload their row samples (`_dcor_gpu.py`, `_hsic_gpu.py`: 28 + 42 uploads).
[prof] `one_resample` 6.46 thread-seconds in 10 calls: KSG 2.62 (host prep 0.53 own + 0.42 `std` + 0.14 copy + 0.12 `unique` + 0.09 `mean`; device 1.68: argsort 0.57, `take_along_axis` 0.31, upload 0.29, transpose 0.16, read 0.15, own 0.22), `column_stack` 0.39, the gather list 0.27, plug-in MI 0.14, device block 0.11. Counted uploads: `mi_cd_columns_device` 342 MB / 28 calls (MR11-17).
Device-born form: the family uploads its columns once as a `(k, n)` float32 block (or takes it from the store of proposal 7); a resample is `block[:, idx_dev]` (one gather, the index is 60 KB); the plug-in MI takes it through its resident entry (`_strict_resident_mi` accepts a device matrix, `_orth_mi_backends.py:436-447`, and `x_is_cm=True` exists in `_plugin_mi_classif_batch_cuda_resident`); KSG computes `std`, the scale and the noise on the device from the uploaded n-vector `one_draw` (0.12 MB instead of the n x k float64 matrix) and sorts; dCor / HSIC gather their 500 rows on the device.
What it needs: a device entry of `ksg_mi_discrete_target_columns` that takes a resident float32 block (widened on the card); `tail_dependence_columns`, xi and copula already take `device_block`.
Gain [est]: about 2.2 thread-seconds on the resample pool (stack 0.66, KSG host prep 1.3, uploads 0.3); the family is 1.7 s sequential on one lane, so 0.3 to 0.5 s of wall.

### Proposal 4. Raw marginal MI once (Q2, the largest repeated upload source found)

[read] `_top_mi_num_cols` (`_extra_fe_families.py:586-610`) copies the numeric block out of the frame (`X[...].to_numpy`, `:607`) and calls `_mi_classif_batch(block, y_bin, nbins=10)` (`:608`). [prof] 10 calls, one from each of `hybrid_sir_direction_fe`, `hybrid_random_fourier_fe`, `hybrid_conditional_residual_fe`, `hybrid_mahalanobis_density_fe`, `hybrid_offset_product_fe`, `hybrid_ordinal_pattern_te_fe`, `hybrid_conditional_dispersion_fe`, `hybrid_rankgauss_fe`, `hybrid_lof_fe`, `hybrid_conditional_quantile_rank_fe`: 0.81 thread-seconds (0.42 pandas copy, 0.34 upload + MI), 20 uploads of 64-column blocks.
[read] the fit-wide memo of raw-column MI baselines exists (`_orthogonal_univariate_fe/_orth_scoring_memo.py:58` `cached_raw_mi_baseline`, entered for the whole fit in `_mrmr_fit_impl/_fit_impl_core.py:64`) and `_top_mi_num_cols` does not call it. Route it through the memo (key: column content, target, bin count 10) or through `_resident_raw_mi.resident_raw_baseline_mi`.

Q2 inventory, callers of `_mi_classif_batch` -> `_strict_resident_mi` that upload a host-built matrix [prof: 160 calls, 110 uploads, 1.12 s; 816 MB / 112 calls in MR11-17]:

| caller | calls | what is uploaded | device twin and why it is bypassed |
|---|---|---|---|
| `_top_mi_num_cols` (`_extra_fe_families.py:608`) | 20 | raw columns, 64 at a time, by ten families | memo exists, not wired (above) |
| `_replicate_mi` slow columns (`_orthogonal_bootstrap_mi_fe.py:166`) | 40 | `sub[idx, :]` per resample for tied columns | the fast route `bootstrap_rank_mi` covers distinct columns only; a device-born resample needs the block resident and a device gather (proposal 3's block) |
| `cheap_row_argmax_scan` -> `_mi` (`_pairwise_modular_fe.py:144-169`) | 40 | one column per call, rank binning | `_mi` accepts a device column (`:164`); the scan hands host columns; batch through `_mi_columns` (`:175`), as MR11-13 did for the ranking |
| `one_resample` (`_orth_auto_scorer_fe.py:324`) | 10 | the resample block, a second time | proposal 3 |
| `_score_plug_in` (`_orth_auto_scorer_fe.py:55-86`) | 10 | one column | per-column fallback of the batch; left |
| `local_mi_gate` (`_unified_fe_gate.py:398`) | 12 | each family's engineered candidates + raw sources | the candidates are born on the host by 11 families (TE, RFF, SIR, LOF, ...); a device generator per family is the L-sized part; the raw half can come from the memo |
| `_lattice_grid_mi` (`_integer_lattice_fe.py:199`) | 10 | residue grids | `residue_grid_mis_resident` is the twin, used by the modular family only |
| `_batch_one_scorer`, `_engineered_mi_scores`, CMIM, JMIM, greedy | 8 | candidate blocks | one call each; left |

Inside `_strict_resident_mi` itself: `np.ascontiguousarray(X)` 0.26 s + upload 0.19 s of the 1.12 s [prof], i.e. 40 percent of the call is getting the matrix to the card.

### Proposal 5. Host cost of the launch path

- [prof] `block_size_for` 4273 calls, 0.44 s cumulative (`_kernel_resources` 0.19 s, `_max_active_blocks_per_sm` 10794 calls): the launch width is recomputed for every launch from the kernel's registers and the device limits. Memoise per (kernel, dynamic shared bytes, power-of-two bucket of `work_items`).
- [read] `_fe_resident_operands.py:263`: after every miss the byte budget loop sums `nbytes` over all entries under the lock; [prof] the generator ran 213357 times, 0.24 s. Keep a running total.
- [read] `resident_code_operand` (`:377-381`): `host.max()` / `host.min()` and `ascontiguousarray` on every call (752 calls); the range is a fact of the content, memoise it with the hash.
- [prof] `sort_room` 890 calls 0.11 s (`_device_sort_safe.py:100-128`: an allocation of the need and its release per sort); keep it, it is the guard against thrust aborting; listed for completeness.

### Proposal 6. Reads

- (6a) give `_plugin_mi_classif_batch_cuda_resident` a device-returning mode (`binned_mi_from_codes_gpu` has `as_device`, `_fe_batched_mi.py:568,622`; `binned_mi_from_values_gpu` ends in `cp.asnumpy` at `:466` / `:473`) and read once per caller loop: residue grid (360 reads -> 45), `gate_grid_mi_resident` (26), `best_existing_op_mi_resident(_many)` (66), `_resident_mi` of the cross basis (24), routing (36).
- (6b) `_gpu_route_bases_batched`: one read for all bases; `Syy` passed as a device scalar or computed in the abs-corr kernel.
- (6c) `_minmax_i8` folded (above). (6d) `null_max` (above). (6e) `_rows_to_host` batched per pair (above).
- (6f) `cmi_device_argmax` (`_fe_batched_mi.py:664-666`): two scalar reads -> one two-element read.
Gain is small because a read on an idle queue is cheap here [bench]; the value is that the host stops slicing its launch stream into pieces, which matters once proposal 2 has shortened the stream.

### Proposal 7. Device column store (Q4)

What the cache does today [read `_fe_resident_operands.py`]: content-keyed (`:243`: device, shape, target dtype, source dtype, contiguity, xxh3 of the buffer), LRU of 192 entries (`:177`), byte budget a quarter of the card (`:181`), entries allocated from whichever lane's pool uploaded them. [prof] `resident_operand` 10106 calls, 1.20 s cumulative, 1192 uploads (0.26 s); `assemble_resident_matrix` column lookups 1486 in 23 calls, 0.65 s; MR11-17 / ro_fit: 948 distinct contents, 260 re-uploads.
Are uploads repeated because of eviction or slot caps? Eviction: 260 of 1208 uploads, about 52 of 241 MB, about 0.06 s; not a cost worth a larger cap (and `raised_capacity`'s note at `:272-273` records that a fit-wide larger cap took the lanes' memory). Slot caps: no upload is repeated because of them; they cause fallbacks (proposal 8).
Why 948 contents for 100 raw columns: the key is content, so every row set (the decision subsample gathered 36 times, [prof] `_gathered_subsample`; resamples of n/2 and 0.8 n rows, `_orth_auto_scorer_fe.py:210`, `_orthogonal_bootstrap_mi_fe.py:172`; fold splits), every dtype pair, and every engineered column is its own entry.
What consumers request [read]: bin counts in the filters package by literal use: 10 (179 sites), 20 (63), 12 (21), 5 (11; the honest fallback `_FALLBACK_NBINS = 5`, `_honest_spec.py:20`), 4, 8, 16, plus data-dependent ones (`max(nbins, k)` of the modular grid, MDLP cuts); dtypes requested from `resident_operand` (grep of the `dtype=` argument): int64 (25 sites), int32 (11), float64 (7), the criterion dtype (float32 by default, 4), uint8 (3), float32 (2), a caller-supplied variable (10); row sets: full n (honest null, dominant candidate, Fleuret, pair step), the 30000-row decision subsample, resamples, 500-row dCor / HSIC samples, held-out halves.
Judgement: **one store can serve the values, not one table of codes.** Codes at 10 / 12 / 20 quantile bins are one gather each from a per-column sorted array (edges by position, codes by counting edges <= x, the `_qbin_block_gpu` kernel), so the shared object is `(row set) -> float32 (k, n) block + sorted values + order`, and codes are derived per request and cached by (column, row set, bin count). The tracker already measured that repeated quantile binning is 34 percent of 0.79 s and repeated device sorts 10 percent of the sorted elements (MR11-04, MR11-19), so the saving in sorts and codes is under 0.4 s; the saving that matters is (i) no hashing / dictionary walk / stack per consumer (about 1.6 thread-seconds: `resident_operand` 1.20 + the 0.40 of the assembly that is not inside it), (ii) no host gather + upload per row set, (iii) the column facts that host code recomputes with numpy: [prof] `np.partition` 3.25 s (through `quantile` / `percentile` / `_detect_heavy_tail_numpy` / `_robust_scale`), `np.sort` 2.78 s (1.60 of it proposal 1), `_var` 5.17 s cumulative in 7115 calls, `np.unique` 2.23 s. One batched device pass per block of columns (a `(k, n)` sort along the rows: 54 ms for 256 x 20000, CUDA.md section 16) yields min, max, mean, std, the finite flag, the distinct count, the median / MAD and any quantile grid.
Sizes: 100 raw float32 columns x 100000 rows = 40 MB; with order (int32) and sorted values 120 MB; the 30000-row subsample of 500 columns with both = 180 MB. It fits beside three 456 MB slots only if the store has its own budget outside the slots (today the operands sit in the slot pools, `_fe_device_scheduler.py:107-108`).
This is the L-sized structural item and overlaps the host agent's "fit-wide shared services"; the device half is the block layout `(k, n)` C-order (CUDA.md section 15), the facts pass and the handle API.

### Proposal 8. Pools and slots (Q4)

[read] a lane allocates from a private pool whose limit is its budget (`_fe_device_scheduler.py:113-125,129-138`; minimum 256 MB, cushion 1 GiB, `:25,55`); `device_room` (`_device_sort_safe.py:63-66`) is the room of the calling thread's pool. 29 modules size their blocks from `device_room` / `pool_column_block`; 36 still read the device-wide free memory, among them modules that run inside lanes: `_dcor_gpu.py:86`, `_hsic_gpu.py:100`, `_hermite_fe_mi.py:274`, `_gpu_resident_select.py:39`, `_mi_greedy_cmi_fe.py:511`, `_family_maxt_gate_device.py:144`, `_fe_accuracy_gate_batched.py:123`, `_binned_numeric_agg_resident.py:649`, `_batch_mi_noise_gate_bits.py:253`, `_gpu_batched.py:132,414`. A block sized that way can exceed the lane's limit and fall back, mostly at debug level (`_fleuret_confirm_gpu.py`, `_pairwise_modular_resident.py:149`, `_pairs_analytic_gpu.py`), which is how the MR11-38 failure stayed silent. Today's logs (`util_1117b.txt`, `util_1117c.txt`) carry no out-of-memory line at warning level; debug-level fallbacks are not visible in them (open question 2).
Proposed: a meta-gate that a device module sizes blocks through `device_room` (py-ci-shared, since it is a generic detector), and one counter of device fallbacks per reason reported at the end of the fit.
Pinned staging: `install_pinned_policy` is called per `resident_operand` (`_fe_resident_operands.py:229`). [bench] uploads: 240 KB 0.103 / 0.092 ms (default / pinned policy), 2.4 MB 0.46 / 0.55 ms, 80 MB 16.8 / 13.5 ms (4.8 / 5.9 GB/s). The fit's 4266 MB would take about 0.9 s at these rates plus about 0.1 ms per call; the 3.3 s measured in the fit is the same work under three lanes. Transfers are bound by the call count (6935), not by bandwidth; nothing to gain from staging.

### Q3. Small launches

The detector's table (`tiny_cupy_high_triage.md`: 42 rows marked REAL) was re-measured on 2026-11-16; in [prof] none of its REAL rows exceeds 0.53 s cumulative: `cheap_row_argmax_scan` 0.53 s (1 call; MR11-13 batched its ranking, the scan's 40 single-column `_mi` calls remain, table above), `build_binagg_oof_matrix_gpu` 0.45 s (13 calls), `residue_grid_mis_resident` 0.42 s (360 MI calls), `gate_grid_mi_resident` 0.40 s (26 calls, 800 tau blocks and 800 small uploads inside the function that starts at `_resident_candidate_mi.py:421`), `_build_usability_cand_step3_pidx_n1_n2` 0.28 s, `generate_conditional_dispersion_features` 0.25 s, `_prepass_gate_and_usability_candidates` 0.23 s, `_null_max_strict_gpu` 0.18 s (one label upload and one MI per permutation), `_gate_round` 0.11 s. `detect_fourier_freqs_for_col_gpu`, `_score_candidates_one_by_one`, `_per_candidate_cmi_argmax` are not reached. So the per-item loops that a detector can see are no longer where the launches are: the 126872 cupy-internal launches come from straight-line cupy code inside functions called a few hundred times (proposal 2), which the loop detector does not flag.
Batched forms where a row is worth it: `gate_grid_mi_resident`: one tau matrix upload per call instead of one per spec (800 -> 26); `residue_grid`: per-column bin count in the MI kernel (open question 3); `_null_max_strict_gpu`: permuted labels as one `(P, n)` upload and one launch over (permutation, column), the `null_mi_perms_bits` pattern.

### Q5. Kernel level (new points only)

- Accept kernel: no new lever; the precomputed variant was tested and rejected (below).
- KSG: `{REAL}` is double by contract and the 2-D float64 argsort holds the lock (0.57 s / 28 calls). No kernel change proposed; the gain is in the gather (2c) and in the upload (proposal 3).
- `analytic_mi_from_hist` (2565 launches at 36 us) and `minmax_i8` (2579 at 14 us) run once per pair call around `analytic_obs_mi_bins_i8`: three launches where the fused kernel could finish its own histogram when `splits == 1` already does; with the bin count passed in, the split route is two launches and no read before them.
- `radix_interp` / `mlframe_edges_rounded_up_f32`: 8007 launches of about 2 us (2b).
- Integer codes widened to int64 before histogram kernels (2a): CUDA.md 19.8 / 19.10 measured 1.3 to 1.6x on the kernels that read them.

### Q6. Is 23 percent the ceiling?

- Device seconds per fit: about 11.3 (23.0 percent of 49.1 s). Lock-held host seconds: about 29 (MR11-06). While the host holds the lock and is not issuing launches, the card has nothing queued (MR11-19: per-lane streams change nothing, the queue is empty).
- Upper bound with today's work split and perfect overlap of every device second with host work: wall >= 29 s, occupancy <= 11.3 / 29 = 39 percent. Device-side changes do not move the 29 s except through the launch path (proposals 2 and 5: about 1 to 1.6 s of it).
- On the serial tail a device second is a wall second: honest null about 1.0 s of kernel + 0.3 s of presort, pair analytic about 0.6 s, the confirm batches about 0.2 s. Proposal 1 removes the host work wrapped around the first; the kernels themselves stay.
- So: **device side alone, about 2.5 to 4 s of wall; occupancy 23 -> about 24 to 26 percent.** Occupancy is the wrong meter for these changes: fewer launches lower the device seconds as well as the wall.
- What would unlock more, in order: (i) host numerics on subsample columns served as device facts (proposal 7 with adoption: about 11.8 thread-seconds [prof: `partition` 3.25 + `sort` 1.18 outside the honest spec + `_var` 5.17 + `unique` 2.23] become a few hundred milliseconds of device work and stop holding the lock; that is the only item here that raises occupancy and cuts wall together); (ii) families written as "issue everything, then collect" so that a lane issues the next family's device program while the previous one's kernels run (needs the device-returning entries of proposal 6 first); (iii) candidate generation on the device for the 11 families behind `local_mi_gate` (Q2 table), which turns their host-built matrices into resident blocks.

### Q-thermal

`util_1117c.txt`: before the run 70 C, SM 1708 / 1911 MHz, no throttle; over the run mean 75 C, max 86 C, **mean SM clock 1702 MHz**, `sw_thermal` seen in at least one sample; CPU 2808 / 2808 MHz. This session, idle and cool (65 to 69 C, no throttle reason): **SM 1721 MHz**. The card does not reach 1911 MHz even idle, so 1911 is not the reference; against the idle 1721 the run's mean clock is 1.1 percent lower. With the card busy 23 percent of the time, that is under 0.1 s of wall [inference from two clock readings; the per-sample clock series is not in the file]. The risk is on the other side of the shared chassis: MR11-38 / 39 record the processor's performance counter between 52 and 124 percent and a 3 s drift of the same code, and the fit is host-bound, so a hot card can cost wall through the CPU's frequency, not through its own clock. Recording the mean CPU frequency per run (not the nominal 2808 MHz that `wmic` prints) would show it.

## (c) Checked and rejected

- **Precomputed sorted-space accept variant at 100000 rows** (`MLFRAME_HONEST_ACCEPT_MODE=pre`): [bench] 119.0 ms against 101.9 ms for the shared variant (48 x 256 x 100000, five interleaved rounds, outputs identical). The default is right below the shared budget.
- **Larger operand cache**: eviction costs about 0.06 s a fit (above).
- **Fit-wide shared sort order / code table for the cascades**: ceiling under 0.4 s by the tracker's own counts (MR11-04, MR11-19); the one real duplicate is host-versus-device and is proposal 1 (the device-sort counter could not see the host `np.sort`).
- **Removing blocking reads for their latency**: 0.02 ms a read on an idle queue [bench]; the 4.62 s inside `get` is device work.
- **Per-lane streams**: consistent with MR11-19; an empty queue has nothing to reorder.
- **Fleuret shuffles on the device**: the shuffle is a sequential generator walk per job (`_fleuret_confirm_gpu.py:127-153`), the class CUDA.md section 16 measured as not faster on the device; its uploads are 0.09 s.
- **`_minmax_i8` as a 0.58 s saving**: it is the wait for the upstream kernels; folding it saves two launches' issue time per pair.
- **KSG on float32 sort keys**: the noise is 1e-10 relative; float32 keys would change the neighbour order (an estimator change, not selection-neutral by construction).
- **Pinned staging for uploads**: call-count bound (above).
- **Thermal throttle of the card**: about 1 percent of clock (above).
- **Tiny-loop REAL rows**: all at or under 0.53 s (Q3).

## (d) Open questions (no evidence yet)

1. Launch count and cupy / RawKernel split of **today's** fit per family. The 183941 comes from the 2026-11-16 nvprof file (possibly with a warm-up), the tracker says 133000 from nsys; proposal 2's ranking of families needs the per-family census.
2. How many device paths fall back to the host in a three-lane fit, and why. Warning-level logs are clean; most fallbacks log at debug. A counter would settle whether slot caps cost anything today.
3. A plug-in MI kernel with a bin count per column (one launch for the residue grid's eight groups and for mixed-bin callers): the shared tile is sized by the largest count; not measured.
4. `cupy.argsort` of a 2-D float64 block below 80000 rows (KSG, 0.57 s): whether a composite-key single sort or per-vector sorts are faster at 15000 x 100; the 80000-row floor of `_device_sort_safe.py:150` was measured on float32.
5. Whether `build_honest_spec`'s edge functions (`_edges_from_quantiles`, `_fallback_edges`) return identical edges from device-sorted values when a column holds both signed zeros (they compare equal; the positions can differ).
6. Which block had the host presort in today's fit (the 0.32 s of MR11-35): the block widths of `_null_columns_blocks` were not read from a run.
7. Share of the 29 lock-held seconds that is launch issuing in the fit (my 4.8 to 6.7 s is count x isolated cost; under three lanes the per-launch cost is higher, MR11-02 measured 1.4x for a small call with a second process).
8. `cheap_row_argmax_scan`'s 40 single-column rank-binned MI calls as one `_mi_columns` batch: the source says the launch count is kept on purpose (triage table); whether that reason still holds was not checked.

## Microbenchmarks of this session

| what | result |
|---|---|
| accept kernel, 48 col x 256 perm x 100000 rows, shared / precomputed | 101.9 / 119.0 ms, outputs identical |
| `accept_fits` / `valid_cut_bits` / order upload inside that call | 31.7 / 10.3 / 5.8 ms |
| presort host / device, 100000 rows, 6 / 16 / 40 columns | 22.6 / 8.3, 58.0 / 19.8, 234.4 / 64.8 ms, identical |
| upload 240 KB / 2.4 MB / 80 MB (default allocator) | 0.103 / 0.462 / 16.8 ms |
| launch + 8-element read against launch alone | 0.108 / 0.086 ms |
| host issue cost per call at 30000 elements: RawKernel / elementwise with `out=` / `astype` / elementwise + alloc / `take` / reduction / reduction read as float / small `asarray` | 18.3 / 31.6 / 45.9 / 55.5 / 60.5 / 54.7 / 92.2 / 70.8 us |

Card state during these: 65 to 69 C, SM 1721 / 1911 MHz, no throttle reason, 3 to 4 percent background utilisation, about 640 MB in use by other processes; each run held the card for a few seconds and under 250 MB.
