# Launch counts of our compiled kernels in a 100000 x 100 every-family fit (2026-11-02)

Method: `D:/Temp/launch_count.py` (a Python subclass of `cupy.RawKernel` plus a counting proxy around `RawModule.get_function`, installed before mlframe imports; warm fit at 20000 x 30 first, then one counted fit at 100000 x 100; wall 145.9 s). It counts only compiled `RawKernel` / `RawModule` kernels: cupy's own elementwise kernels, ufuncs, sorts and reductions are cdef types whose `__call__` cannot be patched, and nsys exports of the whole fit fail on this host (see memory `nsys-export-disk-limits`).

Result: **15,856 launches of our own kernels in the whole fit**, at roughly 10 to 30 us of launch cost each that is under 0.5 s of the 146 s wall. Our compiled kernels are NOT the launch-bound part. The 14 million events nsys recorded for the same fit are therefore cupy's built-in ops (elementwise, sort, reduction, memset, copies), which is where per-column device loops and small temporaries live.

Top kernels: `radix_select_f32_v3` 1021, `radix_interp` 1021, `ent_nnz_1d` 902, `mi_hist_split_f32` 879, `mi_from_counts_col_f32` 879, `cmi_joint_entropies` 864, `periodogram_power` 760, `joint_hist_nnz2` 654, `joint_hist_nnz3` 643, `mi_hist_split` 530, `build_op_cands_cm` 480.

Top callers (launches): `_hermite_fe_mi._plugin_mi_classif_batch_cuda_resident` 3024, `_hermite_fe_mi._resident_mi_candidates` 2411, `_fe_batched_mi_cmi.joint_entropy_gpu` 1804, `_mi_greedy_cmi_fe_gpu._nc` 1308, `_mi_greedy_cmi_fe_gpu._cmi_from_binned_cupy` 1047, `hermite_fe._als_fused_gpu.als_sweep_fused` 742, `_binned_numeric_agg_resident._pair_fold_stat_tables` 696.

Next: find where the cupy built-in ops are issued (a `sys.setprofile` or `cupy.cuda.runtime` hook, or an nsys window of one stage with `--capture-range=cudaProfilerApi`), then replace per-column elementwise sequences (like the one batched in `_qbin_block_gpu`) by one block operation. `joint_entropy_gpu` (1804 launches) and `_nc` / `joint_nnz_gpu` (1308 plus an allocation each) are the first of our own call sites worth batching per candidate block.

## Allocation counts as a proxy for cupy built-in ops (same fit, `D:/Temp/alloc_count.py`, a `cupy.cuda.memory_hook.MemoryHook` counting `malloc` by the nearest mlframe frame)

101,476 device allocations in the 100000 x 100 fit (wall 142.7 s). Top sites: `build_leg_product_matrix_gpu` 13,444 (3.2 GB), `_usability_greedy_gpu_resident._cv_candidates` 4,504, `gate_grid_mi_resident` 4,052 (7.6 GB), the greedy-CMI candidate binning 3,730, `abs_corr_finite_gpu` 3,354, `_radix_select_interior_edges` 3,340, `resident_code_operand` 2,918.

Check on the biggest one: `build_leg_product_matrix_gpu` evaluates one basis leg per call; 60 legs of 30000 rows cost 52 ms (0.85 ms per leg), and the batched `_gpu_evaluate_basis_matrix` takes ONE degree list for all columns (it returns columns x degrees, 3600 columns for 60 x 60), so it cannot replace the per-leg calls and was 87 ms against 52 ms. About 700 legs per fit is under a second: a high allocation count is not a high wall cost. Rank by sampled wall (`idle_frames_agg.py`), use the counts only to see which loops are per-column.

## Done: `gate_grid_mi_resident` builds each tau-grid block in one fused kernel

The column-major tau-grid block was three expressions (compare, cast to double, multiply by the operand, or `where`) followed by the cast to the block dtype: two double temporaries of (k, n) per spec and four launches. `_tau_grid_kernel` computes `(T)((c > t ? 1.0 : 0.0) * a)` (mask) or `(T)(c > t ? a : b)` (select) in one elementwise kernel writing the final block; the result is bit-identical including NaN, +-inf and zero operands (`test_gate_tau_grid_kernel.py`, byte comparison). 11 mask + 11 select specs of 9 taus at 100000 rows: 28 ms -> 20 ms for the whole call (same MI sum). The row-major layout used by the rank-binning kernel keeps the old expression.

## Done: `build_binagg_oof_matrix_gpu` writes its candidate block in place (peak pool 1.79 GB -> 1.02 GB)

Sampled pool use per stage (`idle_frames_agg.py`, `used_bytes` / `total_bytes` of the cupy pool each sample): `_stage_binned_numeric_agg` held 1793 MB live of a 1795 MB pool, every other stage at most about 730 MB. The pool limit is free VRAM at start minus a 1 GB cushion (1.94 GB when the desktop holds 0.9 GB), so with the card shared the stage ran into the limit, the device gate declined and the exact host path took 79 s instead of 6 s. The builder collected every float64 OOF column in a list, then cast each and stacked them: about three copies of the block at the peak. It now allocates the (K, n) block once in the candidate dtype and writes each column into it as it is built (the assignment casts like the old `astype`). Peak live 1793 -> 1019 MB, pool total 1795 -> 1348 MB; the 48 binagg tests (host parity included) pass. Together with the retry after releasing the caches (`_binned_numeric_agg_cands`) a low pool limit no longer sends the stage to the host.

## Done: ext-val factor matrix packed once per factor set (`_pairs_emit_extval._packed_factor_matrix`)

`_extval_host_buffer` repacked the external factors into an `(n, E)` matrix on every pair: 24 calls, 2.4 s in the 100000 x 100 fit (about 100 ms each at E = 106). Instrumenting the calls (`D:/Temp/extval_rep.py`): only 5 distinct factor sets (the same memoised arrays), consecutive calls overlap 100%, one set repeated up to 9 times. The packed matrix of the last call is kept (the arrays are held so identity stays valid, `is` is checked per factor) and reused when every factor is the same object; a fill goes through contiguous rows and one transposing copy (264 -> 144 ms at E = 252). Released under memory pressure through `register_cache_releaser`. Tests: `test_extval_packed_factors.py`.

## BENCH-ATTEMPT-REJECTED (2026-11-02): byte-budgeted resident operand cache (256 MB / 15% of the pool limit, 768 entries) instead of 192 entries

Hypothesis: the 192-entry LRU of `_fe_resident_operands` thrashes (the ext-val factor set is 106 arrays used cyclically). Instrumented `resident_operand` per role over one 100000 x 100 fit (`D:/Temp/extval_miss.py`, calls slower than 0.4 ms counted as uploads): `gate_op` 2320 calls / 382 slow / 0.73 s, `extval_ext` 2544 / 227 / 0.57 s, `op` 1360 / 294 / 0.49 s, all roles about 3 s. With the byte budget (count cap 768) and a hash memo of 1024 entries: `gate_op` 256 slow / 0.70 s, `extval_ext` 328 / 0.67 s, `op` 186 / 0.46 s: about 3.1 s in total, no gain. The hash memo thrash is cheap (9 ms against 2 ms per pass of 106 factors). The slow calls are mostly first uploads (5 distinct ext-val factor sets of 106 arrays = at most 530 compulsory uploads), not evictions, so cache capacity is not the lever; reverted. Retry condition: a fit where the same operands are used again after more than 192 other distinct operands (the instrumentation above would show `slow` falling with capacity).

## Done: NaN-aware device binner for the ext-val candidates (`_gpu_nan_quantile_codes.py`)

Counting the calls of `gpu_materialise_extval_codes_host` in a 30000 x 40 fit (`D:/Temp/extval_dev.py`): 24 of 24 returned `None`, so the device-born ext-val path (no host materialise, no candidate upload) never ran: the finiteness guard added upstream (`2953e9e64`, "keep the host binner for candidates holding NaN") declined every batch, because the factors carry missing values. Every pair then took the host route (njit materialise, host `np.nanpercentile`, block upload).

The new binner reproduces the host codes on the device: per column block, sort (NaN last), edges at `(m - 1) * q` of the column's own non-NaN count `m` with numpy's lerp, code = number of interior edges `<= x`, NaN to the top code. Bit-equal to `discretize_2d_quantile_batch` for NaN-holding, tied, constant, integer and heavy-tailed columns and nbins 3 / 10 / 16 (`test_gpu_nan_quantile_codes.py`); an infinite value or fewer than two numbers in a column still returns `None` (host path). Now 24 of 24 calls run on the device (`test_extval_gpu_declines_non_finite_candidates.py` states the new contract: host-identical codes for NaN, `None` for infinity).

Fit level (100000 x 100, every family): wall 147.3 s, no OutOfMemoryError / "FE raised", the same 6 selected features, sum of the top-30 family walls 74.4 s against 78.1 s before.

## Done: hybrid NaN-aware ext-val binning (`quantile_codes_with_nan_columns`)

The first NaN-aware binner sorted every column, and once the ext-val path ran on the device it became the top sampled frame of `_emit_pair_features_step2_read_here_once_part1` (52 samples, GPU 87%, about 45 of them inside `nan_aware_quantile_codes_gpu`): the f64 row sort costs about 1 ms per 100000-row column (64 ms per 64-column block) and the comparison count 33 ms per block. Only about 2% of the candidate columns hold NaN (276 columns, 6 with NaN: one factor with missing values times its operations; `D:/Temp/nan_frac.py`), so the whole matrix is now coded by the fast radix binner and only the NaN-holding columns are re-coded exactly by the NaN-aware path (`test_mostly_finite_matrix_with_a_few_nan_columns_equals_the_host_binner`). 954 columns x 100000 rows with 2% NaN columns: 1689 ms -> 178 ms (9.5x), codes equal to the host binner.

Sampler (100000 x 100, every family): `_emit_pair_features_step2_read_here_once_part1` no longer in the top 8 stages (52 samples before), wall of the sampled fit 134.0 s (the lowest of the day; run-to-run noise is about 10 s).

## Test-suite note (2026-11-03): order-dependent residency test

`tests/feature_selection/gpu/test_cmi_residency_traffic.py::test_strict_fit_does_not_copy_the_same_content_back_repeatedly` failed once inside a full run of `tests/feature_selection/gpu` (pytest-randomly shuffled the order; 14 repeated copies of one (8867, 10) float64 column from `_fe_retention_subsumption._device_max_conditional_excess`, called from `final_subsumption_prune`) and passes alone with a fixed order (`-p no:randomly`) both on `origin/master` (cc9a64394) and on this branch. The copying code is upstream's final-subsumption stage, not touched here; a leftover cache state from a previous test is the likely cause. The same directory run also stalled for over half an hour at `test_kernel_tuning_cache.py::test_hw_aware_fallback_routes_per_cc_major[5-512]` (1.3 s alone), so a full-directory gpu run needs a per-test timeout; I stopped that process (my own) and ran the contract files separately.

## Measured, no stage-level effect (2026-11-03): memoised content hash in the `joint_cardinalities_cupy` key

Per call the old key (`hash(tobytes())` of y and z) cost 0.45 ms and the memoised content hash 0.004 ms for a stable host array. At stage level `score_prospective_pairs` did not move: `_pair_perm_null_admission` 2.36 s (3567 calls) before, 2.58 s after (wall 127.7 s against 142.2 s, so the run-to-run noise is larger than any effect here). The pair null passes a fresh `np.ascontiguousarray(data[:, anchor], dtype=np.int64)` as z on every call, so the identity-keyed memo misses there and the per-call cost is elsewhere (the observed CMI, the null's launches and syncs). The change stays (it is exact and 100x cheaper for callers that pass stable arrays) but the per-pair cost of about 0.65 ms is not the key; the lever would be one batched CMI-with-null launch per block of pairs sharing a partner.

## Done: pair admissions and asymmetric-synergy relaxations batched per anchor (`_mrmr_fe_step/_pair_admission_batch.py`)

Proposal of the architecture review (2026-11-03). `_pair_perm_null_admission` scored 3567 pairs one by one (2.36 to 2.58 s per fit: one observed CMI plus one null per pair). The eligible pairs (fail prevalence, clear max-T) are now grouped by anchor and scored by one `batched_cmi_gpu(..., return_cards=True)` per block of 64 candidates; each null is the analytic one from those cardinalities (`precomp_cards`), with the per-pair seed, in the analytic regime only (at least `_cmi_analytic_null_min_n()` rows after the same strided subsample); `_pair_perm_null_admission` reads the verdict. The same pass decides the asymmetric-synergy relaxation of `_resolve_pair_prevalence_gate` (grouped by partner) in the pre-pass. Tests: `test_pair_admission_batch.py` (the batched verdict of every pair equals the per-pair verdict, with admitted and rejected pairs present; the gate returns the same state with and without the batched relaxations).

Measured (`D:/Temp/pairs_timing.py`, 100000 x 100, every family): `_pair_perm_null_admission` 2.36-2.58 s -> 0.00 s, `score_prospective_pairs` 6.3-7.9 s -> 4.97 s (that run: wall 116.4 s, the lowest so far). Full fit afterwards: wall 137.7 s (inside the run-to-run noise), the same 6 selected features, no OutOfMemoryError / "FE raised".

## Rejected after measurement (2026-11-03): architecture-review proposals #5 and #7

- #5, batch the per-candidate `_quantile_bin_device` calls of the CMI redundancy gate (`_bin_candidates`): one call over 6 columns, 0.02 s in the 100000 x 100 fit (`D:/Temp/bin_cand_timing.py`). Not worth a change.
- #7, memoise the raw-column bootstrap MI of `_replicate_mi`: 10 calls, 4.50 s, 1318 columns, of which 129 were scored before in the same fit (`D:/Temp/repmi_timing.py`): a memo saves about 0.4 s. The bulk is one 749-column engineered matrix at 100000 rows (2.93 s). Forcing the device bootstrap walk on that shape in one process: host 1.75 s, device 4.21 s, identical MI, so the tuner's host choice is right and the family stays host-bound.

## Done: dedup scan candidate sorts prefetched (architecture-review proposal #4, the cheap part)

`scan_engineered_duplicates` 2.17 s per 100000 x 100 fit (`D:/Temp/dedup_timing.py`): `_pair_collisions` 1.29 s over 156 candidates, `_precompute_full_ranks` 0.49 s, `_masked_corr_prefilter` 0.23 s. Each candidate's own stable finite-row sort (about 8 ms) ran serially inside the loop; `_CandidateOrderPrefetch` sorts the next 32 candidates in scan order together on the edge-jobs pool when the first asks (identical orders, `test_candidate_order_prefetch_equals_the_direct_sort`). After: scan 1.87 s, `_pair_collisions` 0.99 s. The rest is the masked Spearman itself; the block-wise candidate x kept correlation of the proposal (one GEMM per block, sequential keep/drop fixup) is the remaining step, worth about 1 s at most.

## 2026-11-03: fit after the day's changes

`gpu_walls.py 100000 100` with nothing else running: wall 113.7 s (earlier runs today 120.1 / 136.9 / 139.5 s; single walls on this host vary widely), same six
selected features, no out-of-memory or "FE raised" lines. Top family walls: binned_numeric_agg 4.91, modular 4.09 (6.09 this morning), auto-scorer 3.88 (4.33),
numeric_decompose 2.97 (3.90), univariate_fourier 2.81, grouped_quantile 2.78, composite_group_agg 2.66.

Not changed, measured: `batch_pair_usability_corr_cuda_warp` (1.25 s, two calls of 5778 / 3561 pairs x 5 forms on (108, 33334) float32 operands) is float64-bound
on the GTX 1050 Ti (1.9e9 form evaluations in 0.79 s, about 48 GFLOPS of a ~66 GFLOPS fp64 peak; reads are coalesced: the operands are (k, n)). Only a float32
form evaluation would speed it up, which changes the correlations that feed the prevalence relaxation; left as is.
