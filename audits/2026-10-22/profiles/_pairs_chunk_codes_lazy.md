# Pair-search chunk codes: copy back only the columns the family gate reads, 2026-10-29

Whole-fit profile at HEAD (n=20000): `ensure_host_codes_filled` 2.09 s over 17 chunks. `_compute_one_fe_chunk_part2` filled the whole (n, K) int8 codes block on the host for every chunk so the
pair family gate could later read one column per configuration, but the gate reached its exact test (the only reader of the codes) in 12 `family_pass_batches` calls over 7 pairs in the whole fit.

Change: the eager fill is gone. `ChunkCodes.capture` (before the dispatch clears the deferred-fill record) keeps the device array and the registry's filled flag; the pair state stores
`LazyCodeColumn` references; `_apply_pair_family_gate` calls `resolve_code_columns` after its table-floor early exit, which copies the needed columns of one chunk in a single transfer.
Items that are not lazy columns (arrays, `LazyHostCodes` of the extval path) are passed through untouched; a first version converted them with `np.asarray`, which stripped the device
array the device family test reads and raised the strict-fit traffic from 3.3 MB to 46.6 MB (caught by `test_strict_fit_copies_back_only_result_columns`, back at its 3,295,159 B).
`MLFRAME_PAIR_LAZY_CODES=0` restores the eager fill.

Result (all-flags warm fit): `ensure_host_codes_filled` removed from the profile; `resolve_code_columns` 1.4 s over 12 calls. Fit wall 75.5 -> 67.5 s (cProfile run); `fit_sel_gate.py` seeds 0 and 2:
68.5 -> 61.1 s and 83.9 -> 73.8 s, selection identical in both. Part of the wall difference is GPU load noise; the removed copies are about 1.5 s by the profile.

Not changed by this: the strict-fit test (pair-search audit, n=30000) measures 3,275,410 B with the switch on and off, because that path keeps `LazyHostCodes` and never copies chunk codes; its
overrun against the 2.4 MB budget comes from the sites listed in `strict_fit_d2h_traffic_2026_10_29.md` (score tables and the single-column reads). An earlier audit run showed 12.9 GB in one
tuning-cache state that I could not reproduce afterwards.

## Follow-up: device family test, narrow transfers, strict-fit traffic test green (2026-10-29)

- `LazyCodeColumn.dev` is the (n, 1) device view of a column, so `family_pass_batches_device` scores a pair's configurations from the device codes without any copy; the host test asks
  `resolve_code_columns` for its columns in one transfer per chunk. In one tuning-cache state of the pair-search audit a pair's whole configuration set (1944 columns of 30000 rows) was
  requested by the host test, 1.75 GB in all; with the device test that audit copies 3.2 MB.
- `LazyHostCodes.host()` copies wide integer arrays as int8 plus a 16 byte min/max tail in ONE transfer and widens on the host; a range outside int8 discards the attempt and copies densely
  (tests: `test_lazy_host_codes_narrow.py`). A first version read min and max as two extra scalar syncs and broke `test_first_real_read_copies_once_and_matches_the_eager_copy`
  (it counts every device-to-host operation); the range now rides in the same transfer instead of loosening the test.
- `_device_codes_to_host` (survivor block) copies in int8/int16 by `nbins`; `filter_inert_leaf_block` keeps its probe column on the device, bins it there and copies narrow codes instead of the
  float64 column; the device family gate copies the level counts of int8 codes as int16; the honest-null accept block copies `ncand` and `bidx` as int16 when `n` < 32768.
- `test_strict_fit_copies_back_only_result_columns` (budget 2,400,000 B): 3,295,159 B at HEAD, 2,815,175 after the code narrowing, 2,417,035 after the inert-leaf probe, below the budget after the accept block.
  No budget or test was changed. Remaining sites are the genuine results: MI score tables, the engineered columns, the held-out gate's float design columns.
- Rejected: copying only the non-zero entries of the gated MI vector in `bits_gate_mi` (index plus value pairs). The vector is almost never sparse after the gate, bytes were unchanged and 76 extra
  operations were added; reverted.
- Not done: the readers of one lazy array (`_cmi_from_binned_cupy`, `_fe_cmi_redundancy_null._NullInputs`) are not moved to the device. The null needs the host codes for its within-stratum
  shuffles (`_permutation_sources_njit` runs on the host), so the copy stays, now a single narrow transfer of 30 KB.

Full fit (`fit_sel_gate.py`, GPU shared): seed 0 68.5 -> 58.0 s, seed 2 83.9 -> 67.3 s, selection identical on both. 76 tests of the touched modules pass.

## Systematic pass over repeated copies (2026-10-29)

Fixing readers one by one from the top of a site list missed some and moved copies around (a narrowed read that is not shared leaves a later dense read to pay again; the traffic total went up
once). `duplicate_transfer_audit` (`filters/_gpu_strict_fe/_duplicate_transfers.py`, CLI `_benchmarks/profiling/duplicate_d2h.py`) hashes the host array of every device-to-host copy of at
least 2 KB after a normalising cast (floats as float32, integers as int64), so a dense and a narrowed copy of one array are the same content, and lists every group of repeats with its call
sites. On the strict pair-search audit (n=30000) it found 420,000 B of repeated content in 2,362,048 B:

- one engineered column read narrow by the held-out gate and dense float64 by `_assign_support_tail` (`np.asarray(x).shape[0]` copied it just for the length), `_fe_raw_redundancy_drop._cont_of`,
  `_fe_raw_redundancy_anchors` and `_finite_column` (240 KB, then 120 KB);
- the same compound recipe replayed by the residual rescue and then by the final subsumption (120 KB, plus the replay itself).

Fixes: `LazyHostCodes.host_as(dtype)` keeps one copy per dtype (float arrays narrowed on the device, so half the bytes), `__array__(dtype)` goes through it, so `np.asarray(lazy, dtype=...)` anywhere
shares it; `length_of` / `size_of` read a length without a copy (six sites, plus the meta gate `test_no_asarray_for_length.py` that freezes 126 harmless hits on host arrays and rejects new ones);
`replay_value_cache` shares recipe replays between the stages of `_assign_support_tail` for the duration of that call (thread-local, nothing kept on the estimator; a `unary_binary` replay ignores
`quantization`, so the stripped copy of a recipe hits the same entry).

Result: 2,002,048 B in 80 transfers, 60,000 B repeated; `test_strict_fit_copies_back_only_result_columns` passes on repeated runs (2,480,307 B before the pass, over the 2.4 MB budget).
`test_strict_fit_does_not_copy_the_same_content_back_repeatedly` pins the repeated bytes at 3 n.

Left: two int8 code columns (30 KB each) binned by `_discretise_inert_probe` and again by `_materialise_and_fina_step3_cols_space_index`. A cache keyed on the device address would
return stale codes when a pair buffer is rewritten in place, for 60 KB; not done.

The device cast of float64 to float32 flushes results below 1.2e-38 to zero where the host keeps a subnormal. The selection bar is equivalence, not bit identity, and no consumer of a
criterion-dtype column can tell, so `host_as` does not guard against it (a guard was written first and removed; its cost was an extra scalar read per call).

## `_fused_observed_mi_and_bins` wait (2026-10-29)

cProfile attributes 1.7 s of `lock.acquire` / `.get` to `_pairs_analytic_gpu._fused_observed_mi_and_bins` (108 calls). nsys shows the kernel it launches, `analytic_obs_mi_bins_i8`, at 4.8 ms per
launch (26 launches, 0.125 s). The rest is the first read of the function (`cp.stack([min, max]).get()`) waiting for the kernels the chunk producer queued before it: candidate
materialise, binning, the transposes and the radix select. So the function is where the host first synchronises, not where the time is spent; splitting its launch from its reads would only
move the wait to the next read. The levers are shorter producer kernels (the radix select is at its measured floor on this card, see `_gpu_resident_select_kernels.py`) or a deeper chunk
prefetch (a third candidate buffer of n x K float32, about 300 MB at n=20000, K=3888, on a 4 GB card shared with other processes), which needs a quiet device to measure. Not changed.

## Last repeated columns and the CI question (2026-10-29)

`LazyHostCodes` keeps the quantile codes it has already produced (`codes_cached` / `store_codes`, per array and per `(nbins, dtype)`; a one-column view of a block files its column under the block).
The inert-leaf probe bins each candidate column and the survivor block bins the same columns again; with the cache the second step reads the first one's codes. The cache lives on the array object,
so it cannot go stale across objects, and every hit returns a copy. On the strict pair-search audit the repeated content is now 0 B of 1,942,048 B (it was 420,000 B of 2,362,048 B), and the guard
`test_strict_fit_does_not_copy_the_same_content_back_repeatedly` allows one int8 code column.

A nightly CI job for `duplicate_d2h` was considered and not added: the GitHub-hosted runners have no GPU (`fs-benchmark-nightly.yml` says so), so the job would silently measure nothing. What CI can
run is static: `test_no_asarray_for_length.py` rejects a new `np.asarray(x).shape[0]`, the pattern behind most of the repeated copies. The device guard runs wherever the GPU tests run.
