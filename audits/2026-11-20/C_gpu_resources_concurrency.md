# Direction C: GPU / resident paths, concurrency and resources (MRMR)

Read-only audit of `src/mlframe/feature_selection/filters/` at HEAD 981380c4c. No source or test files were edited and no GPU or test process was run.

Scope read: the RawKernel sources in `_honest_null_accept_bits.py`, `_null_max_perm_gpu.py`, `_cell_moments_kernel.py`, `_bin_fingerprint_gpu.py`, `_gpu_resident_rank_bin.py` and `_binned_numeric_agg_resident.py`. Also the stage-concurrency path (`_mrmr_fit_impl/_fe_stage_concurrency.py`, `_fe_device_scheduler.py`, `_edge_pool.py`), `_raw_module_cache.py`, and the identity-keyed device caches in `info_theory/_cmi_resident_cache.py` and `_dcd_pair_su_gpu.py`.

Checked and found sound (no finding): shared-memory sizing against launch arguments in the cell-moment, honest-null and null-max kernels; the last partial block and the `i < n` guards in the honest-null and sorted-bit kernels; the 2-D to 1-D index arithmetic in the rank-bin kernel (`long long` throughout, and the `nb > n` case does not divide by zero); the 256-thread reduction in `bin_fingerprints`; the QCAP overflow path in the honest-null queue; the `__syncthreads` ordering at the end of each pair in the global-scratch variant; the RawModuleCache lock; the weakref co-validation in the `id()`-keyed caches of the `batch_mi_noise_gate` and `_dcd_pair_su_gpu` modules.

Only four findings survived verification, all of them Low or latent. The brief asked for 8 to 14; the rest did not hold up on reading the code path.

### F01 - Edge-pool workers drop the stage's device and allocator (latent on multi-GPU hosts)
- **Severity**: Medium
- **Location**: `src/mlframe/feature_selection/filters/_edge_pool.py:43-52`; `src/mlframe/feature_selection/filters/_mrmr_fit_impl/_fe_stage_concurrency.py:198`
- **Problem**: `run_stages_concurrently` runs each stage inside `stage_device_slot(slot, i)`, which calls `cp.cuda.Device(slot.device_id)` and `cp.cuda.using_allocator(pool.malloc)`. Both are thread-local. A stage that calls `map_on_edge_pool` gets a fresh `ThreadPoolExecutor`, whose threads start on device 0 with the default allocator. `map_on_edge_pool` only publishes the deadline and the fit context to those threads, never the device or the allocator. The two current call sites are host-only (`_fourier_batch_gpu._prepare` is numpy, and `_extra_fe_families._one` is a numpy sort), so nothing triggers this today.
- **Failure scenario**: On a two-GPU host, a stage placed on device 1 whose family hands device arrays (allocated on device 1) to a `map_on_edge_pool` callable would launch kernels on device 0 with pointers from device 1. The result is `cudaErrorInvalidValue` or an illegal address, or the work is silently counted against the default pool instead of the slot budget. On a single GPU the wrong-allocator half still applies: the slot's budget is bypassed.
- **Suggested fix**: Capture the calling thread's current device id and allocator in `map_on_edge_pool` (as `capture_fit_context` does for the context) and re-enter `cp.cuda.Device(id)` and `using_allocator` in `_run`. Alternatively, document that edge callables must be host-only and assert it in tests.
- **Disposition**: FIXED 2026-11-20 - `map_on_edge_pool` re-enters the caller's CUDA device and allocator in every worker (`_capture_device_scope`, `_edge_pool.py`); sensor `test_edge_pool_workers_see_the_callers_allocator`.

### F02 - Null max-permutation launch exceeds the 48 KB dynamic shared-memory default for large nbins x classes
- **Severity**: Low
- **Location**: `src/mlframe/feature_selection/filters/_null_max_perm_gpu.py:288, 319-323`
- **Problem**: `per_launch = max(1, _SHARED_BUDGET_BYTES // (4 * m))` clamps the permutation count to at least one, but nothing caps `m = nbins * n_classes` itself. The launch then requests `shared_mem = block_len * m * 4`. Once `m * 4` exceeds 49152 bytes (`m > 12288`), the CUDA launch fails with an invalid-value error. Both callers wrap the call in a broad `except Exception`, so the failure is silent.
- **Failure scenario**: A multi-class target with `n_classes = 200` and `nbins = 64` gives `m = 12800`, so `shared_mem = 51200` bytes. The launch raises, `_strict_null_max_batched` logs one throttled warning and returns `None`, and the caller falls back to one launch per permutation. The result is correct but the batched speed-up is lost on that shape without a clear reason in the log.
- **Suggested fix**: Check `m * 4 <= 49152` before choosing the batched path (return `None` early), and log the reason through `log_throttle`.
- **Disposition**: FIXED 2026-11-20 - the generic batched path returns `None` with a throttled warning when one permutation's count table is over the 48 KB dynamic shared-memory limit (`_null_max_perm_gpu.py`); sensor `test_null_max_batched_path_declines_a_table_over_the_shared_limit`.

### F03 - Packed-label cache keeps device arrays alive after the fit, and thrashes when two keys alternate
- **Severity**: Low
- **Location**: `src/mlframe/feature_selection/filters/_null_max_perm_gpu.py:212-254`
- **Problem**: `_LABEL_BITS_CACHE["latest"]` holds the packed device label blocks (about 49 MB at 10 million rows, per the docstring) until a different key replaces them. Nothing releases them at the end of a fit, so the memory stays allocated across fits. The cache holds one key at a time, so when two binned-agg nulls with different keys alternate (for example in two stages running concurrently), each store clears the other's blocks and every block is repacked.
- **Failure scenario**: A fit finishes with 49 MB of label blocks still allocated on the device. A second estimator in the same process then runs a different binned-agg null and hits the single-slot cache, repacking every block; with two keys alternating on concurrent stages, the packing runs on every call. No wrong value results, but VRAM stays held and the repacking cost is paid again.
- **Suggested fix**: Release the cache at the end of `null_max_mi_perms_gpu` for keys that are not reused, or tie it to the fit's cleanup (`_fit_scoped_release`), and make the cache key-per-stage rather than a single slot.
- **Disposition**: FIXED 2026-11-20 - the packed-label cache keeps the last two keys and `clear_label_bits_cache` is a fit-end releaser (`_fit_scoped_release.py`); sensor `test_label_bits_cache_keeps_two_keys_and_clears`.

### F04 - Stage-concurrency VRAM check treats an unreadable device as "fits"
- **Severity**: Low
- **Location**: `src/mlframe/feature_selection/filters/_mrmr_fit_impl/_fe_stage_concurrency.py:158-160`
- **Problem**: When reading `mem_info` or the pool limit raises, `_stages_fit_device` logs a throttled warning and returns `True`. The stage cascades then run concurrently with no memory check, which is the case the check exists to prevent.
- **Failure scenario**: At 500000 rows on a 4 GB card, a transient driver error during `mem_info` lets the three cascades run together. Their scaled peaks exceed the card, the slots fall back to host paths, and the fit slows down. The result stays correct, but the fallback is the one the check was added to avoid.
- **Suggested fix**: Return `False` (run the cascades one by one) when the device cannot be read, which is the conservative choice; the sequential path is always correct.
- **Disposition**: FIXED 2026-11-20 - an unreadable device (not a missing cupy) makes `_stages_fit_device` return False, so the cascades run one by one; sensor `test_stage_vram_check_is_conservative_for_an_unreadable_device`.

### F05 - Worker `finally` in the stage runner can replace the stage's own exception
- **Severity**: Low
- **Location**: `src/mlframe/feature_selection/filters/_mrmr_fit_impl/_fe_stage_concurrency.py:194-203`
- **Problem**: The `finally` block calls `close_thread_laps()` and then `_synchronize_device(...)`. If the stage raised and the synchronize then raises (a sticky CUDA error after an illegal address, for example), the exception stored in the future is the synchronize error. The stage's original error, which names the failing family, is lost. `walls[i]` is also left unset in that case.
- **Failure scenario**: Stage `mid_a` raises `ValueError` from a bad recipe, and the device is in a sticky error state. The reported exception is `CUDA error: an illegal memory access`, and the traceback points at the synchronize call, not at the family that caused it.
- **Suggested fix**: Wrap the post-stage cleanup in its own `try/except` that logs at debug and re-raises the stage's exception if one is pending, so the first failure in stage order is the one reported.
- **Disposition**: FIXED 2026-11-20 - the stage worker's cleanup error is logged and swallowed when the stage itself failed, so the stage's exception reaches the caller; a cleanup error after a successful stage still raises; sensor `test_stage_failure_is_not_replaced_by_the_cleanup_error`.

## Not reported (checked, did not hold up)

- `_cmi_resident_cache._resident_factors_device` is keyed on `id(factors_data)` with no content check, and `_dcd_swap_null` mutates `state.factors_data` in place. The mutation window only runs the CPU `conditional_mi`, which does not use that cache, and the restore runs in `finally`. No stale device copy is reachable, so this is dropped. It is a latent hazard if the swap null ever calls the batched dispatch inside the window.
- `_cell_moments_kernel._KERNELS` and `_FOLD_KERNELS` are initialised without a lock. Two threads may compile twice; the result is identical, so this is benign.
