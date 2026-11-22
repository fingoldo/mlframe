# Audit 03: MRMR GPU and device paths (wave 2)

Scope: device memory sizing, circuit breakers, host/device sync and copies, host vs device dtypes. Read-only, Haiku model, nothing executed. Not in the directory: `_fe_gpu_pool*.py` (pool logic is in `_fe_gpu_vram.py`).

Tracker rows: AUD23-02 (breaker, high impact), AUD23-18, AUD23-19, AUD23-20, AUD23-21.

## Memory sizing that ignores temporaries

- `_fe_cmi_perm_null_gpu.py:456, 484-486` (confirmed from code, medium). `Xp_d = cp.empty((n, nperm), int64)` has no free-memory check. At the scatter on line 485, live arrays are `keys` (int32, 4 B/cell), `within` (int64, 8 B), `Xp_d` (8 B), and the gather temp `x_sorted_d[within]` (8 B): about 28 B per n*nperm cell. The VRAM chunking at line 595 applies only afterwards. Trigger: n=1M, nperm=200 gives about 5.6 GB, beyond a 4 GB card. Reachability depends on caller nperm gating, not traced.
- `_permutation_null_pair_resident.py:64-76` (confirmed, fine). Tile budget uses `_TILE_BYTES_PER_CELL = 64` and 0.5 of available memory; the comment at 67 records the earlier OOM.
- `batch_pair_mi_gpu.py:147, 152-163` (confirmed, low). Block size `min(2^24 // cells, 2^24 // n)` is a fixed cap, not free-VRAM aware. Transient peak about 0.5-0.6 GB (lines 154-163) is not counted in `_required_gpu_bytes` (317); the 1 GB post-upload cushion in `_gpu_upload_fits` mostly covers it.
- `_batch_pair_mi_cuda_kernels.py:413` (confirmed, low). Finalize split into 8 blocks is fixed; peak about 0.875x the accumulator, which is budgeted at 0.2 of free memory (line 554). Acceptable.

## Circuit breaker that trips permanently

- `_permutation_null_pair_resident.py:223-233, 241, 275` (confirmed, high impact). The handler catches any `Exception`, including cupy `OutOfMemoryError`, and calls `trip_pair_maxt_gpu_circuit_breaker()`. That sets `_PAIR_MAXT_GPU_FAILED` for the process; line 275 then returns False forever. An OOM is transient, yet it is treated like a poisoned context. The file's own comment (line 67) records that this happened: "every later floor of the process ran on the CPU". The tile fix reduces OOM frequency but does not change the trip. Compare `_fe_gpu_vram.py:57-92` (2-second cooldown for OOM) and `_fe_cmi_perm_null_gpu.py:606-609` (halve and retry). Suggested: trip only on a launch or context fault; treat OOM as retry-with-smaller-block.
- `batch_pair_mi_gpu.py:440-450, 452-457` and `_honest_null_device.py:579-581` (confirmed no breaker; impact is hypothesis). Per-call fallback with no flag. If a fault poisons the context, each later call re-faults once, then falls back; the resident file's comment (236-239) describes about a 1-hour cost.

## Host/device sync or copy in a loop

- `_batch_pair_mi_cuda_kernels.py:733-734, 741-742` (confirmed, medium-high in degenerate cases). Each pair-subchunk re-uploads all of `factors_data` in row chunks with plain `to_device`, not `resident_operand`. Guard at 613-616 limits launches to 2000, not bytes. H2D about `n_pair_subchunks * n * n_cols * 4`. Example: 200 MB factors_data, 2000 subchunks: about 400 GB PCIe traffic, within the launch limit. The comment at 728-729 calls this cheap, which holds only for few subchunks.
- `_fe_cmi_perm_null_gpu.py:547, 565, 567, 573-575` (confirmed, low). Several blocking `int()`/`float()` reductions per null call.

## Host vs device dtype

- `batch_pair_mi_gpu.py:125, 140, 156, 163` (confirmed, medium). The cupy path computes in `crit_float_dtype()`, float32 by default (`_fe_usability_signal.py:154-165`). CUDA and njit paths use float64. So `batch_pair_mi` precision depends on backend. Float32 `cp.sum` over up to 2^24 cells adds error, relevant for exact ties and gate comparisons. The docstring at `_batch_pair_mi_cuda_kernels.py:395-401` already says the paths are not bit-identical. Selection impact is hypothesis; the dtype difference is confirmed.
- `_honest_null_device.py:168-171, 208` (hypothesis, medium). Pass 1 scores splits in float32 with `__logf` and keeps candidates within 0.5 nats of the minimum (`w_min + 0.5f`). The comment's own estimate is about 0.02 nats at 50k rows, scaling roughly with n: about 0.4 nats at 1M, possibly past the 0.5 window at about 3M. If so, the best split could be dropped before the exact pass. Not measured.
- `_device_sort_safe.py:35-41` (confirmed, low). For float32 the estimate `2*nbytes + 32*size` equals the measured 40 B/element peak exactly, leaving no headroom beyond the 1 MiB margin.
