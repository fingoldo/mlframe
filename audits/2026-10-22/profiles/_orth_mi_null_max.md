# `_orth_mi_null_max` (family-wise MI null of the orthogonal-univariate local gate: maximum over columns, per permutation)

The module is a router. It reproduces the backend that the genuine scorer (`_mi_classif_batch`) takes, so the null and the observed MI are computed the same way:
STRICT residency -> `_null_max_strict_gpu` (the fused permutation kernel `null_mi_perms_bits` of `_null_max_perm_gpu`, profile record `_null_max_perm_gpu.md`);
streamed matrices -> `_streamed_gpu_mi.null_max_streamed`; the CPU rank path -> the numba kernels in this module (record `_orth_mi_null_max_cpu_rank_path.md`).
STRICT mode means the GPU is used regardless of what the timing shows, so the STRICT branch has no kernel-tuning entry by design; the non-STRICT choice belongs to the scorer's backend (`_orth_mi_backends`).

## 2026-11-15: STRICT path measured (GTX 1050 Ti, 100000 rows x 40 columns, 39 permutations, 2 classes, 10 bins)

One call, warm, synchronized: 23-24 ms, including the float64 upload of the matrix (32 MB) and the edge computation.

cProfile of one call (28 ms under the profiler): `cupy.array` 13 ms of host time in 40 calls (the matrix upload and one 800 KB upload per permutation), `_permuted_labels` 6 ms, `nanmax` 6 ms (the one device read of the maxima).

nvprof (`--print-gpu-summary`, 4 calls = 1 warm-up + 3): per call
- `[CUDA memcpy HtoD]` 8.1 ms in 41.5 calls (the matrix 5.6 ms, 39 permutation uploads of about 63 us),
- `null_mi_perms_bits` 5.6 ms in 1 launch (the whole null: a block of permutations per launch, bitset path for two classes),
- `radix_select_interp_f64_v3` 2.3 ms in 1 launch (the bin edges, computed once for all permutations),
- `transpose_f64` 0.67 ms, `cupy_take` 39 launches of 13 us (the permuted labels), `cupy_packbits_little` 0.12 ms.

Kernel launches per call: 1 null kernel + 1 edge launch + 1 transpose + 1 pack + 39 label gathers; the per-permutation MI launch loop (one launch per permutation, kept as the `MLFRAME_STRICT_NULL_BATCHED=0` fallback) is gone from the default path.

nvprof metrics (the null kernel): achieved occupancy 0.207 (64-thread blocks, 40 registers and 512 bytes of local memory per thread: the width `fixed_shared_block_size` picks, recorded in `_null_max_perm_gpu.md`), global load efficiency 88 %, device read throughput 2-5 GB/s of the 112 GB/s card (compute and shared-memory bound, not DRAM bound). The cupy label gathers read at 14 % load efficiency but cost 0.5 ms in all.
GPU throughput / utilization: the device is busy 40 ms of the 4 x 24 ms wall (about 40 %), the rest is the host-side permutation staging; the STRICT null runs only in STRICT mode (it is not in the default 100000 x 100 fit), so no further tuning is warranted now.
