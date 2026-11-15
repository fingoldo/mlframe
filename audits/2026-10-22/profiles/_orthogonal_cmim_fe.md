# `_orthogonal_cmim_fe` (CMIM scorer of the orthogonal-basis candidates) with its carved child `_cmim_min_gpu`

2026-11-16. The score of an engineered candidate is `min over the references z of CMI(x; y | z)` (Fleuret 2004), the references of the candidate's own source skipped, the estimator Miller-Madow. Three paths existed: the host loop (`np.unique` factorisation of `x z` and `x y z` per pair), and one `batched_cmi_gpu` call per reference on the device. `_cmim_min_gpu` replaces both with one joint table per (candidate, reference).

## Kernels

`CMI = sum_{c_xyz>0} (c_xyz/n) ln(c_xyz c_z / (c_xz c_yz))`, bias `(k_xyz + k_z - k_xz - k_yz) / (2n)` over the occupied cells, clipped at 0: the algebra of `H_xz + H_yz - H_z - H_xyz - bias` of `_cmi_from_binned_with_cached_z`, agreement 4e-15.
- `_scores_njit`: `prange` over candidates, a joint table per candidate, early exit at a zero.
- `cmim_min_blocks` (CUDA): ONE launch for the whole block, a block of 256 threads per candidate walking the allowed references, the joint table and its three marginals in shared memory (at most `MAX_CELLS` = 6144 int cells, larger tables are declined), warp-shuffle reductions, early exit at a zero.
- `hinge`-style dispatch: kernel-tuning entry `cmim_min_backend` (sweep over rows, 96 informative candidates x 24 references, equivalence 1e-9), `MLFRAME_CMIM_MIN_DEVICE=0|1` forces a side, `MLFRAME_CMIM_MIN=0` restores the older paths. Candidates go through in column blocks of 256 MB of int16 codes.

## cProfile and timings (100000 rows, 150 candidates, 40 references)

Host loop (`np.unique` per pair) 1.5-1.8 s; one `batched_cmi_gpu` call per reference 0.16-0.17 s; fused numba kernel 0.034 s and fused CUDA kernel 0.043 s on noise-like data (the minimum reaches zero at the first references and both exit early); on candidates that all carry the target (no early exit, the worst case) numba 1.25 s, CUDA 0.138 s, so the swept choice is the device. cProfile of the old path: `_factorize_pack` / `np.unique` / `_entropy_from_classes` per pair.

## nvprof (GTX 1050 Ti, sm_61)

Kernel launches per call: 1 (`cmim_min_blocks`) plus the upload of the block (10 copies, 7 % of the device time) and one 1.2 KB read of the scores; the old device path made `S` = 40 `batched_cmi_gpu` calls of several launches each. Achieved occupancy 0.47 (256-thread blocks, shared tables of 4 x 200 ints per block), multiprocessor activity 94.5 %, device read throughput 20-23 GB/s of the 112 GB/s card, shared-memory efficiency 0.16 % (the histogram atomics are scattered by design: the table is 200 cells and the limit is the atomic rate, not the bus). Whole-GPU utilization of the call: the kernel is 93 % of the device time, copies 7 %.

## Equivalence and tests

`tests/feature_selection/filters/test_cmim_min_gpu.py` (both kernels against the host loop to 1e-12, declined inputs, a narrow block budget, `score_features_by_cmim` ranking with the fused scorer on both backends); `tests/feature_selection/gpu/test_cmim_device.py` keeps the per-reference device path under test with the fused scorer switched off.
