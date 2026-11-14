# _fe_interaction_prerank.py (interaction pre-rank) with its carved module `_fe_interaction_prerank_kernels`: profile record (2026-11-14)

Family: `_fe_interaction_prerank` (the pre-rank driver and the choice between the second-moment, marginal and GBM rank bases) with `_fe_interaction_prerank_kernels` (the discrete score `|Zc' Yc| + |Zc2' Yc|` as numpy GEMM, a numba prange loop and a cupy GEMM, and the `kernel_tuner` entry that picks one).

## cProfile

Host variant (numpy GEMM, 100000 rows x 3000 columns, 7 class columns, float32): one call takes 0.342 s and all of it is inside `discrete_score_numpy` (the two BLAS products; 9 function calls in total). The family is not exercised by the default fit (no function of the family reaches 0.05 s cumulative in the whole-fit cProfile of 2026-11-14), it runs only when the pre-rank is switched on for a wide candidate pool.

## Kernels (nvprof, one cupy call, 100000 x 3000, the profiled call after one warm-up)

Kernel launches per call: 14 `sgemm_32x32x32_TN` (cuBLAS, 3.92 ms each, 12.7 % of the GPU time: two products per column block, seven blocks), 14 `cub::DeviceSegmentedReduceKernel` (24.6 us each), 14 `cupy_absolute` and 7 `cupy_add` launches of 1 to 3 us, 15 host-to-device copies of 25.2 ms on average (87.2 % of the GPU time, 377.5 ms in all: the standardized operands live on the host) and 7 small device-to-host copies. The GEMM is the fused kernel of this family; no hand kernel is written for it because cuBLAS runs the product (the detector of the contract accepts an explicit `cp.matmul` call on device arrays as the GPU kernel for that reason).

## Occupancy and throughput

The nvprof counters (achieved occupancy, DRAM throughput) were not collected for the cuBLAS kernel `sgemm_32x32x32_TN`, a library kernel with its own tiling. Throughput from the call itself: the 14 products are 2 x (100000 x 3000 x 7 x 2) = 8.4 GFLOP of arithmetic in 54.85 ms of GEMM time, about 153 GFLOP/s; the host-to-device copies move the two standardized operands, 2 x 1.2 GB = 2.4 GB, in 377.5 ms, about 6.4 GB/s. The call is transfer-bound: 87 % of its GPU time is the upload, 13 % the products.

## Host-or-device choice

`_SPEC` (`kernel_tuner`, the same shape as `random_features.rff_matmul`) picks numpy, numba or cupy per host and per size. Measured medians of 5 calls (n x p, 7 classes, float32): 20000 x 3000: numpy 67.1 ms, numba 5664 ms, cupy 255.1 ms; 100000 x 3000: numpy 350.1 ms, numba 35316 ms, cupy 1443 ms; 20000 x 10000: numpy 243.7 ms, numba 24391 ms, cupy 802.5 ms. With host-resident operands numpy wins at every size measured; the cupy variant wins only when the operands are already on the device (then the 14 GEMMs take 55 ms of the 100000 x 3000 call, 6.4x under numpy), which is the case the large-p, resident pre-rank is meant for.

## Streaming

`discrete_score_cupy` moves the candidate columns to the device in column blocks (`block_cols` from `_device_blocks.first_block`, halved on a failed block by `run_column_blocks`) against one resident class matrix; the column scores are independent, so the blocks join to the same vector as a single product over all columns (max relative difference to numpy 1.6e-6 in float32; the blocked result equals the unblocked one exactly). The test `test_discrete_score_cupy_blocks_equal_unblocked` pins this.
