# Profile record: batched pair joint MI on the device (`_gpu_pairs`, host twin `_pair_search_kernel_njit` of `cat_interactions`, tuner `_cat_pair_search_tuner`)

Host: GTX 1050 Ti (sm_61, 4 GiB, 112 GB/s), 8 logical cores, elevated process (hardware counters). Measured 2026-11-15, medians of 3 to 5 after a warm call. Workload: 60 columns of 12-bin codes, every pair (1770), binary label, the sweep inputs of `_cat_pair_search_tuner`.

## What the module is
`mi_direct_gpu_batched_pairs` returns `I(X_i, X_j; Y)` for every candidate pair of the categorical interaction step: `compute_joint_hist_multi_pair_shared_cuda` fills one flat int32 joint-count buffer for all pairs in one launch, then `pair_joint_mi` reduces each pair's table to its MI.

## Column blocks (streaming)
A request over 65535 pairs is split into column blocks of the pair table (`pair_chunk_bounds`); each block is histogrammed and reduced alone, and the 4 GiB guard refuses a block whose joint tables would not fit. Per block the device holds the transposed code matrix, the pair indices, the joint-count buffer and `n_pairs` doubles.

## What was changed on this path (2026-11-15)
1. The per-pair reduction ran on the host: the whole count buffer was copied down and a Python loop summed each pair. `pair_joint_mi` (one block per pair, exact integer marginals, double cell sum in a fixed-order tree) writes only the `n_pairs` MI values: 3000 pairs x 5 target bins 210.9 -> 0.99 ms, difference 1.7e-18 to the host reference.
2. The code matrix was transposed on the host (`ascontiguousarray(factors_data.T.astype(int32))`, 58 ms at 100000 x 60) and range-checked column by column with strided reductions (94 ms). It is now uploaded as it is, transposed on the card (2.2 ms `cupy_copy__int32_int32`) and range-checked with two row reductions of the transposed copy. One call 209 -> 45 ms.
3. `backend="auto"` took the device only above a hand-set work product (200 columns x 500000 rows). `cat_pair_search_backend` (host_device_sweep, equal joint MI required) measured the device ahead from 20000 rows up, so the gate now asks the tuner (floor of 64 candidate columns kept).

## cProfile (one device call, 100000 rows, 1770 pairs)
0.046 s for 242 function calls: 0.040 s in the blocking copies of the results (the host waiting for the queued kernels), 0.005 s in 5 small uploads, 0.009 s in the part-1 setup. Before: 0.209 s with 0.094 s in 123 numpy reductions and 0.063 s in the host transpose.

## nvprof (hardware counters)
Launches per call: 1 `compute_joint_hist_multi_pair_shared_cuda`, 1 `pair_joint_mi`, 1 transpose copy, 1 `cupy_take` of the referenced rows and 2 cub reductions (range check); no per-pair launch.

| kernel | achieved occupancy | other counters |
|---|---|---|
| `compute_joint_hist_multi_pair_shared_cuda` | 0.93 | global load efficiency 70.8 percent, DRAM read 53 GB/s, 36.5 ms |
| `pair_joint_mi` | 0.74 | global load efficiency 96.5 percent, DRAM read 3.8 GB/s, 0.74 ms |
| `cupy_copy__int32_int32` (transpose) | 0.80 | store efficiency 100 percent, DRAM read 81 GB/s, 2.2 ms |
| `cupy_take` | 0.90 | load efficiency 85 percent, 1.1 ms |

Whole-GPU utilization: about 41 ms of GPU time in a 45 ms call (91 percent busy); the histogram launch is 89 percent of it. It reads two code columns per pair and row at 70.8 percent load efficiency and is the next kernel to look at (it lives in `_gpu_kernels`).

## Wall clock (njit prange kernel against the device, same joint MI, difference under 2e-16)
| rows | host | device |
|---|---|---|
| 20,000 | 49.4 ms | 10.1 ms (4.9x) |
| 100,000 | 590 ms | 45.4 ms (13x) |
| 500,000 | 6.36 s | 244 ms (26x) |
| 2,000,000 | 29.5 s | 968 ms (30x) |
