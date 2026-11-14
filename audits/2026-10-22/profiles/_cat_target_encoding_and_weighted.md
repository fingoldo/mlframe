# Profile record: weighted categorical pair search (`_cat_target_encoding_and_weighted`, device twin `_cat_weighted_pair_gpu`)

Host: GTX 1050 Ti (sm_61, 4 GiB), 8 logical cores, elevated process (hardware counters). Measured 2026-11-14, medians of 5 after a warm call. Workload: six categorical columns of 12 levels, 20,000 rows, 3 classes, positive float weights, pairs drawn over the columns.

## What was added
`weighted_pair_search_gpu`: one 256-thread block per pair, rows add their weight to a shared `(merged cell, class)` table as a 64-bit integer at a power-of-two scale (`2^floor(61 - log2(W))`, total below 2^62), so the sums are exact integers independent of thread order and reproducible; the threads then sum `jw / W * log(jw * W / (marg_m * marg_y))` over the cells as the host kernel does and a warp-shuffle reduction gives the pair's MI and its occupied cells. Pairs whose table exceeds 4096 entries are left to the host kernel. Pairs stream in blocks, each uploading only the code columns it references (column blocks of the code matrix, halved on OOM by `run_column_blocks`). `_cat_interactions_step` calls `weighted_pair_search_dispatch` in place of the numba kernel; kernel-tuning entry `cat_weighted_pair_backend` (rows x pairs). Before this the weighted search could only run on the host (the GPU pair search warned that sample weights are ignored).

## cProfile (device call, 640 pairs)
3 ms in total: `ndarray.get` 1 ms (2 calls), `cupy.array` uploads 1 ms (7 calls), the block function 2 ms cumulative; nothing else above 0.1 ms.

## nvprof (hardware counters)
`cat_weighted_pair` (launches per call: 1 per pair block, all 640 pairs in one): achieved occupancy 73.3 percent, double-precision efficiency 20.2, DRAM read throughput 0.17 GB/s, global load efficiency 99.7 percent. The six code columns are re-read from L2 by every pair, so DRAM utilization is near zero and the kernel is limited by the shared-memory atomics and the logarithms, not by memory bandwidth (112 GB/s peak, 0.2 percent used).

## Wall clock (numba weighted kernel against the device call with uploads)
| pairs | numba | device |
|---|---|---|
| 10 | 0.52 ms | 1.09 ms |
| 40 | 0.89 ms | 1.23 ms |
| 160 | 3.03 ms | 1.61 ms |
| 640 | 12.83 ms | 2.57 ms (5.0x) |

Equality: joint MI, interaction information and occupied cells match the numba kernel to 1e-12 relative for every pair that fits the table, two runs are identical to the bit, and oversize pairs take the host through the dispatcher (`tests/feature_selection/gpu/test_cat_weighted_pair_device.py`). The pre-sweep floor is 2 million row-pairs (crossover between 40 and 160 pairs of 20,000 rows); the per-host cache replaces it after a sweep.
