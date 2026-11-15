# Profile record: shifted-product scan (`_offset_product_fe`, CUDA twin `_offset_product_gpu`)

Host: GTX 1050 Ti (sm_61, 4 GiB), 8 logical cores, elevated process (hardware counters). Measured 2026-11-14, medians of 3 after a warm call. Workload: `_auto_tune_sweeps_c._scan_inputs` (a sign-crossing target on random columns), unary outputs of 7 unaries, all (column pair, unary pair) tasks.

## Kernels and choice
Host: `scan_offset_products` in `_numba_utils_offset.py` (njit, `prange` over the tasks). Device: `scan_offset_products` in `_offset_product_gpu.py`, one block per task: the 14 normal-equation sums by fixed-order tree reductions, a 4x4 solve for the shifts, the per-feature bin edges by a bitonic sort of an even-row subsample, then the MI of the shifted product and of the baselines; nothing is stored per candidate. The kernel text uses `{ACC}` (accumulator type) with `--fmad=false`, so the products and sums round as the CPU's do. The backend is read by `_scan_all` through `lookup_offset_scan_backend`: the per-host verdict of the kernel-tuning entry `offset_product_scan` (swept over (scan rows, pool columns) by `_run_sweep_offset_product_scan`, registered with `kernel_tuner(`), else the strict-resident mode decides.
The device scan launches in blocks of tasks when the pool tables do not fit a quarter of the free VRAM; each block uploads only the columns of `U` (and their clips) that its tasks reference, halved on an out-of-memory error by `run_column_blocks`; when the pool fits (the usual case) it is one launch with no gather.

## cProfile (device call, 100,000 rows, 3 columns, 147 tasks)
40 ms cumulative in `scan_offset_products_gpu`; the host side is the five `cupy.array` uploads (3 ms) and nothing else above 0.1 ms.

## nvprof (hardware counters)
`scan_offset_products` (launches per call: 1 for all tasks): 34.4 ms of the 37.2 ms traced GPU time (92 percent; the uploads 7.6 percent). Achieved occupancy 23.3 percent (one block per task with the sort buffer and six joint histograms in shared memory), double-precision efficiency 12.7, DRAM read throughput 15.0 GB/s (13 percent of the card's 112 GB/s peak bandwidth), global load efficiency 60.9 percent. Utilization: the kernel is neither memory- nor arithmetic-saturated; occupancy is limited by shared memory per block (the design of one block per task).

## Wall clock (numba scan against the device call, uploads included)
| rows x pool columns (tasks) | numba | device |
|---|---|---|
| 5,000 x 3 (147) | 31.5 ms | 11.1 ms (2.8x) |
| 20,000 x 3 (147) | 70.3 ms | 15.5 ms (4.5x) |
| 100,000 x 3 (147) | 271.5 ms | 39.7 ms (6.8x) |
| 100,000 x 6 (735) | 1362.1 ms | 172.2 ms (7.9x) |

Outputs: the MI tables agree with the numba scan to 5.6e-17, the shifts to 5.2e-12; blocks of 40 tasks equal the single launch bit for bit (`tests/feature_selection/gpu/test_offset_product_task_blocks.py`). A first version of the task-block launch re-uploaded the pool for every block and was slower (58.6 ms at 100,000 x 3); sizing the block as the whole task list when the pool fits restored 39.7 ms.
