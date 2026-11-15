# Profile record: joint MI of categorical pairs (`_cat_pair_joint_njit`, device twin `_cat_pair_joint_gpu`)

Host: GTX 1050 Ti (sm_61, 4 GiB), 8 logical cores. Measured 2026-11-14 on an idle machine, medians of 5 after a warm call. Workload: six categorical columns of 12 levels, 20,000 rows, a 3-class target, pairs drawn over the columns.

## What was added
`pair_joint_mi_gpu`: one 256-thread block per pair, the rows fill a shared `(radix_i * radix_j, k_y)` integer count table (integer atomics, so order does not matter), the threads sum the plug-in MI terms over the cells
(the expression of `_plug_in_mi_njit`; the cell numbering does not enter, so no first-appearance renumbering). A pair whose table exceeds 32 KB of shared memory returns NaN and `pair_joint_mi_dispatch` fills it from the numba pass.
Pairs stream in blocks, each uploading only the code columns it references (halved on OOM by `run_column_blocks`). Kernel-tuning entry `cat_pair_joint_backend`, swept by rows x pairs. `_cat_pair_fe` calls the dispatcher.

## cProfile (dispatcher, 640 pairs, device path)
The call is 3 ms in total: `cupy.array` uploads 1 ms (6 calls), `ndarray.get` 1 ms, the launch itself is asynchronous; nothing else exceeds 0.1 ms.

## nvprof / nsys
`cat_pair_joint_mi`: launches per call: 1 per pair block (all 640 pairs in one). nvprof counters fail on this machine today (superseded by the Hardware counters section below), so occupancy is theoretical: 256 threads, launch bounds 256,
dynamic shared memory 4 bytes x cells x classes (up to 32 KB, one resident block per SM at the largest table, 8 at the 12 x 12 x 3 tables of the sweep). Kernel alone, 640 pairs x 20,000 rows: 0.78 ms; effective read
throughput 197 GB/s of code traffic, above the card's 112 GB/s because the six columns (480 KB) are re-read from L2 by every pair, not from DRAM.

## Wall clock (call with uploads)
| pairs | numba | device |
|---|---|---|
| 10 | 0.37 ms | 1.00 ms |
| 40 | 0.83 ms | 1.10 ms |
| 160 | 2.51 ms | 1.40 ms |
| 640 | 9.41 ms | 2.01 ms (4.7x) |

The device wins from about 160 pairs; the swept choice on this host is host at 10 and 40 pairs, device at 160 and 640. Equality: 1e-12 against the numba pass (`tests/feature_selection/gpu/test_cat_pair_joint_device.py`).

## Hardware counters (nvprof --metrics, elevated shell, 2026-11-14)
Superseding the theoretical occupancy stated above (the counters were blocked by the driver's restriction until the process ran as administrator). Kernel alone, last sweep size, one profiled call after a warm call. Pascal sm_61: Nsight Compute reports the device as unsupported, nvprof is the tool.

| kernel | achieved occupancy | SM efficiency | DRAM read | DRAM write | global load eff. | global store eff. | shared eff. |
|---|---|---|---|---|---|---|---|
| cat_pair_joint_mi | 72.6 % | 99.4 % | 0.72 GB/s | 0.34 GB/s | 99.5 % | 25.0 % | 0.92 % |

Shared efficiency 0.92 percent: the integer count table is hit by a handful of cells, per-warp private tables are the lever.
