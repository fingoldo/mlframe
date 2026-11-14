# _orthogonal_xi_fe.py (Chatterjee's Xi screen) with its device twin `_orthogonal_xi_gpu`: profile record (2026-11-14)

Family: `_orthogonal_xi_fe` (the Xi statistic of many float32 columns against one target; the njit host kernels `_xi_columns_njit` / `_xi_columns_radix_njit`) and `_orthogonal_xi_gpu` (the device twin: key kernel, batched cub radix argsort, walk kernel; kernel-tuning entry `fe_xi_columns_device`). Idle machine, GTX 1050 Ti, the module's own sweep inputs (`_sweep_inputs`: 40 float32 columns of n rows, one target).

## cProfile

The host path (`_host`, 40 columns x 100000 rows): 0.079 s, all of it inside the njit kernel `_xi_columns_radix_njit` (7 function calls in total): there is no Python work around the kernel, a call is its kernel.

## Kernels (nvprof, one device call, 40 x 100000)

Kernel launches per call: `xi_sort_keys` 1 launch (0.53 ms), `xi_walk` 1 launch (1.14 ms), and the batched stable argsort of the keys as cub radix-sort launches: 320 launches of 45 us (14.4 ms, 66 percent of the GPU time) plus 40 launches of 20 us (0.8 ms) for the segment bookkeeping, 40 `cupy_copy` launches (0.67 ms) and 40 device-to-device copies (0.65 ms); one host-to-device copy of the block (5 copies, 2.6 ms). So the two kernels of the family are 8 percent of the device time of a call; the rest is the sort of the keys, which is a library call (8 launches per column).

## Occupancy and throughput

nvprof counters (achieved occupancy, DRAM throughput, issue-slot utilization) were not collected on 2026-11-14 (`Internal profiling error 4211` for every kernel today). The kernels take their block size from the occupancy rule of `pyutilz.performance.gpu_hw_launch.block_size_for`; `xi_walk` handles 4 million sorted positions in 1.14 ms, one thread per position, 3.5 G positions/s, and reads a row index, its predecessor's and two small per-class tables (read-only cache) per position, about 12 bytes: 42 GB/s of the card's 112 GB/s. Throughput of the whole device call: 4 million elements (16 MB of values) in 54 ms, 0.3 GB/s of input, because the 64-bit key sort dominates.

## Host-or-device choice

`fe_xi_columns_device` (`kernel_tuner`, swept over the row count; fallback: the host). Measured medians of 7 calls on this machine, 40 columns: 5000 rows host 2.34 ms, device 4.00 ms; 30000 rows host 17.2 ms, device 14.5 ms; 100000 rows host 77.4 ms, device 54.1 ms (1.43x). The device result equals the host value to the last bit (the rank sums are integer counts held exactly in doubles, so the block sums are exact in any order); `np.array_equal` was true at every size.

## Streaming

`xi_columns_device` processes the columns in blocks that fit `_BUDGET_FRACTION` of the free device memory (column blocks) and returns `None` on any device fault so the caller runs the host kernel; the operands may already be resident.
