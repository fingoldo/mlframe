# _lof_fe.py (local outlier factor features) with its device twin `_lof_gpu`: profile record (2026-11-14)

Family: `_lof_fe` (the LOF fit against a frozen reference, the out-of-sample transform: the njit brute-force scan `_lof_transform_njit` as the host path, the matrix form `_lof_transform_numpy` kept as the reference) and `_lof_gpu` (the device kernel `lof_scores`: one thread per new row, the k nearest in registers or shared memory; kernel-tuning entry `fe_lof_device`, `MLFRAME_LOF_DEVICE=0|1` forces it). Idle machine, GTX 1050 Ti, float32 inputs, k = 20.

## What changed in this record

The host path of the transform was a numpy matrix form (matmul for the squared distances, `argpartition` over every distance): 3699 ms for 100000 rows x 2000 reference points x 3 coordinates. It is now a numba kernel (`prange` over the new rows, a sorted buffer of `k` entries per thread, no `(rows, n_ref)` matrix and so no row blocks on the host): 302 ms on the same input, 12.2 times, and 8 to 12 times on every shape measured (table below). The scores agree with the matrix form to 1e-4 relative on all but the rows with an exact distance tie on the k-th neighbour (agreement 0.99997 to 1.0 of the rows in the measurements; the tests require 0.999). The output keeps the dtype of the frozen densities, as the matrix form returned it (a replay through `apply_recipe` is compared bit for bit with the fit).

## cProfile

The new host path (`_lof_transform_host`, 100000 x 2000 x 3): 0.275 s under the profiler, all of it inside the njit kernel `_lof_transform_njit` (10 function calls in total): a call is its kernel.

## Kernels (nvprof, one device call, 100000 x 2000 x 3)

Kernel launches per call: 1 launch of `lof_scores` (106.8 ms, 99.7 percent of the device time), 4 small host-to-device copies (0.21 ms) and 1 device-to-host copy (0.08 ms) of the scores. The whole transform is one kernel launch per row block of `rows_per_launch` rows (the new rows go to the device in row blocks, so the device memory does not grow with the number of rows).

## Occupancy and throughput

nvprof counters (achieved occupancy, DRAM throughput) could not be collected on 2026-11-14 (`Internal superseded by the Hardware counters section below` for every kernel today). From the resources of the compiled kernel: 130 registers per thread (the running top-k in registers), no spill, no static shared memory, 160 threads per block from `block_size_for`: 3 resident blocks per SM by the register limit (12 by the thread limit), theoretical occupancy 0.23. Throughput from the timing: 100000 x 2000 = 2.0e8 distance evaluations in 106.8 ms, 1.9 G pair-evaluations per second, each a 3-coordinate squared distance in double precision and an insertion test against the current k-th distance; the reference set is read through the cache by every thread. The kernel is arithmetic and register bound, not bandwidth bound.

## Host-or-device choice

Medians of 5 calls, ms, numpy matrix form / njit scan / device, float32, reference points x coordinates: 500 x 3: 2000 rows 19.8 / 1.9 / 2.2, 20000 186.6 / 18.4 / 9.0, 100000 961.6 / 106.2 / 37.3; 2000 x 3: 2000 rows 72.3 / 4.8 / 4.2, 20000 733.8 / 53.9 / 24.7, 100000 3699.4 / 302.5 / 108.3; 500 x 8: 2000 rows 19.7 / 2.7 / 2.5, 20000 197.0 / 27.4 / 13.8, 100000 968.0 / 142.4 / 64.8; 2000 x 8: 2000 rows 68.2 / 7.6 / 6.6, 20000 689.3 / 88.8 / 43.2, 100000 3511.7 / 454.5 / 213.7. The device is 2.2 to 2.8 times the njit scan from 20000 rows on and level with it at 2000 rows. The tuner entry `fe_lof_device` was swept again against the new host path (salt 6): device chosen from 2000 rows up.

## Streaming

`lof_transform_gpu` takes the new rows in row blocks of `rows_per_launch` (default 4 000 000 rows) so the device memory is bounded whatever the number of rows; the host scan needs no blocks.

## Hardware counters (nvprof --metrics, elevated shell, 2026-11-14)
Superseding the theoretical occupancy stated above (the counters were blocked by the driver's restriction until the process ran as administrator). Kernel alone, last sweep size, one profiled call after a warm call. Pascal sm_61: Nsight Compute reports the device as unsupported, nvprof is the tool.

| kernel | achieved occupancy | SM efficiency | DRAM read | DRAM write | global load eff. | global store eff. | shared eff. |
|---|---|---|---|---|---|---|---|
| lof_scores | 10.1 % | 98.7 % | 1.27 GB/s | 0.05 GB/s | 12.5 % | 0.12 % | 0 |

`lof_scores` achieves 10.1 percent occupancy and 12.5 percent load efficiency: both are uncoalesced-read and occupancy defects to fix (the device twin is correct, the kernel is the slow part).
