# _group_distance_fe.py (group-distance features) with its device twin `_group_distance_gpu`: profile record (2026-11-14)

Family: `_group_distance_fe` (z of the group mean, KL of the group histogram and Wasserstein-1 to the global sample, per (group column, numeric column); the host loop over the groups and the njit two-pointer kernel `_wasserstein1_sorted_kernel`) and `_group_distance_gpu` (the device twin of the per-group statistics: `gd_stats`, a batched key sort, `gd_w1`; kernel-tuning entry `fe_group_distance_stats_device`). Earlier records: `_group_distance_gpu.md`, `fe_group_distance_cpu_profile.md`. Idle machine, GTX 1050 Ti, the twin's own sweep inputs (40 float32 columns, 12 groups, 20 global bin edges).

## cProfile

The host path (`_host`, 40 columns x 100000 rows): 0.574 s under the profiler, 3365 function calls: the per-cell sorts `numpy.ndarray.sort` 0.216 s (480 cells), the njit Wasserstein kernel `_wasserstein1_sorted_kernel` 0.214 s (480 calls) and the driver's own time 0.136 s. The statistics are cheap next to the sort and the two-pointer merge of every (column, group) cell.

## Kernels (nvprof, one device call, 40 x 100000)

Kernel launches per call: `gd_stats` 1 launch (3.42 ms, 3.9 percent of the device time), `gd_sort_keys` with the batched argsort as 320 cub radix-sort launches of 44 us (14.0 ms, 16.2 percent) and 40 launches of 22 us, `gd_w1` 1 launch (53.2 ms, 61.3 percent), 5 host-to-device copies (9.2 ms, 10.6 percent), the gather (`cupy_take`, `cupy_prepare_array_indexing`) 2.6 ms. The long pole is `gd_w1`: one thread per (column, group) cell runs the host's two-pointer merge serially, so the launch has 480 threads for 40 x 12 cells and fills a fraction of the card.

## Occupancy and throughput

nvprof counters (achieved occupancy, DRAM throughput, issue-slot utilization) could not be collected on 2026-11-14 (`Internal superseded by the Hardware counters section below` for every kernel today). From the launch shape: `gd_w1` runs 480 threads, 15 warps, over 6 SMs (2.5 warps per SM of the 64 an SM holds): occupancy about 4 percent, which is why it takes 61 percent of the call for 4 million elements; a split of each cell's merge into segments is the next step for this kernel and is not done in this record. `gd_stats` reads the 4 million values once in 3.42 ms: 4.7 GB/s of float32 input (shared-memory histogram bound).

## Host-or-device choice

`fe_group_distance_stats_device` (`kernel_tuner`, swept by the row count; fallback: the host). Medians of 7 calls, 40 columns, host / device: 5000 rows 29.1 / 9.4 ms; 30000 rows 171.6 / 41.5 ms; 100000 rows 578.2 / 143.4 ms (3.1x to 4.0x, the device is ahead at every size swept). The results agree to 1e-6 relative with `numpy.allclose` (not bit-equal: the group sums are double atomic adds, 1e-16 relative, schedule dependent).

## Streaming

`group_stats_device` takes the numeric columns in column blocks sized from `_BUDGET_FRACTION` of the free device memory (`block = max(1, min(k, fraction * free // per_col))`), runs the three launches per block and returns `None` on a group count above `MAX_GROUPS`, a non-float32 block or any device fault, so the caller keeps the host loop.

## Hardware counters (nvprof --metrics, elevated shell, 2026-11-14)
Superseding the theoretical occupancy stated above (the counters were blocked by the driver's restriction until the process ran as administrator). Kernel alone, last sweep size, one profiled call after a warm call. Pascal sm_61: Nsight Compute reports the device as unsupported, nvprof is the tool.

| kernel | achieved occupancy | SM efficiency | DRAM read | DRAM write | global load eff. | global store eff. | shared eff. |
|---|---|---|---|---|---|---|---|
| gd_w1 | 55.5 % | 97.0 % | 29.0 GB/s | 0.31 GB/s | 20.5 % | 0 % | 0 |
| gd_stats | 99.0 % | 99.6 % | 7.4 GB/s | 0.13 GB/s | 100 % | 0 % | 0.54 % |

`gd_w1` is 80 percent of the device call and reads at 20.5 percent load efficiency (uncoalesced); occupancy 55 percent, not the 4 percent stated earlier.
