# _orthogonal_copula_mi_fe.py (copula-transformed MI screen) with its device twin `_orthogonal_copula_gpu`: profile record (2026-11-14)

Family: `_orthogonal_copula_mi_fe` (rank-uniformise a column, bin it against the target's bin codes, plug-in MI with the Miller-Madow correction; the njit host kernel `_copula_columns_radix_njit`) and `_orthogonal_copula_gpu` (the device twin: a batched device argsort of a block of columns and the kernel `copula_joint`; kernel-tuning entry for the host-or-device choice, `MLFRAME_COPULA_DEVICE=0|1` forces it). Earlier record of the twin: `_orthogonal_copula_gpu.md`. Idle machine, GTX 1050 Ti, 2026-11-14, the twin's own sweep inputs (40 float32 columns of n rows, one target).

## cProfile

The host path (`_host`, 40 columns x 100000 rows): 0.052 s under the profiler, all of it inside the njit kernel `_copula_columns_radix_njit` (7 function calls in total): a host call is its kernel.

## Kernels (nvprof, one device call, 40 x 100000)

Kernel launches per call: `copula_joint` 1 launch (1.53 ms, 9.9 percent of the device time of 15.5 ms; one thread per sorted position with a shared histogram per block), the batched argsort as 160 cub radix-sort launches of 39 us (6.2 ms, 40.2 percent) plus 40 launches of 11 us for the segment bookkeeping, the gather (`cupy_take`, `cupy_prepare_array_indexing`) 2.6 ms, 40 device-to-device copies (0.65 ms), 2 host-to-device copies (2.5 ms, 16.4 percent). The kernel proper is a tenth of the call; the sort and the gather around it are the rest.

## Occupancy and throughput

nvprof counters (achieved occupancy, DRAM throughput) could not be collected on 2026-11-14 (`Internal superseded by the Hardware counters section below` for every kernel today). `copula_joint` runs one thread per sorted position (4 million threads in blocks sized by `block_size_for`, the occupancy rule of the code base), so the grid fills the card; its 4 million elements in 1.53 ms are 2.6 G elements/s; the bytes it moves per element were not counted here, so no GB/s figure is given for it. Whole device call: 4 million elements in 15.5 ms, 0.26 G elements/s, of which the sort and the gather are most.

## Host-or-device choice

Medians of 7 calls, host njit / device, ms: 5000 rows 1.94 / 4.31; 30000 rows 11.43 / 15.41; 100000 rows 46.8 / 33.64 (1.39x). The device is ahead somewhere between 30000 and 100000 rows on this card (the sweep sizes bracket it); the tuner entry picks per host, and `MLFRAME_COPULA_DEVICE` forces a side. The two agree to `numpy.allclose` at every size (not bit-equal: the MI is formed from float sums).

## Streaming

`copula_columns_device` takes the columns in column blocks that fit `_BUDGET_FRACTION` of the free device memory and returns `None` on any device fault so the caller runs the host kernel.

## Hardware counters (nvprof --metrics, elevated shell, 2026-11-14)
Superseding the theoretical occupancy stated above (the counters were blocked by the driver's restriction until the process ran as administrator). Kernel alone, last sweep size, one profiled call after a warm call. Pascal sm_61: Nsight Compute reports the device as unsupported, nvprof is the tool.

| kernel | achieved occupancy | SM efficiency | DRAM read | DRAM write | global load eff. | global store eff. | shared eff. |
|---|---|---|---|---|---|---|---|
| copula_joint | 73.7 % | 99.5 % | 15.1 GB/s | 0.08 GB/s | 44.1 % | 0 % | 34.7 % |

Load efficiency 44 percent in `copula_joint`.
