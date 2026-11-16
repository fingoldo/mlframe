# `_fe_synergy_exhaustive` (affordability decision for the exhaustive C(p,2) joint-MI synergy sweep)

Measured 2026-11-16, GTX 1050 Ti host (sm_61, 112 GB/s), 8 logical cores. Hardware: GPU 84 C, SM 1683/1911 MHz, mem 3504 MHz, no throttle; CPU 2808/2808 MHz, load 25 %.
Bench: the module's own throughput probes, `_measure_exhaustive_throughput` (CUDA `batch_pair_mi_cuda`) and `_measure_exhaustive_cpu_throughput` (`batch_pair_mi_njit_prange`), 64 features of 8 levels, 4-class target, one warm-up call.

## Measured pair-MI throughput

| n x pairs | device (pairs/s) | numba (pairs/s) | ratio |
|---|---|---|---|
| 20000 x 20000 | 22726 | 5542 | 4.1 |
| 100000 x 20000 | 9470 | 473 | 20.0 |
| 100000 x 100000 | 9062 | 333 | 27.2 |

(The first attempt, 500000 pairs at 100000 rows on the numba side, would have taken about 25 minutes and was stopped; it is the cell the prediction exists to avoid.) At 100000 rows the exhaustive sweep of 100 columns (4950 pairs) takes about 0.5 s on the device and 15 s on the host; the module's prediction compares those per-host numbers with the time budget, so it never hardcodes a rate (the 5e4 / 2000 constants are only the cold-cache fallbacks).

## Host profile

The module's own work is the decision arithmetic and the two timed probes; there is no Python loop over rows or pairs (cProfile of the module's code paths: the probe's kernel call and the cache lookup are the only work). The numba probe is the parallel kernel of `batch_pair_mi_gpu`.

## Launches per call and the GPU profile

One `batch_pair_mi_cuda` launch per pair block; its nvprof table, launches per call, occupancy and throughput are in `batch_pair_mi_gpu.md` and `_batch_pair_mi_cuda_shared_fused.md` (the probe's warm-up call on 64 pairs triggers numba's low-occupancy notice: grid of 64 blocks, by design of a warm-up). Device throughput falls from 22.7k to 9.1k pairs/s between 20000 and 100000 rows: the kernel is per-row bound (work per pair is proportional to n).

## Backend choice and memory

Choice: the measured rates (cache keys `batch_pair_mi_exhaustive_throughput` and `batch_pair_mi_exhaustive_cpu_throughput`, written by `get_or_tune`), then the time budget. Memory: pair blocks of `batch_pair_mi_gpu`, never the whole pair list on the device.
