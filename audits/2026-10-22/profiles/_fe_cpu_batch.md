# `_fe_cpu_batch` / `_fe_gpu_batch` (+ dispatcher `_fe_batch_dispatch`, tuner `_fe_batch_dispatch_ktc`): edge-binned plug-in MI of a candidate matrix

Measured 2026-11-16, GTX 1050 Ti host (sm_61, 112 GB/s), 8 logical cores. Hardware: GPU 80 C, SM 1695/1911 MHz, mem 3504 MHz, no throttle; CPU 2808/2808 MHz, load 18 %.
Bench: `cpu_fe_batch_mi` against `gpu_fe_batch_mi` on an (n, K) float32 normal matrix and a 10-class target, 10 bins, min of 3 warm runs.

## Whole call, by path

| n x K | host (numba) | device | ratio |
|---|---|---|---|
| 20000 x 200 | 0.072 s | 0.010 s | 7.2 |
| 100000 x 400 | 0.771 s | 0.067 s | 11.5 |
| 100000 x 2000 | 3.983 s | 0.578 s | 6.9 |

The MI vectors agree to 6e-16 to 9e-16. The dispatcher's AUTO rule never reached the device: it read an `fe_batch_backend` cache key that nothing wrote ("the GTX 1050 Ti measured 0.5x for the resident pack-and-batch pattern", an older measurement), so every AUTO call took the host path. New kernel-tuning entry `fe_batch_backend_choice` (`_fe_batch_dispatch_ktc`: `cpu_fe_batch_mi` timed against `multi_gpu_fe_batch_mi` over the candidate work from 2e5 to 8e7 cells at identical MI vectors) chose the device at every swept size; the fallback before a sweep is the device from 2e5 cells; `MLFRAME_FE_BATCH_DEVICE=0|1` forces a side.

## Host profile

The host path is one parallel njit kernel call per column chunk (`plugin_mi_classif_batch_edge_njit` of `_fe_edge_mi`, parallel over columns): a cProfile shows the single kernel entry and the finiteness scan (`np.isfinite(X).all(axis=0)`), no Python loop; the chunk (`_cpu_col_chunk`) bounds the scratch.

## Launches per call and the GPU profile

The host path launches no kernel; the nvprof table, launches per call, occupancy and throughput of the device half are in `_fe_gpu_batch.md`.

## Backend choice and memory

Choice: `fe_batch_backend_choice`; STRICT forces the device. Memory: the host scores column chunks of `_cpu_col_chunk` columns (RAM budget); the device scores VRAM column chunks of `_gpu_k_chunk`; neither holds a scratch sized by the full width.
