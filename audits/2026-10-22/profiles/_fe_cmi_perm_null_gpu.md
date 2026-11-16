# `_fe_cmi_perm_null_gpu` (conditional permutation null of the CMI redundancy gate)

Measured 2026-11-16, GTX 1050 Ti (sm_61, 112 GB/s). Shape of a call: 100000 rows, x 10 levels, y 5 levels, 8 strata, 200 permutations, Miller-Madow CMI; (floor, mean) returned.

## Launches per call and the GPU profile (nvprof, 11 calls)

Launches per call: about 63 (33 driver launches of raw kernels and cub reductions plus 30 cupy elementwise/scatter launches); the sparse path sorts the (n, nperm) key matrix in one batched device sort. 77 ms per call, 0.55 of the 0.85 s profiled is the device sort.

Metrics (nvprof) of `flat_strata_key_pack`, the in-module kernel: achieved occupancy 0.89, multiprocessor activity 99.9 %, device-memory read throughput 65 to 70 GB/s and write 22 to 23 GB/s (of 112 GB/s), global load efficiency 97 %, store efficiency 100 %. The cupy scatter/copy/bitwise launches around it are 3 to 6 ms each (3 to 4 % of the time apiece).

## Host profile (cProfile, 10 calls)

Device form: 0.85 s for 10 calls, 7481 function calls; `cupy sort` 0.55 s (the batched sort surfaces the GPU time), the host argsort of the stratum order 0.069 s, `_batched_cmi_resident_chunked` 0.087 s. Host twin (`StratifiedNull(debias=True)`, the contingency-table null): 0.036 s for 10 calls, 441 function calls; `_null_njit` 0.019 s, the log-factorial table 0.011 s, the stratum margins 0.003 s: 3.6 ms per call, 21 times the device.

## Backend choice and streaming

Backend choice: the tuner entry `cmi_perm_null_backend` (`perm_null_on_device`, a sweep over the row count 5000 to 200000 of the table null against `conditional_perm_null_gpu`, means of 400 draws compared at the 4-standard-error tolerance (50 %) derived in the module from the measured 12 % seed-to-seed spread of a 200-draw mean). On this host: host at every swept size (the two means agree within that noise). STRICT-GPU keeps routing to the device and does not consult it. The host path was the row-shuffle loop (200 permutations: 0.43 s at 50000 rows, 2.1 s at 200000); it is now the table draw (3 ms, 16 ms), the same null distribution.

Streaming: permutations are scored in column blocks sized by the free VRAM (`_batched_cmi_resident_chunked`), the sparse path holds O(n x nperm) after the `fe_gpu_has_vram_cushion` admission; the table null keeps strata x levels cells per thread and falls back to the shuffle loop above 4 million cells.
