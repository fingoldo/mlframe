# Profile record: modular-scan pair and triple ranking (CPU kernels and the CUDA kernel)

Host: GTX 1050 Ti (sm_61, 4 GiB), 8 logical cores. Workload: integer columns, n=4000 (ranking subsamples 1000 rows), 17 coarse moduli, plug-in residue MI per operator-candidate.

## Kernels

* CPU: `_residue_rank_scores` (pairs, three operators) and `_residue_rank_scores3` (triples, `sum3`) in `_pairwise_modular_fe.py`, njit with `prange` over candidates.
* GPU: `residue_rank_scores_gpu` in `_pairwise_modular_rank_gpu.py`, one 128-thread block per candidate, (residue, class) counts in shared memory with integer atomics, one block reduction per modulus. Same estimate as the CPU kernels.

## Parity

Max abs difference CPU vs GPU 4.4e-15 (pairs p=200, 19,900 candidates) and 5.0e-15 (triples p=100, 161,700 candidates); the selected top-24 pairs and top-100 triples are identical (`D:/Temp/rank_gpu_parity.py`). Test: `tests/feature_selection/gpu/test_modular_rank_gpu.py`.

## Wall clock (synchronized)

| workload | CPU njit | GPU |
|---|---|---|
| pairs, p=200 (19,900 pairs x 3 operators) | 2.54 s | 0.28 s (9x) |
| triples, p=100 (161,700) | 7.84 s | 0.65 s (12x) |
| triples, p=20 (1,140), fresh process, first call | 0.66 s | 0.17 s |
| triples, p=20, warm | 0.041 s | 0.006 s |

The GPU wins from the smallest swept cell (100 operator-candidates); the kernel-tuning-cache sweep (`modular_rank`, axis `n_work`) picked the GPU in every cell, max diff 4.7e-15. The fresh-process GPU call is cheaper than the CPU one because cupy keeps the compiled kernel on disk and numba has to load its cache.

## cProfile of a whole `cheap_modular_scan` (CPU path, before the GPU dispatch, 100 ranked triples)

| p | total | where the time is |
|---|---|---|
| 20 | 9.33 s | `_residue_grid_mi` 8.16 s cumulative over 192 calls (42 ms each; `_mi_classif_batch_numba` 1740 calls); `_residue_rank_scores3` 0.044 s |
| 200 | 31.75 s | `_residue_rank_scores3` 9.65 s (250,000 sampled triples), `_residue_rank_scores` 2.76 s, `_residue_grid_mi` 16.4 s over 372 calls |

cProfile inflates these numba-heavy calls about 7x: the same 17-modulus residue grid costs 7 ms unprofiled (`D:/Temp/grid_vs_direct.py`), not 42 ms, and the real wall times without a profiler (`D:/Temp/scan_wall.py`) are 0.82-0.92 s (12 triples) and 1.46-1.76 s (100 triples) at p=20, 2.3-2.6 s for a pairs-only scan at p=200 with the GPU ranking. Raising the default `max_triples` from 12 to 100 therefore adds about 0.6-0.8 s to a p=20 scan. The ranking no longer dominates once it runs on the device; the exact residue grid is the remaining cost, and it cannot be swapped for a direct plug-in MI (max diff 0.55 nats, argmax differs in 13 of 30 cases, because the pinned estimator quantile-bins the residues).

## Not yet done

* nvprof/ncu counters for `residue_rank` (nvprof needs an elevated shell on this host).
* The exact residue grid (`_residue_grid_mi`): batching across combiners with the pinned per-column binning is possible but gains little at 7 ms per grid; not pursued.
