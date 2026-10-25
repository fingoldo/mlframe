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

## GPU profiler counters (nvprof on the GTX 1050 Ti, elevated shell, kernel `residue_rank`, two launches: pairs p=200 and triples p=100)

Launch statistics (`--print-gpu-trace`): one launch per call, grid (19,900, 1, 1) for the pairs and (161,700, 1, 1) for the triples, block (128, 1, 1), 1 KB static plus 384 B dynamic shared memory; kernel time 264 ms and 570 ms for version 1, equal to the wall times, so the call is the kernel.

| metric | v1 (64-bit values, 40 regs) | v4 (default: 32-bit values, 32 regs, no spill) |
|---|---|---|
| registers per thread / local bytes | 40 / 0 | 32 / 0 |
| achieved occupancy | 0.749 (register-limited: 12 blocks of 128 per SM) | 0.997 |
| sm_efficiency | 99.7% | 98.7% |
| executed IPC / issue slot utilization | 2.38 / 55.3% | 2.21 / 50.0% |
| dram read / write throughput | 1.9 GB/s / 86 MB/s | 2.0 GB/s / 496 MB/s |
| warp execution efficiency | 86.6% | 94.0% |
| shared memory efficiency | 3.9% | 0.35% |
| gld_efficiency | 99.5% | 99.1% |
| stall: synchronization / exec dependency / memory dependency | 33.7% / 31.9% / 12.8% | 44.5% / 34.9% / 8.1% |
| local load / store transactions | 1.0 M / 1.7 M | 4.6 M / 1.3 M |

## Kernel versions (all kept; `MLFRAME_MODULAR_RANK_KERNEL` selects one)

Interleaved A/B in one process, 15 repeats, 32-bit values, median seconds (`D:/Temp/rank_ab.py`, `D:/Temp/rank_versions.py`):

| version | what changed | regs / local bytes | pairs p=200 | triples p=100 |
|---|---|---|---|---|
| 1 | shared-memory tree reduction (about ten barriers per operator-modulus) | 40 / 0 | 0.206 | 0.586 |
| 2 | warp-shuffle reduction | 38 / 0 | 0.199 | 0.558 |
| 3 | v2 with `__launch_bounds__(128, 16)` | 32 / 16 (spills) | 0.194 | 0.526 |
| 4 | 32-bit row offsets, running best in shared memory, cap kept | 32 / 0 | 0.192 | 0.517 |
| 5 | v4 without the cap | 38 / 0 | not timed | not timed |
| 6 | v4 with one count matrix per warp (four copies) | 32 / 8 | 0.204 | 0.552 |
| 7 | int8 residue table per column and modulus, no division in the row loop | 32 / 0 | 0.201 | 0.534 |
| 8 (default) | v7 with one shared atomic per row (the residue marginal is summed from the joint counts) | 32 / 0 | 0.170 | 0.459 |

Versions 4, 7 and 8 in the same interleaved run (v4 0.195 s / 0.554 s, v7 0.201 s / 0.534 s, v8 0.170 s / 0.459 s). All eight versions are tested equal to the njit kernels in both value widths (`tests/feature_selection/gpu/test_modular_rank_gpu.py`) and on negative values.

The 32-bit value mode (used when no product or sum can overflow an int) is the one consistent gain: pairs 0.24 s -> 0.18 s and triples 0.61 s -> 0.56 s against 64-bit values on version 1. Every kernel version is within about 12 percent of the others. Register granularity on Pascal is 8, so any `__launch_bounds__` minimum of 13 to 16 blocks forces 32 registers; the way to avoid the spill is a body that needs at most 32 by itself (version 4), not a weaker cap.

What the result says: raising occupancy from 0.75 to 1.0, removing the barriers' share, removing the register spills and splitting the shared atomics per warp each moved the wall by at most a few percent. The nvprof stall percentages (synchronization 44 percent) are serialization-distorted (CUDA.md section 6) and did not predict the wall. Two hypotheses were then tested as less work rather than a leaner kernel. Removing the division with a per-column residue table (version 7) cut the kernel time by only 5 to 8 percent (nvprof trace: pairs 159 ms -> 152 ms, triples 456 ms -> 422 ms; building the table takes 1.5 ms), so the division was not the limiter once the values were 32-bit. Halving the shared atomics (version 8: one atomic per row, the residue marginal summed from the joint counts) cut the wall by 13 percent on the pairs and 17 percent on the triples, so the atomics are part of the limit; it is now the default. Against version 1 with 64-bit values the default is 29 percent faster on the pairs and 25 percent on the triples.

## Counters of the default (version 8, nvprof, kernel `residue_rank` only, two launches)

achieved occupancy 0.998, sm_efficiency 97.7%, executed IPC 1.60, issue slot utilization 37.1%, warp execution efficiency 91.4%, DRAM read 5.1 GB/s and write 0.9 GB/s, shared efficiency 1.2%, global load efficiency 74.2% (int8 table reads), stalls: synchronization 43.1% (serialization-distorted), execution dependency 31.5%, memory dependency 15.2%; 32 registers per thread, `local_bytes` 0, table build 1.48 ms (cupy `mod` and cast kernels, one launch of `residue_rank` per call). The residue table costs p x 17 x n bytes of int8 (3.4 MB at p=200, 170 MB at p=10000, refused above 1 GiB).

## Not yet done

* Warp-aggregated atomics (`__match_any_sync`) need sm_70 or newer; on a Turing or newer host the `ncu` sections in CUDA.md section 9 apply.
* The exact residue grid (`_residue_grid_mi`): batching across combiners with the pinned per-column binning is possible but gains little at 7 ms per grid; not pursued.
