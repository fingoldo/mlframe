# Profile record: pair-polynomial search on the device (`_cupy_polynom_optimizer`, host/device choice `_polynom_search_backend`)

Host: GTX 1050 Ti (sm_61, 4 GiB, 112 GB/s, double rate 1:32), 8 logical cores, elevated process (hardware counters). Measured 2026-11-15, medians of 3 after a warm call. Workload: one `run_cupy_kernel_search` over a Chebyshev pair of degree 3, 300 trials = 3 generations of 100 candidates, six binary functions, 100000 rows, 20 bins, binary target.

## What the module is
The default optimizer of the hermite / orthogonal pair search. One generation is two cuBLAS GEMMs (`B_a @ C_a^T`, `B_b @ C_b^T`), then for each binary function the fused `bf_apply` kernel (function and finiteness flag in one pass), the radix-edge binner (`radix_select_f32_v5`, a transpose to column-major, `qbin_code`) and the tile MI kernel (`mi_from_codes_tile` on byte codes). The all-host twin is `_numba_polynom_optimizer.run_numba_kernel_search`; the host generation scorer is `_eval_coef_pair_batch`. A generation is one column block of 100 candidates (the (n, P) matrices exist for one binary function at a time: about 130 MB at 100000 rows), so memory does not grow with the number of generations.

## cProfile (one search, 3 generations)
0.155 s for 5745 function calls. 0.086 s sit in the cupy `sort` of the 21 x 100 edge-key table inside `_edge_keys` (18 calls): the call returns only after the queued kernels of the generation have run, so it is the host waiting for the device, not sorting work (the sort kernels are 23 microseconds each in the GPU trace). The remaining Python is 0.011 s of `_edge_keys` itself, 0.009 s of dtype casts and 0.007 s of the one blocking copy of the generation's scores.

## nvprof (hardware counters), per search
Launches per generation: 2 GEMMs, then per binary function 1 `bf_apply`, 1 `radix_select_f32_v5`, 1 `transpose_f32`, 1 `qbin_code`, 1 `mi_from_codes_tile` (5 x 6 = 30), plus a handful of small table kernels; 108 launches of the five big kernels in 3 generations.

| kernel | share of GPU time | per launch | achieved occupancy | other counters |
|---|---|---|---|---|
| `radix_select_f32_v5` | 29 % | 2.12 ms | not sampled here | the edge selection of the shared binner |
| `qbin_code` | 19 % | 1.43 ms | | shared-memory edge table, byte codes |
| `bf_apply` | 16 % | 1.21 ms | 0.87 | DRAM read 49 GB/s, global load efficiency 100 % |
| `mi_from_codes_tile` | 13 % | 0.97 ms | 0.49 | DRAM read 16 GB/s, global load efficiency 27 % (byte codes: a warp reads 32 consecutive bytes, one 32-byte sector of the 128-byte line) |
| `transpose_f32` | 12 % | 0.89 ms | 0.96 | DRAM read 38 GB/s, global load efficiency 86 % |
| `sgemm_32x32x32_NT_vec` | 8 % | 1.80 ms | | 2 per generation |

Whole-GPU throughput: the GPU time of a search is 133 ms of a 149 ms wall (89 percent busy), so the search is device bound; the kernels that read their operand once (`bf_apply`, `transpose_f32`) run at 49 and 38 GB/s of the card's 112 GB/s.

## What was changed on this path (2026-11-15)
- `col_all_finite`, then folded into `bf_apply`: six binary functions plus the finiteness check in one kernel (about 70 ms of cupy elementwise launches -> 21.7 ms; values identical to the cupy chain, `--fmad=false`).
- `qbin_code`: sorted edge-key table staged in shared memory, binary search, grid-stride over column slabs: 2.6 -> 1.43 ms per launch, same codes.
- Byte codes from the binner into `mi_from_codes_tile`: 1.24 -> 0.97 ms.
- Search wall 0.247 -> 0.149 s (-40 percent).

## Host or device (kernel-tuning entry `polynom_generation_backend`)
Before, `cupy_kernel` took the device whenever cupy imported. The sweep times one generation of 100 candidates on both paths and requires the same top candidates (`selection_view`): 5000 rows host 65.7 ms / device 20.5 ms (3.2x), 20000 rows 210 ms / 20.2 ms (10x), 100000 rows 1056 ms / 50.4 ms (21x). The pair search asks the entry before taking the device route and falls back to `random_batch` when the host is chosen.
