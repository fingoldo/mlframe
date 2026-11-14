# _fe_sufficient_summary.py (sufficient-summary early stop of the FE search) with its pair test `_fe_sufficient_summary_pairs`: profile record (2026-11-14)

Family: `_fe_sufficient_summary` (the stop rule: ridge residual of the selection, entropy and max-T tests of the raw columns against the residual, the call into the pair test) and `_fe_sufficient_summary_pairs` (the joint-MI test of the residual with every pair of coarse raw columns against a Bonferroni chi-square ceiling: njit block kernel, fused device kernel, tuner entry `pair_residual_test`). The pair test is the cost of the stop rule; its own record is `_fe_sufficient_summary_pairs.md` (nvprof 10.x counters of the earlier campaign). Idle machine, GTX 1050 Ti, 2026-11-14, the module's own synthetic inputs (`_make_pair_residual_inputs`: 5 coarse levels, 10 residual codes, a planted pair).

## cProfile

The host path of the pair test (`best_pair_streamed`, njit, p = 400 columns, n = 5000 rows, 79800 pairs): 0.251 s under the profiler, all of it inside 3 calls of the njit block kernel `_pair_mi_block_cpu` (0.084 s each: one diagonal block pair and two rectangles at `BLOCK_COLS`), 39 function calls in total: no Python work around the kernel. The rest of `sufficient_summary_reached` (ridge residual, max-T over the raw columns) is a few columns and small tables per call and does not show next to it.

## Kernels (nvprof, one device call of the pair test, p = 400, n = 5000)

Kernel launches per call: 3 launches of the fused kernel `pair_mi_block_v2` (pair scoring, histogram, MI reduction and the argmax over partners in one kernel per block pair), 15.3 ms in all (5.1 ms each, 96.8 percent of the device time); the remaining device work is a final argmax over the per-first-column results (two cub launches, 0.06 ms) and 5 small host-to-device copies (0.43 ms). The generic dispatcher path this replaced issued several kernels per pair chunk and was about 149 times slower on this shape (earlier campaign, `_fe_sufficient_summary_pairs.md`).

## Occupancy and throughput

nvprof counters (achieved occupancy, DRAM throughput) could not be collected on 2026-11-14 (`Internal profiling error 4211` for every kernel today). The earlier campaign measured the same kernel family with counters: achieved occupancy 0.984 (0.978 to 0.991) for `pair_mi_block`, see `_fe_sufficient_summary_pairs.md`. Throughput from today's timings: 79800 pairs x 5000 rows = 399 million pair-rows scored in 15.3 ms of kernel time, 26 G pair-rows/s; each pair-row is two int8 code reads and a shared-histogram update.

## Host-or-device choice

`pair_residual_test` (`kernel_tuner` in `_fe_sufficient_summary_pairs`, swept over n_samples x n_pairs; fallback by size). Medians of 5 calls, njit / fused device: 400 columns x 5000 rows (79800 pairs) 261.7 ms / 17.5 ms (15.0x); 400 x 20000 728.6 / 68.5 ms (10.6x); 101 columns x 5000 rows (5050 pairs) 17.2 / 1.8 ms (9.6x). The best pair MI is identical in the two backends to the printed 6 digits (0.647856, 0.629746, 0.637444), the tolerance of the tuner is `pair_equiv_tolerance()` per the kernel's real type.

## Streaming

`best_pair_streamed` walks the column pairs in column blocks of `BLOCK_COLS` columns (`_block_cols_for` sizes the block from the free device memory), so the device never holds more than two blocks and their pair table; the same function drives the host and the device backend.
