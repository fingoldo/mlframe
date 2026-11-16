# `_usability_pool_resident` (+ `_usability_pool_resident_pairs`, `_degenerate_combo_gpu`): resident pair-combo MI table of the usability pool

Measured 2026-11-16, GTX 1050 Ti host (sm_61, 112 GB/s), 8 logical cores. Hardware during the runs: GPU 83 C, SM 1683/1911 MHz, mem 3504 MHz, no throttle; CPU 2808/2808 MHz, load 18 %. The device wall varied between runs (5.7 s and 8.4 s for the same call in the last two runs, GPU temperature 83 C): the temperature-dependent clock spread is part of every figure below.
Bench: `score_pair_combos_table_resident` on 100000 rows, 40 pairs (12 uniform columns), medium unary x minimal binary presets (1734 combos per pair), 10 bins, against the njit `score_pair_combos` per pair.

## Whole table, by path

| path | wall for 40 pairs |
|---|---|
| host, njit `score_pair_combos` per pair | 77.1 s |
| resident, before this record | 10.38 s |
| resident, near-constant test and row extremes in one kernel | 5.7 to 8.4 s (6.72 s, 5.72 s in the sequential runs) |

## Host profile (cProfile, resident path, before the change)

`_score_one_pair_combos` 7.6 s cumulative over 40 pairs (2.55 s own time: the cupy reductions waiting on the device), `_write_combo_chunk_mi` 4.31 s over 160 chunks, `binned_mm_mi_from_cm_gpu` 2.17 s, the final `asnumpy` 3.03 s (the queued device work surfacing), `astype` 1.82 s over 960 calls, `min` 1.31 s over 160. The Python loop is not the cost; the device is.

## Launches per call and the GPU profile (nvprof)

Before: `cupy_var_core_float32` 30.6 % of the GPU time (440 launches of 25 ms: cupy's generic `var` over the (450, 100000) combo block read it at 8 GB/s of 112), `radix_select_f32_v5` 20.8 %, `mi_mm_from_values_nek_cm_f32` 17.6 %, `fused_gen_cm_f32` 15.2 %, `cupy_absolute` 6.6 %, three segmented-reduce kernels 9 %. The `var`, `abs`, the segmented `max` and the two `min` / `max` reductions of the MI edge bounds are replaced by one `degenerate_rows` launch per combo chunk (`_degenerate_combo_gpu`): 1.96 ms against 15.8 ms for the old expression on a (450, 100000) float32 block (identical live mask except one row of true constant values at 1e6, which cupy's float32 `var` reported live from rounding noise; row minima and maxima equal), achieved occupancy 0.96 to 0.99, device-memory read throughput 52 to 87 GB/s (mean 82 GB/s, 73 % of the card's 112 GB/s). After (final kernel, 360 launches of each per 40 pairs): `radix_select_f32_v5` 37.0 % (19.3 ms per launch), `mi_mm_from_values_nek_cm_f32` 31.4 % (16.3 ms), `fused_gen_cm_f32` 26.2 % (13.7 ms), `degenerate_rows` 5.3 % (2.75 ms per launch), the whole table 6.31 s: four launches per chunk, whole-GPU utilization near 100 % (the table is device bound). Occupancy and bandwidth of `radix_select_f32_v5`, `mi_mm_from_values_nek_cm_f32` and `fused_gen_cm_f32` are in `_fe_batched_mi.md` and `_gpu_resident_fe.md` (the radix edge selection has no record of its own yet: it is the next lead, 37 % of the table).

## Backend choice and memory

Choice: `fe_usability_pool_combo_mi_table_resident_crossover` (kernel_tuner, sweep over rows, pairs and combos against the njit table). Memory: one combo chunk (`_combo_chunk_cols`, a column block of the VRAM-admitted combos of one pair) of (chunk, n) values at a time; the (npairs, nc) float64 MI table is the only result kept.
