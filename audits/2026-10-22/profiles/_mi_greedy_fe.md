# `_mi_greedy_fe` family (`_mi_greedy_fe`, `_mi_greedy_device`): generic unary / binary transform constructor ranked by MI uplift

Measured 2026-11-16, GTX 1050 Ti host (sm_61, 112 GB/s), 8 logical cores. Hardware at the start of the run: GPU 79 C, SM 1708/1911 MHz, mem 3504 MHz, throttle none; CPU 2808/2808 MHz, load 22%.
Bench `greedy_mi_fe_construct(X, y, seed_cols_count=8, top_k=8)` on 100000 rows x 5 columns (fixture of `test_mi_greedy_device_matches_host`), min of 3 after a warm-up.

## Whole pass, by path

| path | wall per construct |
|---|---|
| host (njit MI kernels) | 0.350 s |
| device (`use_device` True) | 0.066 s |

In the 100000 x 100 STRICT fit cProfile the stage is 0.106 s cumulative: `score_candidates_device` 0.036 s, `div_safe` 0.009 s over 20 calls, `flush` 0.008 s over 2, `_greedy_score_and_select` 0.013 s, `iter_candidates` 0.005 s over 153. No Python loop dominates.

## Launches per call and the GPU profile (nvprof, device path, 4 constructs)

Kernel share of GPU time: `cupy_var_core_float32` 20.9 % (3.5 ms per construct), `radix_select_f32_v5` 16.1 % (2.7 ms: edge selection), `cupy_concatenate_same_size` 13.5 %, `cupy_nan_to_num` 9.7 %, `mi_hist_split_f32` 8.0 % (1.35 ms per call), then about 2 % each for the candidate transform elementwise kernels (subtract, absolute, log1p, add, multiply, sign, where). Launches per call: about 4 heavy kernels per scoring block plus about 25 elementwise launches per construct per transform family, roughly 400 launches per construct, all tiny elementwise launches of 7 to 16 us each, so launch count rather than occupancy is the cost: the heavy kernels hold the card for 14 + 11 + 9 + 6.5 + 5.4 ms of the 68 ms GPU time over 4 constructs. Occupancy and throughput of `radix_select_f32_v5` and `mi_hist_split_f32` are recorded in `_hermite_fe_mi.md`; utilization of the stage is low by design (66 ms wall, GPU busy about a third of it).

## Backend choice and memory

Choice: `use_device` (STRICT / AUTO by fit size). Memory: one candidate block at a time (column-block streaming), bounded by the block width, not by the candidate count.
