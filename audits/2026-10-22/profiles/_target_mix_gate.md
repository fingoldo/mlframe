# `_target_mix_gate` (structure pre-check of the feature-base target mix)

Measured 2026-11-16, GTX 1050 Ti host (sm_61, 112 GB/s), 8 logical cores. Hardware: GPU 83 C, SM 1683/1911 MHz, mem 3504 MHz, no throttle; CPU 2808/2808 MHz, load 19 %.
Bench: `target_has_structure` on 100000 rows x 100 float32 normal columns and a pure-noise target (the worst case: both screens run to the end and the answer is False), min of 2 warm runs.

## Whole check

0.318 s (host-forced run, `MLFRAME_FE_GPU_STRICT=0`) and 0.316 s (STRICT): the device codes are the default path either way, so the check costs about a third of a second at 100000 x 100 and does not grow with the row count (it scores 50000 evenly spaced rows).

## Host profile (cProfile, 0.337 s)

`pair_interaction_screen` 0.217 s cumulative, `quantile_codes_columns` 0.082 s (the device sort of the column block 0.079 s), `main_effect_z_columns` 0.069 s, `target_has_structure` own time 0.023 s (the finiteness / distinct-value checks per column). No Python loop dominates.

## Launches per call and the GPU profile (nvprof)

GPU time 497 ms over three calls (warm-up and two timed): cub segmented / merge sort kernels 39.0 % (132 launches, 1.5 ms) and 28.0 % (12 launches, 11.6 ms: the per-column quantile sort), `pair_residual_t` 6.2 % (3 launches of 10.3 ms), `cupy_power` 5.7 %, `quantile_codes_right` 3.3 % (9 launches), host-to-device copies 4.3 % (58 copies). Launches per call: about 44 sort launches, one `pair_residual_t` and three code launches. The sort dominates, so the check is bounded by the card's sort rate; occupancy and throughput of `pair_residual_t` are in `_pair_residual_screen.md` and of the quantile coder in `_gain_stats_codes_gpu.md`.

## Backend choice and memory

Choice: the helpers' own (`main_effect_z_columns` returns None to send the statistics to the host loop; `pair_interaction_screen` tuned). Memory: 50000 rows x the kept columns' codes, at most `MAX_COLS` columns in the pair screen.
