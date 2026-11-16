# `_fe_stability_vote` (cross-fold replay vote of the surviving `unary_binary` recipes)

Measured 2026-11-16, GTX 1050 Ti host (sm_61, 112 GB/s), 8 logical cores. Hardware: GPU 80 C, SM 1695/1911 MHz, mem 3504 MHz, no throttle; CPU 2808/2808 MHz, load 18 %.
Bench: `confirm_recipes_cross_fold` on 100000 rows x 60 columns, 30 `mul` recipes over disjoint operand pairs, 10 bins, K = 5 folds of 20000 rows, quorum 0.6 (26 of 30 recipes fail the vote), min of the warm second run.

## Whole pass, by path

| path | wall per vote |
|---|---|
| host (`MLFRAME_FE_GPU_STRICT=0`) | 0.767 s |
| STRICT-resident (device replay and binning) | 1.869 s |

## Host profile (cProfile, host path, 164117 function calls, 0.858 s)

`_vote_recipe_over_folds` 0.842 s over 30 recipes; `_quantile_bin` 0.590 s over 451 calls (np.quantile partition 0.216 s of 0.308 s), of which `_raw_fold_codes` (the raw operand codes per fold: 300 calls) 0.435 s; `searchsorted_right` 0.210 s. No Python loop dominates; the cost is the equi-frequency binning of 20000-row folds.

## Launches per call and the GPU profile (nvprof, STRICT-resident pass)

About 30000 launches for the 150 (recipe, fold) jobs of the 30 recipes: the radix sort of the sync-free binner is 3608 launches of 12 us (22.8 %) plus 902 of 21 us (9.8 %), 1779 small host-to-device copies (13.8 %), `marginal_mi_entropies` 572 launches of 18 us (5.4 %), 8088 memsets, 2096 device-to-host copies. Every kernel runs for 1 to 30 us on a 20000-row fold: the device path is launch and transfer bound, so whole-GPU utilization is low and the STRICT pass is 2.4 times slower than the host here. Occupancy of these small kernels is not the limit (they are latency-bound); the batched MI kernel's occupancy and throughput are in `_fe_batched_mi.md`. Optimisation lead: batch all recipes of a fold into one (recipes, fold rows) matrix and bin it with the segmented device binner in one call (recipes x folds launches become folds).

## Backend choice and memory

Choice: STRICT-resident forces the device replay (STRICT ignores the tuning cache); otherwise the host. Memory: one fold block of the operand columns and codes at a time (K folds of n / K rows), cached fold frames; no pool-sized device buffers.
