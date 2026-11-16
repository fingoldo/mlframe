# `_mrmr_fe_step_helpers` (FE-step blocks carved out of `_run_fe_step`: order-2 maxT floor, synergy bootstrap, surrogate seeder, cluster aggregates)

Measured 2026-11-16, GTX 1050 Ti host (sm_61, 112 GB/s), 8 logical cores. Hardware: GPU 81 C, SM 1695/1911 MHz, mem 3504 MHz, no throttle; CPU 2808/2808 MHz, load 33 %.
Bench: `compute_pair_maxt_floor` on 100000 rows x 60 int8 code columns (8 levels), binary target, 1770 candidate pairs, 25 permutations, fresh data per call (the floor memo would otherwise answer).

## Whole call

| call | wall |
|---|---|
| first (kernel build, caches) | 1.45 s |
| steady state | 0.033 s |

## Host profile

The block's own Python work is the pair-index construction (`pair_operand_arrays`, triu indices, no tuple list) and the call into `pooled_pair_permutation_null_joint_mi_floor`; the 33 ms steady-state call is dominated by the device kernel below, so a cProfile of the helper shows only the launch and the quantile. The synergy bootstrap, surrogate seeder and cluster-aggregate blocks call `_conditional_gate_fe`, the batched MI backends and the GBM seeder, whose records hold their profiles.

## Launches per call and the GPU profile (nvprof)

Launches per call: one `pair_mi_perm` launch for all 25 shuffles of all 1770 pairs (18.8 ms per launch, 98.3 % of the device time), plus one reduction kernel and about 19 small host-to-device copies (the pair arrays, codes and shuffles; 0.64 ms in total). Achieved occupancy 0.50; device-memory read throughput 0.76 GB/s of 112 GB/s: the kernel works out of shared memory / cache on the pair block, so its cost is the contingency counting, not the memory bandwidth. Whole-GPU utilization of the call: the kernel occupies the card for 18.8 of the 33 ms (57 %); the rest is the shuffle generation and the host quantile. Its own record, `_pair_mi_perm_cuda.md`, has the sweep against the njit kernel.

## Backend choice and memory

Choice: `floor_backend_choice` (kernel-tuning entry of `_pair_mi_perm_cuda`), shape rules, then the cupy twin, then the njit floor. Memory: the code columns of the pool as a column block and the (K, pairs) float result; no tuple list of pairs.
