# `_ratio_delta_fe` family (`_ratio_delta_fe`, carved `_ratio_delta_screen`): ratio, log-ratio, grouped-delta and lagged-diff emitters

Measured 2026-11-16, GTX 1050 Ti host (sm_61), hardware as in `_ratio_delta_screen_gpu.md` (GPU 80 to 82 C, SM 1683/1911 MHz, no throttle). Bench: ratio pool + grouped delta with the MI gate on 100000 rows x 20 and x 40 float32 columns plus a 50-level group column.

## Host profile (cProfile)

Before, 20 columns: 4.99 s: `pairwise_ratio_with_recipes` 3.28 s (`_gate_ratio_pairs` 2.03 s, `pairwise_ratio_features` 1.20 s), `grouped_delta_with_recipes` 1.71 s of which `_map_group_keys` 1.29 s over 40 calls (a pandas hash join of the 100000 key strings per column and per statistic) and `local_mi_gate` 2.24 s. After: grouped delta gathers the per-group mean / std by once-factorised row codes (bit-identical, test_ratio_delta_fe_identity), the pair redundancy screen is the parallel numba kernel `_screen_njit`: 20 columns 4.94 -> 2.98 s on the host path; 40 columns 6.7 s (host) and 4.4 s (STRICT, device screen). The remaining cost is the MI gate (`local_mi_gate`: 3.1 to 3.5 s of 100 to 1560 candidates, on the batched MI backends) and the candidate assembly.

## Launches per call and the GPU profile

The device screen is one launch for the whole ordered-pair pool; its nvprof table, occupancy (0.50) and throughput are in `_ratio_delta_screen_gpu.md`. The MI gate launches are those of the batched MI backends (`_orth_mi_backends.md`).

## Backend choice and memory

Choice: `ratio_pair_screen_backend` for the screen (STRICT / AUTO engaged); the gate by the batched MI backends. Memory: the ratio pool keeps only the accepted columns (the screen decides on a per-task candidate buffer), the screen's device operand is the (p, n) source matrix.
