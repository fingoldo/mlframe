# Profile record: RelaxMRMR interaction pair loop (`_relaxmrmr_pair_loop`, new device twin `_relaxmrmr_pair_gpu`)

Host: GTX 1050 Ti (sm_61, 4 GiB), 8 logical cores, elevated process (hardware counters). Measured 2026-11-14, medians of 3 after a warm call. Workload: a selection of 5 to 30 features of 6 levels (10 to 435 pairs) over 20,000 rows, a 6-level candidate, a 3-class target.

## What was added
The loop sums, over every pair of already-selected features, `(cmi_y[i] + cmi_y[j] - I(x; z | y)) - (marg[i] + marg[j] - I(x; z))` with `z` the pair's composite code; it ran only as the numba `prange` loop `_pair_interaction_terms_njit` (or a serial loop below 200,000 row-pairs). `relaxmrmr_pair` (in `_relaxmrmr_pair_gpu.py`) scores the two Miller-Madow quantities of every pair on the device: one 256-thread block per pair, a shared `(x, z, y)` integer table (integer atomics) with its six marginal tables derived in shared memory, the terms of `_cmi_mm_njit` and `_mi_mm_njit` summed by the threads, the occupied-cell counts and a warp-shuffle block reduction; the host combines them with the per-feature terms. Pairs whose tables exceed 9000 shared entries, and pairs the host skips as undersampled, stay with the numba loop. The selected-feature code rows go up as bytes (the selection is one column block; pairs go in one launch). `pair_interaction_sum` calls `pair_terms_dispatch`; kernel-tuning entry `relaxmrmr_pair_backend` (rows x pairs).

## cProfile (device call, 435 pairs)
The call is 2.2 ms: six small uploads, one launch, two reads; no host loop above 0.1 ms.

## nvprof (hardware counters)
`relaxmrmr_pair` (launches per call: 1 for all pairs): 0.90 ms of the 1.02 ms traced GPU time (88 percent; uploads 11.5 percent). Achieved occupancy 59.6 percent, double-precision efficiency 39.2, DRAM read throughput 0.10 GB/s (the code rows of the selection live in L2; 0.1 percent of the card's 112 GB/s peak bandwidth), global load efficiency 98.9 percent. The kernel is limited by the shared-memory atomics and the logarithms (39 percent of the double-precision rate), not by memory.

## Wall clock (numba `prange` loop against the device call with uploads)
| pairs | numba prange | device |
|---|---|---|
| 10 | 0.6 ms | 0.8 ms |
| 45 | 1.4 ms | 0.9 ms (1.6x) |
| 190 | 4.4 ms | 1.5 ms (2.9x) |
| 435 | 9.9 ms | 2.2 ms (4.5x) |

Per-pair agreement with the numba pieces is 1e-12 relative; `pair_interaction_sum` returns the same sum with the device on and off, falls back to the numba loop for pairs that do not fit, and the test counts the device call (`tests/feature_selection/gpu/test_relaxmrmr_pair_device.py`).
