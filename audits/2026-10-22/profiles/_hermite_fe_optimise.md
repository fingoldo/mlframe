# `_hermite_fe_optimise` family (`_hermite_fe_optimise`, `_hermite_fe_optimise_pair`, `_hermite_fe_optimise_parts`): the pair-polynomial optimizer dispatcher

Measured 2026-11-16, GTX 1050 Ti host (sm_61, 112 GB/s), 8 logical cores. One `optimise_hermite_pair` call: Chebyshev basis, degree 3 only, 100 trials, plug-in MI over 20 bins, no multi-fidelity, 100000 rows, a binary target `(a^3 - 2a) * b + noise > 0` (the hard "cubic inner" case of the optimizer benchmarks), warm start on. Median of 3 after a warm call.

## Optimizers end to end

| optimizer | wall per pair | result |
|---|---|---|
| `cupy_kernel` (default, `_cupy_polynom_optimizer`) | 0.19 s | found |
| `cma_batch` (host) | 1.02 s | found |
| `random_batch` (host) | 1.23 s | found |

(the all-njit `numba_kernel` takes 6.9 s for a single-pair search of the same size, see `_numba_polynom_optimizer.md`). The device path is 5 to 6 times the host batch optimizers on this card.

## Host profile (cProfile)

`cma_batch`: 1.07 s for 21985 function calls; `_plugin_mi_classif_batch_rows_njit` 0.657 s (6 calls, the plug-in MI of the candidate columns, 61 percent), `_fill_bf_batch_njit` 0.142 s (the binary functions), `_candidate_h_arrays` 0.087 s (the polynomial evaluation), `_eval_coef_pair_batch` 0.027 s self. `cupy_kernel`: 0.195 s for 3866 function calls; the host-side cost is the 50 `rng.permutation` calls of the noise-floor null (0.033 s), the cupy `sort` of the edge-key table (0.028 s, the host waiting for the queued kernels), the warm-start ALS seed (0.031 s), the identity-baseline MI (0.022 s).

## Launches per call and the GPU profile (nvprof, 4 calls of the default optimizer)

Per call: 6 generations' worth of fused kernels: `qbin_code` 1.44 ms, `bf_apply` 1.21 ms, `mi_from_codes_tile` 0.97 ms, `transpose_f32` 0.89 ms and 2 `sgemm_32x32x32_NT_vec` per generation (1.82 ms each); 24 launches of each big kernel across the 4 calls plus 8 GEMM launches. GPU time per pair about 49 ms of the 190 ms wall; the per-kernel occupancy and bandwidth table (`bf_apply` achieved occupancy 0.87, 49 GB/s; `mi_from_codes_tile` 0.49, 16 GB/s; `transpose_f32` 0.96, 38 GB/s of 112 GB/s) is in `_cupy_polynom_optimizer.md`.

## Backend choice and memory

The optimizer is `fe_optimizer` (default `cupy_kernel`); the host-versus-device scoring of one generation is the tuner entry `polynom_generation_backend` (`_polynom_search_backend`). A generation is a column block of `batch_size` candidate columns of `n` rows (host `_candidate_h_arrays`, device `bf_apply` output), so memory is bounded by the batch, not by the trials. Tracker entry CT11-99.
