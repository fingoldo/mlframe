# `_numba_polynom_optimizer` (pair-polynomial search as one `@njit(parallel=True)` kernel; the host twin of `_cupy_polynom_optimizer`)

Measured 2026-11-16, GTX 1050 Ti host (sm_61, 112 GB/s), 8 logical cores. Workload: one `run_numba_kernel_search` over a Chebyshev pair of degree 3 (4 + 4 coefficients), 300 trials (batch 20, 15 iterations), the six default binary functions, plug-in MI over 20 bins, 100000 rows, a binary target depending on the product of the operands.

## Host profile (cProfile, one search)

6.94 s per search (median of 3 after a warm call), 135 function calls: all 6.97 s are inside `_optimize_all_pairs_kernel`, the njit kernel; the Python around it is below the timer resolution. The kernel's outer `prange` runs over PAIRS, so this single-pair call (the only kind any caller makes) runs on one thread; the per-candidate basis evaluation is a scalar Horner recompute per candidate instead of a batched GEMM, which is the design mismatch the dispatcher comment in `_hermite_fe_optimise_pair` records (`numba_kernel` is not recommended for single-pair search and stays an explicit opt-in).

## The device twin, same workload

`_cupy_polynom_optimizer.run_cupy_kernel_search` (default optimizer, `fe_optimizer="cupy_kernel"`): 0.149 s per search, 133 ms of it GPU time (89 percent busy); see `_cupy_polynom_optimizer.md` for the launches per generation (2 GEMMs, then per binary function `bf_apply`, `radix_select_f32_v5`, `transpose_f32`, `qbin_code`, `mi_from_codes_tile`), the nvprof table (achieved occupancy 0.87 for `bf_apply`, 0.49 for `mi_from_codes_tile`, 0.96 for `transpose_f32`; DRAM read 49 / 16 / 38 GB/s of the card's 112 GB/s) and the throughput. The device twin is 47 times faster than this kernel for a single pair on this host.

## Choice and streaming

Choice: the optimizer is selected by `fe_optimizer` (default `cupy_kernel`); the host-versus-device scoring of one generation is the tuner entry `polynom_generation_backend` (`_polynom_search_backend`). Streaming: a search keeps one candidate scratch of `batch_size` (20) columns of `n` rows per pair, a column block bounded by the batch, not by the number of trials; the multi-pair kernel scales that by the pairs in flight (one per thread).
