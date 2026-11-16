# `polynom_pair_fe` (per-pair polynomial feature search: pair dispatch, row subsampling, survivor commit)

Measured 2026-11-16, GTX 1050 Ti host (sm_61, 112 GB/s), 8 logical cores. Hardware: GPU 83 C, SM 1683/1911 MHz, mem 3504 MHz, no throttle; CPU 2808/2808 MHz, load 20 %.
Bench: `run_polynom_pair_fe` on 100000 rows x 6 columns (15 pairs, circle target in (c0, c1)), Hermite basis, 3 iterations x 20 optimisation steps, `n_jobs=1`, min of the warm second run.

## Whole pass, by optimizer

| `fe_optimizer` | wall per pass |
|---|---|
| `cma` (host numba optimizers) | 48.99 s |
| `cupy_kernel` (default, device search) | 16.74 s (16.10 s in the cProfile run) |

## Host profile (cProfile)

The pass is worker-thread bound: 16.3 s total, `_pp_dispatch_pair_evals` 16.23 s, of which the 15 `run_in_big_stack_thread` joins hold 16.2 s (cProfile does not see the worker threads); the orchestration that is this module's own, `_pp_reduce_pair_results`, is 0.056 s. The per-pair search is profiled in `_cupy_polynom_optimizer.md` and `_numba_polynom_optimizer.md`.

## Launches per call and the GPU profile (nvprof, `cupy_kernel` pass)

Launches per pair: 108 of each heavy kernel (1620 over 15 pairs): `radix_select_f32_v5` 31.3 % of the GPU time (5.11 s, 3.16 ms per launch: the edge selection of the quantile binner), `qbin_code` 19.3 %, `bf_apply` 13.6 %, `mi_from_codes_tile` 13.0 %, `transpose_f32` 11.5 % (1.9 ms per launch), cuBLAS sgemm 8.8 %. The GPU is busy for about 16.3 s of the 16.7 s wall: whole-GPU utilization is near 100 % (the pass is device bound, not launch bound). Occupancy and throughput of these kernels are recorded in `_cupy_polynom_optimizer.md` and `_hermite_fe_mi.md`. Optimisation leads from the table: the radix edge selection per generation (a third of the pass) and the `transpose_f32` pass (a tenth).

## Backend choice and memory

Choice: `fe_optimizer` (default `cupy_kernel`; the per-generation host-versus-device scoring is `polynom_generation_backend`). Memory: the shared row subsample (`subsample_n` rows) of the pair operands plus the optimizer's generation block; the full-length columns are evaluated once per surviving pair.
