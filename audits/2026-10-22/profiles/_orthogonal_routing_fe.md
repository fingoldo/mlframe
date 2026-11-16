# `_orthogonal_routing_fe` (conditional basis routing: cells of source column x pre-transform x basis x degree, scored by MI and |corr|; device twin `_orthogonal_routing_device`)

Measured 2026-11-16, GTX 1050 Ti host (sm_61, 112 GB/s). Hardware: GPU 78 C, SM 1695/1911 MHz, mem 3504 MHz, no throttle; CPU 2808/2808 MHz, load 18 %.
Bench: 10 float32 columns, 100000 rows, 4 bases x 2 degrees x all pre-transforms (640 cells), the host path (`_enumerate_routing_cells` + `_stack_candidates` + `_mi_classif_batch` + `_routing_scores`) against `routing_scores_on_device`.

## Whole call, by path

| path | wall |
|---|---|
| host | 1.222 s |
| device (`_orthogonal_routing_device`) | 0.209 s (5.8 times) |

## Host profile (cProfile, host path, 1.27 s, 8497 function calls)

0.603 s waiting on the worker threads of the cell build (82 lock waits), `numpy._var` 0.165 s over 47 calls (the cell standardisation), `_routing_scores` 0.363 s cumulative, ufunc reductions 0.099 s, float casts 0.068 s, quantile partitions 0.059 s. No Python loop over rows.

## Launches per call and the GPU profile

The device twin's nvprof table, launches per call (50 to 60), occupancy (`clenshaw_basis` 0.48) and throughput are in `_orthogonal_routing_device.md`. Choice: `fe_basis_cells_device` (`_basis_cells_ktc`, fallback 5000 rows, `MLFRAME_BASIS_CELLS_DEVICE=0|1`).

## Memory

Device: one column block of source columns, `_gpu_k_chunk(n)` cells per block. Host: all cells stacked (640 cells x 100000 rows x 4 bytes = 256 MB at this size), which is why the device path is also the lower-memory one for wide frames.
