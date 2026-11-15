# `_orthogonal_adaptive_degree_fe` (adaptive-degree orthogonal basis FE) with its carved device module `_orthogonal_adaptive_degree_device`

2026-11-16. The family enumerates every (source column, degree) cell of the basis chosen per column, scores the stacked cells by plug-in MI and keeps the best degree per source. The host path builds each cell as its own vector (`_evaluate_basis_column`, the numba polynomial evaluators of `hermite_fe`) and scores the stacked matrix; the device path (`_orthogonal_adaptive_degree_device`) uploads the columns once, routes the basis per column with the batched device router, builds all cells with the fused kernels of `_gpu_resident_basis` (`clenshaw_basis`, `nonpoly_cells`), checks them with the fused validity pass and scores them with the resident plug-in MI; only the per-cell MI returns and the winning cell of each source is rebuilt by the host path, so values and recipes are the host ones.

The contract is met through three registered relations (`fe_gpu_coverage_inventory`): CPU twin `hermite_fe` (numba evaluators), kernel module `GPU_BACKENDS` -> `_gpu_resident_basis` (which meets the whole contract itself; the test pins that), KTC dispatcher `_basis_cells_ktc` (`fe_basis_cells_device`, swept per host over the row count). The device module's column blocks come from `first_block` / `run_column_blocks` (sized from the room under the pool, halved on an out-of-memory).

## cProfile / timings (`hybrid_orth_mi_adaptive_degree_fe`, 100 columns, warm)

30000 rows (the size it runs at inside a fit): host 1.01 s wall / 0.89 CPU-s. 100000 rows: host 3.04 s / 3.36 CPU-s, strict-resident device 2.18-2.41 s / 2.4 CPU-s. cProfile of the device call (100000 rows): `_gpu_batched_abs_corr` 1.16 s tottime in 32 calls (the host wait for the routing kernels, async attribution), `cupy.get` 0.35 s in 86 reads, `np.partition` 0.11 s, `column_stack` 0.08 s.

## nvprof (GTX 1050 Ti, sm_61), 3 calls = warm-up + 2

Device time of one call about 1.03 s of 2.18 s wall (47 % busy). Kernel launches per call: about 20 `clenshaw_basis` (0.336 s, 33 %), about 23 `nonpoly_cells` (0.274 s, 26 %), about 24 `abs_corr_tiled` (0.071 s), about 58 `cupy_mean`, 46 `cupy_copy`, 18 `radix_select_f32_v3`, and the rest below 2 % each; copies 1.2 %.
- `clenshaw_basis`: achieved occupancy 0.47, multiprocessor activity 98 %, global load efficiency 69 %, device read throughput 1.7 GB/s (0.8-8.9) of the 112 GB/s card: arithmetic bound.
- `nonpoly_cells`: occupancy 0.87, multiprocessor activity 98 %, global load efficiency 42 %, read throughput 2 GB/s: the cells are evaluated in double (`sin`, `exp`, `pow` at the 1/32 FP64 rate of this card) and stored as the criterion dtype; the limit is the FP64 rate, not the bus. Evaluating in float would change the cell values at the 1e-6 level that the host comparison pins, so it is NOT done; recorded as the lever with its cost.
Whole-GPU utilization of the family call: 47 % busy; the idle half is the host routing loop (`bcorr` argmax per basis, 58 mean reductions) between kernel groups.

## Equivalence

The device cells are compared with the host cells by the existing tests of the family (`test_orthogonal_adaptive_degree*`, `test_gpu_resident_basis*`); the kernel module's own record is `_gpu_resident_basis.md`.
