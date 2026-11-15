# `_orthogonal_bootstrap_mi_fe` (bootstrap-LCB MI scorer of the orthogonal-basis candidates) with its carved kernel modules `_bootstrap_rank_mi_njit`, `_bootstrap_rank_mi_gpu`, `_bootstrap_rank_mi_tied_gpu`

2026-11-16. The scorer draws B seeded resamples of the rows (cached per seed and labels), bins every candidate into equi-frequency rank bins of each resample and takes the lower confidence bound of the B plug-in MIs. The work is `bootstrap_rank_mi`: each column is sorted ONCE and all resamples advance over the sorted rows together with their multiplicities (bit-identical to gathering and re-sorting the resample, ties included). Kernels: the numba family of `_bootstrap_rank_mi_njit` (`_bootstrap_rank_mi_f32`, `dense_ranks_f32`, the generic float kernel), the CUDA walk `bootstrap_rank_walk` for fully distinct columns (`_bootstrap_rank_mi_gpu`) and the device path of tied columns (`_bootstrap_rank_mi_tied_gpu`), each behind a kernel-tuning entry (`fe_bootstrap_rank_device`); this record covers the family, the children keep their own records (`_bootstrap_rank_mi_gpu.md`, `_bootstrap_rank_mi_tied_gpu.md`).

## cProfile and timings (`hybrid_orth_mi_bootstrap_fe`, 100 columns, warm)

30000 rows (the size it runs at inside a fit): 0.80-0.84 s wall, 3.1-3.5 CPU-s. 100000 rows: 2.6-2.7 s wall, 13.1-13.4 CPU-s (5 cores busy). cProfile of the 100000-row call: `_bootstrap_rank_mi_f32` (the numba walk, parallel over column tiles) 1.61 s in 2 calls, `dense_ranks_f32` 0.39 s, `_tied_on_device` 0.51 s cumulative (the tied columns), `ascontiguousarray` 0.13 s.

## Kernel-tuning choice, measured both ways

`fe_bootstrap_rank_device` swept on this host (5000 / 30000 / 100000 rows, 60 columns, 10 resamples): the host walk at every size. Forced to the device (`MLFRAME_BOOTSTRAP_RANK_DEVICE=1`) the same call takes 4.56 s against 2.6 s, so the swept choice is right: the walk kernel itself is fast (14.5 ms per call) but the device path needs a sort of every column (1200 radix-sort launches, 49 ms; 10 `cupy_take` 56 ms; 300 int64 copies 37 ms; uploads 46 ms) and the host walk already runs at 5 cores.

## nvprof (GTX 1050 Ti, sm_61), device path forced

Kernel launches per call: about 600 cub radix-sort launches (a sort per column), 150 copies, 5 `cupy_take`, 1 `bootstrap_rank_walk`. `bootstrap_rank_walk`: achieved occupancy 0.04-0.07 (a thread walks a whole resample of a column: few long threads), multiprocessor activity 95 %, global load efficiency 23 % (the walk gathers by sorted row id), device read throughput 2 GB/s of the 112 GB/s card: latency bound. In the default (host) route the device runs only the tied-column path (`cupy_where`, `cupy_equal`, sgemm: occupancy 0.83, SM activity 97 %, 25-58 GB/s). Whole-GPU utilization of the default call is low (the device is busy about 40 ms of 2.6 s); the lever, if one is wanted, is a batched device sort of all columns (the sort is 70 % of the forced device path), not the walk kernel.
