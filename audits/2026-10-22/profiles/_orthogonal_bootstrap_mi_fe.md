# `_orthogonal_bootstrap_mi_fe` (bootstrap-LCB MI scorer of the orthogonal-basis candidates) with its carved kernel modules `_bootstrap_rank_mi_njit`, `_bootstrap_rank_mi_gpu`, `_bootstrap_rank_mi_tied_gpu`

2026-11-16. The scorer draws B seeded resamples of the rows (cached per seed and labels), bins every candidate into equi-frequency rank bins of each resample and takes the lower confidence bound of the B plug-in MIs. The work is `bootstrap_rank_mi`: each column is sorted ONCE and all resamples advance over the sorted rows together with their multiplicities (bit-identical to gathering and re-sorting the resample, ties included). Kernels: the numba family of `_bootstrap_rank_mi_njit` (`_bootstrap_rank_mi_f32`, `dense_ranks_f32`, the generic float kernel), the CUDA walk `bootstrap_rank_walk` for fully distinct columns (`_bootstrap_rank_mi_gpu`) and the device path of tied columns (`_bootstrap_rank_mi_tied_gpu`), each behind a kernel-tuning entry (`fe_bootstrap_rank_device`); this record covers the family, the children keep their own records (`_bootstrap_rank_mi_gpu.md`, `_bootstrap_rank_mi_tied_gpu.md`).

## cProfile and timings (`hybrid_orth_mi_bootstrap_fe`, 100 columns, warm)

30000 rows (the size it runs at inside a fit): 0.80-0.84 s wall, 3.1-3.5 CPU-s. 100000 rows: 2.6-2.7 s wall, 13.1-13.4 CPU-s (5 cores busy). cProfile of the 100000-row call: `_bootstrap_rank_mi_f32` (the numba walk, parallel over column tiles) 1.61 s in 2 calls, `dense_ranks_f32` 0.39 s, `_tied_on_device` 0.51 s cumulative (the tied columns), `ascontiguousarray` 0.13 s.

## Redesign of the device route (same day, owner: do the batched-sort work instead of noting it)

Before: the walk kernel handled FULLY distinct columns only, sorted every column with its own sort launch (1200 launches per call), took each column out with `take_along_axis` and walked one thread per (column, resample) (1000 threads, occupancy 0.05, 14-320 ms). Continuous float32 columns of 100000 rows nearly always hold a few equal neighbours, so almost no column reached the device and the default stayed on the host; forced onto the device the family call took 4.56 s against 2.6 s on the host.
Now:
- ONE sort per column block: the 64-bit key is the column index above the order-preserving 32-bit key of the value (`-0.0` made `+0.0`), the sorted array holds every column's rows one after the other, the tie flags come from the sorted keys (the high word keeps columns apart);
- `bootstrap_rank_walk_blocks`: a block per (column, resample) walks the sorted rows in chunks of 256: the draws before a row are the exclusive prefix sum of the multiplicities (warp shuffles + one shared scan), a run of equal values takes the bin of the draws before its FIRST row (a running maximum of the prefixes over the run starts, which only grow), the (bin, class) table is a shared histogram; the old kernel stays as the reference;
- nearly distinct columns (up to 1 % equal neighbours) are walked on the device by default (`MLFRAME_BOOTSTRAP_RANK_TIES_DEVICE=0` restores the fully-distinct rule): the rank-binner tie rule makes the order inside a run irrelevant, MI equal to the host walk to 1.8e-15 on every column.

Measured, `bootstrap_rank_mi` alone, 100 columns, 10 resamples, host / device: 10000 rows 0.078 / 0.087 s, 30000 rows 0.22 / 0.108 s, 100000 rows 0.755 / 0.396 s, 300000 rows 3.11 / 1.32 s (maxdiff 1.2-1.8e-15). `fe_bootstrap_rank_device` re-swept: device at every swept size (5000 / 30000 / 100000). Family call at 100000 rows: 2.4 s wall / 5.5 CPU-s on the device route against 2.5 s / 12.8 CPU-s on the host (the CPU time is more than halved; the rest of the call is basis generation).

### nvprof (GTX 1050 Ti, sm_61), batched route

Kernel launches per call: 8 radix-sort launches for the block (the one sort), 1 `bootstrap_rank_walk_blocks`, a handful of elementwise kernels (keys, ties, order) and the uploads; was 1200 sort launches. `bootstrap_rank_walk_blocks`: 156 ms per 100 x 100000 x 10 call, achieved occupancy 0.60, multiprocessor activity 98 %, global load efficiency 20 % (the multiplicity gather by sorted row id is random: `counts_bt[b * n + row]`), device read throughput 25 GB/s of the 112 GB/s card; the sort is 44 ms per call (17 % of the device time). Open lever: a block per COLUMN looping over the resamples would read the 10 multiplicities of a row as one 40-byte line instead of 10 random sectors.

### cProfile (default route, 100000 rows)

`_bootstrap_rank_mi_f32` (host walk) no longer dominates: the columns are finished on the device and the host kernel runs only for the columns above the tie limit; `dense_ranks_f32` 0.39 s remains for the tied-column path.
