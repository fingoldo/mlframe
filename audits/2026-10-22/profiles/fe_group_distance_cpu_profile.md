# fe_group_distance_enable: CPU profile inside the all-flags fit (n=20000)

Measured 2026-10-28 with `D:/Temp/gd_allflags.py` (all FE flags on, warm fit on other data). Family cumulative 2.15 s profiled (cProfile inflates small calls; use the shares). Its CPU wall in `fe_family_walls_gpu_vs_cpu_n20000` was 1.88 s; there is no GPU path (same on both modes).

| part | cum s | share of family | note |
|---|---:|---:|---|
| `_wasserstein1` (2400 calls) | 0.66 | 30% | `np.sort` of each group plus `_wasserstein1_sorted_kernel` (0.21 s) |
| `_kl_from_bin_counts` (2400 calls) | 0.15 | 7% | already bincount-based |
| `np.mean` per group (2430 calls) | 0.10 | 5% | the per-group z needs the original row order (float sum), so it stays |
| self time of the generator loop | 0.12 | 6% | per-group Python loop |

Reading: the Wasserstein kernel walks the whole global sorted array for every group, so the cost is O(groups x n) per numeric column. An exact shortcut (shared global breakpoints with prefix sums) would change the float result slightly and is not a drop-in change, so it is not made here. Sorting each group once with a single lexsort would save only the small per-call sorts (about 0.05 s). Next lever for this family: the O(groups x n) kernel, which needs an equivalence design before any change. No code change in this entry.

## 100000 x 100 every-family GPU fit and standalone call: a device twin is not worth writing (2026-11-06)

`module_walls.py` on `_group_distance_fe` (one warm fit, wall 90.8 s): `hybrid_group_distance_fe` 6.59 s inclusive, of which `generate_group_distance_features` 4.77 s (128 pairs: 4 group columns x 32 numeric), `_wasserstein1_all_groups` 1.83 s summed over the pool threads, `_group_distance_global_stats` 0.68 s, `_global_bin_index` 0.24 s, `_kl_from_bin_counts` 0.22 s (2560 calls). In the family table of the same fit the stage is 6.9 s inside `mid_b`, which runs next to the other cascades, so its pool threads compete for the cores.

The same call alone, one thread (`D:/Temp/gd_prof.py`, 100000 rows, 4 group columns with 20 groups, 32 numeric columns): 2.09 s, of which `_pair_columns` self time 0.46 s (the per-group loop), the parallel W1 kernel 0.30 s, the 32 global sorts 0.21 s, the radix order 0.14 s, the broadcast of the three lookups 0.14 s. Nothing in it is a long serial chain: the compute is about 2 s on one thread, the rest of the in-fit 4.8 s is contention with the concurrent stages.

Decision: no GPU kernel. The ceiling of a device twin is the standalone 2 s (less the 0.4 s of upload and the per-group host loop it would keep), on a stage whose cascade is not the longest after early_a. The sort-based pieces (global sort, radix order) are already vectorised host code. Revisit only if `mid_b` becomes the critical cascade by more than this stage's share.
