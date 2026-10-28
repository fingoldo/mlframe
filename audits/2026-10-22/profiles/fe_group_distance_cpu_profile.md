# fe_group_distance_enable: CPU profile inside the all-flags fit (n=20000)

Measured 2026-10-28 with `D:/Temp/gd_allflags.py` (all FE flags on, warm fit on other data). Family cumulative 2.15 s profiled (cProfile inflates small calls; use the shares). Its CPU wall in `fe_family_walls_gpu_vs_cpu_n20000` was 1.88 s; there is no GPU path (same on both modes).

| part | cum s | share of family | note |
|---|---:|---:|---|
| `_wasserstein1` (2400 calls) | 0.66 | 30% | `np.sort` of each group plus `_wasserstein1_sorted_kernel` (0.21 s) |
| `_kl_from_bin_counts` (2400 calls) | 0.15 | 7% | already bincount-based |
| `np.mean` per group (2430 calls) | 0.10 | 5% | the per-group z needs the original row order (float sum), so it stays |
| self time of the generator loop | 0.12 | 6% | per-group Python loop |

Reading: the Wasserstein kernel walks the whole global sorted array for every group, so the cost is O(groups x n) per numeric column. An exact shortcut (shared global breakpoints with prefix sums) would change the float result slightly and is not a drop-in change, so it is not made here. Sorting each group once with a single lexsort would save only the small per-call sorts (about 0.05 s). Next lever for this family: the O(groups x n) kernel, which needs an equivalence design before any change. No code change in this entry.
