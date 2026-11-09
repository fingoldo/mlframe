# `_group_distance_gpu` (device twin of the per-group statistics of `_group_distance_fe`)

Written 2026-11-09. Family `group_distance` had no device code: 162 (group column, numeric column) pairs, each a Python loop over the groups (z of the group mean, KL of the group histogram on the global bins, Wasserstein-1 against the global sample), 0.73 s of wall and 2.6 thread-seconds in the sequential fit at 100000 x 100.

Design (all numeric columns of one group column in one block, `_device_stats` in `_group_distance_fe`, one device call per group column under a lock):

1. `gd_stats`: one pass over the rows, the `G x (1 + bins)` count table and the `G` sums of a block's rows in shared memory, added into the column's tables (integer counts exact, double sums with atomic adds: the last bit depends on the schedule, 1e-16 relative);
2. `gd_sort_keys` + batched stable argsort of a 64-bit key (group id high word, non-finite rows last; order-preserving value code low word): every group's finite values contiguous and ascending;
3. `gd_w1`: Wasserstein-1 by the host's two-pointer merge, parallelised over the merged sequence with a merge-path co-rank (each thread finds its start `(i, j)` on the diagonal by binary search and walks 512 merged elements; the group's total is the atomic sum of the chunks). The first version used one thread per (column, group): 480 threads, achieved occupancy 0.095, 141 ms per call; merge path: occupancy 0.49, 43 ms.

z and KL are formed on the host from the tables with the host's arithmetic (`_kl_from_bin_counts`); the mean is the double sum over the finite count (the host takes the float32 mean: relative difference about 1e-7, inside the 1e-5 of the test). Equivalence: `tests/feature_selection/fe/test_group_distance_device_matches_host.py` (every z / KL / W1 lookup per group key, with ties, NaN rows and a tiny group, 3000 and 20000 rows) plus the four existing group-distance tests. More than 256 groups, or a table that would not fit 40 KB of shared memory, or a non-float32 block: the host loop runs. Blocks of at most a quarter of the free device memory; a fault returns `None`.

Timing, 40 columns x 12 groups, synchronised, mean of 2 (`bench_gd.py` in D:/Temp), host = the W1 loop only (the z and KL loops are small on top): 5000 rows host 26 / device 11 ms; 30000 rows 146 / 31 ms; 100000 rows 507 / 102 ms (all three statistics on the device).

nvprof metrics at 100000 rows (per call): `gd_stats` 3.1 ms, occupancy 0.99, load efficiency 100 %; `gd_sort_keys` 0.8 ms, occupancy 0.90, load efficiency 100 %; `gd_w1` 43 ms, occupancy 0.49, load efficiency 21 % (the merge reads the sorted values of two sequences with data-dependent steps); cub radix sort 14 ms; host-to-device copies 9 ms (the columns, bins and global samples are uploaded per call). Levers left: keep the columns resident between group columns (the same numeric columns are uploaded for each of the four group columns), cut the registers of `gd_w1` for a higher occupancy, sort 32-bit value codes only.

Dispatch: kernel-tuning entry `fe_group_distance_stats_device` (sweep over 5000 / 30000 / 100000 rows, equivalence 1e-9), sweep run with `tune_spec`, read back in a fresh process: device at all three sizes. No environment switch.
