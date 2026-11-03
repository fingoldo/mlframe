# _qbin_block_gpu (batched device quantile binning)

Used by the candidate pool of `greedy_cmi_fe_construct` (one pool of engineered columns binned before the greedy steps). The per-column binner `_sync_free_qbin_codes` costs a full sort plus about fifteen small kernels per column.

100 columns x 100000 rows, float32, GTX 1050 Ti, one process, cache cleared before each run, same-process back to back:

| path | ms |
|---|---|
| per column `resident_qbin_codes` | 246 |
| `resident_qbin_codes_block` | 146 |
| of which one `cp.sort(axis=1)` of the block | 77 |

Codes are bit-identical to the per-column binner for continuous, tied, mass-point, binary, two-value, constant and integer columns at 2, 5, 10 and 16 bins (`tests/feature_selection/fe/test_qbin_block_gpu.py`): a code is the number of kept edges `<= x`, which is what `searchsorted(side="right")` on the sorted kept edges returns.

Measured trap: stacking the host columns as an `(n, k)` matrix (`np.stack(axis=1)`) cost 93 ms and cancelled the whole gain; the `(k, n)` layout uploads each column contiguously and sorts along the last axis (blocked 246 -> 241 ms before the layout change, 246 -> 146 after).

Remaining: the compare-and-count pass is about 45 ms of the 69 ms above the sort; other per-column loops (`_pair_usability_tasks`, `_build_binned_agg_recipes`, `_cheap_mi_group_selection`) can take the same block entry point.

## group_mi_block_gpu (the strict-resident group pre-selection of the binned-aggregate family)

`_cheap_mi_group_selection` under the strict-resident path scored each candidate on its own (sort, searchsorted, bincount and three host reads per column). `group_mi_block_gpu` sorts the block once and hands every column's edges to the existing fused `binned_mi_from_values_gpu` kernel. Values equal the host `_cheap_mi_with_y` to rtol 1e-4 with the same ranking (`test_group_mi_block_matches_the_per_column_cheap_mi`); a failed pass keeps the per-column device loop.

## 2026-11-03: count kernel for the codes

Stage sample of binned_numeric_agg (100000 x 100): 4 samples at `sync_free_qbin_codes_block:50`, the `(kept[:, :, None] <= x[:, None, :]).sum(axis=1)` broadcast
(a (columns, edges, rows) boolean temporary per 16 columns, then its reduction). An elementwise kernel counts the kept edges <= (double) x per value: same codes
(test_qbin_block_gpu, 6 passed); 100 columns x 100000 rows 0.119 -> 0.078 s. A float32 edge table keeps the broadcast.
