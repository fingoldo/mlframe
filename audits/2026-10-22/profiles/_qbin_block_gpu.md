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
