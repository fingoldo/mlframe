# `_cat_pair_fe` profile (cat x cat crosses)

## 2026-11-06: factorisation by hash and by counting, not by sorting every row

Standalone, host path, 100000 rows, the five categorical / small-integer columns of the fit-shaped frame (`family_scale._make`), all unordered pairs, `generate_cat_pair_crosses`: wall 1.05 s, of which `np.unique` (argsort of 100000 strings per column and of the 100000 combined integer keys per pair) was 93 percent (35 calls, 28 ms each).

Changed: the string columns go through `sorted_string_codes` (hash-factorise, sort only the distinct values) and the combined pair key through `unique_inverse_dense` (a `bincount` rank table for an integer vector whose range is small; it falls back to `np.unique` above `4 * rows + 1024`). Same sorted unique values and inverse codes, so the same codes, mappings and cell counts: checked against the previous code on a frame with strings, missing values and categoricals (21 crosses equal), and by `tests/feature_selection/fe/test_unique_inverse_dense.py`. Wall 1.05 -> 0.11 s.

Inside the 100000 x 100 GPU fit the family costs about 0.3 s (`_stage_cat_pair` in the per-family walls), so the fit-level gain is small; it grows with the number of categorical columns and the rows.

GPU twin: none. The work is a factorisation of host strings and small-integer keys; a device version would move the 100000-row key vectors across the bus for a counting pass that now takes 3 ms per pair on the host. Not built.
