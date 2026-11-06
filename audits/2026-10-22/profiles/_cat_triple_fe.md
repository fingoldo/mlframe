# `_cat_triple_fe` profile (cat x cat x cat crosses)

## 2026-11-06: factorisation by hash and by counting, not by sorting every row

Standalone, host path, 100000 rows, five categorical / small-integer columns, all unordered triples (`generate_cat_triple_crosses`): wall 1.62 s, `np.unique` 89 percent of it (45 calls, 31 ms each).

Changed as in `_cat_pair_fe.md`: `sorted_string_codes` for the three string columns and `unique_inverse_dense` for the Horner-packed joint key and the combined key. Same codes, mappings and cell counts (35 triples equal to the previous code on a frame with strings, missing values and categoricals). Wall 1.62 -> 0.18 s.

Inside the 100000 x 100 GPU fit the family costs about 0.75 s (per-family walls). GPU twin: none, for the reason given for the pair crosses.
