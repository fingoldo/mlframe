# `_count_freq_interaction_fe` profile (count, frequency and category x numeric encodings)

## 2026-11-06: hash-factorised codes and a bincount instead of np.unique over every row

Standalone, host path, 100000 rows, five categorical / small-integer columns, `count_encode_with_recipes`: wall 0.26 s, `np.unique(return_inverse, return_counts)` 92 percent of it (10 calls, 24 ms each).

Changed: `sorted_string_codes` for the codes and `np.bincount(inverse)` for the counts (the four call sites: the count and the frequency fits, the category x numeric interaction fit and its transform). The encoded columns, the recipes and the interaction output equal the previous code's on a frame with strings, missing values and categoricals. Wall 0.26 -> 0.04 s.

GPU twin: none; the work is a host-string factorisation and a gather.
