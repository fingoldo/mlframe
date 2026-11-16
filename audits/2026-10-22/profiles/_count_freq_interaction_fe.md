# `_count_freq_interaction_fe` profile (count, frequency and category x numeric encodings)

## 2026-11-06: hash-factorised codes and a bincount instead of np.unique over every row

Standalone, host path, 100000 rows, five categorical / small-integer columns, `count_encode_with_recipes`: wall 0.26 s, `np.unique(return_inverse, return_counts)` 92 percent of it (10 calls, 24 ms each).

Changed: `sorted_string_codes` for the codes and `np.bincount(inverse)` for the counts (the four call sites: the count and the frequency fits, the category x numeric interaction fit and its transform). The encoded columns, the recipes and the interaction output equal the previous code's on a frame with strings, missing values and categoricals. Wall 0.26 -> 0.04 s.

GPU twin: none; the work is a host-string factorisation and a gather.

## 2026-11-16: contract measurement

100000 rows, five string columns (20 / 200 / 2000 / 30 / 8 levels) and five float32 numeric columns: `count_encode_with_recipes` + `frequency_encode_with_recipes` + `cat_num_interaction_with_recipes` (25 category x numeric residual columns) take 0.49 s together. cProfile (0.55 s): `cat_num_interaction_fit` 0.235 s over 25 calls, `factorize_array` 0.114 s, arrow-to-numpy 0.064 s, `_cell_sum_cnt_kernel` (njit) 0.047 s over 150 calls, string null checks 0.038 s. Host path; no GPU twin (host strings). Hardware during the run: GPU 77 C, SM 1695/1911 MHz, no throttle; CPU 2808 MHz.
