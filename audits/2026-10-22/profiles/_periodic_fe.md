
## 2026-11-03: modular gate-first pool built as a row matrix

100000 x 100, periods (7, 12, 24, 30, 365): 1500 candidates. Phase walls of the warm `hybrid_modular_fe_with_recipes` (one process, second call): generate 0.71 s,
family codes 1.00 s (device coder 0.60 s, the DataFrame-to-array copy and transpose the rest), bootstrap baseline 0.71 s, family max-T 0.42 s, survivor scoring 0.25 s.

The gate now fills the `(K, n)` candidate matrix directly (`modular_candidate_block`, same names / order / values as `generate_modular_features`) and codes its rows in place
(`family_codes_of_rows`, same codes and counts as `_family_codes`). Warm wall 3.29 -> 2.90 s, cold 6.02 -> 5.66 s; same six columns appended.

Left: the device coder uploads 1500 float32 candidates (600 MB) and copies the int16 codes back for the max-T pass. Computing the residues on the device from the 100 uploaded
source columns would remove the upload; the max-T pass would have to take resident codes to remove the copy back.

## 2026-11-03: modular candidate codes computed on the device

`_modular_codes_gpu.modular_family_codes_gpu`: the source columns are uploaded once in double precision, one kernel writes mod / sin / cos of every period into the
float32 rows the resident coder reads (`_quantile_codes_gpu.quantile_codes_resident_f32`, same edges and code kernel), blocks of sources within 20% of the pool.
Only rows whose 21 order statistics hold at most `n_bins` distinct values come back to the host for the exact few-level check.

1500 candidates x 100000 rows: pool + codes 1.29 s (host pool, device coder) -> 0.83 s; codes equal to the host pool's in every row. Warm family wall 2.90 -> 2.54 s
(3.29 s before the row-matrix change), cold 5.66 -> 5.07 s, same six columns. `MLFRAME_MODULAR_CODES_DEVICE=0|1` forces the path; any device fault keeps the host pool.
