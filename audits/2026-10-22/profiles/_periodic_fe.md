
## 2026-11-03: modular gate-first pool built as a row matrix

100000 x 100, periods (7, 12, 24, 30, 365): 1500 candidates. Phase walls of the warm `hybrid_modular_fe_with_recipes` (one process, second call): generate 0.71 s,
family codes 1.00 s (device coder 0.60 s, the DataFrame-to-array copy and transpose the rest), bootstrap baseline 0.71 s, family max-T 0.42 s, survivor scoring 0.25 s.

The gate now fills the `(K, n)` candidate matrix directly (`modular_candidate_block`, same names / order / values as `generate_modular_features`) and codes its rows in place
(`family_codes_of_rows`, same codes and counts as `_family_codes`). Warm wall 3.29 -> 2.90 s, cold 6.02 -> 5.66 s; same six columns appended.

Left: the device coder uploads 1500 float32 candidates (600 MB) and copies the int16 codes back for the max-T pass. Computing the residues on the device from the 100 uploaded
source columns would remove the upload; the max-T pass would have to take resident codes to remove the copy back.
