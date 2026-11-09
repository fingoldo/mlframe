# `_orthogonal_meta_scorer_fe.fingerprint_signal` (host vectorisation, no device kernel)

2026-11-09. Family `orth_mi_meta` was listed as a gap (1.9 s, 4 % on the device). The sequential cProfile showed `fingerprint_signal` at 0.7 s of it, all of it pandas per-column Python on a frame the GPU would not help with: the statistics are O(n x p) sums that a device would spend longer uploading than computing.

Measured at 100000 rows x 100 float32 columns, one process, mean of 2 (loop = the old per-column pandas path, kept behind `_VECTORISED = False` as the reference of the equivalence test):

| piece | loop | vectorised |
|---|---:|---:|
| Spearman and symmetric Pearson proxy (`_dcor_proxy`, 500-row subsample) | 0.25 s | 0.01 s (average ranks, one matrix pass) |
| mean |Pearson| with the target | 0.11 s | 0.10 s (dominated by the float64 conversion) |
| skewness and kurtosis | 0.35 s | one centred-power pass in double (bias-corrected G1 / G2 of pandas, constant column 0.0) |
| distinct values per column (`nunique`) | 0.33 s | parallel radix pass (`radix_order_f32` + changes counted) |
| whole `fingerprint_signal` | 1.11 - 1.24 s | 0.61 s |

Equivalence (`tests/feature_selection/fe/test_meta_fingerprint_vectorised.py`): every entry agrees with the loop to 1e-6 relative (pandas accumulates float32 columns in float32; the vectorised pass is the more accurate); a frame with a NaN, a near-constant column, or a non-numeric dtype takes the old path. The meta-scorer rule tests pass (47).

What is left in `orth_mi_meta` is `_dispatch_scorer` after the fingerprint (0.5 s: the chosen scorer runs); the remaining fingerprint cost is the float64 conversion of the block and the 100 x 100 BLAS correlation (`inter_x_max_corr` 0.08 s).
